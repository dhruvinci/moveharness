"""V3 -- LLM-as-judge SEMANTIC correctness scoring (MoveBench v0).

Uses gemini-2.5-flash as a semantic judge. For each converted prediction in
results/predictions_all/, we temporally align predicted segments to the v0 GT
macro segments (greedy max-IoU, threshold 0.3 -- SAME alignment the strict T1
scorer uses, so the two numbers are comparable), then ask the judge to rate the
SEMANTIC correctness of the predicted (position + technique labels) vs the GT
(position + sub_position + rolled-up subsegment labels) on a 0/1/2 scale:

    2 = semantically correct (incl. paraphrase / canonical-vs-surface form)
    1 = partially correct (right family / right position but wrong technique, etc.)
    0 = wrong

Abstain-aware: if a matched prediction's position is an abstention token
(unknown/occluded/unclear) the segment is recorded as ABSTAIN and excluded from
the scored denominator (explicit abstention is not punished as a wrong commit),
mirroring the strict/semantic scorer's abstain handling.

GT-anchored mean: unmatched GT segments (no prediction overlaps at IoU>=0.3) are
a MISS and score 0. The per-system mean semantic-correctness is the mean over
the GT-anchored set (all 32 GT segs minus any whose only matching prediction
abstained), normalized to [0,1] by dividing the 0/1/2 score by 2. This is
directly comparable to the GT-anchored strict macro-F1 in real_scores.md.

Privacy: prompts carry ONLY ontology position/technique tokens + timestamps. No
user ids, no file paths, no raw human-annotator free-text notes. Outputs are
anonymized aggregate stats. Terminology is MMA / "label dataset"; never the raw
domain name.

Robustness: concurrency <=4 in-flight, exponential backoff on 429/5xx/rate
limit, time-boxed, never crashes the run (skip+log failures).
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import random
import re
import threading
import time

# ----------------------------------------------------------------------------- env
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
_REPO = os.path.normpath(os.path.join(_BENCH, os.pardir, "repos", "vlm-experiments"))
ENV_PATH = os.path.join(_REPO, ".env")
GT_PATH = os.path.join(_BENCH, "data", "v0", "mb_v0_0001.json")
PRED_DIR = os.path.join(_HERE, "predictions_all")
OUT_MD = os.path.join(_HERE, "vlm_llm_judge.md")
OUT_JSON = os.path.join(_HERE, "vlm_llm_judge.json")
STRICT_MD = os.path.join(_HERE, "real_scores.md")

IOU_THRESHOLD = 0.3
ABSTAIN_POSITIONS = {"unknown", "occluded", "unclear"}
MODEL = "gemini-2.5-flash"
MAX_INFLIGHT = 4
WALL_CLOCK_BUDGET_SEC = 35 * 60  # time-box (~35 min, under the 40 min cap)
MAX_RETRIES = 5


def load_env():
    key = None
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        os.environ.setdefault("GOOGLE_API_KEY", key)
        os.environ.setdefault("GEMINI_API_KEY", key)
    return key


# ----------------------------------------------------------------------------- data
import sys

sys.path.insert(0, _BENCH)
from movebench.loader import load_sample, load_prediction  # noqa: E402
from movebench.evaluate import _gt_segments_for_t1, _pred_segments  # noqa: E402


def _gt_rich_segments(sample: dict) -> list[dict]:
    """GT macro segments with the extra context the judge sees (position +
    sub_position + rolled-up subsegment technique labels)."""
    out = []
    for s in sample.get("macro_segments", []):
        rolled = sorted(
            {l for sub in s.get("subsegments", []) for l in (sub.get("labels") or [])}
        )
        if not rolled:
            rolled = list(s.get("labels") or [])
        # drop protocol-only annotations from the technique view (kept transparent)
        out.append(
            {
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "position": s.get("position"),
                "sub_position": s.get("sub_position"),
                "labels": [l for l in rolled if l not in ("pre_match", "match_start")],
            }
        )
    return out


def iou(a, b) -> float:
    s = max(a["start_sec"], b["start_sec"])
    e = min(a["end_sec"], b["end_sec"])
    inter = max(0.0, e - s)
    if inter <= 0:
        return 0.0
    uni = (a["end_sec"] - a["start_sec"]) + (b["end_sec"] - b["start_sec"]) - inter
    return inter / uni if uni > 0 else 0.0


def align(gt_segs, pred_segs):
    """For each GT segment, pick the max-IoU predicted segment (>=threshold).
    Returns list of (gt, pred_or_None, iou)."""
    out = []
    for g in gt_segs:
        best_p, best_i = None, 0.0
        for p in pred_segs:
            i = iou(g, p)
            if i > best_i:
                best_p, best_i = p, i
        out.append((g, best_p if best_i >= IOU_THRESHOLD else None, round(best_i, 3)))
    return out


# ----------------------------------------------------------------------------- judge
JUDGE_SYS = (
    "You are an expert judge for a fine-grained MMA / grappling video "
    "understanding label dataset. You compare a model's PREDICTED labels for a "
    "video segment against the GROUND-TRUTH (GT) labels for the same time "
    "window. Both use a controlled ontology of body-position states (e.g. "
    "standing, guard, mount, side_control, back_control, half_guard, turtle, "
    "scramble, takedown, sweep, escape) and short snake_case technique tokens. "
    "Judge SEMANTIC correctness, accepting paraphrase and canonical-vs-surface "
    "synonyms (e.g. 'Mount'=='mount', 'guard_pass'=='passing_guard', "
    "'rear_naked_choke'=='rnc', 'de_la_riva'=='de_la_riva_guard'). Reward "
    "getting the dominant body position right most; technique labels are "
    "secondary. Be strict about clearly different positions (e.g. predicting "
    "'mount' when GT is 'guard' is wrong)."
)

JUDGE_INSTR = (
    "Rate the PREDICTED labels' semantic correctness vs the GT on this scale:\n"
    "  2 = semantically correct: the dominant position matches (incl. synonym/"
    "paraphrase) AND the technique labels are consistent / overlapping in "
    "meaning.\n"
    "  1 = partially correct: the position is right OR in the right family but "
    "techniques are off, OR position is a near-miss but the action described "
    "overlaps.\n"
    "  0 = wrong: the dominant position is clearly different and techniques do "
    "not correspond.\n"
    "Respond with ONLY a compact JSON object: "
    '{"score": 0|1|2, "reason": "<=12 words"}. No other text.'
)


def build_user_prompt(gt, pred) -> str:
    def fmt(seg):
        return json.dumps(
            {
                "start_sec": seg.get("start_sec"),
                "end_sec": seg.get("end_sec"),
                "position": seg.get("position"),
                "sub_position": seg.get("sub_position"),
                "labels": seg.get("labels") or [],
            },
            ensure_ascii=False,
        )

    gt_view = fmt(gt)
    pred_view = json.dumps(
        {
            "start_sec": pred.get("start_sec"),
            "end_sec": pred.get("end_sec"),
            "position": pred.get("position"),
            "labels": pred.get("labels") or [],
        },
        ensure_ascii=False,
    )
    return (
        f"GROUND TRUTH segment:\n{gt_view}\n\n"
        f"PREDICTED segment (same time window):\n{pred_view}\n\n"
        f"{JUDGE_INSTR}"
    )


_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            from google import genai

            _client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
        return _client


_usage_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0,
          "failures": 0, "retries": 0}


def _is_retryable(exc) -> bool:
    s = str(exc).lower()
    return any(
        t in s
        for t in ("429", "rate", "quota", "resource_exhausted", "500", "503",
                  "internal", "unavailable", "deadline", "timeout", "overloaded")
    )


def judge_call(gt, pred):
    """Returns (score:int|None, reason:str, raw:str). score None on hard failure."""
    from google.genai import types

    prompt = build_user_prompt(gt, pred)
    cfg = types.GenerateContentConfig(
        system_instruction=JUDGE_SYS,
        temperature=0.0,
        max_output_tokens=2048,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_client().models.generate_content(
                model=MODEL, contents=prompt, config=cfg
            )
            um = resp.usage_metadata
            with _usage_lock:
                _usage["calls"] += 1
                _usage["prompt_tokens"] += getattr(um, "prompt_token_count", 0) or 0
                _usage["output_tokens"] += getattr(um, "candidates_token_count", 0) or 0
                _usage["total_tokens"] += getattr(um, "total_token_count", 0) or 0
            raw = (resp.text or "").strip()
            score, reason = parse_judge(raw)
            return score, reason, raw
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                with _usage_lock:
                    _usage["retries"] += 1
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                continue
            break
    with _usage_lock:
        _usage["failures"] += 1
    return None, f"FAIL: {str(last_exc)[:120]}", ""


def parse_judge(raw: str):
    if not raw:
        return None, "empty"
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            sc = obj.get("score")
            if isinstance(sc, (int, float)) and int(sc) in (0, 1, 2):
                return int(sc), str(obj.get("reason", ""))[:80]
        except Exception:  # noqa: BLE001
            pass
    m2 = re.search(r"\b([012])\b", raw)
    if m2:
        return int(m2.group(1)), "parsed-fallback"
    return None, f"unparseable: {raw[:40]}"


# ----------------------------------------------------------------------------- run
def main():
    t0 = time.time()
    key = load_env()
    if not key:
        raise SystemExit("no API key")
    print(f"[env] API key loaded (len={len(key)})")

    sample = load_sample(GT_PATH)
    gt_rich = _gt_rich_segments(sample)
    n_gt = len(gt_rich)
    print(f"[data] {n_gt} GT macro segments")

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*.json")))
    print(f"[data] {len(pred_files)} prediction files")

    # Build the full task list (all matched GT->pred pairs across all systems).
    tasks = []  # (system, gt_idx, gt, pred, iou)
    per_system = {}  # system -> {"alignments": [...], "file": ...}
    for pf in pred_files:
        pred = load_prediction(pf)
        system = pred.get("system") or os.path.basename(pf)
        pred_segs = _pred_segments(pred)
        aligned = align(gt_rich, pred_segs)
        rec = {"file": os.path.basename(pf), "system": system,
               "pred_segments": len(pred_segs), "segments": []}
        for gi, (g, p, i) in enumerate(aligned):
            if p is None:
                rec["segments"].append({"gt_idx": gi, "status": "miss",
                                        "iou": i, "score": 0})
                continue
            pos_norm = (p.get("position") or "").strip().lower()
            if pos_norm in ABSTAIN_POSITIONS:
                rec["segments"].append({"gt_idx": gi, "status": "abstain",
                                        "iou": i, "score": None})
                continue
            rec["segments"].append({"gt_idx": gi, "status": "judged",
                                    "iou": i, "score": None, "reason": None})
            tasks.append((system, gi, g, p, i))
        per_system[system] = rec

    print(f"[plan] {len(tasks)} judge calls (matched, non-abstain GT segments)")

    # ---- TEST: run ONE task end-to-end first ----
    test_result = None
    if tasks:
        ts, tgi, tg, tp, ti = tasks[0]
        sc, rs, raw = judge_call(tg, tp)
        test_result = {"system": ts, "gt_idx": tgi, "score": sc, "reason": rs,
                       "raw": raw[:120]}
        print(f"[test] 1-item end-to-end: score={sc} reason={rs!r}")
        if sc is None:
            print("[test] WARNING: test item failed to parse; continuing anyway")

    # ---- main fan-out with bounded concurrency + time-box ----
    results_map = {}  # (system, gt_idx) -> (score, reason)
    # seed with the test result if it succeeded
    if test_result and test_result["score"] is not None:
        results_map[(test_result["system"], test_result["gt_idx"])] = (
            test_result["score"], test_result["reason"])
    remaining = [t for t in tasks
                 if (t[0], t[1]) not in results_map]

    timed_out = False
    done_ct = len(results_map)
    with cf.ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as ex:
        futs = {}
        it = iter(remaining)
        # prime
        for _ in range(MAX_INFLIGHT):
            try:
                t = next(it)
            except StopIteration:
                break
            futs[ex.submit(judge_call, t[2], t[3])] = t
        while futs:
            if time.time() - t0 > WALL_CLOCK_BUDGET_SEC:
                timed_out = True
                break
            for fut in cf.as_completed(list(futs.keys()), timeout=None):
                t = futs.pop(fut)
                try:
                    sc, rs, _ = fut.result()
                except Exception as exc:  # noqa: BLE001
                    sc, rs = None, f"FAIL: {str(exc)[:80]}"
                    with _usage_lock:
                        _usage["failures"] += 1
                results_map[(t[0], t[1])] = (sc, rs)
                done_ct += 1
                if done_ct % 25 == 0:
                    print(f"[progress] {done_ct}/{len(tasks)} "
                          f"({time.time()-t0:.0f}s elapsed)")
                # submit next
                if time.time() - t0 <= WALL_CLOCK_BUDGET_SEC:
                    try:
                        nt = next(it)
                        futs[ex.submit(judge_call, nt[2], nt[3])] = nt
                    except StopIteration:
                        pass
                break  # re-evaluate time-box each completion

    # ---- write scores back into per_system records ----
    skipped_timeout = 0
    for system, rec in per_system.items():
        for seg in rec["segments"]:
            if seg["status"] != "judged":
                continue
            r = results_map.get((system, seg["gt_idx"]))
            if r is None:
                seg["status"] = "skipped_timeout"
                seg["score"] = None
                skipped_timeout += 1
            else:
                sc, rs = r
                seg["reason"] = rs
                if sc is None:
                    seg["status"] = "judge_failed"
                    seg["score"] = None
                else:
                    seg["score"] = sc

    # ---- aggregate per-system ----
    summaries = []
    for system, rec in per_system.items():
        judged = [s for s in rec["segments"]
                  if s["status"] in ("judged",) and s["score"] is not None]
        misses = [s for s in rec["segments"] if s["status"] == "miss"]
        abstains = [s for s in rec["segments"] if s["status"] == "abstain"]
        failed = [s for s in rec["segments"]
                  if s["status"] in ("judge_failed", "skipped_timeout")]
        # GT-anchored scored set = misses (0) + judged (0/1/2); abstains & failed
        # are excluded from the denominator.
        scored = [s["score"] for s in judged] + [0 for _ in misses]
        n_scored = len(scored)
        mean01 = (sum(scored) / (2.0 * n_scored)) if n_scored else 0.0  # /2 -> [0,1]
        # distribution among judged-only
        dist = {0: 0, 1: 0, 2: 0}
        for s in judged:
            dist[s["score"]] += 1
        summaries.append({
            "system": system,
            "pred_file": rec["file"],
            "pred_segments": rec["pred_segments"],
            "gt_segments": n_gt,
            "n_matched_judged": len(judged),
            "n_miss": len(misses),
            "n_abstain": len(abstains),
            "n_failed_or_skipped": len(failed),
            "n_scored_denom": n_scored,
            "judged_score_dist": {"wrong_0": dist[0], "partial_1": dist[1],
                                  "correct_2": dist[2]},
            "mean_semantic_correctness_0to1": round(mean01, 4),
            "mean_judged_only_0to1": round(
                (sum(s["score"] for s in judged) / (2.0 * len(judged))), 4
            ) if judged else None,
        })

    summaries.sort(key=lambda r: -r["mean_semantic_correctness_0to1"])

    # ---- strict macro-F1 from real_scores.md ----
    strict = parse_strict_scores()
    for s in summaries:
        sf = strict.get(s["system"])
        s["strict_macro_f1"] = sf
        s["strict_to_judge_gap"] = (
            round(s["mean_semantic_correctness_0to1"] - sf, 4)
            if sf is not None else None)

    elapsed = time.time() - t0
    out = {
        "benchmark": "MoveBench v0",
        "task": "V3 LLM-as-judge semantic correctness",
        "judge_model": MODEL,
        "judge_config": {"temperature": 0.0, "thinking_budget": 0,
                         "response_format": "json"},
        "gt_path": "data/v0/mb_v0_0001.json",
        "n_samples": 1,
        "gt_macro_segments": n_gt,
        "iou_threshold": IOU_THRESHOLD,
        "alignment": "greedy max-IoU GT->pred, threshold 0.3 (same as strict T1 scorer)",
        "scoring_scale": "0=wrong, 1=partial, 2=correct; normalized /2 to [0,1]",
        "denominator": "GT-anchored: judged matched segs + misses(=0); abstain & "
                       "judge-failed excluded",
        "abstain_positions": sorted(ABSTAIN_POSITIONS),
        "test_one_item": test_result,
        "timed_out": timed_out,
        "skipped_timeout": skipped_timeout,
        "wall_clock_sec": round(elapsed, 1),
        "usage": dict(_usage),
        "approx_cost_usd": estimate_cost(),
        "systems": summaries,
        "detail": per_system,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[write] {OUT_JSON}")
    print(f"[usage] {dict(_usage)}")
    print(f"[time] {elapsed:.0f}s  timed_out={timed_out}  skipped={skipped_timeout}")
    return out


def estimate_cost():
    # gemini-2.5-flash: $0.30 / 1M input, $2.50 / 1M output (standard tier)
    with _usage_lock:
        pin = _usage["prompt_tokens"]
        pout = _usage["output_tokens"]
    cost = pin / 1e6 * 0.30 + pout / 1e6 * 2.50
    return round(cost, 5)


def parse_strict_scores():
    """Pull strict T1 macro-F1 per system from real_scores.md table."""
    out = {}
    if not os.path.exists(STRICT_MD):
        return out
    with open(STRICT_MD) as fh:
        for line in fh:
            if "| strict |" not in line:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            system = cells[0]
            try:
                out[system] = float(cells[2])
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    main()
