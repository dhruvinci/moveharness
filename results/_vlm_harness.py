#!/usr/bin/env python3
"""
Multi-video VLM harness + measured grounding ablation.

For each sampled production video, runs gemini-2.5-flash in TWO conditions:
  (A) GROUNDED   = COMPETITION_PROMPT + Stage0 CV anchors (full mode) appended.
  (B) UNGROUNDED = COMPETITION_PROMPT only (M1 ablated, no CV anchors).

Collects REAL measured: token usage, approx cost, latency, coverage,
#segments, oversegmentation ratio, ontology-violation/decay signals.

Privacy: outputs are anonymized aggregate stats only. Videos keyed V01..VNN.
No user IDs / paths / names in anything written to benchmark/results/.

Robustness: time-boxed, retry w/ exponential backoff on 429/5xx,
skip+log failures, never crash. File API: upload -> poll ACTIVE ->
generateContent -> delete.
"""
import os, sys, json, time, re, traceback, random
from pathlib import Path

# Paths contain no user-specific absolute path:
#   * benchmark/ is resolved relative to this script (results/ -> benchmark/).
#   * The private experiment tree (Stage0 pipeline imports) is WITHHELD; point at
#     it with the VLM_DIR env var (defaults to ../repos/vlm-experiments relative
#     to the repo root, or MOVEBENCH_ROOT if set).
_BENCH = Path(__file__).resolve().parent.parent          # .../benchmark
_ROOT = Path(os.environ.get("MOVEBENCH_ROOT") or _BENCH.parent)
VLM_DIR = Path(os.environ.get("VLM_DIR") or (_ROOT / "repos" / "vlm-experiments"))
EXP4 = VLM_DIR / "experiment" / "experiment4"
DEPS = VLM_DIR / "experiment" / "dependencies"
RESULTS = _BENCH / "results"
ONTOLOGY_PATH = Path(os.environ.get("MOVEBENCH_ONTOLOGY") or (_BENCH / "ontology.json"))
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(EXP4))

# --- API key from the environment (no key value / .env path embedded here) ---
# Export GOOGLE_API_KEY (or GEMINI_API_KEY) before running; if you keep your key
# in an external .env, source it into the environment first, e.g.
#   set -a; source /path/to/your/.env; set +a
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
assert API_KEY, "No API key: set GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment"

MODEL = "gemini-2.5-flash"
# gemini-2.5-flash pricing (per 1M tokens), as of harness run
PRICE_IN = 0.30 / 1_000_000
PRICE_OUT = 2.50 / 1_000_000   # flash output

# --- Imports from pipeline ---
import stage0  # noqa: E402
from stage1_v3 import COMPETITION_PROMPT, parse_stage1_output  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

client = genai.Client(api_key=API_KEY)

# Build valid position vocabulary from ontology.json (L1A macro + aliases +
# L1B sub-positions + v0 working set + generic engagement states). A "position"
# field value is an ontology violation only if it matches NONE of these.
def _build_ontology_vocab():
    vocab = set()

    def norm(x):
        return x.strip().lower().replace(" ", "_").replace("-", "_")

    # generic engagement / always-allowed
    for x in ["standing", "ground", "scramble", "paused", "unknown", "escape",
              "neutral", "transition", "occluded", "unclear"]:
        vocab.add(norm(x))
    try:
        ont = json.loads(ONTOLOGY_PATH.read_text())
        for p in ont.get("L1A_macro_positions", []):
            vocab.add(norm(p["id"]))
            for a in p.get("aliases", []):
                vocab.add(norm(a))
        for vals in ont.get("L1B_subpositions", {}).values():
            for v in vals:
                vocab.add(norm(v))
        for v in ont.get("v0_macro_position_field_values", {}).get("values", []):
            vocab.add(norm(v))
        for v in ont.get("L2_roles", {}).get("values", []):
            vocab.add(norm(v))
    except Exception as e:  # noqa: BLE001
        print(f"WARN: ontology load failed, using minimal vocab: {e}")
    return vocab


ONTOLOGY_POS = _build_ontology_vocab()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_time_to_sec(t):
    """Parse 'M:SS' or 'MM:SS' to seconds."""
    t = t.strip()
    m = re.match(r"(\d+):(\d+)", t)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def analyze_timeline(parsed, duration_sec):
    """Compute coverage / segments / oversegmentation / ontology / decay signals."""
    segs = parsed.get("segments", [])
    n = len(segs)
    spans = []
    pos_violations = 0
    conf_vals = []
    parsed_pos = []
    for s in segs:
        tm = s.get("time", "")
        if "-" in tm:
            a, b = tm.split("-", 1)
            sa, sb = parse_time_to_sec(a), parse_time_to_sec(b)
            if sa is not None and sb is not None and sb >= sa:
                spans.append((sa, sb))
        # ontology violation: position field not in vocab
        pos = (s.get("position") or "").strip().lower().replace(" ", "_")
        parsed_pos.append(pos)
        if pos and pos not in ONTOLOGY_POS:
            pos_violations += 1
        # confidence for decay analysis
        try:
            conf_vals.append(float(s.get("confidence")))
        except (TypeError, ValueError):
            pass

    # Coverage: union of spans / duration
    covered = 0.0
    last_end = 0.0
    max_end = 0.0
    if spans:
        spans_sorted = sorted(spans)
        cur_s, cur_e = spans_sorted[0]
        for a, b in spans_sorted[1:]:
            if a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                covered += cur_e - cur_s
                cur_s, cur_e = a, b
        covered += cur_e - cur_s
        max_end = max(b for _, b in spans_sorted)
        last_end = spans_sorted[-1][1]
    coverage = min(covered / duration_sec, 1.0) if duration_sec > 0 else 0.0
    # end_reach: did last segment reach the end of video?
    end_reach = min(max_end / duration_sec, 1.0) if duration_sec > 0 else 0.0

    # Oversegmentation: segs per minute, vs expectation.
    # Expectation from stage1 prompt: <4min:10-30, 4-8min:20-40, >8min:30-70.
    dur_min = duration_sec / 60.0
    if duration_sec < 240:
        exp_lo, exp_hi = 10, 30
    elif duration_sec < 480:
        exp_lo, exp_hi = 20, 40
    else:
        exp_lo, exp_hi = 30, 70
    exp_mid = (exp_lo + exp_hi) / 2.0
    overseg_ratio = n / exp_mid if exp_mid > 0 else None
    segs_per_min = n / dur_min if dur_min > 0 else None

    # Confidence decay: slope of confidence over segment index (negative = decay
    # toward end of timeline, a known long-context degradation signal).
    conf_decay = None
    conf_first_third = conf_last_third = None
    if len(conf_vals) >= 6:
        k = len(conf_vals)
        xs = list(range(k))
        mx = sum(xs) / k
        my = sum(conf_vals) / k
        num = sum((x - mx) * (y - my) for x, y in zip(xs, conf_vals))
        den = sum((x - mx) ** 2 for x in xs)
        conf_decay = num / den if den else 0.0
        t3 = k // 3
        conf_first_third = sum(conf_vals[:t3]) / t3
        conf_last_third = sum(conf_vals[-t3:]) / t3

    return {
        "n_segments": n,
        "coverage": round(coverage, 4),
        "end_reach": round(end_reach, 4),
        "segs_per_min": round(segs_per_min, 3) if segs_per_min else None,
        "exp_segments": [exp_lo, exp_hi],
        "overseg_ratio": round(overseg_ratio, 3) if overseg_ratio else None,
        "ontology_violations": pos_violations,
        "ontology_violation_rate": round(pos_violations / n, 4) if n else None,
        "avg_confidence": round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None,
        "conf_decay_slope": round(conf_decay, 5) if conf_decay is not None else None,
        "conf_first_third": round(conf_first_third, 4) if conf_first_third is not None else None,
        "conf_last_third": round(conf_last_third, 4) if conf_last_third is not None else None,
    }


def upload_and_wait(path, max_wait=180):
    """Upload to File API, poll until ACTIVE. Returns file object."""
    f = client.files.upload(file=path)
    t0 = time.time()
    while f.state.name == "PROCESSING":
        if time.time() - t0 > max_wait:
            raise TimeoutError("upload processing timeout")
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"file state={f.state.name}")
    return f


def gen_with_retry(contents, max_retries=4):
    """generateContent with exponential backoff on 429/5xx."""
    delay = 5.0
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=25000,
                    temperature=0.5,
                ),
            )
            latency = time.time() - t0
            return resp, latency
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            retriable = any(s in msg for s in ("429", "rate", "quota", "500", "502", "503", "504", "unavailable", "internal", "deadline"))
            if attempt < max_retries - 1 and retriable:
                sl = delay + random.uniform(0, 2)
                log(f"    gen retry {attempt+1}/{max_retries} after {sl:.1f}s ({type(e).__name__})")
                time.sleep(sl)
                delay *= 2
            else:
                raise
    raise last_err


def usage_to_cost(resp):
    u = getattr(resp, "usage_metadata", None)
    pin = getattr(u, "prompt_token_count", 0) or 0
    pout = getattr(u, "candidates_token_count", 0) or 0
    total = getattr(u, "total_token_count", 0) or 0
    cost = pin * PRICE_IN + pout * PRICE_OUT
    return {"prompt_tokens": pin, "output_tokens": pout, "total_tokens": total,
            "cost_usd": round(cost, 6)}


def run_condition(video_file, duration_sec, anchor_text, cond):
    """Run one Gemini condition; returns metrics dict (skip+log on failure)."""
    try:
        # segment range matching stage1 logic
        if duration_sec < 240:
            seg_range = "10-30"
        elif duration_sec < 480:
            seg_range = "20-40"
        else:
            seg_range = "30-70"
        mm, ss = int(duration_sec // 60), int(duration_sec % 60)
        duration_str = f"{mm}:{ss:02d}"
        prompt = COMPETITION_PROMPT.format(duration=duration_str, segment_range=seg_range)
        if cond == "grounded" and anchor_text:
            prompt += (
                "\n\nCV PRE-ANALYSIS ANCHORS (deterministic pose/interaction signals per time window; "
                "S/G=standing-vs-ground prob, Contact/Entangle/Velocity/Action are normalized CV metrics, "
                "Dominant=which athlete is on top). USE THESE to anchor your timestamps, standing/ground "
                "calls, and who is dominant. Do not contradict strong CV signals without clear visual reason.\n"
                + anchor_text
            )
        resp, latency = gen_with_retry([video_file, prompt])
        text = resp.text or ""
        parsed = parse_stage1_output(text)
        tl = analyze_timeline(parsed, duration_sec)
        cost = usage_to_cost(resp)
        out = {"condition": cond, "ok": True, "latency_s": round(latency, 2),
               "output_chars": len(text), **cost, **tl}
        return out
    except Exception as e:  # noqa: BLE001
        log(f"    [SKIP] {cond} failed: {type(e).__name__}: {e}")
        return {"condition": cond, "ok": False, "error": f"{type(e).__name__}: {e}"}


def run_stage0_cv(path, cache_dir):
    """Run Stage0 CV -> suggested segments -> full anchor text. Returns (anchor_text, cv_meta)."""
    t0 = time.time()
    pre = stage0.CVPreprocessor(path, cache_dir=str(cache_dir))
    data = pre.process_video(sample_rate=1)
    segs = pre.create_smart_segments(data["per_second_metrics"])
    anchor_lines = stage0.build_anchor_lines(segs, "full")
    cv_time = time.time() - t0
    summ = data.get("summary", {})
    cv_meta = {
        "cv_time_s": round(cv_time, 1),
        "cv_n_anchor_segments": len(segs),
        "cv_both_detect_rate": summ.get("athlete_detection", {}).get("both_rate"),
        "cv_avg_action": summ.get("avg_action"),
        "cv_anchor_chars": sum(len(x) for x in anchor_lines),
    }
    return "\n".join(anchor_lines), cv_meta


def main():
    # Sample-list input: a JSON array of objects, each with at least
    #   {"idx": <int>, "video_id": "<opaque id>"}  (plus any per-video metadata
    # the Stage0 pipeline needs). Path is configurable via the VLM_SAMPLE_LIST
    # env var; defaults to vlm_sample.json next to this script.
    sample_path = Path(os.environ.get("VLM_SAMPLE_LIST")
                       or (Path(__file__).resolve().parent / "vlm_sample.json"))
    sample = json.loads(sample_path.read_text())
    cache_dir = VLM_DIR / "outputs" / "experiment4.0"
    deadline = time.time() + float(os.environ.get("VLM_BUDGET_SEC", "1800"))

    results = []
    grand_cost = 0.0
    n_calls = 0
    for item in sample:
        if time.time() > deadline:
            log(f"=== TIME-BOX hit before V{item['idx']:02d}; stopping ===")
            break
        vid = f"V{item['idx']:02d}"
        path = item["path"]
        dur = item["dur"]
        log(f"=== {vid} (dur={dur}s, {item['w']}x{item['h']}) ===")
        rec = {"video": vid, "duration_s": dur, "resolution": f"{item['w']}x{item['h']}", "fps": item["fps"]}

        # Stage0 CV (grounded prerequisite)
        try:
            anchor_text, cv_meta = run_stage0_cv(path, cache_dir)
            rec.update(cv_meta)
            log(f"  CV done: {cv_meta['cv_n_anchor_segments']} anchors in {cv_meta['cv_time_s']}s")
        except Exception as e:  # noqa: BLE001
            log(f"  [SKIP CV] {type(e).__name__}: {e}")
            anchor_text = None
            rec["cv_error"] = f"{type(e).__name__}: {e}"

        # Upload once, reuse for both conditions
        vfile = None
        try:
            log("  Uploading video to File API...")
            vfile = upload_and_wait(path)
            log(f"  Upload ACTIVE: {vfile.name}")
        except Exception as e:  # noqa: BLE001
            log(f"  [SKIP VIDEO] upload failed: {type(e).__name__}: {e}")
            rec["upload_error"] = f"{type(e).__name__}: {e}"
            results.append(rec)
            continue

        try:
            # GROUNDED
            if anchor_text:
                log("  Running GROUNDED (with CV anchors)...")
                g = run_condition(vfile, dur, anchor_text, "grounded")
                rec["grounded"] = g
                n_calls += 1
                grand_cost += g.get("cost_usd", 0.0)
                if g.get("ok"):
                    log(f"    grounded: {g['n_segments']} segs, cov={g['coverage']}, "
                        f"${g['cost_usd']:.4f}, {g['latency_s']}s, viol={g['ontology_violations']}")
            else:
                rec["grounded"] = {"condition": "grounded", "ok": False, "error": "no_cv_anchors"}

            # UNGROUNDED
            log("  Running UNGROUNDED (no CV anchors)...")
            ug = run_condition(vfile, dur, None, "ungrounded")
            rec["ungrounded"] = ug
            n_calls += 1
            grand_cost += ug.get("cost_usd", 0.0)
            if ug.get("ok"):
                log(f"    ungrounded: {ug['n_segments']} segs, cov={ug['coverage']}, "
                    f"${ug['cost_usd']:.4f}, {ug['latency_s']}s, viol={ug['ontology_violations']}")
        finally:
            # delete file
            try:
                client.files.delete(name=vfile.name)
                log("  Deleted uploaded file.")
            except Exception as e:  # noqa: BLE001
                log(f"  (delete failed, non-fatal: {e})")

        results.append(rec)
        log(f"  running cost=${grand_cost:.4f}, calls={n_calls}")

    out = {
        "model": MODEL,
        "pricing_per_1M": {"input_usd": PRICE_IN * 1e6, "output_usd": PRICE_OUT * 1e6},
        "n_videos_attempted": len(results),
        "n_gemini_calls": n_calls,
        "total_cost_usd": round(grand_cost, 4),
        "results": results,
    }
    raw_path = RESULTS / "vlm_runs_raw.json"
    raw_path.write_text(json.dumps(out, indent=2))
    log(f"=== DONE. {len(results)} videos, {n_calls} calls, ${grand_cost:.4f}. Wrote {raw_path} ===")


if __name__ == "__main__":
    main()
