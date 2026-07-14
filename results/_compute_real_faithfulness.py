"""D4 — REAL faithfulness, over-segmentation & attention-decay metrics.

Computes, deterministically and from EXISTING artifacts only (no model inference):

  (1) Ontology-violation / hallucination rate per available output: fraction of
      emitted position+technique tokens that are NOT in the v0 ontology. Where a
      CV-anchor checkpoint file exists for the match, also a PHYSICAL-consistency
      rate: predicted ground-plane position while the CV signal says both
      athletes are standing (and vice-versa).

  (2) Over-segmentation ratio vs GT (pred_segments / gt_macro_segments). GT has
      32 macro segments / 959 s.

  (3) Attention decay by NORMALIZED video position (deciles 0->1): per decile,
      segment density (segments/min), label/detail density (labels per segment;
      mean note/commentary chars per segment), and — for the converted,
      scorable predictions — per-decile position agreement with GT (an honest,
      tautology-free signal, since GT != prediction here).

INTEGRITY:
  * No new inference. Everything is parsed from files already on disk.
  * Tokens a pipeline used that the v0 working-set does not contain are passed
    through verbatim so the mismatch is VISIBLE (counts as a violation), not
    hidden.
  * We NEVER copy athlete names, urls, or free text into the emitted artifacts;
    only counts, ratios, and per-decile numeric arrays are written.
  * The 50-video GT==prediction tautology is NOT part of this; all numbers here
    are pred-vs-GT or pred-vs-CV with pred != GT.
"""

from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, _BENCH)

from movebench.loader import Ontology, load_sample  # noqa: E402

_REPOS = os.path.normpath(os.path.join(_BENCH, os.pardir, "repos", "vlm-experiments"))
_OUT = os.path.join(_REPOS, "outputs")
_GT_PATH = os.path.join(_BENCH, "data", "v0", "mb_v0_0001.json")
_CV_PATH = os.path.join(_OUT, "experiment4.0", "stage0_cv_checkpoints_5s.json")

# Ground-plane vs standing-plane classification (canonical ids). Used only for
# the CV physical-consistency check.
_GROUND_POS = {
    "guard", "mount", "side_control", "back_control", "north_south",
    "knee_on_belly", "turtle", "half_guard", "passing_guard", "sweep",
    "escape", "scramble",
}
_STANDING_POS = {"standing", "clinch", "takedown", "single_leg", "double_leg"}


def _parse_ts(ts: str) -> float:
    ts = str(ts).strip()
    if not ts:
        return 0.0
    parts = [float(p) for p in ts.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(ts)


def _clean_token(tok: str) -> str | None:
    if not isinstance(tok, str):
        return None
    t = tok.strip().lower()
    if not t or t in {"-", "n/a", "none", "null"}:
        return None
    return t


def _split_tokens(raw) -> list[str]:
    """Split a comma/semicolon list of technique tokens; drop sentence-like ones."""
    out = []
    if raw is None:
        return out
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,;]", str(raw))
    for it in items:
        t = _clean_token(it)
        if t is None:
            continue
        # drop free-text (multiword/long) — these are not ontology tokens anyway
        if len(t) > 40:
            continue
        if re.fullmatch(r"[a-z0-9_]+", t.replace(" ", "_")):
            out.append(t.replace(" ", "_"))
    return out


# --------------------------------------------------------------------------- #
# Normalizers: turn each raw timeline file into a uniform segment list:
#   {start_sec, end_sec, position, labels:[...], detail_chars}
# detail_chars = length of the human-readable note/commentary (a detail-density
# proxy; we measure its LENGTH, we never copy its content into outputs).
# --------------------------------------------------------------------------- #
def norm_exp3(d: dict) -> list[dict]:
    segs = []
    for s in d.get("segments", []):
        try:
            a, b = _parse_ts(s["start"]), _parse_ts(s["end"])
        except Exception:
            continue
        labels = _split_tokens(s.get("key_actions") or s.get("sub_position"))
        note = " ".join(str(x) for x in (s.get("key_moments") or []))
        segs.append({
            "start_sec": a, "end_sec": b,
            "position": _clean_token(s.get("position")) or "",
            "labels": labels,
            "detail_chars": len(note) if note else len(str(s.get("notes") or "")),
        })
    return segs


def norm_exp4_seg(d: dict) -> list[dict]:
    """Exp4 skeleton/combined: 'time': '0:00-0:07', labels in reasons/focus."""
    segs = []
    for s in d.get("segments", []):
        time = s.get("time", "")
        if "-" not in time:
            continue
        try:
            a_str, b_str = time.split("-", 1)
            a, b = _parse_ts(a_str), _parse_ts(b_str)
        except Exception:
            continue
        labels = _split_tokens(s.get("reasons")) + _split_tokens(s.get("focus"))
        # de-dup preserve order
        seen, uniq = set(), []
        for t in labels:
            if t not in seen:
                seen.add(t); uniq.append(t)
        note = str(s.get("notes") or "")
        segs.append({
            "start_sec": a, "end_sec": b,
            "position": _clean_token(s.get("position")) or "",
            "labels": uniq,
            "detail_chars": len(note),
        })
    return segs


def norm_pass1_timeline(d: dict) -> list[dict]:
    """Exp2 pass1/pass3 style: position_timeline with start_time/end_time."""
    segs = []
    tl = d.get("position_timeline") or d.get("unified_timeline") or []
    for s in tl:
        try:
            a, b = _parse_ts(s["start_time"]), _parse_ts(s["end_time"])
        except Exception:
            continue
        labels = _split_tokens(s.get("sub_position"))
        note = str(s.get("notes") or "")
        segs.append({
            "start_sec": a, "end_sec": b,
            "position": _clean_token(s.get("position")) or "",
            "labels": labels,
            "detail_chars": len(note),
        })
    return segs


# Registry of scorable timeline artifacts. (rel_path, normalizer, pipeline_tag)
ARTIFACTS = [
    ("experiment3/stage1_timeline.json", norm_exp3, "exp3_stage1_single_pass"),
    ("experiment3/stage1_timeline_test.json", norm_exp3, "exp3_stage1_test"),
    ("experiment3/stage1_timeline_v3.3-pass1.json", norm_exp3, "exp3_v3.3_pass1"),
    ("experiment3/stage1_timeline_v3.3-compressed2.json", norm_exp3, "exp3_v3.3_compressed"),
    ("experiment3/stage3_timeline_v3.3.json", norm_exp3, "exp3_v3.3_merged_stage3"),
    ("experiment4.0/stage1_skeleton_competition_run1.json", norm_exp4_seg, "exp4_skeleton_run1"),
    ("experiment4.0/stage1_skeleton_competition_run2.json", norm_exp4_seg, "exp4_skeleton_run2"),
    ("experiment4.0/stage1_skeleton_competition_run3.json", norm_exp4_seg, "exp4_skeleton_run3"),
    ("experiment4.0/stage1v2_skeleton_competition_run1.json", norm_exp4_seg, "exp4_v2_run1"),
    ("experiment4.0/stage1v2_skeleton_competition_run2.json", norm_exp4_seg, "exp4_v2_run2"),
    ("experiment4.0/stage1v3_skeleton_competition_run1.json", norm_exp4_seg, "exp4_v3_run1"),
    ("experiment4.0/stage1v3_skeleton_competition_run2.json", norm_exp4_seg, "exp4_v3_run2"),
    ("experiment4.0/combined_Who's Number One.json", norm_exp4_seg, "exp4_combined"),
    ("pass1_holistic.json", norm_pass1_timeline, "exp2_pass1_holistic"),
    ("experiment2/pass3_synthesis.json", norm_pass1_timeline, "exp2_pass3_synthesis"),
]


def ontology_violations(segs: list[dict], onto: Ontology) -> dict:
    """Fraction of emitted tokens (positions + technique labels) not in ontology."""
    pos_total = pos_viol = 0
    lbl_total = lbl_viol = 0
    viol_examples: dict[str, int] = {}
    for s in segs:
        p = onto.normalize_position(s["position"])
        if p:
            pos_total += 1
            if not onto.is_in_ontology(p):
                pos_viol += 1
                viol_examples[p] = viol_examples.get(p, 0) + 1
        for lbl in s["labels"]:
            nl = onto.normalize_label(lbl)
            if not nl:
                continue
            lbl_total += 1
            if not onto.is_in_ontology(nl):
                lbl_viol += 1
                viol_examples[nl] = viol_examples.get(nl, 0) + 1
    tot = pos_total + lbl_total
    viol = pos_viol + lbl_viol
    top = sorted(viol_examples.items(), key=lambda kv: -kv[1])[:12]
    return {
        "position_tokens": pos_total,
        "position_violations": pos_viol,
        "position_violation_rate": pos_viol / pos_total if pos_total else 0.0,
        "label_tokens": lbl_total,
        "label_violations": lbl_viol,
        "label_violation_rate": lbl_viol / lbl_total if lbl_total else 0.0,
        "combined_tokens": tot,
        "combined_violations": viol,
        "combined_violation_rate": viol / tot if tot else 0.0,
        "top_out_of_ontology_tokens": top,
    }


def load_cv_checkpoints():
    d = json.load(open(_CV_PATH))
    cps = []
    for c in d.get("checkpoints", []):
        t = float(c.get("time_sec", 0))
        a1 = c.get("athlete_1", {}).get("standing")
        a2 = c.get("athlete_2", {}).get("standing")
        cps.append((t, a1, a2))
    return cps, float(d.get("video_duration") or 0.0)


def cv_state_at(cps, t):
    """Nearest CV checkpoint posture call: 'standing','ground','mixed','unknown'."""
    if not cps:
        return "unknown"
    best = min(cps, key=lambda c: abs(c[0] - t))
    a1, a2 = best[1], best[2]
    codes = {a1, a2}
    if codes <= {"S"}:
        return "standing"
    if codes <= {"G"}:
        return "ground"
    if "?" in codes and len(codes) == 1:
        return "unknown"
    return "mixed"


def physical_consistency(segs: list[dict], onto: Ontology, cps) -> dict:
    """Predicted posture vs CV posture at segment midpoint.

    A violation = predicted GROUND-plane position while CV says both standing,
    or predicted STANDING-plane position while CV says both on the ground.
    'mixed'/'unknown' CV calls are not counted as violations (ambiguous).
    """
    checked = viol = 0
    examples = []
    for s in segs:
        p = onto.normalize_position(s["position"])
        if p in _GROUND_POS:
            pred_plane = "ground"
        elif p in _STANDING_POS:
            pred_plane = "standing"
        else:
            continue
        mid = (s["start_sec"] + s["end_sec"]) / 2.0
        cv = cv_state_at(cps, mid)
        if cv in ("standing", "ground"):
            checked += 1
            if cv != pred_plane:
                viol += 1
                if len(examples) < 8:
                    examples.append({
                        "mid_sec": round(mid, 1),
                        "pred_position": p,
                        "pred_plane": pred_plane,
                        "cv_plane": cv,
                    })
    return {
        "cv_anchored_segments_checked": checked,
        "physically_inconsistent": viol,
        "physical_inconsistency_rate": viol / checked if checked else None,
        "examples": examples,
    }


def decile_curves(segs: list[dict], duration: float, gt_pos_at, onto: Ontology):
    """Bin segment MIDPOINTS into 10 normalized-position deciles.

    Per decile: n_segments, segments-per-minute density, mean labels/segment,
    mean detail_chars/segment, and GT-position agreement (fraction of segments
    whose normalized position equals GT's position at that midpoint).
    """
    bins = [{"n": 0, "labels": 0, "detail": 0, "agree": 0, "agree_den": 0,
             "covered_sec": 0.0} for _ in range(10)]
    for s in segs:
        mid = (s["start_sec"] + s["end_sec"]) / 2.0
        if duration <= 0:
            continue
        idx = min(9, max(0, int((mid / duration) * 10)))
        b = bins[idx]
        b["n"] += 1
        b["labels"] += len(s["labels"])
        b["detail"] += s["detail_chars"]
        b["covered_sec"] += max(0.0, s["end_sec"] - s["start_sec"])
        gtp = gt_pos_at(mid)
        if gtp is not None:
            b["agree_den"] += 1
            if onto.normalize_position(s["position"]) == onto.normalize_position(gtp):
                b["agree"] += 1
    bin_sec = duration / 10.0 if duration > 0 else 0.0
    out = {
        "decile_edges_sec": [round(i * bin_sec, 1) for i in range(11)],
        "n_segments": [b["n"] for b in bins],
        "segments_per_min": [
            round(b["n"] / (bin_sec / 60.0), 3) if bin_sec else 0.0 for b in bins
        ],
        "mean_labels_per_segment": [
            round(b["labels"] / b["n"], 3) if b["n"] else 0.0 for b in bins
        ],
        "mean_detail_chars_per_segment": [
            round(b["detail"] / b["n"], 1) if b["n"] else 0.0 for b in bins
        ],
        "gt_position_agreement": [
            round(b["agree"] / b["agree_den"], 3) if b["agree_den"] else None
            for b in bins
        ],
    }
    return out


def main():
    onto = Ontology.load()
    gt = load_sample(_GT_PATH)
    duration = float(gt["duration_sec"])
    gt_macro = gt["macro_segments"]
    n_gt = len(gt_macro)

    # GT position lookup by time
    gt_spans = [(s["start_sec"], s["end_sec"], s.get("position")) for s in gt_macro]

    def gt_pos_at(t):
        for a, b, p in gt_spans:
            if a <= t < b:
                return p
        if gt_spans and t >= gt_spans[-1][1]:
            return gt_spans[-1][2]
        return None

    cps, cv_dur = load_cv_checkpoints()

    # ---- GT decile baseline (its own density curve, for reference) --------- #
    gt_segs_uniform = [
        {"start_sec": s["start_sec"], "end_sec": s["end_sec"],
         "position": s.get("position"),
         "labels": sorted({l for sub in s.get("subsegments", [])
                           for l in (sub.get("labels") or [])}),
         "detail_chars": sum(len(sub.get("note") or "")
                             for sub in s.get("subsegments", []))}
        for s in gt_macro
    ]
    gt_curve = decile_curves(gt_segs_uniform, duration, gt_pos_at, onto)

    results = []
    for rel, normfn, tag in ARTIFACTS:
        path = os.path.join(_OUT, rel)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        segs = normfn(d)
        if not segs:
            continue
        oviol = ontology_violations(segs, onto)
        phys = physical_consistency(segs, onto, cps)
        curve = decile_curves(segs, duration, gt_pos_at, onto)
        # global agreement
        agree = agree_den = 0
        for s in segs:
            mid = (s["start_sec"] + s["end_sec"]) / 2.0
            gtp = gt_pos_at(mid)
            if gtp is not None:
                agree_den += 1
                if onto.normalize_position(s["position"]) == onto.normalize_position(gtp):
                    agree += 1
        results.append({
            "artifact": rel,
            "pipeline_tag": tag,
            "n_pred_segments": len(segs),
            "n_gt_macro_segments": n_gt,
            "oversegmentation_ratio": round(len(segs) / n_gt, 3),
            "ontology": oviol,
            "physical_consistency": phys,
            "global_position_agreement": round(agree / agree_den, 3) if agree_den else None,
            "decile_curve": curve,
        })

    # ---- Cross-run aggregated decay curve (mean over scorable runs) -------- #
    # A single clean curve a plot can use: per decile, the mean GT-position
    # agreement and mean labels/detail density, averaged over all artifacts that
    # have data in that decile. This is the REAL decay curve (replaces the
    # illustrative figure).
    agg = {"gt_position_agreement": [], "mean_labels_per_segment": [],
           "mean_detail_chars_per_segment": [], "n_runs_contributing": [],
           "segments_per_min": []}
    for di in range(10):
        ag_vals, lbl_vals, det_vals, spm_vals = [], [], [], []
        for r in results:
            c = r["decile_curve"]
            if c["gt_position_agreement"][di] is not None:
                ag_vals.append(c["gt_position_agreement"][di])
            if c["n_segments"][di] > 0:
                lbl_vals.append(c["mean_labels_per_segment"][di])
                det_vals.append(c["mean_detail_chars_per_segment"][di])
                spm_vals.append(c["segments_per_min"][di])
        agg["gt_position_agreement"].append(
            round(sum(ag_vals) / len(ag_vals), 4) if ag_vals else None)
        agg["mean_labels_per_segment"].append(
            round(sum(lbl_vals) / len(lbl_vals), 4) if lbl_vals else None)
        agg["mean_detail_chars_per_segment"].append(
            round(sum(det_vals) / len(det_vals), 2) if det_vals else None)
        agg["segments_per_min"].append(
            round(sum(spm_vals) / len(spm_vals), 4) if spm_vals else None)
        agg["n_runs_contributing"].append(len(ag_vals))

    report = {
        "title": "D4 real faithfulness, over-segmentation & attention-decay",
        "source": "deterministic re-analysis of existing pipeline outputs; no model inference",
        "ground_truth": {
            "sample_id": gt.get("sample_id"),
            "n_macro_segments": n_gt,
            "duration_sec": duration,
        },
        "cv_anchor": {
            "file": os.path.relpath(_CV_PATH, _BENCH),
            "n_checkpoints": len(cps),
            "interval_sec": 5,
            "cv_video_duration_sec": cv_dur,
        },
        "headline_summary": {
            "n_artifacts_scored": len(results),
            "oversegmentation_ratio_min": round(min(r["oversegmentation_ratio"] for r in results), 3),
            "oversegmentation_ratio_max": round(max(r["oversegmentation_ratio"] for r in results), 3),
            "worst_oversegmentation": max(results, key=lambda r: r["oversegmentation_ratio"])["pipeline_tag"]
                + f" ({max(r['n_pred_segments'] for r in results)} vs {n_gt} GT)",
            "position_violation_rate_max": round(
                max(r["ontology"]["position_violation_rate"] for r in results), 4),
            "label_violation_rate_min": round(
                min(r["ontology"]["label_violation_rate"] for r in results), 4),
            "label_violation_rate_max": round(
                max(r["ontology"]["label_violation_rate"] for r in results), 4),
            "physical_inconsistency_rates": {
                r["pipeline_tag"]: r["physical_consistency"]["physical_inconsistency_rate"]
                for r in results
                if r["physical_consistency"]["cv_anchored_segments_checked"] >= 4
            },
            "decay_first_vs_last_decile_agreement": {
                "decile0_mean_agreement": agg["gt_position_agreement"][0],
                "decile9_mean_agreement": agg["gt_position_agreement"][9],
            },
        },
        "gt_decile_baseline": gt_curve,
        "aggregated_decay_curve": agg,
        "per_artifact": results,
        "integrity_notes": [
            "All numbers computed from artifacts already on disk; no inference run.",
            "Out-of-ontology tokens are passed through verbatim and counted as "
            "violations, never silently remapped to flatter the score.",
            "Physical-consistency uses CV posture (standing/ground) at segment "
            "midpoint; 'mixed'/'unknown' CV calls are excluded from the denominator.",
            "GT-position agreement is a non-tautological signal (pred != GT here); "
            "the 50-video GT==prediction case is excluded entirely.",
            "Detail density measures the LENGTH of notes/commentary, not content; "
            "no free text is copied into outputs.",
        ],
    }

    out_json = os.path.join(_HERE, "real_faithfulness.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out_json)
    # quick console summary
    for r in results:
        o = r["ontology"]
        p = r["physical_consistency"]
        print(f"{r['pipeline_tag']:32s} segs={r['n_pred_segments']:4d} "
              f"overseg={r['oversegmentation_ratio']:5.2f}x "
              f"ontoViol={o['combined_violation_rate']*100:5.1f}% "
              f"physInc={(p['physical_inconsistency_rate'] or 0)*100:5.1f}% "
              f"(n={p['cv_anchored_segments_checked']}) "
              f"agree={r['global_position_agreement']}")
    return report


if __name__ == "__main__":
    main()
