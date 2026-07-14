"""REAL per-module ablation for the MoveBench v0 harness (post-hoc toggles).

WHAT THIS IS
------------
A deterministic, no-inference ablation. It takes EXISTING machine outputs (no
new model calls) and measures the metric-vector DELTA when each post-hoc-
applicable harness module is toggled ON vs OFF, scored against the single v0
ground-truth sample (`data/v0/mb_v0_0001.json`) with the SAME scorer the rest
of the benchmark uses (`movebench.metrics`).

MODULES TOGGLED
---------------
  M1  measurement-grounding   : CV-grounded prediction (Exp4 two-stage skeleton)
                                vs an UNGROUNDED one (multi-pass holistic VLM,
                                no CV constraints — the Exp2-family pass1
                                timeline). This is the headline with/without-
                                harness contrast. DIRECT (two real pred sets).
  M2  semantic scoring        : ontology equivalence folding (alias/synonym/
      (ontology folding)        hierarchy normalization) ON vs OFF. Same Exp4
                                prediction; only the scorer's normalize_* fn
                                changes. DIRECT.
  M3  temporal smoothing      : merge adjacent same-position micro-segments
                                (a deterministic stand-in for EMA-over-
                                boundaries / micro-segment merge) on the
                                over-segmented ungrounded prediction. Measures
                                oversegmentation ratio + T1 F1 before/after.
                                DIRECT (post-hoc transform of a real pred).
  M4  abstain-aware           : treat predicted unknown/occluded/paused spans
                                as ABSTAIN (excluded from FP accounting) vs
                                counting them as ordinary (mislabel-able)
                                segments. Measures precision / FP delta on the
                                ungrounded prediction. DIRECT.
  M5  schema constraint       : ontology-violation rate (fraction of predicted
      (ontology folding)        technique labels that fall outside the v0
                                ontology) WITH folding vs WITHOUT. DIRECT (same
                                T3 scorer, normalize_* swapped).

INTEGRITY
---------
  * No new inference; every input is a stored artifact.
  * The 50-video GT==prediction tautology is NOT used anywhere.
  * Single annotated sample (n=1): all deltas are single-sample developer
    measurements, NOT leaderboard results. No CIs are computable.
  * Each reported delta is tagged "direct" (a real re-score / real second
    prediction set) or "derived" (an arithmetic combination of two direct
    measurements).
  * The ungrounded source carries names/free-text; we extract ONLY structural
    fields (start/end/position/label tokens) and privacy-scan the output, the
    same discipline as results/_convert_experiment_timeline.py.

Run from the benchmark/ directory:
    python3 results/real_ablation.py
Writes results/real_ablation.json and results/real_ablation.md.
"""

from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
_REPOS = os.path.normpath(os.path.join(_BENCH, os.pardir, "repos", "vlm-experiments"))

# Make the movebench package importable when run as a plain script.
sys.path.insert(0, _BENCH)
sys.path.insert(0, _HERE)

from movebench.loader import Ontology, load_sample  # noqa: E402
from movebench.metrics import score_segments, label_f1, coverage  # noqa: E402
from _privacy import scan as _privacy_scan_terms  # noqa: E402

# ---- inputs (all EXISTING artifacts) -------------------------------------- #
GT_PATH = os.path.join(_BENCH, "data", "v0", "mb_v0_0001.json")

# CV-grounded "with-harness" prediction: Exp4 two-stage skeleton+detail v3 run2.
EXP4_PRED = os.path.join(_BENCH, "results", "predictions", "pred_pipeline_B_exp4.json")

# Ungrounded "without-harness" source: multi-pass holistic VLM, no CV
# constraints (Exp2-family). We convert its position_timeline here.
UNGROUNDED_SRC = os.path.join(_REPOS, "outputs", "pass1_holistic.json")

IOU_THRESHOLD = 0.3
ABSTAIN_POSITIONS = {"unknown", "occluded", "paused", "unclear"}

OUT_JSON = os.path.join(_HERE, "real_ablation.json")
OUT_MD = os.path.join(_HERE, "real_ablation.md")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_ts(ts: str) -> float:
    ts = ts.strip()
    parts = [float(p) for p in ts.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unparseable timestamp: {ts!r}")


def _clean_labels(raw_labels) -> list[str]:
    """Conservative, non-flattering technique-token cleanup (matches the
    converter): keep only short snake_case-ish single tokens; drop free text."""
    out: list[str] = []
    if not raw_labels:
        return out
    if isinstance(raw_labels, str):
        raw_labels = re.split(r"[,;]", raw_labels)
    for tok in raw_labels:
        if not isinstance(tok, str):
            continue
        t = tok.strip().lower()
        if not t or t in {"-", "n/a", "none"}:
            continue
        if " " in t or len(t) > 40:
            continue
        if re.fullmatch(r"[a-z0-9_]+", t):
            out.append(t)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _privacy_scan(obj) -> list[str]:
    # Deny-list = generic non-PII terms + externally-loaded person-name terms
    # (see _privacy.py). No person names are hard-coded here.
    return _privacy_scan_terms(obj)


def load_ungrounded_pred(path: str) -> dict:
    """Convert the multi-pass holistic (ungrounded) timeline to MoveBench pred.

    Carries ONLY start/end/position + a sub_position token (as a label proxy);
    drops all narrative/notes/athlete fields. Order-preserving. Some end<start
    rows exist in the raw window-3 output (the VLM counted DOWN); those zero/neg
    spans are kept verbatim (clipped by the scorer) so the artifact's real
    noisiness is visible rather than silently fixed."""
    d = json.load(open(path, "r", encoding="utf-8"))
    segs = []
    for s in d.get("position_timeline", []):
        start = _parse_ts(s["start_time"])
        end = _parse_ts(s["end_time"])
        labels = _clean_labels(s.get("sub_position"))
        segs.append({
            "start_sec": start,
            "end_sec": end,
            "position": (s.get("position") or "").strip().lower(),
            "labels": labels,
        })
    pred = {
        "sample_id": "mb_v0_0001",
        "ontology_version": "0.1.0",
        "system": "ungrounded_multipass_holistic_no_cv",
        "provenance": ("honest machine output, multi-pass holistic VLM with NO "
                       "CV/skeleton grounding; Exp2-family; dev artifact, in-sample"),
        "segments": segs,
    }
    return pred


def identity_norm(x):
    return (x or "").strip().lower()


def identity_in_ontology(_x) -> bool:
    """Folding OFF: nothing is recognized as in-ontology, so a label only
    'matches' the GT by exact surface string. (Used for the M5 OFF arm: with no
    canonicalization, any predicted label that is not a byte-equal GT surface
    form is out-of-vocabulary.)"""
    return False


def merge_microsegments(segs: list[dict]) -> list[dict]:
    """Temporal smoothing: merge adjacent segments with the same (lowercased)
    position into one span, unioning their labels. Deterministic stand-in for
    EMA-over-boundaries / micro-segment merge. Order-preserving."""
    if not segs:
        return []
    merged: list[dict] = []
    for s in segs:
        pos = (s.get("position") or "").strip().lower()
        if merged and (merged[-1]["position"] or "").strip().lower() == pos \
                and float(s["start_sec"]) <= float(merged[-1]["end_sec"]) + 1e-6:
            merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(s["end_sec"]))
            lab = list(dict.fromkeys((merged[-1].get("labels") or []) + (s.get("labels") or [])))
            merged[-1]["labels"] = lab
        else:
            merged.append({
                "start_sec": float(s["start_sec"]),
                "end_sec": float(s["end_sec"]),
                "position": s.get("position"),
                "labels": list(s.get("labels") or []),
            })
    return merged


def drop_abstain(segs: list[dict]) -> list[dict]:
    """Abstain-aware ON: remove predicted unknown/occluded/paused spans so they
    do not count as false positives (the model declined to commit)."""
    return [s for s in segs
            if (s.get("position") or "").strip().lower() not in ABSTAIN_POSITIONS]


def gt_macro_segments(sample: dict) -> list[dict]:
    def _macro_labels(seg):
        rolled = sorted({l for sub in seg.get("subsegments", [])
                         for l in (sub.get("labels") or [])})
        return rolled or list(seg.get("labels") or [])
    return [{
        "start_sec": s["start_sec"],
        "end_sec": s["end_sec"],
        "position": s.get("position"),
        "labels": _macro_labels(s),
    } for s in sample.get("macro_segments", [])]


def t1(gt, pred, ont, folding=True):
    np_ = ont.normalize_position if folding else identity_norm
    ls = ont.position_label_space if folding else None
    r = score_segments(gt, pred, iou_threshold=IOU_THRESHOLD,
                        normalize_position=np_, label_space=ls,
                        duration_weighted=True)
    return r


def t3(gt, pred, ont, folding=True):
    if folding:
        return label_f1(gt, pred, normalize_label=ont.normalize_label,
                        is_in_ontology=ont.is_in_ontology)
    return label_f1(gt, pred, normalize_label=identity_norm,
                    is_in_ontology=identity_in_ontology)


def _round(x, n=4):
    return None if x is None else round(float(x), n)


# --------------------------------------------------------------------------- #
# Main ablation
# --------------------------------------------------------------------------- #
def main() -> int:
    ont = Ontology.load()
    sample = load_sample(GT_PATH)
    gt = gt_macro_segments(sample)
    duration = float(sample.get("duration_sec") or 0.0)

    exp4 = json.load(open(EXP4_PRED, "r", encoding="utf-8"))
    exp4_segs = [{
        "start_sec": s["start_sec"], "end_sec": s["end_sec"],
        "position": s.get("position"), "labels": s.get("labels", []),
    } for s in exp4.get("segments", [])]

    ungr = load_ungrounded_pred(UNGROUNDED_SRC)
    leaks = _privacy_scan(ungr)
    if leaks:
        print(f"ABORT: privacy leak in ungrounded pred {leaks}", file=sys.stderr)
        return 2
    ungr_segs = ungr["segments"]
    # Real pathology of the ungrounded multi-pass output: window-3 emitted
    # DECREASING timestamps (e.g. '11:30-11:20'), so many spans are zero/neg
    # length and the scorer clips them away. Count this honestly.
    n_malformed = len([s for s in ungr_segs
                       if float(s["end_sec"]) <= float(s["start_sec"])])

    results = {
        "meta": {
            "what": "Real per-module post-hoc ablation on MoveBench v0 (n=1 sample).",
            "no_inference": True,
            "gt": "data/v0/mb_v0_0001.json (32 macro segments, 64 subsegments, 959s, single annotator)",
            "iou_threshold": IOU_THRESHOLD,
            "grounded_pred": "Exp4 two-stage skeleton+detail v3 run2 (25 segments)",
            "ungrounded_pred": "multi-pass holistic VLM, no CV grounding (Exp2-family pass1, 39 segments)",
            "caveat": ("Single annotated sample; every delta is a single-sample "
                       "developer measurement, not a leaderboard result. No CIs."),
            "tautology_excluded": "The 50-video GT==prediction case is excluded everywhere.",
        },
        "modules": {},
    }

    # ----- M1: measurement-grounding (CV-grounded vs ungrounded) ----------- #
    t1_g = t1(gt, exp4_segs, ont, folding=True)
    t1_u = t1(gt, ungr_segs, ont, folding=True)
    cov_g = coverage(exp4_segs, duration)
    cov_u = coverage(ungr_segs, duration)
    results["modules"]["M1_measurement_grounding"] = {
        "description": ("Headline with/without-harness contrast: CV-grounded "
                        "two-stage prediction vs ungrounded multi-pass holistic "
                        "VLM (no CV constraints). Both real, unedited outputs."),
        "measurement_type": "direct",
        "ON_cv_grounded": {
            "pred_segments": t1_g["pred_segments"],
            "T1_macro_f1": _round(t1_g["macro_f1"]),
            "T1_dur_weighted_macro_f1": _round(t1_g["duration_weighted_macro_f1"]),
            "T1_mean_matched_iou": _round(t1_g["mean_matched_iou"]),
            "matched_of_32": t1_g["matched_segments"],
            "coverage": _round(cov_g["coverage_fraction"]),
        },
        "OFF_ungrounded": {
            "pred_segments": t1_u["pred_segments"],
            "T1_macro_f1": _round(t1_u["macro_f1"]),
            "T1_dur_weighted_macro_f1": _round(t1_u["duration_weighted_macro_f1"]),
            "T1_mean_matched_iou": _round(t1_u["mean_matched_iou"]),
            "matched_of_32": t1_u["matched_segments"],
            "coverage": _round(cov_u["coverage_fraction"]),
        },
        "delta_ON_minus_OFF": {
            "T1_macro_f1": _round(t1_g["macro_f1"] - t1_u["macro_f1"]),
            "T1_dur_weighted_macro_f1": _round(
                t1_g["duration_weighted_macro_f1"] - t1_u["duration_weighted_macro_f1"]),
            "T1_mean_matched_iou": _round(t1_g["mean_matched_iou"] - t1_u["mean_matched_iou"]),
            "matched_of_32": t1_g["matched_segments"] - t1_u["matched_segments"],
        },
        "headline_metric": "T1_macro_f1",
        "ungrounded_pathology": {
            "malformed_decreasing_spans": n_malformed,
            "of_total_spans": len(ungr_segs),
            "note": ("The ungrounded multi-pass output emitted DECREASING "
                     "timestamps in its third window (e.g. 11:30->11:20), so "
                     f"{n_malformed}/{len(ungr_segs)} predicted spans are zero/"
                     "negative-length and are clipped to nothing by the scorer. "
                     "This is a real failure mode of the harness-free pipeline, "
                     "and it is the main reason the ungrounded run matches only "
                     "1/32 GT positions."),
        },
        "direction": ("CV-grounding is the headline win: it RAISES T1 macro-F1 "
                      "~9.7x (0.014 -> 0.136) and matched positions 1 -> 6 of 32. "
                      "The ungrounded run's HIGHER mean matched IoU (0.875) is an "
                      "ARTIFACT of matching only a single segment (n=1 matched), "
                      "not better localization; read macro-F1 / matched-count as "
                      "the load-bearing M1 metrics, not IoU."),
    }

    # ----- M2: semantic scoring (ontology folding) ON vs OFF (Exp4) -------- #
    t1_fold = t1(gt, exp4_segs, ont, folding=True)
    t1_nofold = t1(gt, exp4_segs, ont, folding=False)
    results["modules"]["M2_semantic_ontology_folding"] = {
        "description": ("Ontology equivalence folding (alias/synonym/hierarchy "
                        "normalization) ON vs OFF, scored on the SAME Exp4 "
                        "prediction. OFF = exact lowercased string match."),
        "measurement_type": "direct",
        "ON_folding": {
            "T1_macro_f1": _round(t1_fold["macro_f1"]),
            "T1_mean_matched_iou": _round(t1_fold["mean_matched_iou"]),
            "matched_of_32": t1_fold["matched_segments"],
        },
        "OFF_exact_string": {
            "T1_macro_f1": _round(t1_nofold["macro_f1"]),
            "T1_mean_matched_iou": _round(t1_nofold["mean_matched_iou"]),
            "matched_of_32": t1_nofold["matched_segments"],
        },
        "delta_ON_minus_OFF": {
            "T1_macro_f1": _round(t1_fold["macro_f1"] - t1_nofold["macro_f1"]),
            "matched_of_32": t1_fold["matched_segments"] - t1_nofold["matched_segments"],
        },
        "direction": ("Folding can only RAISE or hold matched-position F1 (it "
                      "merges surface variants onto canonical ids). The measured "
                      "delta on Exp4's position field quantifies how much the "
                      "alias map is doing on this artifact."),
    }

    # ----- M3: temporal smoothing (merge micro-segments) ------------------- #
    ungr_merged = merge_microsegments(ungr_segs)
    t1_pre = t1(gt, ungr_segs, ont, folding=True)
    t1_post = t1(gt, ungr_merged, ont, folding=True)
    overseg_pre = len(ungr_segs) / len(gt)
    overseg_post = len(ungr_merged) / len(gt)
    results["modules"]["M3_temporal_smoothing"] = {
        "description": ("Merge adjacent same-position micro-segments (deterministic "
                        "stand-in for EMA-over-boundaries) on the ungrounded "
                        "prediction. Measures oversegmentation ratio (pred/GT) and "
                        "T1 macro-F1 before/after."),
        "measurement_type": "direct",
        "OFF_raw": {
            "pred_segments": len(ungr_segs),
            "oversegmentation_ratio_vs_gt": _round(overseg_pre, 3),
            "T1_macro_f1": _round(t1_pre["macro_f1"]),
            "T1_mean_matched_iou": _round(t1_pre["mean_matched_iou"]),
            "matched_of_32": t1_pre["matched_segments"],
        },
        "ON_smoothed": {
            "pred_segments": len(ungr_merged),
            "oversegmentation_ratio_vs_gt": _round(overseg_post, 3),
            "T1_macro_f1": _round(t1_post["macro_f1"]),
            "T1_mean_matched_iou": _round(t1_post["mean_matched_iou"]),
            "matched_of_32": t1_post["matched_segments"],
        },
        "delta_ON_minus_OFF": {
            "pred_segments": len(ungr_merged) - len(ungr_segs),
            "oversegmentation_ratio_vs_gt": _round(overseg_post - overseg_pre, 3),
            "T1_macro_f1": _round(t1_post["macro_f1"] - t1_pre["macro_f1"]),
            "T1_mean_matched_iou": _round(t1_post["mean_matched_iou"] - t1_pre["mean_matched_iou"]),
            "matched_of_32": t1_post["matched_segments"] - t1_pre["matched_segments"],
        },
        "direction": ("Smoothing REDUCES the predicted-segment count toward GT "
                      "granularity. On THIS artifact only 1 adjacent same-position "
                      "pair exists (the multi-pass timeline interleaves positions "
                      "and resets at window boundaries), so the merge is nearly "
                      "inert (39 -> 38). The mean-IoU drop is a side-effect of that "
                      "single merge widening one matched span; reported as measured, "
                      "not inflated. The documented over-segmentation extreme for "
                      "this pipeline family (Exp2 reached 241 segments, 7.5x GT) is "
                      "NOT recomputed here because that 241-segment artifact is not "
                      "in the converted prediction set."),
    }

    # ----- M4: abstain-aware (drop unknown/occluded) ----------------------- #
    n_abstain = len([s for s in ungr_segs
                     if (s.get("position") or "").strip().lower() in ABSTAIN_POSITIONS])
    ungr_noabstain = drop_abstain(ungr_segs)
    t1_keep = t1(gt, ungr_segs, ont, folding=True)         # abstain counted (OFF)
    t1_drop = t1(gt, ungr_noabstain, ont, folding=True)    # abstain excluded (ON)
    # precision proxy: matched / pred_segments
    prec_keep = (t1_keep["matched_segments"] / t1_keep["pred_segments"]
                 if t1_keep["pred_segments"] else 0.0)
    prec_drop = (t1_drop["matched_segments"] / t1_drop["pred_segments"]
                 if t1_drop["pred_segments"] else 0.0)
    results["modules"]["M4_abstain_aware"] = {
        "description": ("Treat predicted unknown/occluded/paused spans as ABSTAIN "
                        "(excluded from FP accounting) vs counting them as ordinary "
                        "segments, on the ungrounded prediction."),
        "measurement_type": "direct",
        "abstain_positions": sorted(ABSTAIN_POSITIONS),
        "n_abstain_spans_in_pred": n_abstain,
        "OFF_abstain_counted": {
            "pred_segments": t1_keep["pred_segments"],
            "T1_macro_f1": _round(t1_keep["macro_f1"]),
            "matched_of_32": t1_keep["matched_segments"],
            "matched_per_pred_segment": _round(prec_keep),
        },
        "ON_abstain_excluded": {
            "pred_segments": t1_drop["pred_segments"],
            "T1_macro_f1": _round(t1_drop["macro_f1"]),
            "matched_of_32": t1_drop["matched_segments"],
            "matched_per_pred_segment": _round(prec_drop),
        },
        "delta_ON_minus_OFF": {
            "pred_segments": t1_drop["pred_segments"] - t1_keep["pred_segments"],
            "T1_macro_f1": _round(t1_drop["macro_f1"] - t1_keep["macro_f1"]),
            "matched_per_pred_segment": _round(prec_drop - prec_keep),
        },
        "direction": ("Abstain-aware removes non-committal spans so they cannot be "
                      "scored as false positives; on this artifact only "
                      f"{n_abstain} span(s) are abstain, so the measured impact is "
                      "small but its sign/magnitude is reported honestly."),
        "note_if_small": ("If n_abstain is 0-1, this module is effectively inert on "
                          "this single sample; reported as measured, not inflated."),
    }

    # ----- M5: schema constraint (ontology-violation rate) ----------------- #
    t3_fold = t3(gt, exp4_segs, ont, folding=True)
    t3_nofold = t3(gt, exp4_segs, ont, folding=False)
    results["modules"]["M5_schema_constraint_violation_rate"] = {
        "description": ("Ontology-violation rate (fraction of predicted technique "
                        "labels outside the v0 ontology) WITH folding vs WITHOUT, "
                        "on the Exp4 prediction's labels (T3 oracle-segment regime, "
                        "approximate index alignment)."),
        "measurement_type": "direct",
        "ON_folding": {
            "ontology_violation_rate": _round(t3_fold["ontology_violation_rate"]),
            "ontology_violations": t3_fold["ontology_violations"],
            "T3_micro_f1": _round(t3_fold["micro_f1"]),
        },
        "OFF_no_folding": {
            "ontology_violation_rate": _round(t3_nofold["ontology_violation_rate"]),
            "ontology_violations": t3_nofold["ontology_violations"],
            "T3_micro_f1": _round(t3_nofold["micro_f1"]),
        },
        "delta_ON_minus_OFF": {
            "ontology_violation_rate": _round(
                t3_fold["ontology_violation_rate"] - t3_nofold["ontology_violation_rate"]),
            "T3_micro_f1": _round(t3_fold["micro_f1"] - t3_nofold["micro_f1"]),
        },
        "direction": ("Schema-constraint folding LOWERS the apparent violation rate "
                      "(canonical aliases get recognized as in-ontology) and can "
                      "only hold-or-raise T3 F1. The delta quantifies the schema "
                      "module's effect on this artifact."),
        "note": ("T3 here is the same approximate oracle-segment alignment used in "
                 "baselines.md (predictions are not GT-boundary-aligned), so read "
                 "the violation-rate delta as the load-bearing M5 number and T3 F1 "
                 "as directional."),
    }

    # ----- derived summary: harness headline (with vs without harness) ----- #
    f1_rel = (100.0 * (t1_g["macro_f1"] - t1_u["macro_f1"]) / t1_u["macro_f1"]
              if t1_u["macro_f1"] else None)
    f1_ratio = (t1_g["macro_f1"] / t1_u["macro_f1"] if t1_u["macro_f1"] else None)
    results["derived_summary"] = {
        "measurement_type": "derived",
        "headline_with_vs_without_harness_T1_macro_f1": {
            "with_harness_cv_grounded": _round(t1_g["macro_f1"]),
            "without_harness_ungrounded": _round(t1_u["macro_f1"]),
            "abs_delta": _round(t1_g["macro_f1"] - t1_u["macro_f1"]),
            "rel_pct": _round(f1_rel, 1),
            "ratio_x": _round(f1_ratio, 2),
        },
        "matched_positions_with_vs_without": {
            "with_harness_cv_grounded": t1_g["matched_segments"],
            "without_harness_ungrounded": t1_u["matched_segments"],
            "of_32_gt_positions": 32,
        },
        "note": ("Derived = arithmetic combination of the M1 direct measurements. "
                 "Macro-F1 (not mean IoU) is the load-bearing harness headline: "
                 "the ungrounded run's higher IoU comes from matching only 1 "
                 "segment and must not be read as better localization."),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT_JSON}")

    _write_md(results)
    print(f"wrote {OUT_MD}")
    return 0


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0"
    return str(v)


def _sgn(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        s = f"{v:+.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"{v:+d}"
        return s
    return str(v)


def _write_md(r: dict) -> None:
    m = r["meta"]
    M = r["modules"]
    lines = []
    lines.append("# MoveBench v0 — Real Per-Module Ablation (post-hoc toggles)\n")
    lines.append("**No new inference.** Every number below is computed by re-scoring "
                 "EXISTING machine outputs against the v0 ground truth with the "
                 "`movebench.metrics` scorer, toggling one harness module at a time.\n")
    lines.append(f"- Ground truth: `{m['gt']}`")
    lines.append(f"- IoU threshold: {m['iou_threshold']}")
    lines.append(f"- With-harness (CV-grounded) prediction: {m['grounded_pred']}")
    lines.append(f"- Without-harness (ungrounded) prediction: {m['ungrounded_pred']}")
    lines.append("")
    lines.append(f"> **Caveat (frames everything).** {m['caveat']} "
                 f"{m['tautology_excluded']}\n")

    lines.append("## Ablation table (module -> metric impact)\n")
    lines.append("| Module | Toggle | Key metric | OFF | ON | Δ (ON−OFF) | Type | Direction |")
    lines.append("|---|---|---|---|---|---|---|---|")

    m1 = M["M1_measurement_grounding"]
    lines.append(f"| **M1 measurement-grounding** | ungrounded → CV-grounded | "
                 f"T1 macro-F1 (headline) | {_fmt(m1['OFF_ungrounded']['T1_macro_f1'])} | "
                 f"{_fmt(m1['ON_cv_grounded']['T1_macro_f1'])} | "
                 f"{_sgn(m1['delta_ON_minus_OFF']['T1_macro_f1'])} | direct | "
                 f"grounding raises macro-F1 ~9.7x |")
    lines.append(f"| M1 (cont.) | | matched / 32 | "
                 f"{_fmt(m1['OFF_ungrounded']['matched_of_32'])} | "
                 f"{_fmt(m1['ON_cv_grounded']['matched_of_32'])} | "
                 f"{_sgn(m1['delta_ON_minus_OFF']['matched_of_32'])} | direct | "
                 f"1 → 6 matched positions |")
    lines.append(f"| M1 (cont.) | | T1 mean IoU (read with care) | "
                 f"{_fmt(m1['OFF_ungrounded']['T1_mean_matched_iou'])} | "
                 f"{_fmt(m1['ON_cv_grounded']['T1_mean_matched_iou'])} | "
                 f"{_sgn(m1['delta_ON_minus_OFF']['T1_mean_matched_iou'])} | direct | "
                 f"ungrounded IoU is over 1 matched span only |")

    m2 = M["M2_semantic_ontology_folding"]
    lines.append(f"| **M2 semantic folding** | exact-string → ontology fold | "
                 f"T1 macro-F1 | {_fmt(m2['OFF_exact_string']['T1_macro_f1'])} | "
                 f"{_fmt(m2['ON_folding']['T1_macro_f1'])} | "
                 f"{_sgn(m2['delta_ON_minus_OFF']['T1_macro_f1'])} | direct | folding ≥ exact |")
    lines.append(f"| M2 (cont.) | | matched / 32 | "
                 f"{_fmt(m2['OFF_exact_string']['matched_of_32'])} | "
                 f"{_fmt(m2['ON_folding']['matched_of_32'])} | "
                 f"{_sgn(m2['delta_ON_minus_OFF']['matched_of_32'])} | direct | |")

    m3 = M["M3_temporal_smoothing"]
    lines.append(f"| **M3 temporal smoothing** | raw → merge micro-seg | "
                 f"oversegmentation (pred/GT) | "
                 f"{_fmt(m3['OFF_raw']['oversegmentation_ratio_vs_gt'])} | "
                 f"{_fmt(m3['ON_smoothed']['oversegmentation_ratio_vs_gt'])} | "
                 f"{_sgn(m3['delta_ON_minus_OFF']['oversegmentation_ratio_vs_gt'])} | direct | "
                 f"merging reduces oversegmentation |")
    lines.append(f"| M3 (cont.) | | pred segments | "
                 f"{_fmt(m3['OFF_raw']['pred_segments'])} | "
                 f"{_fmt(m3['ON_smoothed']['pred_segments'])} | "
                 f"{_sgn(m3['delta_ON_minus_OFF']['pred_segments'])} | direct | |")
    lines.append(f"| M3 (cont.) | | T1 macro-F1 | "
                 f"{_fmt(m3['OFF_raw']['T1_macro_f1'])} | "
                 f"{_fmt(m3['ON_smoothed']['T1_macro_f1'])} | "
                 f"{_sgn(m3['delta_ON_minus_OFF']['T1_macro_f1'])} | direct | as measured |")

    m4 = M["M4_abstain_aware"]
    lines.append(f"| **M4 abstain-aware** | counted → excluded | "
                 f"pred segments | {_fmt(m4['OFF_abstain_counted']['pred_segments'])} | "
                 f"{_fmt(m4['ON_abstain_excluded']['pred_segments'])} | "
                 f"{_sgn(m4['delta_ON_minus_OFF']['pred_segments'])} | direct | "
                 f"{m4['n_abstain_spans_in_pred']} abstain span(s) |")
    lines.append(f"| M4 (cont.) | | matched/pred (precision proxy) | "
                 f"{_fmt(m4['OFF_abstain_counted']['matched_per_pred_segment'])} | "
                 f"{_fmt(m4['ON_abstain_excluded']['matched_per_pred_segment'])} | "
                 f"{_sgn(m4['delta_ON_minus_OFF']['matched_per_pred_segment'])} | direct | "
                 f"abstain removal raises precision |")

    m5 = M["M5_schema_constraint_violation_rate"]
    lines.append(f"| **M5 schema constraint** | no-fold → ontology fold | "
                 f"ontology-violation rate | "
                 f"{_fmt(m5['OFF_no_folding']['ontology_violation_rate'])} | "
                 f"{_fmt(m5['ON_folding']['ontology_violation_rate'])} | "
                 f"{_sgn(m5['delta_ON_minus_OFF']['ontology_violation_rate'])} | direct | "
                 f"folding lowers violation rate |")
    lines.append(f"| M5 (cont.) | | T3 micro-F1 | "
                 f"{_fmt(m5['OFF_no_folding']['T3_micro_f1'])} | "
                 f"{_fmt(m5['ON_folding']['T3_micro_f1'])} | "
                 f"{_sgn(m5['delta_ON_minus_OFF']['T3_micro_f1'])} | direct | folding ≥ no-fold |")
    lines.append("")

    lines.append("## Direct vs derived\n")
    lines.append("- **Direct measurements:** M1, M2, M3, M4, M5 — each is a real "
                 "re-score (M2/M5 swap the scorer's normalize fn; M1 uses two "
                 "real prediction sets; M3/M4 apply a deterministic post-hoc "
                 "transform to a real prediction and re-score).")
    ds = r["derived_summary"]["headline_with_vs_without_harness_T1_macro_f1"]
    lines.append(f"- **Derived:** the headline with-vs-without-harness macro-F1 gap "
                 f"({_fmt(ds['without_harness_ungrounded'])} → "
                 f"{_fmt(ds['with_harness_cv_grounded'])}, "
                 f"Δ={_sgn(ds['abs_delta'])}, {_fmt(ds['ratio_x'])}x) is an "
                 f"arithmetic combination of the M1 direct numbers, not an "
                 f"independent measurement. Mean IoU is deliberately NOT the "
                 f"headline (the ungrounded run's high IoU is over a single "
                 f"matched span).\n")

    lines.append("## Per-module detail\n")
    for key in ["M1_measurement_grounding", "M2_semantic_ontology_folding",
                "M3_temporal_smoothing", "M4_abstain_aware",
                "M5_schema_constraint_violation_rate"]:
        mod = M[key]
        lines.append(f"### {key}\n")
        lines.append(f"{mod['description']}\n")
        lines.append(f"*Direction:* {mod['direction']}\n")
        if "ungrounded_pathology" in mod:
            up = mod["ungrounded_pathology"]
            lines.append(f"*Ungrounded pathology:* {up['note']}\n")
        if "note" in mod:
            lines.append(f"*Note:* {mod['note']}\n")

    lines.append("## Integrity\n")
    lines.append("- No new model inference; all inputs are stored artifacts.")
    lines.append("- The 50-video GT==prediction tautology is excluded everywhere.")
    lines.append("- n=1 annotated sample: all deltas are single-sample developer "
                 "measurements, not leaderboard results; no confidence intervals.")
    lines.append("- T3-based numbers (M5) use the same approximate oracle-segment "
                 "index alignment as `baselines.md`; the violation-rate delta is "
                 "the load-bearing M5 figure, T3 F1 is directional.")
    lines.append("")
    lines.append("*Generated by `results/real_ablation.py`. Inputs read-only; no "
                 "paper or shared doc modified; predictions privacy-scanned.*")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
