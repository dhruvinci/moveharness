"""MoveBench v0 evaluation CLI.

Usage:
    python -m movebench.evaluate --gt <gt.json> --pred <pred.json> [options]

Scores a single MoveBench prediction file against a MoveBench v0 sample
(ground-truth) file and prints a metric VECTOR — never a single conflated
scalar. Output covers:

  * T1 macro-position segment F1 (segment- and duration-weighted) + temporal IoU
  * T3 ontology-aware label F1 (oracle-segment regime) + ontology-violation rate
  * T2 transition-F1 (reports `no_v0_ground_truth` when the GT layer is empty)
  * T5 coverage (completeness, NOT correctness)

INTEGRITY GUARDS (mandatory, per the MoveBench spec):
  1. No degenerate baselines. If the sample's `annotation.label_source ==
     "prediction_review"` AND the GT segments are byte-identical to the
     predicted segments, the run is flagged `degenerate: true` and the headline
     accuracy is SUPPRESSED (set to null). This is exactly why the P1 50-video
     1.0 is tautological and must never be reported as a result.
  2. Empty GT layers are not 0/1. Tasks whose v0 GT layer is empty report
     `task_status: "no_v0_ground_truth"`, not a misleading F1.

Standard library only; no external services.
"""

from __future__ import annotations

import argparse
import json
import sys

from .loader import Ontology, load_sample, load_prediction
from .metrics import score_segments, score_transitions, label_f1, coverage


def _macro_labels(seg: dict) -> list[str]:
    """Roll up a macro segment's technique labels.

    In the v0 seed, technique labels live on SUBSEGMENTS (the macro `labels`
    field is empty). For the oracle-segment T3 regime we score over the macro
    boundary, so the GT label set for a macro segment is the union of its
    subsegment labels (falling back to any macro-level `labels`).
    """
    macro_lvl = seg.get("labels") or []
    rolled = sorted({l for sub in seg.get("subsegments", []) for l in (sub.get("labels") or [])})
    return rolled or list(macro_lvl)


def _gt_segments_for_t1(sample: dict) -> list[dict]:
    """Macro segments in scorer shape (start_sec/end_sec/position/labels)."""
    return [
        {
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "position": s.get("position"),
            "labels": _macro_labels(s),
        }
        for s in sample.get("macro_segments", [])
    ]


def _pred_segments(pred: dict) -> list[dict]:
    return [
        {
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "position": s.get("position"),
            "labels": s.get("labels", []),
        }
        for s in pred.get("segments", [])
    ]


def _segments_byte_identical(a: list[dict], b: list[dict]) -> bool:
    """True iff the two segment lists are structurally identical (start/end/pos)."""
    if len(a) != len(b):
        return False
    norm = lambda s: (
        round(float(s["start_sec"]), 6),
        round(float(s["end_sec"]), 6),
        (s.get("position") or "").strip().lower(),
    )
    return [norm(s) for s in a] == [norm(s) for s in b]


def evaluate(
    sample: dict,
    pred: dict,
    ontology: Ontology,
    iou_threshold: float = 0.3,
    transition_tolerance_sec: float = 5.0,
) -> dict:
    gt_segs = _gt_segments_for_t1(sample)
    pred_segs = _pred_segments(pred)

    # --- Integrity guard 1: degenerate-baseline detection ----------------- #
    label_source = (sample.get("annotation") or {}).get("label_source")
    is_degenerate = (
        label_source == "prediction_review"
        and _segments_byte_identical(gt_segs, pred_segs)
    )

    # --- T1: macro-position segments -------------------------------------- #
    t1 = score_segments(
        gt_segs,
        pred_segs,
        iou_threshold=iou_threshold,
        normalize_position=ontology.normalize_position,
        label_space=ontology.position_label_space,
        duration_weighted=True,
    )
    if is_degenerate:
        # Suppress headline accuracy: GT == predictions by construction.
        t1["degenerate"] = True
        t1["macro_f1"] = None
        t1["duration_weighted_macro_f1"] = None
        t1["mean_matched_iou"] = None
        t1["note"] = (
            "GT == predictions by construction (label_source=prediction_review); "
            "headline accuracy suppressed as tautological."
        )

    # --- T3: ontology-aware label F1, oracle-segment ----------------------- #
    # Oracle-segment regime: align predictions to GT macro boundaries. Here we
    # score over macro segments when both sides expose `labels`; if predictions
    # do not align 1:1 with GT boundaries, the caller should submit an
    # oracle-segment prediction. We score the aligned prefix.
    t3 = label_f1(
        gt_segs,
        pred_segs,
        normalize_label=ontology.normalize_label,
        is_in_ontology=ontology.is_in_ontology,
    )
    if is_degenerate:
        t3["degenerate"] = True
        t3["micro_f1"] = None
        t3["macro_f1"] = None
        t3["note"] = "suppressed: degenerate (GT == predictions)."

    # --- T2: transitions (INERT in v0 when GT layer empty) ----------------- #
    gt_trans = sample.get("transitions", []) or []
    if not gt_trans:
        t2 = {
            "task_status": "no_v0_ground_truth",
            "note": "v0 transition ground-truth layer is empty; not scored.",
        }
    else:
        t2 = score_transitions(
            gt_trans,
            pred.get("transitions", []) or [],
            tolerance_sec=transition_tolerance_sec,
            normalize_transition=ontology.normalize_transition,
        )

    # --- Scoring task (INERT in v0 when GT empty) -------------------------- #
    gt_scoring = sample.get("scoring", {}) or {}
    if not gt_scoring.get("events"):
        scoring = {
            "task_status": "no_v0_ground_truth",
            "note": "v0 scoring ground-truth layer is empty; not scored.",
        }
    else:
        scoring = {"task_status": "scored", "note": "scoring scorer is v1; placeholder."}

    # --- T5: coverage (completeness, NOT correctness) ---------------------- #
    duration = float(sample.get("duration_sec") or 0.0)
    cov = coverage(pred_segs, duration)

    return {
        "sample_id": sample.get("sample_id"),
        "ontology_version": ontology.version,
        "iou_threshold": iou_threshold,
        "degenerate": is_degenerate,
        "T1_position_segments": t1,
        "T3_label_f1_oracle_segment": t3,
        "T2_transitions": t2,
        "scoring_task": scoring,
        "T5_coverage": cov,
        "integrity_notes": [
            "Metric VECTOR, not a single scalar.",
            "Coverage is COMPLETENESS, not correctness; read it next to T1/T3.",
            "Empty GT layers report no_v0_ground_truth, never a misleading 0/1.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="movebench.evaluate",
        description="Score a MoveBench v0 prediction against a sample ground truth.",
    )
    parser.add_argument("--gt", required=True, help="path to a MoveBench sample (GT) json")
    parser.add_argument("--pred", required=True, help="path to a MoveBench prediction json")
    parser.add_argument("--ontology", default=None, help="path to ontology.json (default: bundled)")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--transition-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--output", default=None, help="optional path to write full JSON report")
    args = parser.parse_args(argv)

    ontology = Ontology.load(args.ontology)
    sample = load_sample(args.gt)
    pred = load_prediction(args.pred)

    if sample.get("sample_id") and pred.get("sample_id") and sample["sample_id"] != pred["sample_id"]:
        print(
            f"WARNING: sample_id mismatch (gt={sample['sample_id']!r} "
            f"pred={pred['sample_id']!r})",
            file=sys.stderr,
        )

    report = evaluate(
        sample,
        pred,
        ontology,
        iou_threshold=args.iou_threshold,
        transition_tolerance_sec=args.transition_tolerance_sec,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
