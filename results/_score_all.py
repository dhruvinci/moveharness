"""D1: REAL metric-vector scoring of every converted prediction against v0 GT,
in TWO regimes (STRICT exact-match vs SEMANTIC ontology-normalized,
abstain-aware).

Uses the bundled MoveBench scorer functions directly (no model inference, no
network). Standard library only.

  STRICT regime
    - position matching uses an IDENTITY normalizer (lowercase only; NO ontology
      alias-folding). This is the raw exact-string-match floor.
    - label F1 uses the same identity normalizer; ontology-violation tracking is
      disabled (strict does not consult the ontology).

  SEMANTIC regime
    - position + label matching use the ontology alias-folding normalizer (the
      scorer default): synonyms / aliases / hierarchical surface forms fold to a
      canonical id (e.g. "Mount"->"mount", "de_la_riva"->"de_la_riva_guard",
      "guard_pass"->"passing_guard").
    - ABSTAIN-AWARE: predicted segments whose position normalizes to the
      always-allowed abstention class ("unknown"/"occluded") are removed before
      scoring, so an explicit abstention is not penalised as a wrong commitment.
      The count of dropped abstain segments is reported.
    - ontology-violation rate is reported (fraction of predicted technique labels
      outside the v0 ontology).

Metric vector per (system x regime):
    macro_f1 (T1 position), duration_weighted_macro_f1, temporal_iou
    (mean matched IoU), label_f1 (T3 micro), label_macro_f1,
    ontology_violation_rate, oversegmentation_ratio = pred_segs / gt_segs,
    coverage, matched (k/32).
"""

from __future__ import annotations

import glob
import json
import os

from movebench.loader import Ontology, load_sample, load_prediction
from movebench.evaluate import _gt_segments_for_t1, _pred_segments
from movebench.metrics import score_segments, label_f1, coverage

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
GT_PATH = os.path.join(_BENCH, "data", "v0", "mb_v0_0001.json")
PRED_DIR = os.path.join(_HERE, "predictions_all")

IOU_THRESHOLD = 0.3
ABSTAIN_POSITIONS = {"unknown", "occluded", "unclear"}


def _identity(x):
    return (x or "").strip().lower()


def score_one(gt_segs, pred_segs, ontology, regime: str) -> dict:
    """Score one prediction's segment list against GT in the named regime."""
    if regime == "strict":
        norm_pos = _identity
        norm_lbl = _identity
        in_ont = None          # strict: no ontology consulted
        label_space = None
        abstain_dropped = 0
        scored_pred = pred_segs
    elif regime == "semantic":
        norm_pos = ontology.normalize_position
        norm_lbl = ontology.normalize_label
        in_ont = ontology.is_in_ontology
        label_space = ontology.position_label_space
        # abstain-aware: drop predicted segments that explicitly abstain
        scored_pred = []
        abstain_dropped = 0
        for s in pred_segs:
            if ontology.normalize_position(s.get("position")) in ABSTAIN_POSITIONS:
                abstain_dropped += 1
                continue
            scored_pred.append(s)
    else:
        raise ValueError(regime)

    t1 = score_segments(
        gt_segs, scored_pred,
        iou_threshold=IOU_THRESHOLD,
        normalize_position=norm_pos,
        label_space=label_space,
        duration_weighted=True,
    )
    t3 = label_f1(
        gt_segs, scored_pred,
        normalize_label=norm_lbl,
        is_in_ontology=in_ont,
    )
    return {
        "regime": regime,
        "pred_segments_scored": len(scored_pred),
        "abstain_dropped": abstain_dropped,
        "macro_f1": round(t1["macro_f1"], 4),
        "duration_weighted_macro_f1": round(t1["duration_weighted_macro_f1"], 4),
        "temporal_iou_mean_matched": round(t1["mean_matched_iou"], 4),
        "matched_positions": t1["matched_segments"],
        "label_micro_f1": round(t3["micro_f1"], 4),
        "label_macro_f1": round(t3["macro_f1"], 4),
        "ontology_violation_rate": round(t3["ontology_violation_rate"], 4),
        "ontology_violations": t3["ontology_violations"],
    }


def _pooled_token_f1(gt_segs, pred_segs, normalize):
    """Alignment-free set F1 over all position+label tokens.

    Pools every GT position+label token and every predicted position+label token
    into multisets (by normalized form), then computes micro-F1 on the multiset
    intersection. This isolates the alias-folding (strict vs semantic) effect
    WITHOUT depending on the broken oracle-index alignment that T3 label_f1 uses.
    Tokens are matched as multisets so repeated tokens count proportionally.
    """
    from collections import Counter

    def toks(segs):
        c = Counter()
        for s in segs:
            for t in [s.get("position")] + list(s.get("labels") or []):
                n = normalize(t)
                if n:
                    c[n] += 1
        return c

    g = toks(gt_segs)
    p = toks(pred_segs)
    tp = sum((g & p).values())
    fp = sum((p - g).values())
    fn = sum((g - p).values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


def main():
    ontology = Ontology.load()
    sample = load_sample(GT_PATH)
    gt_segs = _gt_segments_for_t1(sample)
    n_gt = len(gt_segs)
    duration = float(sample.get("duration_sec") or 0.0)

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_*.json")))
    results = []
    for pf in pred_files:
        pred = load_prediction(pf)
        pred_segs = _pred_segments(pred)
        n_pred = len(pred_segs)
        cov = coverage(pred_segs, duration)
        row = {
            "system": pred.get("system"),
            "pred_file": os.path.basename(pf),
            "provenance": pred.get("provenance"),
            "gt_segments": n_gt,
            "pred_segments": n_pred,
            "oversegmentation_ratio": round(n_pred / n_gt, 4) if n_gt else None,
            "coverage_fraction": round(cov["coverage_fraction"], 4),
            "strict": score_one(gt_segs, pred_segs, ontology, "strict"),
            "semantic": score_one(gt_segs, pred_segs, ontology, "semantic"),
        }
        # Alignment-free pooled-token F1 (isolates alias-folding effect cleanly).
        pooled_strict = _pooled_token_f1(gt_segs, pred_segs, _identity)
        pooled_semantic = _pooled_token_f1(gt_segs, pred_segs, ontology.normalize_label)
        row["pooled_token_f1_strict"] = pooled_strict
        row["pooled_token_f1_semantic"] = pooled_semantic
        row["pooled_token_f1_gap"] = round(
            pooled_semantic["f1"] - pooled_strict["f1"], 4
        )

        # strict->semantic gap on the load-bearing T1 macro-F1
        row["strict_to_semantic_macro_f1_gap"] = round(
            row["semantic"]["macro_f1"] - row["strict"]["macro_f1"], 4
        )
        row["strict_to_semantic_label_f1_gap"] = round(
            row["semantic"]["label_micro_f1"] - row["strict"]["label_micro_f1"], 4
        )
        results.append(row)

    out = {
        "benchmark": "MoveBench v0",
        "gt_path": "data/v0/mb_v0_0001.json",
        "n_samples": 1,
        "gt_macro_segments": n_gt,
        "iou_threshold": IOU_THRESHOLD,
        "regimes": {
            "strict": "identity (lowercase) position/label matching; ontology NOT consulted; raw exact-string floor",
            "semantic": "ontology alias-folding normalization + abstain-aware (drop predicted 'unknown' positions); ontology-violation tracked",
        },
        "ontology_version": ontology.version,
        "results": results,
    }
    out_json = os.path.join(_HERE, "real_scores_data.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_json} ({len(results)} systems scored)")
    return out


if __name__ == "__main__":
    main()
