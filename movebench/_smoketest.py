"""Smoke test: prove the MoveBench v0 scorer is not degenerate.

Constructs two predictions against the v0 seed:
  (A) an IDENTITY prediction (predicts exactly the GT macro segments + labels);
      since the seed's label_source is 'human_review' (NOT 'prediction_review'),
      this is a legitimate perfect prediction and scores ~1.0 — it is NOT
      flagged degenerate (the degenerate guard only fires for
      prediction_review GT, which is what makes the P1 1.0 tautological).
  (B) a PERTURBED prediction (some positions wrong, some labels dropped/added,
      one segment shifted) — must score strictly LOWER on T1 macro-F1, IoU, and
      T3 label-F1. If A and B scored identically the scorer would be degenerate.

Prints both metric vectors and an assertion result.
"""

from __future__ import annotations

import copy
import json
import os

from .loader import Ontology, load_sample
from .evaluate import evaluate

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEED = os.path.normpath(os.path.join(_HERE, os.pardir, "data", "v0", "mb_v0_0001.json"))


def _identity_pred(sample: dict) -> dict:
    """Predict exactly the GT macro segments (oracle perfect)."""
    segs = []
    for m in sample["macro_segments"]:
        # union of subsegment labels as the segment's predicted labels
        labels = sorted({l for s in m["subsegments"] for l in s["labels"]})
        segs.append(
            {
                "start_sec": m["start_sec"],
                "end_sec": m["end_sec"],
                "position": m["position"],
                "labels": labels,
            }
        )
    return {"sample_id": sample["sample_id"], "ontology_version": "0.1.0", "segments": segs,
            "transitions": [], "scoring": {"ruleset": "adcc", "events": []}}


def _gt_labels_per_macro(sample: dict) -> list[list[str]]:
    return [sorted({l for s in m["subsegments"] for l in s["labels"]})
            for m in sample["macro_segments"]]


def _perturb(pred: dict) -> dict:
    """Introduce realistic errors: wrong positions, shifted span, label noise."""
    p = copy.deepcopy(pred)
    segs = p["segments"]
    # 1. Flip several positions to a plausibly-wrong class.
    flip = {"guard": "mount", "side_control": "guard", "escape": "standing",
            "sweep": "scramble", "back_control": "mount", "standing": "guard"}
    for i, s in enumerate(segs):
        if i % 3 == 0:  # corrupt ~1/3 of segments' positions
            s["position"] = flip.get(s["position"], s["position"])
    # 2. Temporally shift a couple of segments (degrade IoU).
    for i in (5, 11, 17):
        if i < len(segs):
            segs[i]["start_sec"] += 12.0
            segs[i]["end_sec"] += 12.0
    # 3. Label noise: drop some, add an out-of-ontology hallucination.
    for i, s in enumerate(segs):
        if s["labels"]:
            s["labels"] = s["labels"][: max(0, len(s["labels"]) - 1)]  # drop one
        if i % 5 == 0:
            s["labels"] = list(s["labels"]) + ["flying_unicorn_armbar"]  # hallucination
    return p


def main() -> int:
    ont = Ontology.load()
    sample = load_sample(_SEED)

    pred_id = _identity_pred(sample)
    pred_perturbed = _perturb(pred_id)

    rep_id = evaluate(sample, pred_id, ont)
    rep_pt = evaluate(sample, pred_perturbed, ont)

    def row(tag, rep):
        t1 = rep["T1_position_segments"]
        t3 = rep["T3_label_f1_oracle_segment"]
        cov = rep["T5_coverage"]
        return {
            "regime": tag,
            "degenerate": rep["degenerate"],
            "T1_macro_f1": t1["macro_f1"],
            "T1_dur_weighted_macro_f1": t1["duration_weighted_macro_f1"],
            "T1_mean_matched_iou": t1["mean_matched_iou"],
            "T1_matched": f'{t1["matched_segments"]}/{t1["gt_segments"]}',
            "T3_micro_f1": t3["micro_f1"],
            "T3_ontology_violation_rate": t3["ontology_violation_rate"],
            "T2_status": rep["T2_transitions"].get("task_status"),
            "coverage": cov["coverage_fraction"],
        }

    out = {"identity": row("identity", rep_id),
           "perturbed": row("perturbed", rep_pt)}
    print(json.dumps(out, indent=2))

    a, b = out["identity"], out["perturbed"]
    checks = {
        "identity_not_flagged_degenerate": a["degenerate"] is False,
        "identity_perfect_macro_f1": a["T1_macro_f1"] == 1.0,
        "perturbed_lower_macro_f1": b["T1_macro_f1"] < a["T1_macro_f1"],
        "perturbed_lower_iou": b["T1_mean_matched_iou"] < a["T1_mean_matched_iou"],
        "perturbed_lower_label_f1": b["T3_micro_f1"] < a["T3_micro_f1"],
        "perturbed_has_ontology_violation": b["T3_ontology_violation_rate"] > 0.0,
        "t2_inert_no_gt": a["T2_status"] == "no_v0_ground_truth",
    }
    print("\nASSERTIONS:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\nSCORER NON-DEGENERATE: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
