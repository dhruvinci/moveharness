#!/usr/bin/env python3
"""E2 (camera-ready): position macro-F1 restricted to GT segments matchable
under the pipelines' position output vocabulary.

The MoveBench v0 GT has 32 macro_segments; 12 of them (6 sweep + 6 escape,
37.5%) carry positions that no evaluated pipeline can emit (the pipelines'
18-position output vocabulary has no sweep/escape token), so they are
unmatchable by construction and depress macro-F1 as a benchmark artifact.
This script re-scores the SAME stored predictions with the SAME movebench
scorer against (a) the full 32-segment GT and (b) the 20-segment matchable
subset. Deterministic; no inference, no network.

Usage: python3 results/e2_matchable_subset.py   (from benchmark/)
Output: results/e2_matchable_subset.json
"""
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from movebench.evaluate import evaluate           # noqa: E402
from movebench.loader import Ontology, load_sample, load_prediction  # noqa: E402

UNMATCHABLE = {"sweep", "escape"}  # absent from every pipeline's output vocabulary
PREDS = {
    "headline_best_vlm_exp4_v2_run2": "pred_exp4_v2_run2.json",   # 0.1888, 6/32
    "exp4_v3_run2":                   "pred_exp4_v3_run2.json",   # 0.1361, 6/32
}

ontology = Ontology.load(str(BENCH / "ontology.json"))
gt = load_sample(str(BENCH / "data/v0/mb_v0_0001.json"))
segs = gt["macro_segments"]
assert len(segs) == 32, f"expected 32 GT segments, got {len(segs)}"

keep = [s for s in segs if s["position"] not in UNMATCHABLE]
dropped = [s["position"] for s in segs if s["position"] in UNMATCHABLE]
gt_sub = dict(gt)
gt_sub["macro_segments"] = keep
assert len(keep) == 20 and len(dropped) == 12, (len(keep), len(dropped))

out = {
    "method": ("GT macro_segments filtered to positions inside the pipelines' "
               "output vocabulary (drops sweep/escape, 12/32 = 37.5%); same "
               "movebench scorer, same stored predictions, IoU 0.3"),
    "dropped": {p: dropped.count(p) for p in sorted(set(dropped))},
    "systems": {},
}
for name, fname in PREDS.items():
    pred = load_prediction(str(BENCH / "results/predictions_all" / fname))
    rep_full = evaluate(gt, pred, ontology)
    rep_sub = evaluate(gt_sub, pred, ontology)
    if rep_full.get("degenerate") or rep_sub.get("degenerate"):
        raise SystemExit(f"unexpected degenerate flag for {name}")
    full = rep_full["T1_position_segments"]
    sub = rep_sub["T1_position_segments"]
    row = {
        "pred_file": fname,
        "full":      {"n_gt": 32, "macro_f1": round(full["macro_f1"], 4),
                      "matched": full["matched_segments"]},
        "matchable": {"n_gt": 20, "macro_f1": round(sub["macro_f1"], 4),
                      "matched": sub["matched_segments"]},
    }
    out["systems"][name] = row
    print(f"{name}: full={row['full']['macro_f1']}  "
          f"matchable={row['matchable']['macro_f1']}")

json.dump(out, open(BENCH / "results/e2_matchable_subset.json", "w"), indent=2)
print("wrote results/e2_matchable_subset.json")
