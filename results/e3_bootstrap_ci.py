#!/usr/bin/env python3
"""E3 (camera-ready): segment-level bootstrap CIs for dense-match macro-F1.

Resamples the 32 GT macro_segments with replacement (10,000 resamples, fixed
seed) and recomputes position macro-F1 against the FIXED stored predictions.
The CI captures within-match variation only: segments of one match are not
independent samples of the domain, and the paper states this caveat wherever
the interval is reported. Deterministic; no inference, no network.

Usage: python3 results/e3_bootstrap_ci.py   (from benchmark/)
Output: results/e3_bootstrap_ci.json
"""
import json
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from movebench.evaluate import _gt_segments_for_t1, _pred_segments  # noqa: E402
from movebench.loader import Ontology, load_sample, load_prediction  # noqa: E402
from movebench.metrics import score_segments  # noqa: E402

N_BOOT = 10_000
SEED = 20260714
SYSTEMS = {
    "grounded_best_vlm_exp4_v2_run2": "pred_exp4_v2_run2.json",   # 0.1888 (6/32)
    "grounded_exp4_v3_run2":          "pred_exp4_v3_run2.json",   # 0.1361 (6/32)
    "ungrounded_naive_single_pass":   "pred_naive_pass1.json",    # 0.0165 (1/32)
}

ontology = Ontology.load(str(BENCH / "ontology.json"))
gt = load_sample(str(BENCH / "data/v0/mb_v0_0001.json"))
gt_segs = _gt_segments_for_t1(gt)
assert len(gt_segs) == 32, len(gt_segs)


def macro_f1(segs, pred_segs):
    t1 = score_segments(
        segs, pred_segs, iou_threshold=0.3,
        normalize_position=ontology.normalize_position,
        label_space=ontology.position_label_space,
        duration_weighted=True,
    )
    return t1["macro_f1"]


out = {"n_boot": N_BOOT, "seed": SEED, "unit": "gt_segment",
       "caveat": ("within-match resampling only; segments of one match are not "
                  "independent samples of the domain"),
       "systems": {}}

for name, fname in SYSTEMS.items():
    pred_segs = _pred_segments(load_prediction(str(BENCH / "results/predictions_all" / fname)))
    point = macro_f1(gt_segs, pred_segs)
    rng = random.Random(SEED)
    stats = []
    for _ in range(N_BOOT):
        sample = [gt_segs[rng.randrange(32)] for _ in range(32)]
        stats.append(macro_f1(sample, pred_segs))
    stats.sort()
    lo, hi = stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]
    out["systems"][name] = {"pred_file": fname, "macro_f1": round(point, 4),
                            "ci95": [round(lo, 4), round(hi, 4)]}
    print(f"{name}: {point:.4f} [{lo:.4f}, {hi:.4f}]")

# The gap that matters: grounded-vs-ungrounded delta under PAIRED resampling
# (same resampled GT applied to both systems per draw).
g = _pred_segments(load_prediction(str(BENCH / "results/predictions_all/pred_exp4_v2_run2.json")))
u = _pred_segments(load_prediction(str(BENCH / "results/predictions_all/pred_naive_pass1.json")))
rng = random.Random(SEED)
deltas = []
for _ in range(N_BOOT):
    sample = [gt_segs[rng.randrange(32)] for _ in range(32)]
    deltas.append(macro_f1(sample, g) - macro_f1(sample, u))
deltas.sort()
dlo, dhi = deltas[int(0.025 * N_BOOT)], deltas[int(0.975 * N_BOOT) - 1]
out["grounded_minus_ungrounded_delta"] = {
    "point": round(macro_f1(gt_segs, g) - macro_f1(gt_segs, u), 4),
    "ci95": [round(dlo, 4), round(dhi, 4)],
    "excludes_zero": bool(dlo > 0 or dhi < 0),
}
print(f"delta grounded-ungrounded: {out['grounded_minus_ungrounded_delta']}")

json.dump(out, open(BENCH / "results/e3_bootstrap_ci.json", "w"), indent=2)
print("wrote results/e3_bootstrap_ci.json")
