"""Judge self-consistency: judge the SAME 10 segments TWICE and measure
exact-agreement, within-1 agreement, and Cohen's-kappa-style stats. Reuses the
judge from _llm_judge.py. Time-boxed, retry, never crashes."""

from __future__ import annotations

import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, os.pardir)))
sys.path.insert(0, _HERE)

import results._llm_judge as J  # noqa: E402
from movebench.loader import load_sample, load_prediction  # noqa: E402
from movebench.evaluate import _pred_segments  # noqa: E402

N_SEGMENTS = 10
SEED = 17


def main():
    key = J.load_env()
    if not key:
        raise SystemExit("no API key")
    sample = load_sample(J.GT_PATH)
    gt = J._gt_rich_segments(sample)

    # Build the same pool of matched (gt,pred) pairs across all systems.
    import glob

    pairs = []
    for pf in sorted(glob.glob(os.path.join(J.PRED_DIR, "pred_*.json"))):
        pred = load_prediction(pf)
        system = pred.get("system") or os.path.basename(pf)
        ps = _pred_segments(pred)
        for gi, (g, p, i) in enumerate(J.align(gt, ps)):
            if p is None:
                continue
            pos = (p.get("position") or "").strip().lower()
            if pos in J.ABSTAIN_POSITIONS:
                continue
            pairs.append((system, gi, g, p, i))

    rng = random.Random(SEED)
    sample_pairs = rng.sample(pairs, min(N_SEGMENTS, len(pairs)))
    print(f"[consistency] judging {len(sample_pairs)} segments TWICE")

    rows = []
    for idx, (system, gi, g, p, i) in enumerate(sample_pairs):
        s1, r1, _ = J.judge_call(g, p)
        s2, r2, _ = J.judge_call(g, p)
        rows.append({
            "idx": idx, "system": system, "gt_idx": gi, "iou": i,
            "score_run1": s1, "score_run2": s2,
            "reason_run1": r1, "reason_run2": r2,
            "agree_exact": (s1 == s2) if (s1 is not None and s2 is not None) else None,
            "agree_within1": (abs(s1 - s2) <= 1)
            if (s1 is not None and s2 is not None) else None,
        })
        print(f"  seg{idx}: {system[:28]:28s} run1={s1} run2={s2} "
              f"{'OK' if s1==s2 else 'DIFF'}")

    valid = [r for r in rows if r["agree_exact"] is not None]
    n = len(valid)
    exact = sum(1 for r in valid if r["agree_exact"])
    within1 = sum(1 for r in valid if r["agree_within1"])
    # Cohen's kappa over the 0/1/2 labels (two raters = run1, run2)
    kappa = cohen_kappa([r["score_run1"] for r in valid],
                        [r["score_run2"] for r in valid])
    # mean abs diff
    mad = (sum(abs(r["score_run1"] - r["score_run2"]) for r in valid) / n) if n else None

    out = {
        "task": "LLM judge self-consistency (same segment judged twice)",
        "judge_model": J.MODEL,
        "judge_config": {"temperature": 0.0, "thinking_budget": 0},
        "n_segments": n,
        "seed": SEED,
        "exact_agreement": round(exact / n, 4) if n else None,
        "within_1_agreement": round(within1 / n, 4) if n else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "mean_abs_score_diff": round(mad, 4) if mad is not None else None,
        "usage": dict(J._usage),
        "approx_cost_usd": J.estimate_cost(),
        "pairs": rows,
    }
    out_path = os.path.join(_HERE, "vlm_llm_judge_consistency.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[consistency] exact={out['exact_agreement']} "
          f"within1={out['within_1_agreement']} kappa={out['cohens_kappa']} "
          f"mad={out['mean_abs_score_diff']}")
    print(f"[write] {out_path}")
    return out


def cohen_kappa(a, b):
    labels = [0, 1, 2]
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for l in labels:
        pa = sum(1 for x in a if x == l) / n
        pb = sum(1 for y in b if y == l) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0  # perfect by-chance-degenerate (all same label)
    return (po - pe) / (1 - pe)


if __name__ == "__main__":
    main()
