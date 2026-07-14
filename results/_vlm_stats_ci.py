#!/usr/bin/env python3
"""
Deterministic error-bar / confidence-interval computation for the VLM benchmark.

Inputs (all under benchmark/results/):
  - vlm_multivideo.json        (N=10 per-video grounded metrics)
  - vlm_ablation_measured.json (N=10 paired grounded vs ungrounded)
  - vlm_technique_id.json       (N=119 clips, position/technique match)
  - vlm_runs_raw.json           (raw per-call, cross-check)
  - real_scores.json            (v0 GT match: 32 GT macro-segments, best CV-grounded)

Methods:
  - Multi-video distributions (N=10): mean, std (sample, ddof=1), and 95% CI by
    nonparametric bootstrap (10,000 resamples, fixed seed 1234), percentile method.
  - Paired ablation deltas (grounded - ungrounded, N=10): mean delta + 95% CI by
    bootstrap on paired differences (10,000 resamples, seed 1234). Significant if CI excludes 0.
  - Technique-ID accuracy: point estimate over N=119 with Wilson score 95% interval.
    Per-tag Wilson where the sub-N is reported.
  - v0 GT segment match: matched-position proportion 6/32 with Wilson 95% interval.
    (Per-segment IoU bootstrap is NOT possible: individual matched-IoU values are not
     stored in real_scores.json -- only the mean is available -- so we do not fabricate them.)

All numbers are computed from the source files; nothing is hand-entered.
"""

import json
import math
import os

import numpy as np

RESULTS = os.path.dirname(os.path.abspath(__file__))
SEED = 1234
N_BOOT = 10000
Z = 1.959963984540054  # 95% two-sided normal quantile


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def bootstrap_ci_mean(values, seed=SEED, n_boot=N_BOOT):
    """Percentile bootstrap 95% CI for the mean of a 1-D array."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi), boot_means


def wilson_interval(k, n, z=Z):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def fmt(x, nd=4):
    return round(float(x), nd)


def main():
    mv = load("vlm_multivideo.json")
    ab = load("vlm_ablation_measured.json")
    tid = load("vlm_technique_id.json")
    raw = load("vlm_runs_raw.json")  # noqa: F841 (loaded for provenance / cross-check)
    scores = load("real_scores.json")

    out = {
        "method": {
            "bootstrap": "nonparametric percentile bootstrap, 10000 resamples, numpy default_rng seed=1234",
            "wilson": "Wilson score interval, z=1.95996 (two-sided 95%)",
            "std": "sample standard deviation, ddof=1",
            "seed": SEED,
            "n_boot": N_BOOT,
            "sources": [
                "vlm_multivideo.json",
                "vlm_ablation_measured.json",
                "vlm_technique_id.json",
                "vlm_runs_raw.json",
                "real_scores.json",
            ],
        },
    }

    # ---------- (a) Multi-video distributions, N=10 ----------
    per_video = mv["per_video"]
    metrics_a = {
        "cost_usd": "Cost per match-edit (USD per video)",
        "coverage": "Timeline coverage fraction",
        "segs_per_min": "Segments per minute (segmentation density)",
        "overseg_ratio": "Oversegmentation ratio (pred_segs / expected)",
        "latency_s": "End-to-end latency (s)",
        "n_segments": "Segments per match (count)",
    }
    a = {}
    for key, label in metrics_a.items():
        vals = [v[key] for v in per_video if v.get(key) is not None]
        arr = np.asarray(vals, dtype=float)
        lo, hi, _ = bootstrap_ci_mean(arr)
        a[key] = {
            "label": label,
            "n": int(len(arr)),
            "mean": fmt(arr.mean(), 4),
            "std_ddof1": fmt(arr.std(ddof=1), 4),
            "ci95_lo": fmt(lo, 4),
            "ci95_hi": fmt(hi, 4),
            "min": fmt(arr.min(), 4),
            "max": fmt(arr.max(), 4),
        }
    out["a_multivideo_distributions"] = a

    # ---------- (b) Paired ablation deltas (grounded - ungrounded), N=10 ----------
    pv = ab["per_video"]
    paired_metrics = {
        "n_segments": ("Segments per match (count)", False),
        "segs_per_min": ("Segments per minute", False),
        "overseg_ratio": ("Oversegmentation ratio", True),
        "ontology_violation_rate": ("Ontology-violation rate", True),
        "latency_s": ("Latency (s)", True),
        "cost_usd": ("Cost (USD)", True),
        "output_tokens": ("Output tokens", True),
        "coverage": ("Coverage fraction", False),
    }
    b = {}
    for key, (label, lower_is_better) in paired_metrics.items():
        diffs = []
        for rec in pv:
            g = rec["grounded"].get(key)
            u = rec["ungrounded"].get(key)
            if g is None or u is None:
                continue
            diffs.append(g - u)
        arr = np.asarray(diffs, dtype=float)
        lo, hi, _ = bootstrap_ci_mean(arr)
        significant = (lo > 0) or (hi < 0)
        b[key] = {
            "label": label,
            "lower_is_better": lower_is_better,
            "n": int(len(arr)),
            "delta_mean_grounded_minus_ungrounded": fmt(arr.mean(), 4),
            "delta_std_ddof1": fmt(arr.std(ddof=1), 4),
            "ci95_lo": fmt(lo, 4),
            "ci95_hi": fmt(hi, 4),
            "significant_at_95": bool(significant),
            "ci_excludes_zero": bool(significant),
        }
    out["b_paired_ablation_deltas"] = b

    # ---------- (c) Technique-ID accuracy with Wilson CI ----------
    summary = tid["summary"]
    per_clip = tid["per_clip"]
    n_total = summary["n_clips_scored"]

    # Recompute the technique-match and position-acc counts directly from per_clip
    # to verify the reported 57.1% headline.
    n_tech = sum(1 for c in per_clip if c.get("technique_correct"))
    n_pos = sum(1 for c in per_clip if c.get("position_correct"))
    n_either = sum(
        1 for c in per_clip if c.get("technique_correct") or c.get("position_correct")
    )

    p_tech, tlo, thi = wilson_interval(n_tech, n_total)
    p_pos, plo, phi = wilson_interval(n_pos, n_total)
    p_either, elo, ehi = wilson_interval(n_either, n_total)

    c = {
        "n_total": int(n_total),
        "technique_match": {
            "k": int(n_tech),
            "n": int(n_total),
            "pct": fmt(100 * p_tech, 1),
            "ci95_lo_pct": fmt(100 * tlo, 1),
            "ci95_hi_pct": fmt(100 * thi, 1),
            "reported_headline_pct": summary["technique_match_rate_pct"],
        },
        "position_top1": {
            "k": int(n_pos),
            "n": int(n_total),
            "pct": fmt(100 * p_pos, 1),
            "ci95_lo_pct": fmt(100 * plo, 1),
            "ci95_hi_pct": fmt(100 * phi, 1),
            "reported_headline_pct": summary["position_top1_accuracy_pct"],
        },
        "either_match": {
            "k": int(n_either),
            "n": int(n_total),
            "pct": fmt(100 * p_either, 1),
            "ci95_lo_pct": fmt(100 * elo, 1),
            "ci95_hi_pct": fmt(100 * ehi, 1),
        },
        "per_tag": {},
    }

    # Per-tag Wilson on technique_match_pct (n reported per tag).
    for tag, rec in summary["per_tag_accuracy_top_tags"].items():
        n_tag = rec["clips"]
        # reported pct -> reconstruct integer count of matches
        for axis, pct_key in [
            ("technique_match", "technique_match_pct"),
            ("position_acc", "position_acc_pct"),
        ]:
            k = round(rec[pct_key] / 100.0 * n_tag)
            p, lo, hi = wilson_interval(k, n_tag)
            c["per_tag"].setdefault(tag, {"clips": n_tag})
            c["per_tag"][tag][axis] = {
                "k_reconstructed": int(k),
                "n": int(n_tag),
                "pct": fmt(100 * p, 1),
                "ci95_lo_pct": fmt(100 * lo, 1),
                "ci95_hi_pct": fmt(100 * hi, 1),
                "reported_pct": rec[pct_key],
            }
    out["c_technique_id_accuracy"] = c

    # ---------- (d) v0 GT segment match (Wilson on matched-position proportion) ----------
    head = scores["headline"]["contrast_a_naive_vs_cv_grounded"]
    best = head["best_cv_grounded"]
    naive = head["naive_single_pass"]
    # matched_positions strings like "6/32"
    bk, bn = (int(x) for x in best["matched_positions"].split("/"))
    nk, nn = (int(x) for x in naive["matched_positions"].split("/"))
    bp, blo, bhi = wilson_interval(bk, bn)
    np_, nlo, nhi = wilson_interval(nk, nn)
    d = {
        "note": (
            "v0 GT video has 32 macro-segments (n_samples=1 video). Matched-position rate "
            "is a binomial proportion -> Wilson 95% CI. Per-segment matched-IoU bootstrap is "
            "NOT computed: real_scores.json stores only the MEAN matched IoU, not the per-segment "
            "array, so resampling individual IoUs would require fabricating data."
        ),
        "best_cv_grounded": {
            "system": best["system"],
            "matched_positions": best["matched_positions"],
            "k": bk,
            "n": bn,
            "match_rate_pct": fmt(100 * bp, 1),
            "ci95_lo_pct": fmt(100 * blo, 1),
            "ci95_hi_pct": fmt(100 * bhi, 1),
            "mean_matched_iou_point": best["mean_matched_iou"],
            "mean_matched_iou_ci": "not computable (per-segment IoUs not stored)",
        },
        "naive_single_pass": {
            "matched_positions": naive["matched_positions"],
            "k": nk,
            "n": nn,
            "match_rate_pct": fmt(100 * np_, 1),
            "ci95_lo_pct": fmt(100 * nlo, 1),
            "ci95_hi_pct": fmt(100 * nhi, 1),
            "mean_matched_iou_point": naive["mean_matched_iou"],
        },
    }
    out["d_v0_segment_match"] = d

    # write JSON
    with open(os.path.join(RESULTS, "vlm_stats_ci.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------- Markdown ----------
    L = []
    L.append("# VLM benchmark: error bars and 95% confidence intervals\n")
    L.append(
        "All intervals are 95%. Bootstrap = nonparametric percentile bootstrap, "
        f"{N_BOOT:,} resamples, numpy `default_rng(seed={SEED})`. "
        "Wilson = Wilson score interval (z=1.95996). "
        "std = sample std (ddof=1). Source files: `vlm_multivideo.json`, "
        "`vlm_ablation_measured.json`, `vlm_technique_id.json`, `vlm_runs_raw.json`, "
        "`real_scores.json`.\n"
    )

    L.append("## (a) Multi-video distributions (N=10, bootstrap CI on the mean)\n")
    L.append("| Metric | N | Mean | Std (ddof=1) | 95% CI (bootstrap) |")
    L.append("|---|---|---|---|---|")
    for key, rec in a.items():
        L.append(
            f"| {rec['label']} | {rec['n']} | {rec['mean']} | {rec['std_ddof1']} | "
            f"[{rec['ci95_lo']}, {rec['ci95_hi']}] |"
        )
    L.append("")

    L.append(
        "## (b) Paired grounded-vs-ungrounded ablation deltas (N=10 paired, bootstrap CI on paired differences)\n"
    )
    L.append(
        "Delta = grounded - ungrounded. Significant = 95% bootstrap CI excludes 0.\n"
    )
    L.append("| Metric | N | Mean delta (G-U) | 95% CI | Significant? |")
    L.append("|---|---|---|---|---|")
    for key, rec in b.items():
        sig = "YES" if rec["significant_at_95"] else "no"
        L.append(
            f"| {rec['label']} | {rec['n']} | {rec['delta_mean_grounded_minus_ungrounded']} | "
            f"[{rec['ci95_lo']}, {rec['ci95_hi']}] | {sig} |"
        )
    L.append("")
    sig_list = [rec["label"] for rec in b.values() if rec["significant_at_95"]]
    L.append(
        "**Significant deltas (CI excludes 0): "
        + (", ".join(sig_list) if sig_list else "none")
        + ".** All other paired deltas have 95% CIs spanning 0 (not significant at N=10).\n"
    )

    L.append("## (c) Technique-ID accuracy (N=119 clips, Wilson 95% CI)\n")
    L.append("| Axis | k / n | Accuracy | 95% Wilson CI |")
    L.append("|---|---|---|---|")
    for axis, label in [
        ("position_top1", "Position top-1"),
        ("technique_match", "Technique match"),
        ("either_match", "Either match"),
    ]:
        r = c[axis]
        L.append(
            f"| {label} | {r['k']}/{r['n']} | {r['pct']}% | "
            f"[{r['ci95_lo_pct']}%, {r['ci95_hi_pct']}%] |"
        )
    L.append("")
    L.append("Per-tag technique-match (Wilson 95% CI; counts reconstructed from reported per-tag pct):\n")
    L.append("| Tag | clips | Technique match | 95% CI | Position acc | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for tag, rec in c["per_tag"].items():
        t = rec["technique_match"]
        p = rec["position_acc"]
        L.append(
            f"| {tag} | {rec['clips']} | {t['pct']}% | [{t['ci95_lo_pct']}%, {t['ci95_hi_pct']}%] | "
            f"{p['pct']}% | [{p['ci95_lo_pct']}%, {p['ci95_hi_pct']}%] |"
        )
    L.append("")

    L.append("## (d) v0 GT segment match (32 GT macro-segments, Wilson 95% CI on matched-position rate)\n")
    L.append("| System | Matched positions | Rate | 95% Wilson CI | Mean matched IoU |")
    L.append("|---|---|---|---|---|")
    bb = d["best_cv_grounded"]
    nn_ = d["naive_single_pass"]
    L.append(
        f"| {bb['system']} (best CV-grounded) | {bb['matched_positions']} | {bb['match_rate_pct']}% | "
        f"[{bb['ci95_lo_pct']}%, {bb['ci95_hi_pct']}%] | {bb['mean_matched_iou_point']} |"
    )
    L.append(
        f"| naive single-pass | {nn_['matched_positions']} | {nn_['match_rate_pct']}% | "
        f"[{nn_['ci95_lo_pct']}%, {nn_['ci95_hi_pct']}%] | {nn_['mean_matched_iou_point']} |"
    )
    L.append("")
    L.append("> " + d["note"] + "\n")

    with open(os.path.join(RESULTS, "vlm_stats_ci.md"), "w") as f:
        f.write("\n".join(L))

    # console echo of headline CIs
    print("=== HEADLINE CIs ===")
    print("cost/match (a):", a["cost_usd"]["mean"], a["cost_usd"]["ci95_lo"], a["cost_usd"]["ci95_hi"])
    print("coverage (a):", a["coverage"]["mean"], a["coverage"]["ci95_lo"], a["coverage"]["ci95_hi"])
    print("segs/min (a):", a["segs_per_min"]["mean"], a["segs_per_min"]["ci95_lo"], a["segs_per_min"]["ci95_hi"])
    print("overseg (a):", a["overseg_ratio"]["mean"], a["overseg_ratio"]["ci95_lo"], a["overseg_ratio"]["ci95_hi"])
    print("latency (a):", a["latency_s"]["mean"], a["latency_s"]["ci95_lo"], a["latency_s"]["ci95_hi"])
    print("--- ablation significant:", sig_list)
    print("output_tokens delta (b):", b["output_tokens"]["delta_mean_grounded_minus_ungrounded"],
          b["output_tokens"]["ci95_lo"], b["output_tokens"]["ci95_hi"], b["output_tokens"]["significant_at_95"])
    print("latency delta (b):", b["latency_s"]["delta_mean_grounded_minus_ungrounded"],
          b["latency_s"]["ci95_lo"], b["latency_s"]["ci95_hi"], b["latency_s"]["significant_at_95"])
    print("--- technique match (c):", c["technique_match"]["pct"], c["technique_match"]["ci95_lo_pct"], c["technique_match"]["ci95_hi_pct"])
    print("--- v0 best match (d):", bb["match_rate_pct"], bb["ci95_lo_pct"], bb["ci95_hi_pct"])
    print("n_tech recomputed:", n_tech, "/", n_total, "= ", round(100*n_tech/n_total,1), "%")
    print("n_pos recomputed:", n_pos, "/", n_total, "= ", round(100*n_pos/n_total,1), "%")


if __name__ == "__main__":
    main()
