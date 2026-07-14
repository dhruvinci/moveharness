#!/usr/bin/env python3
"""Build vlm_multivideo.{md,json} and vlm_ablation_measured.{md,json} from raw runs.
Anonymized aggregate stats only (V01.. keys, no paths/names/transcripts)."""
import json, statistics as st
from pathlib import Path
from datetime import datetime

# Results dir resolved relative to this script (no user-specific absolute path).
RES = Path(__file__).resolve().parent
raw = json.loads((RES / "vlm_runs_raw.json").read_text())
MODEL = raw["model"]


def dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals)
    return {
        "n": n,
        "mean": round(st.mean(vals), 4),
        "median": round(st.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "std": round(st.pstdev(vals), 4) if n > 1 else 0.0,
        "p25": round(vals_sorted[int(0.25 * (n - 1))], 4),
        "p75": round(vals_sorted[int(0.75 * (n - 1))], 4),
    }


def fmt_dist(d, unit=""):
    if not d:
        return "n/a"
    return (f"mean {d['mean']}{unit} | median {d['median']}{unit} | "
            f"min {d['min']} | max {d['max']} | std {d['std']} (n={d['n']})")


# Collect grounded-condition distributions (system metrics "across N matches")
g_ok = []
for r in raw["results"]:
    g = r.get("grounded", {})
    if g.get("ok"):
        g_ok.append((r, g))

N = len(g_ok)
metrics = {
    "cost_usd": [g["cost_usd"] for _, g in g_ok],
    "coverage": [g["coverage"] for _, g in g_ok],
    "end_reach": [g["end_reach"] for _, g in g_ok],
    "n_segments": [g["n_segments"] for _, g in g_ok],
    "segs_per_min": [g["segs_per_min"] for _, g in g_ok],
    "overseg_ratio": [g["overseg_ratio"] for _, g in g_ok],
    "latency_s": [g["latency_s"] for _, g in g_ok],
    "total_tokens": [g["total_tokens"] for _, g in g_ok],
    "output_tokens": [g["output_tokens"] for _, g in g_ok],
    "ontology_violation_rate": [g["ontology_violation_rate"] for _, g in g_ok],
    "avg_confidence": [g["avg_confidence"] for _, g in g_ok],
    "conf_decay_slope": [g["conf_decay_slope"] for _, g in g_ok],
}
cv_times = [r.get("cv_time_s") for r, _ in g_ok if r.get("cv_time_s") is not None]
durations = [r["duration_s"] for r, _ in g_ok]

dists = {k: dist(v) for k, v in metrics.items()}
dists["cv_time_s"] = dist(cv_times)
dists["duration_s"] = dist(durations)

multivideo = {
    "model": MODEL,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "condition": "GROUNDED (Stage0 CV anchors + Gemini)",
    "n_videos_grounded_ok": N,
    "n_videos_attempted": raw["n_videos_attempted"],
    "total_footage_min": round(sum(durations) / 60, 1),
    "total_gemini_calls": raw["n_gemini_calls"],
    "total_cost_usd": raw["total_cost_usd"],
    "distributions": dists,
    "per_video": [
        {
            "video": r["video"], "duration_s": r["duration_s"],
            "cv_time_s": r.get("cv_time_s"), "cv_anchors": r.get("cv_n_anchor_segments"),
            "n_segments": g["n_segments"], "coverage": g["coverage"],
            "end_reach": g["end_reach"], "segs_per_min": g["segs_per_min"],
            "overseg_ratio": g["overseg_ratio"], "latency_s": g["latency_s"],
            "cost_usd": g["cost_usd"], "total_tokens": g["total_tokens"],
            "ontology_violations": g["ontology_violations"],
            "ontology_violation_rate": g["ontology_violation_rate"],
            "avg_confidence": g["avg_confidence"],
            "conf_decay_slope": g["conf_decay_slope"],
        }
        for r, g in g_ok
    ],
}
(RES / "vlm_multivideo.json").write_text(json.dumps(multivideo, indent=2))

# ---- Markdown for multivideo ----
md = []
md.append("# VLM Harness: Multi-Video Distributions (GROUNDED condition)\n")
md.append(f"- Model: **{MODEL}**")
md.append(f"- Generated: {multivideo['generated_at']}")
md.append(f"- Condition: GROUNDED = Stage0 CV anchors (YOLO11n-pose, per-frame) -> Gemini timeline with anchors injected")
md.append(f"- Videos with valid grounded run: **N = {N}** (attempted {raw['n_videos_attempted']})")
md.append(f"- Total footage analyzed: {multivideo['total_footage_min']} min")
md.append(f"- Total Gemini calls (both conditions): {raw['n_gemini_calls']}; total measured cost: **${raw['total_cost_usd']}**\n")
md.append("## Across-N-matches distributions (grounded)\n")
md.append("| Metric | mean | median | min | max | std | p25 | p75 |")
md.append("|---|---|---|---|---|---|---|---|")
order = ["duration_s", "cv_time_s", "latency_s", "cost_usd", "total_tokens",
         "output_tokens", "n_segments", "segs_per_min", "overseg_ratio",
         "coverage", "end_reach", "ontology_violation_rate", "avg_confidence",
         "conf_decay_slope"]
for k in order:
    d = dists.get(k)
    if not d:
        continue
    md.append(f"| {k} | {d['mean']} | {d['median']} | {d['min']} | {d['max']} | {d['std']} | {d['p25']} | {d['p75']} |")
md.append("\n### Headline (grounded, across N matches)")
md.append(f"- Cost/match: **${dists['cost_usd']['mean']:.4f}** mean (median ${dists['cost_usd']['median']:.4f}); range ${dists['cost_usd']['min']:.4f}-${dists['cost_usd']['max']:.4f}")
md.append(f"- Coverage: **{dists['coverage']['mean']*100:.1f}%** mean (median {dists['coverage']['median']*100:.1f}%)")
md.append(f"- Segments/match: **{dists['n_segments']['mean']:.1f}** mean (median {dists['n_segments']['median']:.0f}); {dists['segs_per_min']['mean']:.2f} segs/min")
md.append(f"- Oversegmentation ratio (segs / prompt-expected midpoint): **{dists['overseg_ratio']['mean']:.2f}x** mean")
md.append(f"- Gemini latency: **{dists['latency_s']['mean']:.1f}s** mean; CV (stage0) time: {dists['cv_time_s']['mean']:.1f}s mean")
md.append(f"- Ontology-violation rate (position field not in ontology): {dists['ontology_violation_rate']['mean']*100:.1f}% mean")
md.append("\n## Per-video (anonymized)\n")
md.append("| Vid | dur(s) | cv(s) | anchors | segs | segs/min | overseg | cov | end | lat(s) | $ | tokens | viol_rate | conf | conf_decay |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for pv in multivideo["per_video"]:
    md.append(f"| {pv['video']} | {pv['duration_s']:.0f} | {pv['cv_time_s']} | {pv['cv_anchors']} | "
              f"{pv['n_segments']} | {pv['segs_per_min']} | {pv['overseg_ratio']} | {pv['coverage']} | "
              f"{pv['end_reach']} | {pv['latency_s']} | {pv['cost_usd']:.4f} | {pv['total_tokens']} | "
              f"{pv['ontology_violation_rate']} | {pv['avg_confidence']} | {pv['conf_decay_slope']} |")
md.append("\n_Privacy: anonymized aggregate stats; no user IDs, paths, athlete names, or raw transcripts. "
          "Label dataset = MMA/grappling production clips._")
(RES / "vlm_multivideo.md").write_text("\n".join(md) + "\n")

# ============ MEASURED GROUNDING ABLATION ============
paired = []
for r in raw["results"]:
    g = r.get("grounded", {})
    u = r.get("ungrounded", {})
    if g.get("ok") and u.get("ok"):
        paired.append((r, g, u))

NP = len(paired)


def delta_stats(key, lower_better=False):
    gs = [g[key] for _, g, _ in paired if g.get(key) is not None and _ is not None]
    # build aligned pairs
    gv, uv, dv = [], [], []
    for _, g, u in paired:
        if g.get(key) is None or u.get(key) is None:
            continue
        gv.append(g[key]); uv.append(u[key]); dv.append(g[key] - u[key])
    if not dv:
        return None
    return {
        "n": len(dv),
        "grounded_mean": round(st.mean(gv), 4),
        "ungrounded_mean": round(st.mean(uv), 4),
        "delta_mean_g_minus_u": round(st.mean(dv), 4),
        "delta_median": round(st.median(dv), 4),
        "delta_std": round(st.pstdev(dv), 4) if len(dv) > 1 else 0.0,
        "n_grounded_better": sum(1 for d in dv if (d < 0 if lower_better else d > 0)),
        "n_ties": sum(1 for d in dv if d == 0),
        "lower_is_better": lower_better,
    }


abl_keys = {
    "n_segments": False,
    "segs_per_min": False,
    "overseg_ratio": True,         # closer-to-1 / lower over-seg better
    "coverage": False,
    "end_reach": False,
    "ontology_violation_rate": True,
    "ontology_violations": True,
    "avg_confidence": False,
    "conf_decay_slope": False,     # less negative (higher) = less decay
    "latency_s": True,
    "cost_usd": True,
    "output_tokens": True,
    "total_tokens": True,
}
ablation = {
    "model": MODEL,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "design": "Within-video paired ablation of M1 (CV grounding). GROUNDED = COMPETITION_PROMPT + Stage0 full CV anchors. UNGROUNDED = COMPETITION_PROMPT only. Same model, same video upload, same parser.",
    "n_paired_videos": NP,
    "deltas": {k: delta_stats(k, lower_better=lb) for k, lb in abl_keys.items()},
    "per_video": [
        {
            "video": r["video"], "duration_s": r["duration_s"],
            "grounded": {kk: g.get(kk) for kk in abl_keys},
            "ungrounded": {kk: u.get(kk) for kk in abl_keys},
        }
        for r, g, u in paired
    ],
}
(RES / "vlm_ablation_measured.json").write_text(json.dumps(ablation, indent=2))

amd = []
amd.append("# Measured Grounding Ablation (M1: CV anchors) — gemini-2.5-flash\n")
amd.append(f"- Model: **{MODEL}**")
amd.append(f"- Generated: {ablation['generated_at']}")
amd.append(f"- Design: within-video **paired** ablation. GROUNDED = prompt + Stage0 CV anchors; UNGROUNDED = prompt only. Same model, same uploaded video, same parser.")
amd.append(f"- Paired videos (both conditions valid): **N = {NP}**\n")
amd.append("## Grounded vs Ungrounded deltas (averaged across videos)\n")
amd.append("| Metric | grounded mean | ungrounded mean | Δ (g−u) mean | Δ median | better? | #g-better/N |")
amd.append("|---|---|---|---|---|---|---|")
label_dir = {True: "lower=better", False: "higher=better"}
for k, lb in abl_keys.items():
    d = ablation["deltas"][k]
    if not d:
        continue
    direction = "grounded" if ((d["delta_mean_g_minus_u"] < 0) == lb and d["delta_mean_g_minus_u"] != 0) else ("ungrounded" if d["delta_mean_g_minus_u"] != 0 else "tie")
    amd.append(f"| {k} ({label_dir[lb]}) | {d['grounded_mean']} | {d['ungrounded_mean']} | "
               f"{d['delta_mean_g_minus_u']:+} | {d['delta_median']:+} | {direction} | {d['n_grounded_better']}/{d['n']} |")
amd.append("\n## Interpretation\n")
ov = ablation["deltas"].get("overseg_ratio")
cov = ablation["deltas"].get("coverage")
viol = ablation["deltas"].get("ontology_violation_rate")
decay = ablation["deltas"].get("conf_decay_slope")
seg = ablation["deltas"].get("n_segments")
if seg:
    amd.append(f"- **Segmentation**: grounded {seg['grounded_mean']:.1f} vs ungrounded {seg['ungrounded_mean']:.1f} segments/match (Δ {seg['delta_mean_g_minus_u']:+.1f}).")
if ov:
    amd.append(f"- **Oversegmentation ratio**: grounded {ov['grounded_mean']:.2f}x vs ungrounded {ov['ungrounded_mean']:.2f}x (Δ {ov['delta_mean_g_minus_u']:+.2f}; lower better).")
if cov:
    amd.append(f"- **Coverage**: grounded {cov['grounded_mean']*100:.1f}% vs ungrounded {cov['ungrounded_mean']*100:.1f}% (Δ {cov['delta_mean_g_minus_u']*100:+.1f} pts).")
if viol:
    amd.append(f"- **Ontology-violation rate**: grounded {viol['grounded_mean']*100:.1f}% vs ungrounded {viol['ungrounded_mean']*100:.1f}% (Δ {viol['delta_mean_g_minus_u']*100:+.1f} pts; lower better).")
if decay:
    amd.append(f"- **Confidence decay slope** (per-segment, less-negative=better): grounded {decay['grounded_mean']} vs ungrounded {decay['ungrounded_mean']} (Δ {decay['delta_mean_g_minus_u']:+}).")
amd.append("\n_Privacy: anonymized aggregate stats only. No user IDs, paths, athlete names, or raw transcripts._")
(RES / "vlm_ablation_measured.md").write_text("\n".join(amd) + "\n")

print(f"multivideo N={N}, ablation paired N={NP}")
print("Wrote vlm_multivideo.{md,json}, vlm_ablation_measured.{md,json}")
