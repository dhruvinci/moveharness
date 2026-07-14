"""Build real_scores.json (machine) + real_scores.md (human) from
real_scores_data.json. Adds a headline summary block, group rollups, and the
two highlighted contrasts. No model inference; pure aggregation of computed
scores. Standard library only.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "real_scores_data.json")
OUT_JSON = os.path.join(_HERE, "real_scores.json")
OUT_MD = os.path.join(_HERE, "real_scores.md")

GROUPS = {
    "naive_single_pass_holistic": "Naive (no CV grounding)",
    "exp2_multipass_pass1_enhanced": "Exp2 multi-pass",
    "exp2_multipass_pass1_fixed": "Exp2 multi-pass",
    "exp2_multipass_pass3_synthesis": "Exp2 multi-pass",
    "exp3_stage1_cv_cached_timeline": "Exp3 (CV-cached)",
    "exp3_stage1_v33_pass1": "Exp3 (CV-cached)",
    "exp3_stage1_v33_compressed": "Exp3 (CV-cached)",
    "exp3_stage3_merged_timeline": "Exp3 (CV-cached)",
    "exp4_v1_skeleton_run1": "Exp4 v1",
    "exp4_v1_skeleton_run2": "Exp4 v1",
    "exp4_v1_skeleton_run3": "Exp4 v1",
    "exp4_v2_skeleton_run1": "Exp4 v2",
    "exp4_v2_skeleton_run2": "Exp4 v2",
    "exp4_v3_skeleton_run1": "Exp4 v3 (CV-grounded)",
    "exp4_v3_skeleton_run2": "Exp4 v3 (CV-grounded)",
    "exp4_combined_stage1_stage2": "Exp4 composite",
}
GROUP_ORDER = [
    "Naive (no CV grounding)", "Exp2 multi-pass", "Exp3 (CV-cached)",
    "Exp4 v1", "Exp4 v2", "Exp4 v3 (CV-grounded)", "Exp4 composite",
]


def main():
    d = json.load(open(DATA, "r", encoding="utf-8"))
    results = d["results"]
    n_gt = d["gt_macro_segments"]

    # group rollups on semantic T1 macro-F1
    by_group = defaultdict(list)
    for r in results:
        by_group[GROUPS[r["system"]]].append(r)
    group_summary = []
    for g in GROUP_ORDER:
        rs = by_group.get(g, [])
        if not rs:
            continue
        mf = [r["semantic"]["macro_f1"] for r in rs]
        ious = [r["semantic"]["temporal_iou_mean_matched"] for r in rs]
        matched = [r["semantic"]["matched_positions"] for r in rs]
        group_summary.append({
            "group": g,
            "n_runs": len(rs),
            "best_semantic_macro_f1": max(mf),
            "mean_semantic_macro_f1": round(sum(mf) / len(mf), 4),
            "best_mean_matched_iou": max(ious),
            "max_matched_positions": max(matched),
        })

    # headline contrasts
    R = {r["system"]: r for r in results}
    naive = R["naive_single_pass_holistic"]
    best_cv = max(results, key=lambda r: r["semantic"]["macro_f1"])
    pooled_gaps = [(r["system"],
                    r["pooled_token_f1_strict"]["f1"],
                    r["pooled_token_f1_semantic"]["f1"],
                    r["pooled_token_f1_gap"]) for r in results]
    unders = []
    for r in results:
        pm = r["pooled_token_f1_semantic"]["f1"]
        ps = r["pooled_token_f1_strict"]["f1"]
        if pm > 0:
            unders.append((r["system"], round(100 * (pm - ps) / pm, 1)))
    max_under = max(unders, key=lambda x: x[1])
    mean_under = round(sum(u for _, u in unders) / len(unders), 1)
    overseg = sorted(((r["system"], r["oversegmentation_ratio"]) for r in results),
                     key=lambda x: x[1])

    headline = {
        "contrast_a_naive_vs_cv_grounded": {
            "naive_single_pass": {
                "semantic_T1_macro_f1": naive["semantic"]["macro_f1"],
                "matched_positions": f"{naive['semantic']['matched_positions']}/{n_gt}",
                "mean_matched_iou": naive["semantic"]["temporal_iou_mean_matched"],
            },
            "best_cv_grounded": {
                "system": best_cv["system"],
                "semantic_T1_macro_f1": best_cv["semantic"]["macro_f1"],
                "matched_positions": f"{best_cv['semantic']['matched_positions']}/{n_gt}",
                "mean_matched_iou": best_cv["semantic"]["temporal_iou_mean_matched"],
            },
            "relative_improvement_x": round(
                best_cv["semantic"]["macro_f1"] / naive["semantic"]["macro_f1"], 1
            ) if naive["semantic"]["macro_f1"] else None,
            "note": "CV-grounded pipelines match up to 6/32 GT positions vs 1/32 for naive single-pass; position macro-F1 rises ~11x.",
        },
        "contrast_b_strict_to_semantic_gap": {
            "measure": "alignment-free pooled position+label token micro-F1",
            "max_underCount_system": max_under[0],
            "max_underCount_pct": max_under[1],
            "mean_underCount_pct": mean_under,
            "note": ("Exact-string matching under-counts correctness by up to "
                     f"{max_under[1]}% (system {max_under[0]}), mean {mean_under}% "
                     "across systems, recovered by ontology alias-folding "
                     "(mixed-case 'Mount'->'mount'; surface forms "
                     "'de_la_riva'->'de_la_riva_guard', 'guard_pass'->'passing_guard'). "
                     "On the IoU-matched T1 position axis the gap is near-zero "
                     "because both the v0 GT macro-position field and the pipeline "
                     "outputs already use canonical lowercase tokens; the gap is "
                     "concentrated on the technique-label axis."),
        },
        "oversegmentation": {
            "min": {"system": overseg[0][0], "ratio": overseg[0][1]},
            "max": {"system": overseg[-1][0], "ratio": overseg[-1][1]},
            "note": ("pred_segs/gt_segs; exp2 pass1_fixed emits 241 segments = "
                     "7.53x the 32 GT macro segments (a pathology MoveBench surfaces)."),
        },
    }

    out = dict(d)
    out["group_summary_semantic_T1"] = group_summary
    out["headline"] = headline
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ---------------- Markdown ---------------- #
    def row(r, regime):
        s = r[regime]
        return (f"| {r['system']} | {regime} | {s['macro_f1']:.4f} | "
                f"{s['duration_weighted_macro_f1']:.4f} | "
                f"{s['temporal_iou_mean_matched']:.4f} | {s['matched_positions']}/{n_gt} | "
                f"{s['label_micro_f1']:.4f} | {s['ontology_violation_rate']:.4f} | "
                f"{r['oversegmentation_ratio']:.2f} | {r['coverage_fraction']:.4f} |")

    lines = []
    lines.append("# MoveBench v0 — Real Metric-Vector Scores (D1)")
    lines.append("")
    lines.append("**What this is.** Every available model output in the experiment "
                 "repository, converted to the MoveBench prediction schema "
                 "(timestamps + ontology-mappable position/technique tokens only; "
                 "all names/ids/free-text dropped, privacy-scanned) and scored "
                 "against the v0 ground truth with the bundled MoveBench scorer, in "
                 "**two regimes**: STRICT (exact string match) and SEMANTIC "
                 "(ontology-normalized equivalence-class / alias-folding, "
                 "abstain-aware). All numbers are **recomputed this session** by "
                 "deterministic scoring of stored artifacts — no model inference, "
                 "no network.")
    lines.append("")
    lines.append("**Hard caveat (frames everything below).** MoveBench v0 has "
                 "exactly **one** annotated sample (a single ~16-min grappling "
                 "match: 32 macro segments / 62 subsegments / 122 labels, single "
                 "annotator). These are **single-sample developer measurements, "
                 "not a leaderboard.** No confidence intervals are computable from "
                 "n=1. All scored predictions are in-sample dev artifacts on the "
                 "same match. The tautological P1 50-video GT==prediction case is "
                 "**excluded** (degenerate by construction).")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    ha = headline["contrast_a_naive_vs_cv_grounded"]
    lines.append(f"- **(a) Naive vs CV-grounded.** Naive single-pass holistic VLM "
                 f"(no CV grounding) reaches **T1 position macro-F1 "
                 f"{ha['naive_single_pass']['semantic_T1_macro_f1']:.4f}**, matching "
                 f"only **{ha['naive_single_pass']['matched_positions']}** GT "
                 f"positions. The best CV-grounded pipeline "
                 f"(`{ha['best_cv_grounded']['system']}`) reaches "
                 f"**{ha['best_cv_grounded']['semantic_T1_macro_f1']:.4f}** "
                 f"(**{ha['best_cv_grounded']['matched_positions']}** matched) — "
                 f"a **~{ha['relative_improvement_x']}x** relative gain in position "
                 f"macro-F1. There is a clean monotone trend across the development "
                 f"lineage (naive < Exp2 multi-pass < Exp3 CV-cached < Exp4 v2/v3).")
    hb = headline["contrast_b_strict_to_semantic_gap"]
    lines.append(f"- **(b) Strict -> semantic gap (how much exact-match "
                 f"under-counts).** On an alignment-free pooled position+label "
                 f"token F1, exact-string matching under-counts correctness by up "
                 f"to **{hb['max_underCount_pct']}%** "
                 f"(`{hb['max_underCount_system']}`), mean "
                 f"**{hb['mean_underCount_pct']}%** across the 16 runs, recovered "
                 f"purely by ontology alias-folding (mixed-case `Mount`->`mount`; "
                 f"surface forms `de_la_riva`->`de_la_riva_guard`, "
                 f"`guard_pass`->`passing_guard`). On the IoU-matched T1 position "
                 f"axis the gap is near-zero — both the GT macro-position field and "
                 f"the pipelines already emit canonical lowercase position tokens, "
                 f"so the semantic uplift is concentrated on the technique-label axis.")
    ho = headline["oversegmentation"]
    lines.append(f"- **Oversegmentation.** pred_segs/gt_segs ranges "
                 f"**{ho['min']['ratio']:.2f}x** (`{ho['min']['system']}`) to "
                 f"**{ho['max']['ratio']:.2f}x** (`{ho['max']['system']}`, 241 segs "
                 f"vs 32 GT — a pathology the benchmark surfaces).")
    lines.append("")
    lines.append("## Per-system metric vector (STRICT and SEMANTIC)")
    lines.append("")
    lines.append("| system | regime | T1 macro-F1 | T1 dur-wtd F1 | T1 mean IoU | "
                 "matched | T3 label micro-F1 (oracle-idx) | ontology-violation rate | "
                 "overseg (pred/gt) | coverage |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -x["semantic"]["macro_f1"]):
        lines.append(row(r, "strict"))
        lines.append(row(r, "semantic"))
    lines.append("")
    lines.append("## Strict -> semantic gap, isolated (alignment-free pooled token F1)")
    lines.append("")
    lines.append("The T1 position axis hides the alias-folding effect (positions "
                 "already canonical) and the T3 oracle-index label-F1 is unreliable "
                 "for these non-GT-aligned predictions. This pooled measure pools "
                 "ALL GT position+label tokens vs ALL predicted tokens (as "
                 "multisets) and computes micro-F1 — isolating exactly what "
                 "alias-folding recovers, with no temporal alignment dependency.")
    lines.append("")
    lines.append("| system | strict pooled-F1 | semantic pooled-F1 | gap | "
                 "exact-match under-counts by |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -x["pooled_token_f1_gap"]):
        ps = r["pooled_token_f1_strict"]["f1"]
        pm = r["pooled_token_f1_semantic"]["f1"]
        under = f"{round(100*(pm-ps)/pm,1)}%" if pm > 0 else "n/a"
        lines.append(f"| {r['system']} | {ps:.4f} | {pm:.4f} | "
                     f"{r['pooled_token_f1_gap']:.4f} | {under} |")
    lines.append("")
    lines.append("## Group rollup (semantic T1 position macro-F1)")
    lines.append("")
    lines.append("| group | runs | best macro-F1 | mean macro-F1 | best mean IoU | "
                 "max matched / 32 |")
    lines.append("|---|---|---|---|---|---|")
    for g in group_summary:
        lines.append(f"| {g['group']} | {g['n_runs']} | "
                     f"{g['best_semantic_macro_f1']:.4f} | "
                     f"{g['mean_semantic_macro_f1']:.4f} | "
                     f"{g['best_mean_matched_iou']:.4f} | "
                     f"{g['max_matched_positions']} |")
    lines.append("")
    lines.append("## Method & integrity")
    lines.append("")
    lines.append("- **Conversion.** `results/_convert_all_outputs.py` (extends "
                 "`_convert_experiment_timeline.py`) maps each raw output to the "
                 "MoveBench prediction schema, carrying only start/end seconds, "
                 "position, and short snake_case technique tokens "
                 "(`sub_position`/`key_actions`/`reasons`/`focus`/`labels`). Names, "
                 "video ids, narratives, coaching free-text, and URLs are dropped; "
                 "name-bearing tokens (e.g. `<athlete-name>_single_leg`) are "
                 "stripped; a "
                 "privacy scan aborts on any leak. Positions are passed through "
                 "verbatim (lowercased) so vocabulary mismatch stays VISIBLE in the "
                 "score, not hidden.")
    lines.append("- **STRICT regime.** Identity normalizer (lowercase only); the "
                 "ontology is NOT consulted; raw exact-string match. This is the "
                 "syntactic floor.")
    lines.append("- **SEMANTIC regime.** Ontology alias-folding "
                 "(`ontology.json`) for positions and labels, plus **abstain-aware** "
                 "(predicted segments whose position folds to the always-allowed "
                 "abstention class `unknown`/`occluded` are removed before scoring, "
                 "so explicit abstention is not penalised as a wrong commitment), "
                 "plus ontology-violation tracking.")
    lines.append("- **Scoring.** `results/_score_all.py` calls the bundled scorer "
                 "(`movebench.metrics.score_segments` / `label_f1` / `coverage`) "
                 "with IoU threshold 0.3. Reproduce: "
                 "`PYTHONPATH=. python3 results/_score_all.py && python3 "
                 "results/_build_report.py`.")
    lines.append("- **T3 caveat.** The T3 label micro-F1 column is "
                 "oracle-segment-by-INDEX; these pipelines do not emit GT-aligned "
                 "segments, so that column is a weak lower bound, not a clean "
                 "oracle-segment score. Read T1 position macro-F1 and the pooled "
                 "token-F1 as the load-bearing numbers; T3 is directional.")
    lines.append("- **Vocabulary cap.** The v0 GT uses `escape` and `sweep` as "
                 "macro-position classes for **12 of 32 segments (37.5%)**; no "
                 "pipeline's 18-position vocabulary emits those tokens, so those 12 "
                 "segments are unmatchable by construction and cap achievable "
                 "macro-F1 below 1.0. This is itself a measured finding.")
    lines.append("- **Excluded.** The P1 50-video GT==prediction 1.0 is "
                 "tautological (degenerate by construction) and is excluded.")
    lines.append("")
    lines.append("*All inputs read-only. No paper, shared doc, or repo was "
                 "modified. Generated by the MoveBench D1 stage.*")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
