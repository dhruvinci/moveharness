# MoveBench v0 — Real Metric-Vector Scores (D1)

**What this is.** Every available model output in the experiment repository, converted to the MoveBench prediction schema (timestamps + ontology-mappable position/technique tokens only; all names/ids/free-text dropped, privacy-scanned) and scored against the v0 ground truth with the bundled MoveBench scorer, in **two regimes**: STRICT (exact string match) and SEMANTIC (ontology-normalized equivalence-class / alias-folding, abstain-aware). All numbers are **recomputed this session** by deterministic scoring of stored artifacts — no model inference, no network.

**Hard caveat (frames everything below).** MoveBench v0 has exactly **one** annotated sample (a single ~16-min grappling match: 32 macro segments / 62 subsegments / 122 labels, single annotator). These are **single-sample developer measurements, not a leaderboard.** No confidence intervals are computable from n=1. All scored predictions are in-sample dev artifacts on the same match. The tautological P1 50-video GT==prediction case is **excluded** (degenerate by construction).

## Headline numbers

- **(a) Naive vs CV-grounded.** Naive single-pass holistic VLM (no CV grounding) reaches **T1 position macro-F1 0.0165**, matching only **1/32** GT positions. The best CV-grounded pipeline (`exp4_v2_skeleton_run2`) reaches **0.1888** (**6/32** matched) — a **~11.4x** relative gain in position macro-F1. There is a clean monotone trend across the development lineage (naive < Exp2 multi-pass < Exp3 CV-cached < Exp4 v2/v3).
- **(b) Strict -> semantic gap (how much exact-match under-counts).** On an alignment-free pooled position+label token F1, exact-string matching under-counts correctness by up to **31.6%** (`naive_single_pass_holistic`), mean **6.6%** across the 16 runs, recovered purely by ontology alias-folding (mixed-case `Mount`->`mount`; surface forms `de_la_riva`->`de_la_riva_guard`, `guard_pass`->`passing_guard`). On the IoU-matched T1 position axis the gap is near-zero — both the GT macro-position field and the pipelines already emit canonical lowercase position tokens, so the semantic uplift is concentrated on the technique-label axis.
- **Oversegmentation.** pred_segs/gt_segs ranges **0.44x** (`exp2_multipass_pass3_synthesis`) to **7.53x** (`exp2_multipass_pass1_fixed`, 241 segs vs 32 GT — a pathology the benchmark surfaces).

## Per-system metric vector (STRICT and SEMANTIC)

| system | regime | T1 macro-F1 | T1 dur-wtd F1 | T1 mean IoU | matched | T3 label micro-F1 (oracle-idx) | ontology-violation rate | overseg (pred/gt) | coverage |
|---|---|---|---|---|---|---|---|---|---|
| exp4_v2_skeleton_run2 | strict | 0.1888 | 0.2305 | 0.5276 | 6/32 | 0.0000 | 0.0000 | 0.69 | 1.0000 |
| exp4_v2_skeleton_run2 | semantic | 0.1888 | 0.2305 | 0.5276 | 6/32 | 0.0000 | 0.8824 | 0.69 | 1.0000 |
| exp4_v2_skeleton_run1 | strict | 0.1650 | 0.2208 | 0.6183 | 6/32 | 0.0106 | 0.0000 | 0.75 | 0.9447 |
| exp4_v2_skeleton_run1 | semantic | 0.1650 | 0.2208 | 0.6183 | 6/32 | 0.0213 | 0.8393 | 0.75 | 0.9447 |
| exp4_v3_skeleton_run2 | strict | 0.1361 | 0.1999 | 0.7229 | 6/32 | 0.0195 | 0.0000 | 0.78 | 1.0000 |
| exp4_v3_skeleton_run2 | semantic | 0.1361 | 0.1999 | 0.7229 | 6/32 | 0.0364 | 0.7422 | 0.78 | 1.0000 |
| exp4_v1_skeleton_run3 | strict | 0.1331 | 0.1646 | 0.6661 | 5/32 | 0.0102 | 0.0000 | 0.78 | 1.0000 |
| exp4_v1_skeleton_run3 | semantic | 0.1331 | 0.1646 | 0.6661 | 5/32 | 0.0206 | 0.8333 | 0.78 | 1.0000 |
| exp4_v3_skeleton_run1 | strict | 0.1317 | 0.1663 | 0.3901 | 4/32 | 0.0301 | 0.0000 | 0.47 | 1.0000 |
| exp4_v3_skeleton_run1 | semantic | 0.1317 | 0.1663 | 0.3901 | 4/32 | 0.0625 | 0.8118 | 0.47 | 1.0000 |
| exp3_stage1_cv_cached_timeline | strict | 0.0991 | 0.1223 | 0.4627 | 5/32 | 0.0225 | 0.0000 | 1.81 | 1.0000 |
| exp3_stage1_cv_cached_timeline | semantic | 0.0991 | 0.1223 | 0.4627 | 5/32 | 0.0305 | 0.6104 | 1.81 | 1.0000 |
| exp4_combined_stage1_stage2 | strict | 0.0774 | 0.1079 | 0.6476 | 4/32 | 0.0090 | 0.0000 | 2.16 | 1.0000 |
| exp4_combined_stage1_stage2 | semantic | 0.0774 | 0.1079 | 0.6476 | 4/32 | 0.0183 | 0.9333 | 2.16 | 1.0000 |
| exp4_v1_skeleton_run1 | strict | 0.0774 | 0.1079 | 0.6476 | 4/32 | 0.0090 | 0.0000 | 2.16 | 1.0000 |
| exp4_v1_skeleton_run1 | semantic | 0.0774 | 0.1079 | 0.6476 | 4/32 | 0.0183 | 0.9333 | 2.16 | 1.0000 |
| exp3_stage1_v33_compressed | strict | 0.0564 | 0.0548 | 0.4722 | 2/32 | 0.0135 | 0.0000 | 0.75 | 0.4692 |
| exp3_stage1_v33_compressed | semantic | 0.0564 | 0.0548 | 0.4722 | 2/32 | 0.0561 | 0.5694 | 0.75 | 0.4692 |
| exp2_multipass_pass3_synthesis | strict | 0.0220 | 0.0444 | 0.5333 | 1/32 | 0.0000 | 0.0000 | 0.44 | 0.3754 |
| exp2_multipass_pass3_synthesis | semantic | 0.0220 | 0.0444 | 0.5333 | 1/32 | 0.0000 | 0.0000 | 0.44 | 0.3754 |
| exp4_v1_skeleton_run2 | strict | 0.0171 | 0.0239 | 0.8750 | 1/32 | 0.0090 | 0.0000 | 4.91 | 1.0000 |
| exp4_v1_skeleton_run2 | semantic | 0.0171 | 0.0239 | 0.8750 | 1/32 | 0.0172 | 0.8770 | 4.91 | 1.0000 |
| naive_single_pass_holistic | strict | 0.0152 | 0.0282 | 0.8750 | 1/32 | 0.0000 | 0.0000 | 0.81 | 0.7247 |
| naive_single_pass_holistic | semantic | 0.0165 | 0.0282 | 0.8750 | 1/32 | 0.0000 | 0.9231 | 0.81 | 0.7247 |
| exp3_stage1_v33_pass1 | strict | 0.0119 | 0.0148 | 0.5000 | 1/32 | 0.0761 | 0.0000 | 1.53 | 0.4484 |
| exp3_stage1_v33_pass1 | semantic | 0.0119 | 0.0148 | 0.5000 | 1/32 | 0.1053 | 0.6145 | 1.53 | 0.4484 |
| exp3_stage3_merged_timeline | strict | 0.0111 | 0.0155 | 0.5000 | 1/32 | 0.0773 | 0.0000 | 1.50 | 0.4380 |
| exp3_stage3_merged_timeline | semantic | 0.0111 | 0.0155 | 0.5000 | 1/32 | 0.1069 | 0.6250 | 1.50 | 0.4380 |
| exp2_multipass_pass1_enhanced | strict | 0.0084 | 0.0222 | 0.6250 | 1/32 | 0.0427 | 0.0000 | 3.78 | 0.5787 |
| exp2_multipass_pass1_enhanced | semantic | 0.0084 | 0.0222 | 0.6250 | 1/32 | 0.0621 | 0.5489 | 3.78 | 0.5787 |
| exp2_multipass_pass1_fixed | strict | 0.0042 | 0.0164 | 0.6250 | 1/32 | 0.0420 | 0.0000 | 7.53 | 1.0000 |
| exp2_multipass_pass1_fixed | semantic | 0.0042 | 0.0164 | 0.6250 | 1/32 | 0.0633 | 0.5839 | 7.53 | 1.0000 |

## Strict -> semantic gap, isolated (alignment-free pooled token F1)

The T1 position axis hides the alias-folding effect (positions already canonical) and the T3 oracle-index label-F1 is unreliable for these non-GT-aligned predictions. This pooled measure pools ALL GT position+label tokens vs ALL predicted tokens (as multisets) and computes micro-F1 — isolating exactly what alias-folding recovers, with no temporal alignment dependency.

| system | strict pooled-F1 | semantic pooled-F1 | gap | exact-match under-counts by |
|---|---|---|---|---|
| naive_single_pass_holistic | 0.1512 | 0.2209 | 0.0697 | 31.6% |
| exp2_multipass_pass3_synthesis | 0.1333 | 0.1733 | 0.0400 | 23.1% |
| exp3_stage1_v33_compressed | 0.2707 | 0.2969 | 0.0262 | 8.8% |
| exp4_v1_skeleton_run3 | 0.1871 | 0.2086 | 0.0215 | 10.3% |
| exp4_v3_skeleton_run2 | 0.2238 | 0.2378 | 0.0140 | 5.9% |
| exp4_v1_skeleton_run2 | 0.0739 | 0.0873 | 0.0134 | 15.3% |
| exp4_v2_skeleton_run1 | 0.1561 | 0.1636 | 0.0075 | 4.6% |
| exp2_multipass_pass1_fixed | 0.0696 | 0.0724 | 0.0028 | 3.9% |
| exp2_multipass_pass1_enhanced | 0.0841 | 0.0865 | 0.0024 | 2.8% |
| exp3_stage1_cv_cached_timeline | 0.2078 | 0.2078 | 0.0000 | 0.0% |
| exp3_stage1_v33_pass1 | 0.2045 | 0.2045 | 0.0000 | 0.0% |
| exp3_stage3_merged_timeline | 0.2193 | 0.2193 | 0.0000 | 0.0% |
| exp4_combined_stage1_stage2 | 0.1330 | 0.1330 | 0.0000 | 0.0% |
| exp4_v1_skeleton_run1 | 0.1330 | 0.1330 | 0.0000 | 0.0% |
| exp4_v2_skeleton_run2 | 0.1533 | 0.1533 | 0.0000 | 0.0% |
| exp4_v3_skeleton_run1 | 0.1631 | 0.1631 | 0.0000 | 0.0% |

## Group rollup (semantic T1 position macro-F1)

| group | runs | best macro-F1 | mean macro-F1 | best mean IoU | max matched / 32 |
|---|---|---|---|---|---|
| Naive (no CV grounding) | 1 | 0.0165 | 0.0165 | 0.8750 | 1 |
| Exp2 multi-pass | 3 | 0.0220 | 0.0115 | 0.6250 | 1 |
| Exp3 (CV-cached) | 4 | 0.0991 | 0.0446 | 0.5000 | 5 |
| Exp4 v1 | 3 | 0.1331 | 0.0759 | 0.8750 | 5 |
| Exp4 v2 | 2 | 0.1888 | 0.1769 | 0.6183 | 6 |
| Exp4 v3 (CV-grounded) | 2 | 0.1361 | 0.1339 | 0.7229 | 6 |
| Exp4 composite | 1 | 0.0774 | 0.0774 | 0.6476 | 4 |

## Method & integrity

- **Conversion.** `results/_convert_all_outputs.py` (extends `_convert_experiment_timeline.py`) maps each raw output to the MoveBench prediction schema, carrying only start/end seconds, position, and short snake_case technique tokens (`sub_position`/`key_actions`/`reasons`/`focus`/`labels`). Names, video ids, narratives, coaching free-text, and URLs are dropped; name-bearing tokens (e.g. `<athlete-name>_single_leg`) are stripped; a privacy scan aborts on any leak. Positions are passed through verbatim (lowercased) so vocabulary mismatch stays VISIBLE in the score, not hidden.
- **STRICT regime.** Identity normalizer (lowercase only); the ontology is NOT consulted; raw exact-string match. This is the syntactic floor.
- **SEMANTIC regime.** Ontology alias-folding (`ontology.json`) for positions and labels, plus **abstain-aware** (predicted segments whose position folds to the always-allowed abstention class `unknown`/`occluded` are removed before scoring, so explicit abstention is not penalised as a wrong commitment), plus ontology-violation tracking.
- **Scoring.** `results/_score_all.py` calls the bundled scorer (`movebench.metrics.score_segments` / `label_f1` / `coverage`) with IoU threshold 0.3. Reproduce: `PYTHONPATH=. python3 results/_score_all.py && python3 results/_build_report.py`.
- **T3 caveat.** The T3 label micro-F1 column is oracle-segment-by-INDEX; these pipelines do not emit GT-aligned segments, so that column is a weak lower bound, not a clean oracle-segment score. Read T1 position macro-F1 and the pooled token-F1 as the load-bearing numbers; T3 is directional.
- **Vocabulary cap.** The v0 GT uses `escape` and `sweep` as macro-position classes for **12 of 32 segments (37.5%)**; no pipeline's 18-position vocabulary emits those tokens, so those 12 segments are unmatchable by construction and cap achievable macro-F1 below 1.0. This is itself a measured finding.
- **Excluded.** The P1 50-video GT==prediction 1.0 is tautological (degenerate by construction) and is excluded.

*All inputs read-only. No paper, shared doc, or repo was modified. Generated by the MoveBench D1 stage.*
