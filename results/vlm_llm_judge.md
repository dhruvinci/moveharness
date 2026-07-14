# MoveBench v0 — V3 LLM-as-Judge Semantic Correctness (D3)

**What this is.** A real *semantic-correctness* number for every model output in the experiment repository, produced by an LLM judge (gemini-2.5-flash) rather than by string-matching or alias-folding. For each system we temporally align its predicted segments to the v0 ground-truth (GT) macro segments (greedy max-IoU, threshold 0.3 — the **same** alignment the strict T1 scorer uses, so the two numbers are directly comparable), then ask the judge to rate the predicted (position + technique labels) vs the GT (position + sub_position + rolled-up subsegment labels) on a **0/1/2** scale (2 = semantically correct incl. paraphrase, 1 = partial, 0 = wrong). Scores are normalized /2 to [0,1]. This gives the SEMANTIC-correctness quantity we previously only had as a strict exact-match *floor*.

**Hard caveat (frames everything).** MoveBench v0 has exactly **one** annotated sample (a single ~16-min grappling match, 32 macro segments, single annotator). These are **single-sample developer measurements, not a leaderboard**; no confidence intervals are computable from n=1. All scored predictions are in-sample dev artifacts on the same match. Everything below is anonymized aggregate statistics on ontology tokens + timestamps only — no user ids, paths, or raw annotator free-text entered any judge prompt.

## Headline numbers

- **Real semantic score.** The judge's mean semantic-correctness (GT-anchored, [0,1]) tops out at **0.2031** (`hmm_cv_only_baseline_raw`). Among CV-grounded pipelines the leaders cluster at **0.1875** (exp4 v1/v2/v3 skeleton runs). Mean across the 16 strict-scored systems is **0.1094**.
- **Strict -> judge gap (the thing we wanted).** The LLM judge is systematically *higher* than the strict exact-match macro-F1: mean gap **+0.0372**, median **+0.0352**, positive for **15/16** systems. The largest uplift is **+0.0827** (`exp3_stage3_merged_timeline`) — a system whose exact-string macro-F1 is near-zero but which the judge credits with partially/fully correct positions. The single negative gap (**-0.0326**, `exp4_v2_skeleton_run2`) is the strict-F1 leader: its high IoU-match count includes matched-but-semantically-wrong segments the judge penalizes. This is a **real-judge** strict->semantic gap, not the alias-folding gap reported in real_scores.md.
- **Ranking is preserved.** Spearman rho between judge and strict rankings = **0.871** — the judge lifts absolute scores but largely keeps the development-lineage ordering (naive / Exp2 < Exp3 < Exp4).
- **Judge self-consistency.** Re-judging 10 segments twice (temperature 0, thinking disabled): exact agreement **1.00**, within-1 **1.00**, Cohen kappa **1.00**, mean abs score diff **0.00**. At T=0 with thinking off the judge is effectively deterministic; this measures judge *stability*, not inter-annotator agreement.

## Per-system: LLM-judge semantic score vs strict macro-F1

Judge score is GT-anchored over all 32 GT macro segments: matched segments get the judge's 0/1/2 (normalized /2), unmatched GT segments (no prediction overlaps at IoU>=0.3) are a miss scored 0. `judged 0/1/2` is the score distribution among judged (matched) segments only.

| system | judge semantic [0,1] | strict macro-F1 | strict->judge gap | judged 0/1/2 | miss | abstain |
|---|---|---|---|---|---|---|
| hmm_cv_only_baseline_raw | 0.2031 | n/a (not in real_scores.md) | n/a | 14/13/0 | 5 | 0 |
| exp4_v1_skeleton_run3 | 0.1875 | 0.1331 | +0.0544 | 19/6/3 | 4 | 0 |
| exp4_v2_skeleton_run1 | 0.1875 | 0.1650 | +0.0225 | 12/10/1 | 9 | 0 |
| exp4_v3_skeleton_run1 | 0.1875 | 0.1317 | +0.0558 | 6/6/3 | 17 | 0 |
| exp4_v3_skeleton_run2 | 0.1875 | 0.1361 | +0.0514 | 16/2/5 | 9 | 0 |
| exp3_stage1_cv_cached_timeline | 0.1562 | 0.0991 | +0.0571 | 23/4/3 | 2 | 0 |
| exp4_v2_skeleton_run2 | 0.1562 | 0.1888 | -0.0326 | 15/8/1 | 8 | 0 |
| hmm_cv_only_baseline | 0.1250 | n/a (not in real_scores.md) | n/a | 9/6/1 | 16 | 0 |
| exp3_stage1_v33_pass1 | 0.0938 | 0.0119 | +0.0819 | 3/4/1 | 24 | 0 |
| exp3_stage3_merged_timeline | 0.0938 | 0.0111 | +0.0827 | 2/4/1 | 25 | 0 |
| exp4_combined_stage1_stage2 | 0.0938 | 0.0774 | +0.0164 | 25/6/0 | 1 | 0 |
| exp4_v1_skeleton_run1 | 0.0938 | 0.0774 | +0.0164 | 25/6/0 | 1 | 0 |
| exp3_stage1_v33_compressed | 0.0781 | 0.0564 | +0.0217 | 11/3/1 | 17 | 0 |
| exp2_multipass_pass3_synthesis | 0.0625 | 0.0220 | +0.0405 | 8/2/1 | 21 | 0 |
| naive_single_pass_holistic | 0.0625 | 0.0152 | +0.0473 | 9/2/1 | 20 | 0 |
| exp4_v1_skeleton_run2 | 0.0469 | 0.0171 | +0.0298 | 12/1/1 | 18 | 0 |
| exp2_multipass_pass1_enhanced | 0.0312 | 0.0084 | +0.0228 | 3/0/1 | 28 | 0 |
| exp2_multipass_pass1_fixed | 0.0312 | 0.0042 | +0.0270 | 5/0/1 | 26 | 0 |

## Strict (exact-match) vs alias-fold (semantic) vs LLM-judge — three regimes

real_scores.md established two regimes: STRICT exact-string macro-F1, and a SEMANTIC regime that was *alias-folding only* (ontology synonym normalization). On the T1 position axis that semantic uplift was near-zero (positions already canonical lowercase). This V3 adds a THIRD, stronger regime — a real LLM judge that credits paraphrase, family-membership, and partial correctness that no alias table encodes. The judge gap (mean +0.0372) is the first non-trivial strict->semantic lift on the position axis, because the judge can award partial credit (score 1) for right-family / near-miss positions that both exact-match and alias-folding score as 0.

## Method & integrity

- **Judge.** gemini-2.5-flash, temperature 0.0, thinking disabled (thinking_budget=0), JSON response format. System prompt frames the task as an MMA / grappling "label dataset" with a controlled position+technique ontology and instructs synonym/paraphrase tolerance. The judge sees only normalized ontology tokens + timestamps for both GT and prediction.
- **Alignment.** Greedy max-IoU GT->pred at threshold 0.3 (identical to the strict T1 scorer in _score_all.py), so judge scores and strict macro-F1 share a denominator and matching rule. Total judged matched segments across all systems: 325; total misses (GT segs with no IoU>=0.3 prediction): 251.
- **Abstain-aware.** A matched prediction whose position folds to unknown/occluded/unclear is recorded as ABSTAIN and excluded from the scored denominator (explicit abstention is not punished as a wrong commit). In this run **0** matched segments abstained — no pipeline emits abstention positions on matched windows.
- **Robustness.** Concurrency capped at 4 in-flight; exponential backoff with jitter on 429/5xx/rate-limit/quota; wall-clock time-boxed to ~35 min; failures are skipped+logged, never crash the run. This run: **325 judge calls, 0 failures, 0 retries, 97.9s wall-clock, timed_out=False**.
- **Measured cost.** 151,094 input + 9,527 output tokens (judge) -> **~$0.06915** at gemini-2.5-flash standard pricing ($0.30/1M in, $2.50/1M out). Self-consistency added 20 more calls. Total well under the $10 budget.
- **Privacy.** No athlete names, video ids, gym names, annotator id, local paths, or raw human-annotator free-text notes entered any prompt or output. GT free-text `note` fields were dropped; only position/sub_position/technique tokens + timestamps were sent. Outputs are anonymized aggregate statistics.
- **Strict baseline source.** strict macro-F1 column parsed from results/real_scores.md (the 16 strict-regime rows). The two HMM CV baselines are not in real_scores.md (they live in real_hmm_baseline.md), so their gap is marked n/a.

*All inputs read-only. No paper.tex, shared doc, or repo source was modified. Judge artifacts: results/vlm_llm_judge.json (per-segment detail), results/vlm_llm_judge_consistency.json. Generated by the MoveBench V3 LLM-judge stage.*
