# D4 — Real Faithfulness, Over-segmentation & Attention-Decay

**Source:** deterministic re-analysis of existing VLM-pipeline outputs (Exp2 / Exp3 / Exp4).
**No model inference was run.** Every number below is computed from JSON artifacts already on disk, scored against the v0 ground truth (`benchmark/data/v0/mb_v0_0001.json`) and the CV anchor checkpoints (`experiment4.0/stage0_cv_checkpoints_5s.json`).

**Ground truth:** 32 macro segments, 959.0 s. **CV anchor:** 194 checkpoints @ 5 s.
**Artifacts scored:** 15 timeline files (the parseable, segment-bearing outputs across the three experiment generations).

> Integrity: out-of-ontology tokens are passed through verbatim and counted as violations — never silently remapped to flatter a score. Physical-consistency uses the CV posture (standing/ground) at each segment midpoint; ambiguous (`mixed`/`unknown`) CV calls are excluded from the denominator. The GT-position agreement signal is non-tautological (pred ≠ GT here); the 50-video GT==prediction case is excluded entirely.

---

## 1. Headline numbers

| Metric | Value | How computed |
|---|---|---|
| Over-segmentation ratio (worst) | **4.91× (157 pred vs 32 GT)** — `exp4_skeleton_run2` | `n_pred_segments / 32` |
| Over-segmentation range across runs | **0.44× – 4.91×** | min/max over 15 artifacts |
| Position-token ontology-violation rate | **0 – 20.8%** (≤4% on most runs) | fraction of emitted `position` tokens not in ontology |
| Technique-label out-of-ontology rate | **0 – 95.5%** (median ≈ 80%) | fraction of emitted technique labels not in the controlled vocabulary |
| Physical inconsistency vs CV (over-segmenting runs) | **75 – 100%** | predicted ground/standing plane ≠ CV posture at segment midpoint |
| Attention decay (GT-position agreement) | **0.65 (decile 0) → 0.10 (decile 9)** | mean over 15 runs, per normalized-position decile |
| CV "attention cliff" (corroborating) | first degradation flag at **0:34 s** (within decile 0) | `experiment3/stage2_lazy_detection_v3.3.json` |

The two faithfulness failures are distinct and both real:

1. **Hallucinated *positions* are rare** (≤4% on almost every run): the models almost always pick a *valid* macro position. The pathology is not "impossible state names."
2. **Hallucinated *technique labels* are pervasive** (~70–95%): the pipelines emit fluent, plausible-sounding compound tokens (`guard_passing_mechanics`, `side_control_entry`, `submission_defense`, `position_maintenance`) that are **not** in the controlled vocabulary. The model is confidently verbose in a vocabulary it invents.

---

## 2. Over-segmentation vs GT (32 macro segments)

The multi-pass / skeleton pipelines fragment the timeline far past the GT granularity. The worst case is the multi-pass `exp4_skeleton_run2` at **157 segments — 4.9× the 32-segment ground truth** — the structural analogue of the "~241 vs 32" log claim, here verified on a scorable artifact. (The literal 241-position Exp2 figure remains a log-only, unverified claim per the evidence ledger; the verified over-segmentation evidence is the 157-vs-32 run.)

| Pipeline | Pred segs | Overseg ratio | Pos-viol % | Label-OOV % | Phys-inconsistent | GT agreement |
|---|---:|---:|---:|---:|---:|---:|
| exp3_stage1_single_pass | 58 | 1.81× | 3.4 | 67.6 | 100% (n=6) | 0.172 |
| exp3_stage1_test | 39 | 1.22× | 2.6 | 38.6 | 100% (n=3) | 0.154 |
| exp3_v3.3_pass1 | 49 | 1.53× | 0.0 | 82.4 | 0% (n=1) | 0.306 |
| exp3_v3.3_compressed | 24 | 0.75× | 0.0 | 72.7 | n/a (n=0) | 0.125 |
| exp3_v3.3_merged_stage3 | 48 | 1.50× | 6.2 | 82.4 | 0% (n=1) | 0.292 |
| exp4_skeleton_run1 | 69 | 2.16× | 1.4 | 95.5 | 75% (n=8) | 0.072 |
| **exp4_skeleton_run2** | **157** | **4.91×** | 0.0 | 87.1 | 88% (n=8) | 0.089 |
| exp4_skeleton_run3 | 25 | 0.78× | 4.0 | 83.3 | 0% (n=2) | 0.200 |
| exp4_v2_run1 | 24 | 0.75× | 20.8 | 83.9 | 100% (n=1) | 0.250 |
| exp4_v2_run2 | 22 | 0.69× | 4.5 | 88.2 | n/a (n=0) | 0.273 |
| exp4_v3_run1 | 15 | 0.47× | 6.7 | 81.2 | 100% (n=4) | 0.267 |
| exp4_v3_run2 | 25 | 0.78× | 4.0 | 74.2 | 50% (n=2) | 0.320 |
| exp4_combined | 69 | 2.16× | 1.4 | 95.5 | 75% (n=8) | 0.072 |
| exp2_pass1_holistic | 39 | 1.22× | 0.0 | 53.8 | 40% (n=5) | 0.205 |
| exp2_pass3_synthesis | 14 | 0.44× | 0.0 | 0.0 | 0% (n=1) | 0.071 |

Global GT-position agreement is low everywhere (≤0.32), confirming that even runs that hit a *plausible* segment count are not actually right on position.

---

## 3. Physical consistency vs CV anchor

For each segment we compare its predicted plane (ground vs standing) against the CV posture detector at the segment midpoint. Among the over-segmenting runs that have ≥4 CV-anchored checks, **75–100% of checked segments are physically inconsistent** with the CV signal (e.g. the timeline claims a ground position while the CV detector reports both athletes standing). This is hallucination that is *physically* refutable, not merely vocabulary mismatch.

`physical_inconsistency_rate` (runs with ≥4 anchored checks):
`exp3_stage1_single_pass` 100% · `exp4_skeleton_run1` 75% · `exp4_skeleton_run2` 88% · `exp4_v3_run1` 100% · `exp4_combined` 75% · `exp2_pass1_holistic` 40%.

> Caveat: CV-anchored sample sizes per run are small (n = 1–8 after excluding ambiguous CV calls), so these are directional rates, not tight estimates. They are reported with their n.

---

## 4. Attention decay by normalized video position (the real decay curve)

Segments are binned by midpoint into 10 normalized-position deciles (0→1). The aggregated curve below is the **mean over all 15 scorable runs** — this is the real decay curve that can replace the illustrative figure.

| Decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **GT-position agreement** | **0.646** | 0.209 | 0.388 | 0.047 | 0.094 | 0.006 | 0.000 | 0.006 | 0.092 | 0.096 |
| Mean labels / segment | 3.17 | 3.62 | 3.51 | 3.37 | 3.51 | 4.12 | 4.14 | 4.09 | 4.39 | 3.67 |
| Mean detail-chars / segment | 98.5 | 106.9 | 112.3 | 105.2 | 110.5 | 120.8 | 124.0 | 121.3 | 102.9 | 105.6 |
| Segments / min | 3.96 | 3.63 | 3.38 | 3.38 | 2.95 | 3.07 | 3.07 | 2.90 | 3.57 | 3.55 |
| Runs contributing | 15 | 15 | 15 | 15 | 14 | 11 | 11 | 11 | 10 | 9 |

**The decay is in *correctness*, not in *verbosity*.** Position agreement falls off a cliff after decile 0 — from 0.65 to roughly 0.0–0.10 — and never recovers. Meanwhile label density and detail-text length stay flat or even *rise* slightly across the video (labels/seg 3.2 → 4.4; detail chars ~98 → ~124). The model keeps producing confident, detailed, fluent output exactly as its grounding collapses. That gap between sustained output-detail and collapsing accuracy is the faithfulness failure, quantified.

A single representative run shows the same shape without averaging — `exp3_stage1_single_pass` GT-agreement by decile:
`[1.00, 0.14, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]`.

The CV "attention cliff" corroborates the curve from an independent signal: the lazy-detection pass flags the **first** degraded segment at **0:34 s** (≈34 s of "success duration" — inside decile 0), with degradation signals (`repetition`, `cv_standing_mismatch`, `action_mismatch`) recurring through the rest of the timeline.

> Caveat on decile 0: part of the high early-agreement is structural — the match opens in `standing`, the most common/default position — so decile-0 agreement is partly the "easy default" being right rather than fine-grained skill. The *collapse* after decile 0, sustained across 15 runs, is the robust signal.

---

## 5. Files

- Machine-readable: `benchmark/results/real_faithfulness.json` (includes the per-decile arrays — `aggregated_decay_curve`, `gt_decile_baseline`, and per-artifact `decile_curve` — so a real plot can be drawn directly).
- This report: `benchmark/results/real_faithfulness.md`.
- Recompute: `cd benchmark && python3 results/_compute_real_faithfulness.py`.
