# MoveBench v0 — Real Per-Module Ablation (post-hoc toggles)

**No new inference.** Every number below is computed by re-scoring EXISTING machine outputs against the v0 ground truth with the `movebench.metrics` scorer, toggling one harness module at a time.

- Ground truth: `data/v0/mb_v0_0001.json (32 macro segments, 64 subsegments, 959s, single annotator)`
- IoU threshold: 0.3
- With-harness (CV-grounded) prediction: Exp4 two-stage skeleton+detail v3 run2 (25 segments)
- Without-harness (ungrounded) prediction: multi-pass holistic VLM, no CV grounding (Exp2-family pass1, 39 segments)

> **Caveat (frames everything).** Single annotated sample; every delta is a single-sample developer measurement, not a leaderboard result. No CIs. The 50-video GT==prediction case is excluded everywhere.

## Ablation table (module -> metric impact)

| Module | Toggle | Key metric | OFF | ON | Δ (ON−OFF) | Type | Direction |
|---|---|---|---|---|---|---|---|
| **M1 measurement-grounding** | ungrounded → CV-grounded | T1 macro-F1 (headline) | 0.014 | 0.1361 | +0.1221 | direct | grounding raises macro-F1 ~9.7x |
| M1 (cont.) | | matched / 32 | 1 | 6 | +5 | direct | 1 → 6 matched positions |
| M1 (cont.) | | T1 mean IoU (read with care) | 0.875 | 0.7229 | -0.1521 | direct | ungrounded IoU is over 1 matched span only |
| **M2 semantic folding** | exact-string → ontology fold | T1 macro-F1 | 0.1361 | 0.1361 | +0 | direct | folding ≥ exact |
| M2 (cont.) | | matched / 32 | 6 | 6 | +0 | direct | |
| **M3 temporal smoothing** | raw → merge micro-seg | oversegmentation (pred/GT) | 1.219 | 1.188 | -0.031 | direct | merging reduces oversegmentation |
| M3 (cont.) | | pred segments | 39 | 38 | -1 | direct | |
| M3 (cont.) | | T1 macro-F1 | 0.014 | 0.0154 | +0.0014 | direct | as measured |
| **M4 abstain-aware** | counted → excluded | pred segments | 39 | 38 | -1 | direct | 1 abstain span(s) |
| M4 (cont.) | | matched/pred (precision proxy) | 0.0256 | 0.0263 | +0.0007 | direct | abstain removal raises precision |
| **M5 schema constraint** | no-fold → ontology fold | ontology-violation rate | 1 | 0.6076 | -0.3924 | direct | folding lowers violation rate |
| M5 (cont.) | | T3 micro-F1 | 0 | 0.037 | +0.037 | direct | folding ≥ no-fold |

## Direct vs derived

- **Direct measurements:** M1, M2, M3, M4, M5 — each is a real re-score (M2/M5 swap the scorer's normalize fn; M1 uses two real prediction sets; M3/M4 apply a deterministic post-hoc transform to a real prediction and re-score).
- **Derived:** the headline with-vs-without-harness macro-F1 gap (0.014 → 0.1361, Δ=+0.1221, 9.73x) is an arithmetic combination of the M1 direct numbers, not an independent measurement. Mean IoU is deliberately NOT the headline (the ungrounded run's high IoU is over a single matched span).

## Per-module detail

### M1_measurement_grounding

Headline with/without-harness contrast: CV-grounded two-stage prediction vs ungrounded multi-pass holistic VLM (no CV constraints). Both real, unedited outputs.

*Direction:* CV-grounding is the headline win: it RAISES T1 macro-F1 ~9.7x (0.014 -> 0.136) and matched positions 1 -> 6 of 32. The ungrounded run's HIGHER mean matched IoU (0.875) is an ARTIFACT of matching only a single segment (n=1 matched), not better localization; read macro-F1 / matched-count as the load-bearing M1 metrics, not IoU.

*Ungrounded pathology:* The ungrounded multi-pass output emitted DECREASING timestamps in its third window (e.g. 11:30->11:20), so 13/39 predicted spans are zero/negative-length and are clipped to nothing by the scorer. This is a real failure mode of the harness-free pipeline, and it is the main reason the ungrounded run matches only 1/32 GT positions.

### M2_semantic_ontology_folding

Ontology equivalence folding (alias/synonym/hierarchy normalization) ON vs OFF, scored on the SAME Exp4 prediction. OFF = exact lowercased string match.

*Direction:* Folding can only RAISE or hold matched-position F1 (it merges surface variants onto canonical ids). The measured delta on Exp4's position field quantifies how much the alias map is doing on this artifact.

### M3_temporal_smoothing

Merge adjacent same-position micro-segments (deterministic stand-in for EMA-over-boundaries) on the ungrounded prediction. Measures oversegmentation ratio (pred/GT) and T1 macro-F1 before/after.

*Direction:* Smoothing REDUCES the predicted-segment count toward GT granularity. On THIS artifact only 1 adjacent same-position pair exists (the multi-pass timeline interleaves positions and resets at window boundaries), so the merge is nearly inert (39 -> 38). The mean-IoU drop is a side-effect of that single merge widening one matched span; reported as measured, not inflated. The documented over-segmentation extreme for this pipeline family (Exp2 reached 241 segments, 7.5x GT) is NOT recomputed here because that 241-segment artifact is not in the converted prediction set.

### M4_abstain_aware

Treat predicted unknown/occluded/paused spans as ABSTAIN (excluded from FP accounting) vs counting them as ordinary segments, on the ungrounded prediction.

*Direction:* Abstain-aware removes non-committal spans so they cannot be scored as false positives; on this artifact only 1 span(s) are abstain, so the measured impact is small but its sign/magnitude is reported honestly.

### M5_schema_constraint_violation_rate

Ontology-violation rate (fraction of predicted technique labels outside the v0 ontology) WITH folding vs WITHOUT, on the Exp4 prediction's labels (T3 oracle-segment regime, approximate index alignment).

*Direction:* Schema-constraint folding LOWERS the apparent violation rate (canonical aliases get recognized as in-ontology) and can only hold-or-raise T3 F1. The delta quantifies the schema module's effect on this artifact.

*Note:* T3 here is the same approximate oracle-segment alignment used in baselines.md (predictions are not GT-boundary-aligned), so read the violation-rate delta as the load-bearing M5 number and T3 F1 as directional.

## Integrity

- No new model inference; all inputs are stored artifacts.
- The 50-video GT==prediction tautology is excluded everywhere.
- n=1 annotated sample: all deltas are single-sample developer measurements, not leaderboard results; no confidence intervals.
- T3-based numbers (M5) use the same approximate oracle-segment index alignment as `baselines.md`; the violation-rate delta is the load-bearing M5 figure, T3 F1 is directional.

*Generated by `results/real_ablation.py`. Inputs read-only; no paper or shared doc modified; predictions privacy-scanned.*
