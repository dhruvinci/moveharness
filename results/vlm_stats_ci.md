# VLM benchmark: error bars and 95% confidence intervals

All intervals are 95%. Bootstrap = nonparametric percentile bootstrap, 10,000 resamples, numpy `default_rng(seed=1234)`. Wilson = Wilson score interval (z=1.95996). std = sample std (ddof=1). Source files: `vlm_multivideo.json`, `vlm_ablation_measured.json`, `vlm_technique_id.json`, `vlm_runs_raw.json`, `real_scores.json`.

## (a) Multi-video distributions (N=10, bootstrap CI on the mean)

| Metric | N | Mean | Std (ddof=1) | 95% CI (bootstrap) |
|---|---|---|---|---|
| Cost per match-edit (USD per video) | 10 | 0.0205 | 0.0154 | [0.0117, 0.03] |
| Timeline coverage fraction | 10 | 0.9895 | 0.0174 | [0.9779, 0.9973] |
| Segments per minute (segmentation density) | 10 | 7.053 | 3.5818 | [5.1392, 9.2962] |
| Oversegmentation ratio (pred_segs / expected) | 10 | 0.5433 | 0.385 | [0.32, 0.7766] |
| End-to-end latency (s) | 10 | 45.922 | 26.5177 | [31.0698, 62.6802] |
| Segments per match (count) | 10 | 12.8 | 10.6019 | [6.9, 19.4] |

## (b) Paired grounded-vs-ungrounded ablation deltas (N=10 paired, bootstrap CI on paired differences)

Delta = grounded - ungrounded. Significant = 95% bootstrap CI excludes 0.

| Metric | N | Mean delta (G-U) | 95% CI | Significant? |
|---|---|---|---|---|
| Segments per match (count) | 10 | -2.5 | [-10.2, 3.2] | no |
| Segments per minute | 10 | -0.5834 | [-2.2072, 0.8046] | no |
| Oversegmentation ratio | 10 | -0.0867 | [-0.3551, 0.135] | no |
| Ontology-violation rate | 10 | 0.1809 | [-0.07, 0.4525] | no |
| Latency (s) | 10 | 5.006 | [0.885, 9.676] | YES |
| Cost (USD) | 10 | -0.0002 | [-0.0037, 0.0023] | no |
| Output tokens | 10 | -450.0 | [-1856.3, 546.1225] | no |
| Coverage fraction | 10 | 0.0 | [0.0, 0.0] | no |

**Significant deltas (CI excludes 0): Latency (s).** All other paired deltas have 95% CIs spanning 0 (not significant at N=10).

## (c) Technique-ID accuracy (N=119 clips, Wilson 95% CI)

| Axis | k / n | Accuracy | 95% Wilson CI |
|---|---|---|---|
| Position top-1 | 68/119 | 57.1% | [48.2%, 65.7%] |
| Technique match | 68/119 | 57.1% | [48.2%, 65.7%] |
| Either match | 90/119 | 75.6% | [67.2%, 82.5%] |

Per-tag technique-match (Wilson 95% CI; counts reconstructed from reported per-tag pct):

| Tag | clips | Technique match | 95% CI | Position acc | 95% CI |
|---|---|---|---|---|---|
| standing | 36 | 77.8% | [61.9%, 88.3%] | 41.7% | [27.1%, 57.8%] |
| guardpass | 20 | 40.0% | [21.9%, 61.3%] | 70.0% | [48.1%, 85.5%] |
| singleleg | 18 | 72.2% | [49.1%, 87.5%] | 50.0% | [29.0%, 71.0%] |
| destabilization | 19 | 73.7% | [51.2%, 88.2%] | 52.6% | [31.7%, 72.7%] |
| turtleentry | 19 | 73.7% | [51.2%, 88.2%] | 36.8% | [19.1%, 59.0%] |
| underhook | 16 | 56.2% | [33.2%, 76.9%] | 75.0% | [50.5%, 89.8%] |
| wrestling | 24 | 87.5% | [69.0%, 95.7%] | 33.3% | [18.0%, 53.3%] |
| legentanglement | 14 | 71.4% | [45.4%, 88.3%] | 92.9% | [68.5%, 98.7%] |
| takedown | 17 | 94.1% | [73.0%, 99.0%] | 47.1% | [26.2%, 69.0%] |
| mount | 11 | 36.4% | [15.2%, 64.6%] | 54.5% | [28.0%, 78.7%] |
| openguard | 13 | 69.2% | [42.4%, 87.3%] | 84.6% | [57.8%, 95.7%] |
| backtake | 11 | 45.5% | [21.3%, 72.0%] | 63.6% | [35.4%, 84.8%] |
| sweep | 12 | 58.3% | [32.0%, 80.7%] | 91.7% | [64.6%, 98.5%] |
| submission | 12 | 41.7% | [19.3%, 68.0%] | 58.3% | [32.0%, 80.7%] |
| gripfight | 15 | 46.7% | [24.8%, 69.9%] | 66.7% | [41.7%, 84.8%] |

## (d) v0 GT segment match (32 GT macro-segments, Wilson 95% CI on matched-position rate)

| System | Matched positions | Rate | 95% Wilson CI | Mean matched IoU |
|---|---|---|---|---|
| exp4_v2_skeleton_run2 (best CV-grounded) | 6/32 | 18.8% | [8.9%, 35.3%] | 0.5276 |
| naive single-pass | 1/32 | 3.1% | [0.6%, 15.7%] | 0.875 |

> v0 GT video has 32 macro-segments (n_samples=1 video). Matched-position rate is a binomial proportion -> Wilson 95% CI. Per-segment matched-IoU bootstrap is NOT computed: real_scores.json stores only the MEAN matched IoU, not the per-segment array, so resampling individual IoUs would require fabricating data.
