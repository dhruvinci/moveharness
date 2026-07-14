# VLM Harness: Multi-Video Distributions (GROUNDED condition)

- Model: **gemini-2.5-flash**
- Generated: 2026-05-30T18:46:45
- Condition: GROUNDED = Stage0 CV anchors (YOLO11n-pose, per-frame) -> Gemini timeline with anchors injected
- Videos with valid grounded run: **N = 10** (attempted 10)
- Total footage analyzed: 22.7 min
- Total Gemini calls (both conditions): 20; total measured cost: **$0.4122**

## Across-N-matches distributions (grounded)

| Metric | mean | median | min | max | std | p25 | p75 |
|---|---|---|---|---|---|---|---|
| duration_s | 136.44 | 112.8 | 10.6 | 321.7 | 108.8191 | 30.7 | 183.3 |
| cv_time_s | 107.23 | 95.6 | 7.3 | 307.8 | 92.4553 | 20.5 | 120.0 |
| latency_s | 45.922 | 38.975 | 18.21 | 86.68 | 25.1569 | 27.56 | 39.79 |
| cost_usd | 0.0205 | 0.0164 | 0.0036 | 0.0464 | 0.0146 | 0.0068 | 0.0292 |
| total_tokens | 51481 | 42651.0 | 9719 | 108970 | 35143.0057 | 15520 | 74634 |
| output_tokens | 2752.2 | 1904.5 | 672 | 6478 | 1996.335 | 1109 | 2891 |
| n_segments | 12.8 | 8.0 | 2 | 30 | 10.0578 | 5 | 17 |
| segs_per_min | 7.053 | 5.706 | 3.435 | 14.218 | 3.398 | 4.431 | 7.201 |
| overseg_ratio | 0.5433 | 0.4 | 0.1 | 1.1 | 0.3652 | 0.25 | 0.85 |
| coverage | 0.9895 | 0.9963 | 0.9434 | 0.9991 | 0.0165 | 0.9924 | 0.9978 |
| end_reach | 0.9895 | 0.9963 | 0.9434 | 0.9991 | 0.0165 | 0.9924 | 0.9978 |
| ontology_violation_rate | 0.3435 | 0.1666 | 0.0 | 1.0 | 0.3939 | 0.0 | 0.3929 |
| avg_confidence | 0.9757 | 1.0 | 0.8111 | 1.0 | 0.0568 | 0.9955 | 1.0 |
| conf_decay_slope | -0.0009 | 0.0 | -0.0067 | 0.0012 | 0.0026 | 0.0 | 0.0 |

### Headline (grounded, across N matches)
- Cost/match: **$0.0205** mean (median $0.0164); range $0.0036-$0.0464
- Coverage: **99.0%** mean (median 99.6%)
- Segments/match: **12.8** mean (median 8); 7.05 segs/min
- Oversegmentation ratio (segs / prompt-expected midpoint): **0.54x** mean
- Gemini latency: **45.9s** mean; CV (stage0) time: 107.2s mean
- Ontology-violation rate (position field not in ontology): 34.4% mean

## Per-video (anonymized)

| Vid | dur(s) | cv(s) | anchors | segs | segs/min | overseg | cov | end | lat(s) | $ | tokens | viol_rate | conf | conf_decay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V01 | 11 | 7.3 | 9 | 2 | 11.321 | 0.1 | 0.9434 | 0.9434 | 20.54 | 0.0036 | 9719 | 0.0 | 0.95 | None |
| V02 | 21 | 15.1 | 17 | 5 | 14.218 | 0.25 | 0.9953 | 0.9953 | 27.56 | 0.0065 | 14892 | 0.8 | 1.0 | None |
| V03 | 31 | 20.5 | 25 | 5 | 9.772 | 0.25 | 0.9772 | 0.9772 | 18.21 | 0.0068 | 15520 | 0.0 | 1.0 | None |
| V04 | 52 | 38.3 | 42 | 3 | 3.435 | 0.15 | 0.9924 | 0.9924 | 39.0 | 0.0086 | 26255 | 1.0 | 1.0 | None |
| V05 | 86 | 74.8 | 35 | 7 | 4.861 | 0.35 | 0.9954 | 0.9954 | 27.68 | 0.0125 | 34535 | 0.0 | 1.0 | 0.0 |
| V06 | 139 | 120.0 | 56 | 9 | 3.879 | 0.45 | 0.9986 | 0.9986 | 38.95 | 0.0202 | 50767 | 0.0 | 0.8111 | -0.00667 |
| V07 | 183 | 116.4 | 49 | 22 | 7.201 | 1.1 | 0.9984 | 0.9984 | 74.36 | 0.0300 | 74634 | 0.9091 | 0.9955 | 0.00119 |
| V08 | 230 | 166.3 | 38 | 17 | 4.431 | 0.85 | 0.9991 | 0.9991 | 39.79 | 0.0292 | 77952 | 0.0 | 1.0 | 0.0 |
| V09 | 289 | 205.8 | 49 | 28 | 5.817 | 0.933 | 0.9972 | 0.9972 | 86.45 | 0.0412 | 101566 | 0.3929 | 1.0 | 0.0 |
| V10 | 322 | 307.8 | 44 | 30 | 5.595 | 1.0 | 0.9978 | 0.9978 | 86.68 | 0.0464 | 108970 | 0.3333 | 1.0 | 0.0 |

_Privacy: anonymized aggregate stats; no user IDs, paths, athlete names, or raw transcripts. Label dataset = MMA/grappling production clips._
