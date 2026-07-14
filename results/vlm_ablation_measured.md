# Measured Grounding Ablation (M1: CV anchors) — gemini-2.5-flash

- Model: **gemini-2.5-flash**
- Generated: 2026-05-30T18:46:45
- Design: within-video **paired** ablation. GROUNDED = prompt + Stage0 CV anchors; UNGROUNDED = prompt only. Same model, same uploaded video, same parser.
- Paired videos (both conditions valid): **N = 10**

## Grounded vs Ungrounded deltas (averaged across videos)

| Metric | grounded mean | ungrounded mean | Δ (g−u) mean | Δ median | better? | #g-better/N |
|---|---|---|---|---|---|---|
| n_segments (higher=better) | 12.8 | 15.3 | -2.5 | -0.5 | ungrounded | 3/10 |
| segs_per_min (higher=better) | 7.053 | 7.6364 | -0.5834 | -0.2155 | ungrounded | 3/10 |
| overseg_ratio (lower=better) | 0.5433 | 0.63 | -0.0867 | -0.025 | grounded | 5/10 |
| coverage (higher=better) | 0.9895 | 0.9895 | +0.0 | +0.0 | tie | 0/10 |
| end_reach (higher=better) | 0.9895 | 0.9895 | +0.0 | +0.0 | tie | 0/10 |
| ontology_violation_rate (lower=better) | 0.3435 | 0.1627 | +0.1809 | +0.0123 | ungrounded | 3/10 |
| ontology_violations (lower=better) | 4.8 | 2.1 | +2.7 | +1.0 | ungrounded | 3/10 |
| avg_confidence (higher=better) | 0.9757 | 0.9773 | -0.0017 | +0.0 | ungrounded | 2/10 |
| conf_decay_slope (higher=better) | -0.0009 | 0.0073 | -0.0082 | +0.0 | ungrounded | 1/6 |
| latency_s (lower=better) | 45.922 | 40.916 | +5.006 | +1.465 | ungrounded | 3/10 |
| cost_usd (lower=better) | 0.0205 | 0.0207 | -0.0002 | +0.0006 | grounded | 3/10 |
| output_tokens (lower=better) | 2752.2 | 3202.2 | -450 | -47.0 | grounded | 7/10 |
| total_tokens (lower=better) | 51481 | 48773.7 | +2707.3 | +3054.5 | ungrounded | 1/10 |

## Interpretation

- **Segmentation**: grounded 12.8 vs ungrounded 15.3 segments/match (Δ -2.5).
- **Oversegmentation ratio**: grounded 0.54x vs ungrounded 0.63x (Δ -0.09; lower better).
- **Coverage**: grounded 99.0% vs ungrounded 99.0% (Δ +0.0 pts).
- **Ontology-violation rate**: grounded 34.4% vs ungrounded 16.3% (Δ +18.1 pts; lower better).
- **Confidence decay slope** (per-segment, less-negative=better): grounded -0.0009 vs ungrounded 0.0073 (Δ -0.0082).

_Privacy: anonymized aggregate stats only. No user IDs, paths, athlete names, or raw transcripts._
