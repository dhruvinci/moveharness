# E1: flash vs pro on the 496-clip technique-ID set (camera-ready second-model run)

Same 496 scored clips for both models (the same 2 of 498 on-disk clips fail Gemini
File-API processing under both models: `file state FAILED`). Same prompt, same
ontology-normalized semantic matching vs `gt_tags`, temperature 0. Wilson score 95%
intervals (z=1.95996), computed from per-clip correctness (flash CIs recomputed from
`vlm_technique_id.json` per_clip; the stored `vlm_stats_ci.json` predates the 496-clip
run and is N=119 — do not use it for this comparison). One model-forced config
difference: gemini-2.5-pro cannot disable thinking, so it ran with the minimum
thinking budget (128 tokens; flash ran with thinking disabled).

| Metric | gemini-2.5-flash (n=496) | gemini-2.5-pro (n=496) |
|---|---|---|
| Position top-1 accuracy | 58.9% [54.5%, 63.1%] (292/496) | **66.3% [62.1%, 70.3%]** (329/496) |
| Technique match (semantic) | **48.0% [43.6%, 52.4%]** (238/496) | 40.3% [36.1%, 44.7%] (200/496) |
| Either match (position OR technique) | 71.0% [66.8%, 74.8%] (352/496) | 72.0% [67.9%, 75.7%] (357/496) |
| Both correct (position AND technique) | 35.9% [31.8%, 40.2%] (178/496) | 34.7% [30.6%, 39.0%] (172/496) |
| n_scored / n_failed | 496 / 2 | 496 / 2 |
| Actual API cost (USD) | $0.6127 | $2.9511 ($0.0301 pilot + $2.9210 bulk) |

Reading:

- **Position**: pro is higher by 7.4 points; the 95% CIs barely overlap
  (flash [54.5, 63.1] vs pro [62.1, 70.3], 1.0-point overlap). Paired per-clip
  disagreements (same 496 clips): pro-only-correct 86 vs flash-only-correct 49
  (net +37 clips for pro).
- **Technique**: flash is higher by 7.7 points; CIs barely overlap
  ([43.6, 52.4] vs [36.1, 44.7], 1.1-point overlap). Paired disagreements:
  flash-only-correct 100 vs pro-only-correct 62 (net +38 clips for flash).
- **Either / both**: statistically indistinguishable (heavily overlapping CIs).
- Net: swapping flash for pro at ~4.8x the cost redistributes errors
  (better positions, worse technique naming) but does not move combined
  accuracy — either-match 71.0% vs 72.0%, both-correct 35.9% vs 34.7%. On this
  tier, model scale does not buy what grounding buys.

Sources (all under `benchmark/results/`):

- `vlm_technique_id.json` — flash run (untouched).
- `vlm_technique_id_pro.json` — pro run (summary embeds the Wilson CIs above).
- `vlm_technique_id_pro.checkpoint.jsonl` — per-clip resume checkpoint (498 lines).
- `_vlm_technique_id_scorer_v2.py` — parameterized scorer (env: VLM_MODEL,
  VLM_OUT_TAG, VLM_MAX_CLIPS, VLM_TIME_BUDGET, VLM_COST_ABORT_USD; default cap $5.00).

Privacy: anonymized aggregate stats only; no user IDs, paths, or raw transcripts.
