# MoveBench v0 — HMM CV-only Baseline (X1, real run)

**What this is.** An actually-fitted, actually-decoded, actually-scored CV-only
Hidden Markov Model baseline for the macro-position task on MoveBench v0. This
replaces the paper's "HMM specified but not run" hedge with a real number. The
HMM is fit over the **computer-vision anchor features** that the Exp4 stage0 pass
already cached for the v0 match (`mb_v0_0001`, ~970 s, 194 checkpoints at
5 s) — so no new stage0 / video inference was needed (fast path). No VLM, no
network: this is the pure-CV floor (well, ceiling — see caveats) the language
models are competing against.

All numbers below are **recomputed this session** by deterministic fitting
(fixed seeds) + the bundled MoveBench scorer (`python -m movebench.evaluate`,
greedy position+IoU matching at IoU>=0.3, macro-F1 over the 7 v0 macro-position
classes).

## Method

- **Features (5, the paper's anchor set):** `standing_prob` (mean of A1/A2),
  `contact_intensity`, `action_score`, `dominance` (|neutral=0, A1/A2=1|),
  `entanglement`. Missing `standing_prob` (athlete-not-detected frames)
  column-mean imputed; all columns z-scored. 194 timesteps x 5 dims.
- **Model:** multivariate Gaussian-emission HMM (`hmmlearn 0.3.3`,
  **diagonal covariance**, `min_covar=1e-2`). Full covariance overfits badly at
  this N (log-L spikes to ~4978, fragmented decode), so diagonal is used.
- **Model selection:** fit K = 2..8 hidden states, 6 random restarts each (keep
  best log-likelihood per K), select #states by **AIC**.
- **Decode:** Viterbi most-likely state path. Reported in two variants: the raw
  path, and a min-run=2 (10 s) duration-smoothed path (standard segmentation
  post-processing tuned to the 5 s feature scale, **not** to GT).
- **State -> position:** each HMM state mapped to the GT macro position with which
  it has the largest total temporal overlap (oracle cluster->class assignment).
  This makes the result a **CV-only ceiling**, reported honestly as such.
- **Scoring:** bundled MoveBench scorer vs `benchmark/data/v0/mb_v0_0001.json`
  (32 GT macro segments; 7 position classes: standing, guard, sweep, escape,
  side_control, back_control, mount). T3 labels are intentionally empty (a CV
  macro-position decoder emits no technique labels).

## Model selection (AIC)

| K | log-L | n_params | AIC | BIC |
|---|---|---|---|---|
| 2 | 528.0 | 23 | -1010.0 | -934.8 |
| 3 | 1206.9 | 38 | -2337.9 | -2213.7 |
| 4 | 1244.9 | 55 | -2379.9 | -2200.1 |
| 5 | 1295.1 | 74 | -2442.2 | -2200.3 |
| **6** | **1328.3** | **95** | **-2466.6** | -2156.2 |
| 7 | 1324.5 | 118 | -2412.9 | -2027.3 |
| 8 | 1328.8 | 143 | -2371.7 | -1904.4 |

**Selected: K = 6 states** (minimum AIC = -2466.65; logL = 1328.32; BIC = -2156.20).
(BIC would prefer K=3/4 — noted for honesty; AIC is the requested criterion.)

## Headline numbers (the real CV-only baseline)

| variant | T1 macro-F1 | dur-wtd F1 | mean matched IoU | matched | pred segs (vs 32 GT) | coverage |
|---|---|---|---|---|---|---|
| **smoothed (min-run 10 s) — HEADLINE** | **0.2126** | 0.2011 | 0.4005 | 7/32 | 18 (0.56x) | 1.00 |
| raw Viterbi | 0.0999 | 0.1314 | 0.5016 | 11/32 | 81 (2.53x) | 1.00 |

**Comparison to the VLM/harness numbers in `real_scores.md`:**

| system | T1 macro-F1 |
|---|---|
| **HMM CV-only (smoothed) — this run** | **0.2126** |
| best CV-grounded VLM pipeline (`exp4_v2_skeleton_run2`) | 0.1888 |
| ... | ... |
| naive single-pass holistic VLM | 0.0165 |

**Finding.** The CV-only HMM macro-position decoder (smoothed, oracle state->position
mapping) reaches **macro-F1 0.2126**, which is **above** the best CV-grounded VLM
pipeline (0.1888) and **~12.9x** the naive single-pass VLM (0.0165). The raw
(un-smoothed) HMM lands at 0.0999 — between the Exp3 CV-cached timeline (0.0991)
and the Exp4 v1 pipelines. Read with the caveats: the state->position mapping is
oracle, so 0.2126 is the CV-only **ceiling**, not an end-to-end unsupervised score.

## Per-position breakdown (smoothed, headline)

| position | tp | fp | fn | F1 |
|---|---|---|---|---|
| escape | 4 | 4 | 2 | 0.571 |
| back_control | 1 | 0 | 1 | 0.667 |
| guard | 2 | 7 | 5 | 0.250 |
| standing | 0 | 0 | 6 | 0.000 |
| sweep | 0 | 0 | 6 | 0.000 |
| side_control | 0 | 0 | 4 | 0.000 |
| mount | 0 | 0 | 1 | 0.000 |

The CV anchors carry strong signal for the dynamic / engaged states
(escape, back_control, guard) but the decoder never wins standing, sweep,
side_control, or mount — `sweep`/`escape` are event-style GT positions the
anchors cannot cleanly separate from `guard`, and `standing` segments are short
and get absorbed by smoothing. This is exactly the structure-without-semantics
limitation the paper argues for: CV gives the macro skeleton, not the labels.

## Fit diagnostic (pseudo-residuals)

Smoothed-posterior PIT pseudo-residuals on `standing_prob` (mixture predictive
CDF -> normal quantile): mean ~0.000, std ~0.999, Shapiro-Wilk W = 0.982,
p = 0.014. Close to N(0,1) in the first two moments; the small Shapiro p
reflects mild non-normality (heavy tails / the imputed-frame mass), consistent
with a diagonal-Gaussian approximation to bounded CV scores. Adequate, not
perfect — expected for a 5-feature CV emission model.

## Caveats (frame everything above)

- **n = 1.** Single annotated sample (one ~16-min match). Developer measurement,
  not a leaderboard. No confidence intervals.
- **In-sample.** The CV cache and the GT are the same match.
- **Oracle state->position mapping.** 0.2126 is the CV-only **ceiling** (best-case
  state-to-class assignment), not an end-to-end unsupervised number. The
  oversegmented raw variant (0.0999) is the more conservative read.
- **Smoothing.** Headline uses min-run=2 (10 s) Viterbi smoothing tuned to the
  5 s feature time-scale, not to GT. Raw-Viterbi numbers are reported alongside.
- **AIC vs BIC.** AIC (requested) picks K=6; BIC would pick K=3/4.
- **Diagonal covariance.** Full-covariance overfits at this N (degenerate
  variance spikes, fragmented decode); diagonal + 1e-2 variance floor is used.

## Artifacts

- `real_hmm_baseline.py` — the script (this run)
- `real_hmm_baseline.json` — full machine record (AIC table, transition matrix,
  state durations, mapping, both decode variants, full MoveBench score vector)
- `predictions_all/pred_hmm_cv_baseline.json` — scored prediction (smoothed)
- `predictions_all/pred_hmm_cv_baseline_raw.json` — scored prediction (raw Viterbi)
- CV source: Stage0 CV anchor cache for the v0 seed (`mb_v0_0001`; withheld),
  194 checkpoints @5 s
