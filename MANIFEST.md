# MoveBench v0 — Anonymized Reproducibility Bundle

This is the release manifest for the **MoveBench v0** benchmark for fine-grained
vision–language understanding of technical human movement. It documents every
released artifact, gives the exact commands and fixed experimental settings
needed to reproduce the reported numbers, and reports the result of a privacy /
cleanliness scan over every file the bundle references.

The bundle is designed to satisfy reproducibility and open-access requirements:
the **tool**, the **public ground truth**, the **benchmark-construction recipe**,
the **scorers/harness**, and **exact reproduce commands** are all included, while
private production videos and any consent-gated material are withheld and
referenced by aggregate counts only.

> **Terminology.** Throughout the release, the domain is referred to as **MMA /
> grappling** and the author's clip corpus as a **label dataset**. Athletes are
> anonymized as `A1` / `A2`. The single public seed match is referenced only by
> an opaque `sample_id` (`mb_v0_0001`).

> **Honesty contract.** v0 = one annotated public match. All scored predictions
> are in-sample developer artifacts (n=1, single annotator, no confidence
> intervals); these are **measurements, not a leaderboard**. The strict-syntactic
> numbers are a conservative lower bound; the semantic standard is the intended
> measure. Production-scale figures are *operational* evidence, not *correctness*
> evidence.

All paths below are relative to the repository root unless noted. Run
commands from the repository root with `PYTHONPATH=.` so the `movebench`
package resolves.

---

## 0. What this bundle contains (at a glance)

| # | Component | Released here | Status |
|---|-----------|---------------|--------|
| 1 | **MoveBench tool** — `movebench/` package, `ontology.json`, `SCHEMA.md` | YES (released) | clean |
| 2 | **Public v0 ground truth** — `data/v0/mb_v0_0001.json` (+ examples) | YES (released) | clean |
| 3 | **Technique-ID benchmark construction** — download script + manifest schema | RECIPE ONLY (script + schema; clips & raw manifest NOT redistributed) | construction released; data withheld for consent |
| 4 | **Scorers / harness** — `movebench.evaluate`, `_vlm_*`, `_llm_judge*`, `_score_all`, `real_*` | YES, after scrubbing path/key/ID constants (§7) | needs scrub before release |
| 5 | **Exact reproduce commands + fixed settings** | YES (this README, §6) | clean |
| — | Private production videos, raw exports, instructional SQLite corpus | **WITHHELD** (counts only) | not redistributed |

Headline reproducible numbers (all recomputed deterministically or from logged
runs; see §5):

- Best CV-grounded VLM pipeline T1 position macro-F1 = **0.1888** (6/32 matched)
  vs naive single-pass **0.0165** (1/32) — a **~11.4x** relative gain.
- HMM CV-only baseline (smoothed, oracle state→position) T1 macro-F1 = **0.2126**
  (7/32); raw Viterbi = **0.0999**.
- LLM-as-judge mean semantic correctness tops at **0.2031**; mean strict→judge
  gap **+0.0372** (positive for 15/16 systems); Spearman ρ(judge, strict) =
  **0.871**; judge self-consistency exact-agreement **1.00** at T=0.
- Technique-ID accuracy (gemini-2.5-flash, 496 clips): position top-1 **58.9%**
  [54.5, 63.1], technique match **48.0%** [43.6, 52.4], both-correct **35.9%**,
  either-match **71.0%** [66.8, 74.8] (352/496); measured cost **~$0.61**.
- OOD in-the-wild correctness (n=200 user-reviewed uploads, 4,620 segments): exact **46.7%**
  [42.5,51.4], coarse **59.7%** [56.4,63.4], adjacent-cluster **92.9%** [91.9,93.8]
  (video-clustered 95% CIs); 93% of errors land on an adjacent position.
- Grounding ablation (N=10 paired videos): grounded 12.8 vs ungrounded 15.3
  segments/match; ontology-violation rate grounded 34.4% vs ungrounded 16.3%.

---

## 1. The MoveBench tool

A small, self-contained, **standard-library-only** Python package (Python ≥ 3.9,
no `pip install`, no network) that loads the v0 seed, scores predictions with a
metric **vector** (never one scalar), and enforces two mandatory integrity guards.

### 1.1 Package — `movebench/`

| File | Role |
|------|------|
| `movebench/__init__.py` | Package surface: `load_sample`, `load_prediction`, `Ontology`, `interval_iou`, `score_segments`, `score_transitions`, `coverage`, `label_f1`. |
| `movebench/loader.py` | Loads v0 samples / predictions; `Ontology` alias-folding from `ontology.json`; **counts-only** external-corpus enumerator (returns aggregate integers only, never raw ids/urls/titles/clip text). |
| `movebench/metrics.py` | Metric vector. `interval_iou`, `score_segments` (greedy position+IoU matching → per-position TP/FP/FN → macro-F1 + mean matched IoU), `score_transitions` (timestamp-tolerant event F1, actor-aware), plus new `label_f1` (T3 ontology-aware set F1 + ontology-violation rate) and `coverage` (T5, completeness). Lineage cited in-file: adapted from the P1 measurement-reasoning scorer (`score_p1_eval.py`). |
| `movebench/evaluate.py` | CLI: `python -m movebench.evaluate --gt <gt.json> --pred <pred.json>`. Emits the full metric-vector report. |
| `movebench/requirements.txt` | Requirements note (stdlib only; Python ≥ 3.9). |
| `movebench/_smoketest.py` | Non-degeneracy smoke test (identity vs perturbed prediction). |
| `movebench/_convert_seed.py` | Build-time tool: raw GT → anonymized `data/v0/mb_v0_0001.json` (strips names/url/id). Not part of the runtime surface. |
| `movebench/README.md` | Package-level usage doc. |

### 1.2 Ontology — `ontology.json` (`ontology_version` 0.1.0)

- **18 macro positions (L1A states)** + `unknown/occluded` always allowed.
- **~91-term technique/action vocabulary** (L1B sub-positions, takedowns, passes,
  sweeps, submissions, grips/frames, transitions, lifecycle).
- **42 unique labels used in v0** (excluding the 2 protocol annotations
  `pre_match`/`match_start`), over 122 label instances across 62 subsegments.
- The v0 `position` field uses a reduced **7-value working set** (`standing,
  guard, sweep, back_control, escape, mount, side_control`) — `sweep`/`escape`
  are event-style states used as segment labels. See
  `ontology.json → v0_macro_position_field_values`.
- The alias map (`L1A_macro_positions[*].aliases`, `v0_used_labels[*].normalized`)
  is the **first (syntactic-normalization) step of semantic matching**, not a
  string comparison.
- Ontology is domain-specific and does **not** transfer to judo/wrestling/
  striking/dance/surgery without re-authoring the label set.

### 1.3 Schema — `SCHEMA.md`

Defines the **sample schema** (a benchmark example + ground truth), the
**prediction / submission schema** (what a model emits + `submission_meta.json`),
the **ontology binding**, and the **integrity guards**. All times are stored
both as raw strings (`MM:SS`) and normalized seconds (float); scorers use seconds.
The privacy invariant (no names/URLs/ids/emails/paths; `A1`/`A2`; opaque
`sample_id`) is binding on every released file.

---

## 2. Public v0 ground truth

| File | Contents | Status |
|------|----------|--------|
| `data/v0/mb_v0_0001.json` | The single public seed match. `duration_sec` 959.0, `source_type` competition, `modality` no_gi, `rights_status` public_seed; **32 macro segments**, 64 raw subsegments (62 benchmark, excl. 2 protocol), 124 raw label instances (122 benchmark), 44 raw / 42 unique technique labels; single annotator (`expert_reviewer`, `label_source: human_review`). T2 transitions and ADCC scoring layers are **empty** (planned v1). | clean (anonymized; A1/A2; opaque sample_id) |
| `data/v0/examples/pred_oracle_identity.json` | Identity prediction (scores macro-F1 = 1.0; **not** flagged degenerate because the seed is `human_review`). | clean |
| `data/v0/examples/pred_perturbed.json` | Perturbed prediction (wrong positions, shifted spans, dropped + hallucinated labels) — scores strictly lower; proves the metrics respond to errors. | clean |
| `data/v0/examples/pred_degenerate_demo.json` + `sample_degenerate_demo.json` | A `prediction_review` GT == prediction pair that demonstrates the degenerate-baseline guard firing (`mb_v0_degenerate_demo`). | clean (synthetic demo) |

The v0 seed and its examples carry **no** athlete names, URLs, YouTube IDs, gym
names, user IDs, emails, or local paths (verified — see §7).

---

## 3. Technique-ID benchmark CONSTRUCTION (recipe only; clips NOT redistributed)

The technique-ID task evaluates a VLM on short (~12 s) grappling clips, each
carrying ground-truth technique/position tags. **The clips themselves are the
author's own private label dataset and are NOT redistributed** — only the
construction recipe and the manifest *schema* are released, so the benchmark can
be re-built by a holder of the underlying label dataset.

### 3.1 Construction script (released) — `data/download_technique_clips.py`

Reads a selection file, downloads a ~12 s clip (±6 s) centered on each labeled
timestamp at ≤480p, and emits a ground-truth manifest. Robustness: one retry,
skip+log on failure, ~18 min global wall-clock cap, early-stop on repeated
download blocks. The script resolves its `BASE` directory relative to itself
(overridable via the `MOVEBENCH_BASE` env var) — no absolute path — and reads IDs
from a private selection file kept out of the release (see §7).

Construction run summary (`results/technique_clips_summary.json`,
`results/technique_clips_download.log`): **630 target clips → 498 downloaded**
(2 failed: source video unavailable; stopped at the wall-clock cap), of which
**496 were scored**. Target tags span guards, passes, sweeps, takedowns,
submissions, controls, and transitions across many athletes and source videos.

### 3.2 Manifest schema (released as a template; values withheld)

The construction emits `data/technique_clips/manifest.json`, keyed by clip
filename. **The populated manifest is WITHHELD** (it contains YouTube video IDs,
free-text expert descriptions, and athlete-name tag tokens). The released
*schema* is:

```jsonc
// manifest.json  (one entry per clip; VALUES below are the schema, not real data)
{
  "<clip_id>.mp4": {
    "gt_tags":       ["<technique/position hashtag tokens>"],  // multi-label GT
    "gt_label_text": "<free-text expert description>",          // WITHHELD: prose
    "seconds":       0,                                         // timestamp center
    "video_youtube_id": "<source video id>"                     // WITHHELD: id
  }
}
```

The selection file (`data/technique_clips_selection.json`) has the same fields
plus a `url`. **Both the populated manifest and the selection file are WITHHELD**
because they embed video IDs, URLs, prose, and athlete-name tokens.

A consumer with the private label dataset reconstructs the benchmark by writing a
`technique_clips_selection.json` (the documented schema), running
`download_technique_clips.py`, then scoring with §4.1.

---

## 4. Scorers / harness scripts

All scorer/harness scripts live under `results/`. Scripts that call a model
(`_vlm_*`, `_llm_judge*`) load an API key from an **external** `.env` (outside
the benchmark directory) — **no key value is stored in the bundle** (verified,
§7). The deterministic re-scoring scripts (`_score_all`, `real_*`,
`_convert_*`, `_compute_real_faithfulness`, `_build_report`) need **no key** and
**no network**.

### 4.1 Inventory

| Script | Needs key/network | What it does |
|--------|-------------------|--------------|
| `movebench.evaluate` (CLI) | no | Score one prediction vs one sample → full metric vector (T1/T3/T2/T5 + integrity guards). |
| `results/_vlm_technique_id_scorer.py` | **yes** (gemini-2.5-flash) | Technique-ID task: per-clip File-API call, ontology-constrained `{position, technique, confidence}`, semantic family matching vs `gt_tags`. → `vlm_technique_id.{json,md}`. |
| `results/_vlm_harness.py` | **yes** | Multi-video grounding harness; per video runs GROUNDED (prompt + Stage0 CV anchors) and UNGROUNDED (prompt only). → `vlm_runs_raw.json`. |
| `results/_vlm_report.py` | no | Renders `vlm_multivideo.{json,md}` + `vlm_ablation_measured.{json,md}` from `vlm_runs_raw.json`. |
| `results/_llm_judge.py` | **yes** | LLM-as-judge semantic correctness (0/1/2), greedy max-IoU GT→pred align at 0.3, abstain-aware. → `vlm_llm_judge.json`. |
| `results/_llm_judge_consistency.py` | **yes** | Re-judge the same 10 segments twice (seed 17) → agreement / kappa. → `vlm_llm_judge_consistency.json`. |
| `results/_llm_judge_report.py` | no | Renders `vlm_llm_judge.md` from the two judge JSONs. |
| `results/_convert_all_outputs.py` | no | Convert every available pipeline output → MoveBench prediction files in `predictions_all/` (timestamps + ontology tokens only; privacy scan aborts on leak). |
| `results/_convert_experiment_timeline.py` | no | Single-pipeline variant of the converter → `predictions/`. |
| `results/_score_all.py` | no | D1: real metric-vector scoring of every converted prediction vs v0 GT in two regimes (strict vs semantic, abstain-aware). → `real_scores_data.json`. |
| `results/_build_report.py` | no | Aggregates `real_scores_data.json` → `real_scores.{json,md}` (headline, group rollups, strict→semantic gap). |
| `results/real_ablation.py` | no | Real per-module ablation (M1–M5) by re-scoring stored artifacts. → `real_ablation.{json,md}`. |
| `results/real_hmm_baseline.py` | no (numpy + hmmlearn) | HMM CV-only baseline: fit Gaussian HMM over cached CV anchors, Viterbi-decode, oracle state→position, score with bundled scorer. → `real_hmm_baseline.{json,md}`, `predictions_all/pred_hmm_cv_baseline*.json`. |
| `results/_compute_real_faithfulness.py` | no | D4: ontology-violation/hallucination rate, over-segmentation, attention-decay by decile. → `real_faithfulness.{json,md}`. |

### 4.2 Integrity guards (in `movebench.evaluate`)

1. **No degenerate baselines.** If `annotation.label_source == "prediction_review"`
   AND the GT segments are byte-identical to the prediction, the run is flagged
   `degenerate: true` and the headline accuracy is suppressed to `null`. (This is
   why the P1 50-video 1.0 is tautological and never reported.)
2. **Empty GT layers are not 0/1.** v0 layers with no ground truth (T2
   transitions, scoring) report `task_status: "no_v0_ground_truth"`, never a
   misleading F1.

---

## 5. Released results (deterministic or logged)

| File | Numbers |
|------|---------|
| `results/real_scores.{json,md}` | 16 systems × 2 regimes; best CV-grounded T1 macro-F1 **0.1888** vs naive **0.0165** (~11.4x); pooled-token strict→semantic gap up to **31.6%**; oversegmentation 0.44x–7.53x. |
| `results/real_ablation.{json,md}` | M1 grounding: T1 macro-F1 0.014→0.1361 (~9.7x), matched 1→6/32; M2/M5 alias-folding deltas; M3 smoothing; M4 abstain. |
| `results/real_hmm_baseline.{json,md}` | K=6 by AIC; smoothed macro-F1 **0.2126** (7/32), raw **0.0999** (11/32). |
| `results/vlm_llm_judge.{json,md}` + `vlm_llm_judge_consistency.json` | Judge top **0.2031**; mean strict→judge gap **+0.0372** (15/16 positive); ρ=**0.871**; self-consistency exact 1.00; 325 calls, ~$0.069. |
| `results/vlm_technique_id.{json,md}` | Position top-1 **58.9%** [54.5,63.1], technique **48.0%** [43.6,52.4], both-correct **35.9%**, either-match **71.0%** [66.8,74.8] (352/496); 496/496 scored; ~$0.61. |
| `results/vlm_technique_id_pro.json` + `vlm_technique_id_pro_summary.md` | E1: `gemini-2.5-pro` on the same 496 clips (~$2.95): position top-1 **66.3%** vs flash 58.9%, technique **40.3%** vs 48.0%, both/either unchanged — scale redistributes errors. Scorer: `results/_vlm_technique_id_scorer_v2.py`. |
| `results/e2_matchable_subset.{py,json}` | E2: matchable-subset re-score — best CV-grounded T1 macro-F1 0.189 → **0.243** on the subset of positions present in GT. |
| `results/e3_bootstrap_ci.{py,json}` | E3: bootstrap CI on the grounding delta — **+0.174** [0.031, 0.209], excludes zero. |
| `results/vlm_multivideo.{json,md}`, `results/vlm_ablation_measured.{json,md}` | Grounded vs ungrounded over N=10 paired videos. |
| `results/real_faithfulness.{json,md}` | Ontology-violation / over-segmentation / attention-decay deciles. |
| `results/baselines.md` | Narrative baseline summary. |
| `results/predictions_all/*.json`, `results/predictions/*.json` | Converted MoveBench prediction files (timestamps + ontology tokens only). |

---

## 6. EXACT reproduce commands + FIXED experimental settings

### 6.1 Fixed settings (used everywhere)

| Setting | Value |
|---------|-------|
| Model (all VLM/judge runs) | `gemini-2.5-flash` |
| Temporal-match IoU threshold | **0.3** (greedy position+IoU) |
| Transition tolerance | 5.0 s (inert in v0 — empty GT) |
| Ablation conditions | **GROUNDED** = competition prompt + Stage0 CV anchors; **UNGROUNDED** = prompt only (CV anchors ablated) |
| Splits | v0 = single match assigned to `test` (no `train`/`dev` in v0). v1 splits **by video** (channel/source-disjoint proxy), stratified 70/15/15. |
| LOO | v0 leave-one-video-out is **degenerate** (n=1); reported as the v1 protocol over the full corpus. |
| Judge protocol | gemini-2.5-flash, **temperature 0.0**, thinking disabled (`thinking_budget=0`), JSON response; greedy max-IoU GT→pred at 0.3 (same as strict T1); 0/1/2 → normalized /2; abstain-aware; GT-anchored denominator (misses score 0). |
| Technique-ID inference | gemini-2.5-flash, **temperature 0.0**, `max_output_tokens=256`, thinking disabled; ontology-constrained single-line JSON. |
| Harness inference | gemini-2.5-flash, **temperature 0.5**, `max_output_tokens=25000`. |
| Concurrency / robustness | ≤4 in-flight; exponential backoff + jitter on 429/5xx/quota; wall-clock time-box (~35 min); skip+log on failure. |
| Seeds | HMM baseline: deterministic, fixed seeds, 6 random restarts per K, K=2..8, **K selected by AIC** (K=6). Judge self-consistency: **seed 17**, 10 segments. Bootstrap CIs are **not computed** (n=1 sample; no CIs are claimed). |
| Pricing (for cost estimates) | gemini-2.5-flash $0.30 / 1M input, $2.50 / 1M output. |

### 6.2 No-key, deterministic reproduce (works offline)

```bash
# from the repository root
# (1) Evaluator sanity check — should print macro_f1 = 1.0, degenerate = false
PYTHONPATH=. python3 -m movebench.evaluate \
    --gt data/v0/mb_v0_0001.json \
    --pred data/v0/examples/pred_oracle_identity.json

# (2) Prove the scorer is not degenerate (identity vs perturbed)
PYTHONPATH=. python3 -m movebench._smoketest

# (3) Convert all stored pipeline outputs -> MoveBench predictions
PYTHONPATH=. python3 results/_convert_all_outputs.py

# (4) D1 metric-vector scores (strict vs semantic) + report
PYTHONPATH=. python3 results/_score_all.py
python3 results/_build_report.py            # -> results/real_scores.{json,md}

# (5) Per-module ablation (M1-M5), no inference
PYTHONPATH=. python3 results/real_ablation.py

# (6) HMM CV-only baseline (numpy + hmmlearn; deterministic, K by AIC)
PYTHONPATH=. python3 results/real_hmm_baseline.py

# (7) Faithfulness / over-segmentation / attention-decay
PYTHONPATH=. python3 results/_compute_real_faithfulness.py
```

### 6.3 Key-required reproduce (needs an external Gemini API key)

These read a key from an external `.env` (outside the bundle). Export the key
in the environment instead — no key is stored in the release:

```bash
export GOOGLE_API_KEY=...        # your own key; never committed
# from the repository root
PYTHONPATH=. python3 results/_vlm_technique_id_scorer.py   # needs the (private) clips
PYTHONPATH=. python3 results/_vlm_harness.py               # needs the (private) videos
python3 results/_vlm_report.py
PYTHONPATH=. python3 results/_llm_judge.py
PYTHONPATH=. python3 results/_llm_judge_consistency.py
python3 results/_llm_judge_report.py
```

> The technique-ID and harness runs require the private label dataset (clips /
> videos), which is **not** in this bundle. The LLM-judge runs only on the
> already-converted, anonymized prediction files (no private media) and is
> therefore fully reproducible with a key.

---

## 7. Cleanliness report (privacy / leak scan)

A scan was run over every file this bundle references, looking for: API key
values, absolute home-directory paths, user IDs, the legacy label-corpus
identifier, the sport string, athlete names, emails, and video URLs / IDs. **All
items flagged below have been scrubbed** (see §7.2 for the resolution log); the
released benchmark tree now contains none of those literals as data.

> **Release-build correction (final pass).** A whole-tree re-scan over *every*
> file extension (not just `.json`/`.csv`) caught leaks an earlier scan missed.
> The shipped zip therefore **additionally**: (a) **withholds**
> `results/technique_clips_download.log` (it logged raw `yt=<id>` YouTube IDs per
> clip) and `results/vlm_run.log` (it logged internal `<id>.mp4` clip filenames);
> (b) **anonymizes** the `per_clip[].clip` field of `results/vlm_technique_id.json`
> to opaque `clip_NNNN` and scrubs the two failed-clip ObjectIDs from
> `summary.failures`; (c) **does not redistribute** `results/yolo11n-pose.pt`
> (third-party AGPL pose weights — fetch from the upstream project); and
> (d) drops the stale `vlm_technique_id_119.json` backup. The zip was verified to
> contain **zero** `.mp4`/`.pyc`/`.pt`/`.log` entries and no `yt=`/ObjectID/
> home-path/product-name strings. The env-var knob is `MOVEBENCH_ROOT`.

### 7.1 Verified clean (safe to release as-is)

- **No API key value** is stored anywhere in the benchmark tree. Scripts that
  call a model read the key from the **environment** (`GOOGLE_API_KEY` /
  `GEMINI_API_KEY`); no key value and no user-specific `.env` path are embedded.
- **No email addresses** appear in any tracked file.
- `data/v0/mb_v0_0001.json` and `data/v0/examples/*` — anonymized (A1/A2, opaque
  `sample_id`); no names/URLs/IDs/paths.
- `ontology.json`, `SCHEMA.md` — clean (the strings "YouTube ID", "URL" appear
  only inside privacy-invariant prose, not as values).
- `movebench/metrics.py`, `movebench/evaluate.py`, `movebench/_smoketest.py`,
  `movebench/requirements.txt`, `movebench/_convert_seed.py` — clean.
- `results/_score_all.py`, `_build_report.py`, `_llm_judge_consistency.py`,
  `_llm_judge_report.py`, `_compute_real_faithfulness.py` — clean (paths resolved
  relative to the file via `os.path`).
- `results/predictions_all/*.json` and `results/predictions/*.json` — clean,
  **except** the two HMM baseline predictions (see below).
- `results/real_scores.{json,md}`, `real_ablation.{json,md}`,
  `vlm_technique_id.{json,md}`, `vlm_ablation_measured.{json,md}`,
  `vlm_multivideo.{json,md}`, `vlm_runs_raw.json`, `vlm_llm_judge.{json,md}`,
  `vlm_llm_judge_consistency.json`, `real_scores_data.json`,
  `real_faithfulness.{json,md}`, `baselines.md`,
  `results/technique_clips_summary.json`
  — clean (anonymized aggregate stats; V01.. / system keys only). (`vlm_technique_id.json`
  per-clip IDs are anonymized to `clip_NNNN`; the two download logs are **withheld** — see the
  Release-build correction above.)

### 7.2 Scrub resolution log (RESOLVED before release)

Every item below was found in an earlier draft and has been fixed. None of the
flagged literals remain as data in the released tree.

| File | Issue (now fixed) | Resolution |
|------|-------------------|------------|
| `movebench/__init__.py` | legacy label-corpus identifier (2x) | Renamed function → `enumerate_label_dataset_candidates` (import + `__all__` updated). |
| `movebench/loader.py` | legacy identifier (def + docstring) + an error string naming the legacy corpus | Renamed function; error string reworded to "external label-dataset candidate corpus". |
| `movebench/README.md` | legacy identifier + a home-dir-style example DB path | Renamed references; example path replaced with the `<LABEL_DATASET_DB>` placeholder / `os.environ["LABEL_DATASET_DB"]`. |
| `results/_vlm_technique_id_scorer.py` | home-dir paths (ENV_PATH, CLIP_DIR, RESULTS_DIR); loaded the key from a hard-coded `.env` | `benchmark/` resolved relative to the script; clip dir via `TECHNIQUE_CLIP_DIR`; key read from the environment (no `.env` path). |
| `results/_vlm_harness.py` | home-dir paths (VLM_DIR, RESULTS, ontology); `.env` load; tmp sample path | Paths via `MOVEBENCH_ROOT` / `VLM_DIR` / `MOVEBENCH_ONTOLOGY` (script-relative defaults); key from env; sample list via `VLM_SAMPLE_LIST` (format documented in-file). |
| `results/_vlm_report.py` | home-dir `RES` path | `RES` resolved relative to the script. |
| `results/real_hmm_baseline.py` | home-dir `ROOT`; raw video id in docstring + emitted note | `ROOT` via `MOVEBENCH_ROOT` (script-relative default); raw id replaced everywhere with the opaque `mb_v0_0001`; emitted `cv_source` no longer carries the raw cache path. |
| `results/real_hmm_baseline.{json,md}` | emitted raw-id provenance string | Provenance / CV-source strings rewritten to the opaque sample id. |
| `results/predictions_all/pred_hmm_cv_baseline.json`, `…_raw.json` | `provenance` carried the raw video id | Provenance rewritten to the opaque sample id. |
| `results/_convert_all_outputs.py`, `_convert_experiment_timeline.py`, `real_ablation.py` | converters embedded a **person-name deny-list** as literals in the privacy scrubber | Deny-list externalized to `results/_privacy.py`, which loads the person-name terms from an external, comment-free file (`PRIVACY_DENYLIST` env var, else `<repo root>/.privacy_denylist`) kept **out of the release**. No person names are hard-coded; the scrubber still strips them when the private list is present. |
| `data/download_technique_clips.py` | home-dir `BASE` path | `BASE` resolved relative to the script (overridable via `MOVEBENCH_BASE`). |

> The configurable root is exposed via the `MOVEBENCH_ROOT` environment variable
> (the documented knob); it is a variable **name**, not a path or value, and is
> always overridable. The deny-terms `smrbzebxepa` / `/users/` that appear inside
> `results/_privacy.py` are lowercase **scrubber patterns** (used to *reject*
> those strings), not data.

### 7.3 Withheld for consent / privacy (NOT redistributed)

- **`data/technique_clips/*.mp4`** (498 downloaded clips) and the **populated**
  `data/technique_clips/manifest.json` — the author's private label dataset
  (embeds video IDs, expert prose, athlete-name tags). Released as **recipe +
  schema only** (§3). These withheld files are NOT scrubbed and NOT part of the
  release scan.
- **`data/technique_clips_selection.json`** — embeds 630 video IDs + URLs +
  prose + athlete-name tokens; **withheld** (not scrubbed; not released).
- **`p1-measurement-reasoning/gt_v2/*`** (200 raw per-video review labels, `release_status=private_do_not_release`) and **`benchmark/qa/ood_v2_extracted.json`** / **`ood_v2_fuzzy*.json`** (embed raw reviewer correction text + clip ids) --- **withheld**. The OOD aggregate outputs (`qa/confusion_data_v2.json`, `qa/ood_v2_metrics.json`, `qa/ood_clustered_ci.json`) and the deterministic scorers (`qa/score_ood_v2.py`, `qa/ood_clustered_ci.py`) ARE released; raw inputs are not (recipe pattern).
- **`results/technique_clips_download.log`** (raw `yt=<id>` YouTube IDs per clip)
  and **`results/vlm_run.log`** (internal `<id>.mp4` clip filenames) — **withheld**
  (the construction/run counts they summarize are released via
  `technique_clips_summary.json` and the aggregate result JSONs).
- **`results/yolo11n-pose.pt`** — third-party AGPL-licensed pose-model weights;
  **not redistributed** (fetch from the upstream project; the CV layer is described
  in the paper appendix). **`results/vlm_technique_id_119.json`** — stale earlier
  backup; dropped.
- The **production candidate corpus** (private export) and the **instructional
  SQLite label dataset** — referenced by **aggregate counts only**
  (e.g. 313 / 269 / 213 production videos; 430 videos / 5,604 clips / 1,653 tags
  for the label dataset). Never copied.
- **`results/real_dataset.{json,md}`** — characterizes the private label dataset.
  **Scrubbed for release:** the two athlete-name tags were removed (top clip-tag
  list shrank 30 → 28; remaining counts unchanged) and the raw sport string was
  replaced with `grappling`.
- The external API key (environment / external `.env`), the `<repo root>/.privacy_denylist`
  name file, and any production videos / raw exports — never in the bundle.

### 7.4 Net cleanliness verdict

The released benchmark tree is **clean**: the v0 ground truth, examples,
ontology, schema, the metrics/evaluate/smoketest code, the converted predictions
(incl. the 2 HMM files), and all aggregate result reports carry **no** secrets,
emails, home-dir paths, raw video IDs, the legacy label-corpus identifier, the
raw sport string, or athlete names. The mechanical scrub is complete: (a) the
label-dataset identifier was renamed across the 3 `movebench` files, (b) the raw
video ID was replaced with the opaque sample id in all HMM artifacts, (c) the
home-dir paths and `.env` key loads were parameterized in the scripts, (d) the
converter deny-list was externalized, and (e) `real_dataset.{json,md}` was
scrubbed. No private video is copied into the bundle; private media and
consent-gated corpora are referenced by counts only.
