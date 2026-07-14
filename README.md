# MoveHarness

**MoveHarness** is the measurement-and-reasoning harness behind the CAISc 2026 paper
*"Harnessing Vision–Language Models for Faithful Human-Movement Understanding"*
(Dhruva Chakravarthi, accepted 2026). It bundles:

- **MoveBench v0** — a fine-grained benchmark seed + a standard-library-only
  scorer for vision–language understanding of technical human movement
  (MMA / grappling), and
- the **evaluation harness** — the deterministic scorers, stored model outputs,
  and re-scoring scripts that reproduce every headline number in the paper
  without inference, network access, API keys, or raw media.

The Python package keeps its original name, `movebench/`; MoveHarness is the
umbrella for the package plus the harness scripts and stored results.

## Quickstart (no key, offline)

Requires Python ≥ 3.9. No `pip install`, no network.

```bash
# Evaluator sanity check — should print macro_f1 = 1.0, degenerate = false
PYTHONPATH=. python3 -m movebench.evaluate \
    --gt data/v0/mb_v0_0001.json \
    --pred data/v0/examples/pred_oracle_identity.json

# Non-degeneracy smoke test — identity scores 1.0, perturbed strictly lower
PYTHONPATH=. python3 -m movebench._smoketest
```

The scorer always emits a **metric vector, never one conflated scalar**, and
carries integrity guards: predictions byte-identical to review-derived ground
truth are flagged `degenerate` and headline accuracy is suppressed; empty
ground-truth layers report a `task_status`, never a misleading F1.

## Layout

| Path | Contents |
|------|----------|
| `movebench/` | The scorer package (stdlib-only): loader, metric vector, CLI, smoke test. |
| `data/v0/` | Public v0 seed (`mb_v0_0001.json`) + example predictions. |
| `ontology.json`, `SCHEMA.md` | Label ontology (18 macro positions, ~91-term vocabulary) and the sample/prediction schemas. |
| `results/` | Stored model outputs + deterministic re-scoring scripts (`_score_all.py`, `real_ablation.py`, `real_hmm_baseline.py`, `_vlm_stats_ci.py`, E1/E2/E3 below). |
| `qa/` | Out-of-distribution scorers and aggregate outputs (`score_ood_v2.py`, `ood_clustered_ci.py`, `qa_score.py`). |
| `MANIFEST.md` | The full reproduce manifest: every released artifact, exact commands, fixed settings, and the privacy scan. |
| `OVERVIEW.md` | Narrative overview of the benchmark design. |

## Headline results (all reproducible from this repo)

- **Grounding gain (seed match):** best CV-grounded VLM pipeline T1 position
  macro-F1 **0.1888** vs naive single-pass **0.0165** (~11.4×, i.e. 6/32 vs 1/32
  matched segments); bootstrap CI on the delta **+0.174 [0.031, 0.209]**,
  excludes zero (E3: `results/e3_bootstrap_ci.py`). Matchable-subset re-score
  0.189 → **0.243** (E2: `results/e2_matchable_subset.py`).
- **In-distribution technique-ID (n=496 clips, gemini-2.5-flash, ~$0.61):**
  position top-1 **58.9%** [54.5, 63.1], technique match **48.0%** [43.6, 52.4],
  both-correct **35.9%**, either-match **71.0%** [66.8, 74.8].
- **Model-scale comparison (E1, gemini-2.5-pro on the same 496 clips, ~$2.95):**
  position top-1 **66.3%** vs 58.9%, technique **40.3%** vs 48.0%, both/either
  unchanged — scale redistributes errors rather than removing them
  (`results/vlm_technique_id_pro.json`, scored by
  `results/_vlm_technique_id_scorer_v2.py`).
- **Out-of-distribution correctness (n=200 videos / 4,620 expert-reviewed
  segments):** exact **46.7%** [42.5, 51.4], coarse **59.7%** [56.4, 63.4],
  adjacent-cluster **92.9%** [91.9, 93.8] (video-clustered 95% CIs) — 93% of
  errors land on an adjacent position: graceful, systematic degradation.
- **CV-only HMM baseline:** smoothed T1 macro-F1 **0.2126** ≥ best VLM 0.1888.

See `MANIFEST.md` §5–§6 for the full result inventory and exact reproduce
commands.

## Data and privacy stance

The v0 seed is a single public competition match, anonymized (athletes `A1`/`A2`,
opaque `sample_id`). Raw production videos, user data, and the instructional-clip
corpus are **withheld** and referenced by aggregate counts only; scripts that
need them are released as recipes. No file in this repository contains names,
URLs, user identifiers, or local paths.

## License

Dual-licensed by content type:

- **Code — [MIT](LICENSE).** All Python source: the `movebench/` package and
  every script under `results/`, `qa/`, and `data/` (e.g.
  `download_technique_clips.py`).
- **Data and documentation — [CC BY 4.0](LICENSE-DATA).** The benchmark data
  and annotations (`data/v0/`, `ontology.json`), the stored result and
  aggregate JSON files under `results/` and `qa/`, and the Markdown
  documentation (`README.md`, `MANIFEST.md`, `OVERVIEW.md`, `SCHEMA.md`,
  `results/*.md`).

Attribution for CC BY 4.0 material: cite the paper (see below) or credit
"MoveHarness / MoveBench v0, Dhruva Chakravarthi, 2026" with a link to this
repository.

## Citation

```bibtex
@inproceedings{chakravarthi2026moveharness,
  title     = {Harnessing Vision--Language Models for Faithful Human-Movement Understanding},
  author    = {Chakravarthi, Dhruva},
  booktitle = {Conference for AI Scientists (CAISc)},
  year      = {2026}
}
```
