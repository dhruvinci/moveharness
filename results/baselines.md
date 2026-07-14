# MoveBench v0 — Baseline Results

**Scope.** This file reports the *only* honest, machine-verified baseline numbers
available for MoveBench v0. Every number here is either (a) read from a stored
machine-generated evaluation artifact, or (b) recomputed this session by running
`movebench.evaluate` over the v0 ground truth (`data/v0/mb_v0_0001.json`) against
an honest pipeline output. No number here is hand-tuned toward the ground truth,
and the one tautological signal in the project (the P1 50-video 1.0) is
explicitly excluded and explained.

**Hard caveat that frames the whole section.** MoveBench v0 contains exactly
**one** annotated sample — a single ~16-minute grappling match (32 macro
positions / 62 subsegments / 122 technique labels, single annotator). Everything
below is therefore a **single-sample developer measurement, not a leaderboard
result.** Treat these as *sanity signals and a difficulty floor*, not as model
rankings. No confidence intervals are computable from n=1.

**Scoring framing: every accuracy number below is a STRICT-SYNTACTIC FLOOR.**
MoveBench judges correctness **semantically, not syntactically** (see
`README.md §3` and `../../shared/semantic-evaluation.md`): positions/techniques by
ontology-normalized **equivalence classes** (synonyms, aliases, hierarchical
parent/child equivalence — the scorer's `ontology.json` alias-folding is the first
step), and free-text by **semantic adjudication** (expert / LLM-as-judge meaning
equivalence, abstain-aware). Generative VLM output is non-deterministic in surface
form, so the exact-class / segment-level / string-match numbers reported here
**systematically under-count correctness** and are a **conservative LOWER BOUND**,
explicitly labeled "strict / syntactic". The **semantic** score is the **intended
standard** and is **expected to be higher**, but it is **not yet rigorously
measured — no semantic number is invented here.** We report **two regimes**
(strict-syntactic floor vs semantic standard); the gap between them is itself an
informative result that motivates a semantics-aware benchmark.

---

## 1. Reference baseline (stored, machine-verified): multi-pass pipeline

The one pre-existing machine-checkable evaluation in the experiment repository is
a stored `evaluation_report.json` from a multi-pass VLM pipeline ("Exp2"),
evaluated against this same single match.

| Metric | Value |
|---|---|
| GT macro positions | 32 |
| Detected positions | 33 |
| Matched | 22 |
| Recall | **0.688** |
| Precision | **0.667** |
| F1 | **0.677** |
| Position accuracy (matched-and-correct-class, **strict-syntactic floor**) | **0.273** |
| Average temporal IoU | **0.703** |
| Composite score | 52.9 / 120 |

**Provenance / limits (state these every time these numbers are cited):**
- Single match; one pipeline; one annotator's ground truth.
- A **development artifact**, computed by that pipeline's *own* harness — not by
  the MoveBench scorer and not under a controlled, multi-model protocol.
- The metric here is span-level: a GT position "matches" a detection on temporal
  overlap; "position accuracy" then asks whether the *class label* of matched
  spans is right — **under strict exact-class matching, i.e. the syntactic floor**.

**Honest one-line reading:** *good span localization, weak position labeling
under strict syntactic matching* — the pipeline finds *where* the salient segments
are (IoU 0.70, recall 0.69) but is scored as getting the *position class* right
only ~27% of the time **on exact-class matching**. That gap (find-the-span ≫
name-the-position) is the central difficulty MoveBench is built to measure. The
~27% is a **strict-syntactic lower bound**, not a correctness rate: the semantic
score (ontology equivalence classes + abstain-aware adjudication) is the intended
standard and is expected to be higher but is *not yet measured*.

---

## 2. MoveBench-scored baselines (recomputed this session)

We ran `movebench.evaluate` over the v0 ground truth against two **honest,
unedited** machine outputs taken from real pipeline runs on the same match and
converted to the MoveBench prediction schema (timestamps + ontology-mappable
position/technique tokens only; all athlete names, the video id, narratives, and
coaching free-text were dropped — see `results/_convert_experiment_timeline.py`).

- **Pipeline A** — single-pass CV-cached timeline, 58 segments.
- **Pipeline B** — two-stage skeleton+detail timeline, 25 segments.

Both are real outputs, not gold-aligned. The position field was normalized only
through the same `ontology.json` the scorer uses; nothing was remapped to flatter
the score.

| Regime / system | T1 macro-F1 | T1 dur-wtd F1 | T1 mean IoU | matched | pred segs | T3 micro-F1 (oracle) | T3 ontology-violation rate | coverage |
|---|---|---|---|---|---|---|---|---|
| **Pipeline A (single-pass, 58 seg)** | **0.099** | 0.122 | 0.463 | 5 / 32 | 58 | 0.017 | 0.694 | 1.00 |
| **Pipeline B (two-stage, 25 seg)** | **0.136** | 0.200 | **0.723** | 6 / 32 | 25 | 0.037 | 0.608 | 1.00 |

(Full machine reports: `results/reports/report_pipeline_A_exp3.json`,
`results/reports/report_pipeline_B_exp4.json`. Reproduce with
`python -m movebench.evaluate --gt data/v0/mb_v0_0001.json --pred results/predictions/pred_pipeline_<A|B>.json`.)

### What these MoveBench numbers show
> All T1/T3 F1 numbers in this table are the **strict-syntactic floor**: positions
> are normalized through `ontology.json` alias-folding (the first step of semantic
> matching) and then matched on exact equivalence-class id. They are a conservative
> lower bound, not the semantic correctness rate, which is the intended standard and
> is not yet measured.

1. **Position labeling is the bottleneck, again (under strict matching).** Both
   pipelines hit ~100% *coverage* (they tile the whole timeline) yet land
   **macro-F1 ≤ 0.14** and match only **5–6 of 32** GT positions. Coverage ≠
   correctness, exactly as the scorer's caveat warns; a complete timeline can still
   be mostly mislabeled under exact-class scoring.
2. **Pipeline B's mean matched IoU (0.723) reproduces the ~0.70 IoU** seen in the
   stored Exp2 report — independent confirmation that *localization is the
   strong axis and classification is the weak one*, on a second pipeline and the
   MoveBench scorer.
3. **A visible degenerate failure mode.** Pipeline A's source labels **43 of 58
   segments "mount"** (a single run of "mount" from 3:50 to 15:11). This is a
   real collapse in the upstream artifact, not a conversion bug — MoveBench
   surfaces it as `mount: 1 tp / 42 fp` and a near-zero macro-F1. This is the
   kind of pathology the benchmark is meant to catch.
4. **Vocabulary mismatch is a real, measured cost.** The v0 GT uses `escape` and
   `sweep` as macro-position classes for **12 of 32 segments (37.5%)**. Neither
   pipeline's 18-position vocabulary emits those tokens, so those 12 segments are
   **unmatchable by construction** and cap the achievable macro-F1 well below
   1.0. This is itself a finding: the benchmark's ontology and the production
   pipelines' label space are not yet aligned, and MoveBench makes that gap
   quantitative. (When restricted to the 20 vocab-accessible GT segments, the
   pipelines still match only 5–6 — so vocabulary is *a* cap, not the whole
   story.)
5. **T3 is oracle-segment and approximate here.** The T3 micro-F1 numbers align
   predicted segments to GT macro boundaries *by index*, but these pipelines do
   not emit GT-aligned segments, so the T3 row is a weak lower bound, not a clean
   oracle-segment score. The high ontology-violation rate (0.61–0.69) mostly
   reflects pipeline technique tokens (`collar_ties`, `referee_standup`, etc.)
   that fall outside the v0 used-label set, plus this misalignment. Read T1 as
   the load-bearing position result; treat T3 here as directional only.

---

## 3. Integrity guards (verified firing)

- **Degenerate-baseline guard.** On the bundled `*_degenerate_demo` pair
  (`label_source = prediction_review`, GT byte-identical to prediction), the
  scorer returns `degenerate: true` and **suppresses headline T1/T3 accuracy to
  `null`**. This is the exact mechanism that makes the **P1 50-video macro-F1 /
  IoU / transition-F1 of 1.0 non-reportable** — those GT files are accepted model
  predictions (GT == prediction by construction), so the 1.0 is tautological, not
  accuracy. **It is excluded from every table here and must never be cited as a
  baseline.**
- **Empty-GT guard.** The T2 (transition) and scoring tasks have **empty v0 GT
  layers** and report `task_status: "no_v0_ground_truth"` — never a misleading 0
  or 1.
- **Non-degeneracy proof.** `python -m movebench._smoketest` passes all seven
  assertions (`SCORER NON-DEGENERATE: True`): the identity prediction scores 1.0
  but the perturbed prediction scores strictly lower on macro-F1, IoU, and
  label-F1 with a nonzero ontology-violation rate. The two honest pipelines above
  independently produce *low, differentiated* scores, confirming the scorer
  responds to real error.

---

## 4. What is NOT yet measurable (state plainly in the paper)

- **No multi-model leaderboard.** n=1 sample; the "baselines" are two dev
  artifacts plus one stored report, not a controlled head-to-head. No ranking
  claim is supportable.
- **Transition detection (T2): no ground truth.** The v0 transition layer is
  empty; transition-F1 is uncomputable. (Reported as `no_v0_ground_truth`.)
- **Scoring (ADCC/IBJJF): no ground truth.** Scoring-event layer is empty;
  scoring accuracy is uncomputable.
- **Role / identity consistency (T4): not scored here.** v0 carries top/bottom
  roles but no adversarial identity-swap labels; the full role track is v1.
- **Coaching critique (T5): only coverage is scored.** Grounded-critique quality
  and hallucination rate need reference critiques (v1); v0 scores only timeline
  *completeness*, which is explicitly *not* correctness.
- **Cross-sport / cross-style portability: untested.** The single sample is one
  no-gi grappling match; judo / wrestling / MMA / striking portability is a
  hypothesis with zero labeled evidence.
- **Inter-annotator agreement: unknown.** Single annotator; no second-annotator
  data, so no reliability estimate.
- **Confidence intervals: none.** Every number is a point estimate on one sample.

---

## 5. Citable summary (paste-ready for the P2 evidence section)

> On MoveBench v0 (one annotated ~16-min match; 32 macro positions / 62
> subsegments / 122 labels; single annotator), the only honest baselines are
> developer-grade, single-sample measurements, and every accuracy figure is a
> **strict-syntactic floor** — MoveBench scores correctness *semantically*
> (ontology-normalized equivalence classes for positions/techniques; abstain-aware
> semantic adjudication for free-text), so exact-class numbers are a conservative
> lower bound, not a correctness rate. A stored multi-pass pipeline evaluation
> reports recall 0.688, precision 0.667, F1 0.677, **strict** position accuracy
> 0.273, and average temporal IoU 0.703 — i.e. **strong span localization
> (IoU ≈ 0.70) but weak position-class labeling under exact matching (≈ 27%,
> syntactic floor).** Re-scoring two honest, unedited pipeline timelines with the
> MoveBench scorer reproduces this profile: both reach ~100% timeline coverage yet
> only **macro-F1 0.10–0.14** (strict), matching 5–6 of 32 ground-truth positions,
> with the better pipeline again landing mean IoU ≈ 0.72. The semantic-correctness
> rate is the intended standard, is expected to be higher, and is *not yet
> measured* (no semantic number is claimed); results are reported in **two regimes**
> (strict-syntactic floor vs semantic standard) and the gap is itself a finding.
> The benchmark's integrity guards fire as designed: the
> tautological P1 50-video 1.0 (ground truth == predictions) is detected and
> suppressed and is *not* reported; empty transition and scoring layers report
> `no_v0_ground_truth` rather than a misleading score. **Transition detection,
> scoring accuracy, role-consistency, critique quality, cross-sport portability,
> and any multi-model ranking are not yet measurable** and are deferred to v1.

---

*Generated by the MoveBench BASELINE stage. All inputs are read-only; no paper,
shared doc, or repo was modified. No athlete names, video ids, URLs, or local
paths appear in any shipped prediction or report (privacy-scanned in the
converter; aborts on leak).*
