# MoveBench v0

**The first benchmark for fine-grained vision–language understanding of technical human movement.**

MoveBench measures whether a vision–language model (VLM) can recover a *temporally coherent,
technically correct, and visually grounded* account of long-form, contact-rich expert human
movement — the capability coaching, rehabilitation, surgery, skilled manual work, and sports
officiating actually need, and the one that fluent VLM captions fail to deliver.

Fluency is a weak proxy for technical-movement understanding, and no existing video benchmark
(Kinetics, Something-Something, EgoSchema, MVBench, Video-MME, LongVideoBench, FineGym, FineDiving,
MultiSports, …) forces a model to prove the difference. MoveBench does.

**v0 is a reproducible seed + protocol, not a finished leaderboard.** No-gi submission grappling
(MMA / grappling) is the v0 *stress-test instantiation* (it maximizes all four difficulty axes at
once), not the scope. The ambition is technical human movement broadly.

> **Honesty contract.** v0 = one annotated public match. No generalization claims across athletes,
> camera views, styles, or sports. Production-scale numbers are *operational* evidence, not
> *correctness* evidence. The production corpus is a **candidate pool**, private, not release-cleared.
> Transition and scoring ground truth are **not yet annotated** (planned for v1). Existing model
> outputs are **development artifacts**, not controlled leaderboard runs.

> **Privacy / double-blind.** No athlete names, video URLs/IDs, gym names, user IDs, emails, or local
> paths appear anywhere in this directory. Athletes are `A1`/`A2`. The single public match is referenced
> by an opaque `sample_id`. External private resources (the production export, the instructional SQLite
> corpus) are referenced **by aggregate counts only** and are never redistributed.

Files in this directory:
- `README.md` — this file: task suite, metric vector, splits, expansion protocol.
- `SCHEMA.md` — sample / submission / prediction data schema.
- `ontology.json` — machine-readable label ontology (18 macro positions, ~91-term vocabulary, 42 used in v0).

---

## 1. Task suite

Five **input-video understanding** tasks. Each marked with its v0 ground-truth status.

| ID | Task | Unit | v0 ground truth | Status |
|----|------|------|-----------------|--------|
| **T1** | **Position-timeline labeling** | macro-position segments `[start,end]` | **YES** — 32 macro segments on the seed match | **scorable in v0** |
| **T2** | **Transition detection** | timestamped events (actor + type) | **NO** — `ground_truth_transitions` empty | **planned v1** |
| **T3** | **Technique identification** | per-segment / per-subsegment label sets | **YES** — 62 subsegments, 122 label instances, 42 unique | **scorable in v0 (oracle-segment)** |
| **T4** | **Athlete-role / identity consistency** | per-athlete (A1/A2) roles over time | **partial** — top/bottom present on macro segments; full role track planned | **partial v0 → v1** |
| **T5** | **Grounded coaching critique / long-form coverage** | coaching note + coverage of the whole timeline | **NO** reference critiques; coverage is computable | **planned v1 (critique); coverage scorable in v0** |
| — | Positional control / scoring (ADCC/IBJJF) | timestamped scored events + point totals | **NO** — `ground_truth_scoring_adcc` empty | **planned v1** |
| — | MoveBench-Gen (text→video / motion generation) | generated demonstration | **NO** | **future work (out of scope for v0)** |

### Task definitions

- **T1 — Position-timeline labeling.** Given a full match or clipped round, produce a temporally
  ordered list of macro positions (L1A) with start/end times. *A position is a STATE* (stable),
  contrasted with a transition (an event). Report both segment-weighted and duration-weighted scores
  (long static positions vs short scrambles create different failure incentives). **This is the
  anchor task with real v0 ground truth.**

- **T2 — Transition detection.** Identify discrete events: takedown, guard-pull, sweep, guard-pass,
  back-take, mount-achieve, escape, reguard, leg-entry, submission-on/off, tap, reset. Scored as
  timestamped events with an actor and a tolerance window. *Planned v1* — the v0 seed has an empty
  transition layer; T2 is **defined but not yet evaluated**.

- **T3 — Technique identification.** Given full video or oracle temporal segments, assign ontology
  labels to each segment/subsegment. v0 supplies 62 subsegments and 122 label instances over 42
  unique labels. Evaluate **oracle-segment** (GT boundaries given → isolates semantic understanding)
  and **end-to-end** (boundaries + labels together).

- **T4 — Athlete-role / identity consistency.** Track A1/A2 roles (top/bottom/neutral/passing/
  retaining/attacking/defending/escaping) over time. Measure role F1, **identity-swap count**
  (A1↔A2 exchanged), and **contradiction count** (incompatible roles in one interval, e.g. same
  athlete labeled bottom-guard and top-passer). v0 has top/bottom on macro segments; the full role
  track is *planned v1*.

- **T5 — Grounded coaching critique & long-form coverage.** Two linked sub-goals: (a) *coaching
  critique* — produce a short, visually grounded note per segment, screened for hallucination
  (references to visible positions/grips/transitions; no impermissible claims under occlusion);
  reference critiques are *planned v1*. (b) *long-form coverage* — what fraction of the full
  (minutes-to-tens-of-minutes) video is spanned by valid non-overlapping segments. Coverage is
  computable in v0. **Coverage measures completeness, NOT correctness** (see §3).

- **Positional control / scoring (planned v1).** Reconstruct ADCC/IBJJF scored events and point
  totals. The v0 seed's `ground_truth_scoring_adcc` is empty; this task is *defined but not yet
  evaluated*.

- **MoveBench-Gen (future work).** Generating a correct technical demonstration from a prompt
  (text→video, motion editing). Deliberately **out of scope for v0** — the spine is *understanding*
  of input video. One clearly-labeled future-work track only.

---

## 2. Label ontology (summary)

Full machine-readable definition: **`ontology.json`** (`ontology_version` 0.1.0).

- **18 macro positions (L1A, states):** guard, mount, side_control, back_control, north_south,
  knee_on_belly, turtle, half_guard, standing, clinch, takedown, single_leg, double_leg,
  passing_guard, sweep, submission_attempt, scramble, unknown/unclear. `unknown/occluded` is always
  a legal label.
- **~91-term technique vocabulary** (L1B sub-positions, takedowns, passes, sweeps, submissions,
  grips/frames, transitions, match lifecycle) — the full label space MoveBench draws from.
- **42 unique labels used in v0** (excluding the 2 protocol annotations `pre_match`/`match_start`),
  over 122 label instances across 62 subsegments. The exact 42 with frequencies and normalized
  mappings are in `ontology.json → v0_used_labels`.
- **Reconciliation (use consistently):** *"an ontology of 18 macro positions and a ~91-term technique
  vocabulary, of which 42 unique labels appear in the v0 seed."* Raw JSON has 44 unique labels and
  64/124 subsegments/labels; the 2 protocol annotations are excluded to reach 42 and 62/122. The v0
  macro `position` field itself uses a reduced 7-value working set (a subset of the 18 plus the
  event-style states `sweep`/`escape`); see `ontology.json → v0_macro_position_field_values`.
- **Hierarchy:** L0 engagement / L1A position / L1B sub-position / L2 role+action / transition events
  / scoring events. Always cite an ontology *version*.
- **Domain-specificity:** this ontology is BJJ-specific and does **not** transfer to judo, wrestling,
  striking, dance, or surgery without re-authoring the label set. Ontology transfer is a v1 open
  question, not an assumption.

---

## 3. Metric vector

MoveBench reports a **vector, not a single scalar.** A cheap model that gets boundaries right but
labels fewer techniques may be preferable in a product setting to a verbose model that hallucinates
unseen submissions; a scalar would hide that tradeoff. A scalar leaderboard score may be defined
later, but early reporting exposes the axes.

> **Correctness is judged SEMANTICALLY, not syntactically.** Generative VLM output is
> non-deterministic in surface form: re-running the same model on the same video yields different
> words for the same meaning, so exact-string / strict-syntactic comparison **systematically
> under-counts correctness**. Correctness is a *semantic* property — did the model identify the
> **right position / transition / role** and make the **right (semantically correct) coaching call**,
> regardless of exact wording. MoveBench scores meaning, not strings:
> - **Positions / techniques** are scored by **ontology-normalized equivalence classes** — synonyms,
>   aliases, and hierarchical parent/child equivalence count as matches. The scorer's existing
>   `ontology.json` **alias-folding is the first step** of this normalization, extended conceptually
>   to full semantic equivalence; it is *not* a string match.
> - **Free-text coaching / critique** is scored by **semantic adjudication** — expert judgment or an
>   LLM-as-judge scoring *meaning equivalence* against the reference, **abstain-aware** (no penalty on
>   genuinely ambiguous / occluded spans).
> - **Temporal boundaries** use tolerance (temporal IoU); the *label* match within a matched span is
>   judged semantically.
>
> See `../shared/semantic-evaluation.md` for the authoritative scoring philosophy.

| Metric | Task | Definition | v0 |
|--------|------|------------|----|
| **Macro-F1 over positions** | T1 | Mean per-position F1 after greedy position+IoU matching at IoU≥0.3. Report segment-weighted **and** duration-weighted. | ✅ |
| **Temporal IoU** | T1 | Mean IoU of matched segment pairs (and boundary error). | ✅ |
| **Label F1 (ontology-aware)** | T3 | Micro/macro F1 over `labels` sets; hierarchical partial credit; **ontology-violation rate** as a sub-metric. | ✅ (oracle-segment) |
| **Transition-F1** | T2 | Event F1 within timestamp tolerance (default 5 s), with actor correctness and false-transition rate. | ⏳ planned v1 (empty GT) |
| **Role consistency** | T4 | Role F1 + identity-swap count + contradiction count. | partial |
| **Coverage** | T5 | Fraction of video duration spanned by valid, non-overlapping segments. **Completeness, NOT correctness.** | ✅ |
| **Hallucination rate** | T5 | Fraction of coaching/labels referring to non-visible / impossible technique under occlusion (expert-screened; auto-screen for v1). | ⏳ planned v1 |
| **Cost / latency per minute** | system | Estimated $ and seconds per video-minute (mark estimated vs billed). | ✅ (operational) |

> **The coverage ≠ correctness caveat is load-bearing.** High coverage means the timeline is
> *complete*, not *accurate*. Always report coverage next to T1/T3 accuracy, never as a stand-in for
> it. (Operational evidence from the development trace: a multi-pass run reached ~100% coverage with
> 241 segments — 7.5× the ground truth — while position accuracy stayed low.)

### Reporting regimes
- **Oracle-segment** — model is given GT macro boundaries; tests semantic labels only (T3). Isolates
  understanding from localization.
- **End-to-end** — model produces the full timeline from raw video (T1 + T3 + T2/T4/T5).
- **System** — measures cost, latency, coverage, and output-schema validity for production-style
  pipelines.

#### Strict-syntactic vs semantic (two-regime reporting)
Every accuracy axis is reported under **two regimes**, and the gap between them is itself an
informative result that motivates a semantics-aware benchmark:
- **Strict-syntactic floor** — exact / segment-level / string match (e.g. raw exact-match macro-F1,
  the ~30% segment acceptance below). This is a deliberately conservative **lower bound**, never the
  intended standard.
- **Semantic** — ontology-normalized equivalence classes for positions/techniques plus semantic
  adjudication (expert / LLM-as-judge) for free-text, abstain-aware. **This is the intended scored
  standard.** The semantic-correctness rate is expected to be higher than the syntactic floor but is
  **not yet rigorously measured**, so no semantic number is asserted here — the semantic protocol is
  the standard, and the strict-syntactic number is the only concrete figure in hand.

### Reference scorer & integrity guards
The v0 scorer adapts the P1 harness (`score_p1_eval.py`): `interval_iou`, `score_segments` (greedy
position+IoU → per-position TP/FP/FN → macro-F1 + mean matched IoU), `score_transitions`
(timestamp-tolerant event F1), and alias-folding normalization (now sourced from `ontology.json`).
See `SCHEMA.md §4` for field bindings.

The scorer's `ontology.json` **alias-folding is the first step of semantic normalization**: it folds
surface forms into canonical ids before matching, so the loader never compares raw strings. The
semantic standard extends this to full ontology-normalized equivalence classes (synonyms, aliases,
hierarchical parent/child equivalence) for positions/techniques, and to expert / LLM-as-judge
semantic adjudication for free-text. The current `metrics.py` implements the alias-folding floor;
the full semantic free-text adjudication is the design standard (pilot/planned), not yet a number.

Two **mandatory integrity guards**:
1. **No degenerate baselines.** Any sample where the ground truth was produced by accepting the model
   output (`label_source == "prediction_review"`) and is byte-identical to the prediction is flagged
   `degenerate: true` and **excluded** from aggregate accuracy. (The P1 50-video slice scores 1.0
   *because* GT == predictions by construction — that 1.0 is tautological and must never be reported
   as a result.)
2. **Empty GT layers are not 0/1.** Tasks with no v0 ground truth (T2 transitions, scoring) report
   `task_status: "no_v0_ground_truth"`, not a misleading F1.

### The only honest v0 baseline signal — a STRICT SYNTACTIC FLOOR
The single machine-verified, non-tautological baseline in v0 is the stored multi-pass evaluation over
the 32 macro positions of the seed match:

| recall | precision | F1 | position-accuracy | mean IoU |
|--------|-----------|----|-------------------|----------|
| 0.688 | 0.667 | 0.677 | 0.273 | 0.703 |

> **These are strict-syntactic-floor numbers, a conservative LOWER BOUND — not a correctness rate.**
> The ~0.27 position-accuracy and the ~30% segment acceptance (14 of 47 reviewed segments accepted on
> one match) are **strict, syntactic, segment-level** counts. They must be labeled "strict / syntactic
> floor", **never** read as "the model is ~27–30% correct". Because surface form is non-deterministic,
> the **semantic** correctness rate (ontology-normalized equivalence classes + abstain-aware semantic
> adjudication) is the **intended standard** and is **expected to be higher** — but it is **not yet
> rigorously measured, so no semantic number is invented here.** Report both regimes; the gap is the
> result.

Read the floor honestly: detection F1 ~0.68 but **strict position accuracy only ~0.27** — the model
finds roughly the right *spans* but, *under exact-class matching*, is frequently scored wrong about
*which position* they are. That gap (good IoU, weak strict accuracy) is exactly the failure MoveBench
exists to surface, and the strict-vs-semantic gap is exactly why a semantics-aware benchmark is
needed rather than string-matching on existing pipelines. The earlier naive single-pass baseline
scored ~16.3/100 (a project-log composite, distinct from the ground-truth quality score) with ~13.8%
strict position accuracy and zero subsegments/labels — useful as a seed error case, not a leaderboard
entry.

---

## 4. Splits

- **v0 (current).** One public no-gi match → assigned to the **`test`** split (the seed). There is
  no `train` split in v0; v0 is for evaluation only. A `dev` split is created in v1.
- **Split discipline (v1).** Splits are **by video, not by segment**, to prevent leakage. Stratify
  train/dev/test by competition vs training, gi vs no-gi, camera view, length, and movement density,
  with a source/uploader constraint so the same athlete or gym does not dominate multiple splits.
- **Hidden test set.** A held-out test set is maintained for leaderboard comparisons; only aggregate
  metrics are reported for it.

---

## 5. Consent-safe expansion protocol (v0 → v1)

v1 grows MoveBench from one match to a multi-video, consent-cleared benchmark. The expansion uses two
**external private resources by reference only** — neither is redistributed here, and only aggregate
counts are cited:

- **Production candidate corpus** (private export): 313 BJJ videos; 269 completed analyses; 229 with
  completed timelines; 236 with compressed files; **213 immediate candidates** (ready + nonzero
  duration + completed timeline + compressed file). A *candidate pool*, **not** release data.
- **Instructional clip corpus** (external SQLite, expert-annotated): 430 videos / **5,604** described
  clips / **1,653** distinct technique tags. Useful for ontology expansion, rare-technique discovery,
  and a possible separate clip-level task **if rights allow** — referenced by counts only, never copied.

### Selection procedure
1. **Start from the 213 immediate candidates** (the eligible production subset).
2. **Consent / rights filter.** Remove any video without explicit research permission, any video
   whose uploader or visible athletes decline release, and any file with ambiguous ownership. No
   per-video consent ledger exists yet → this step gates v1.
3. **Stratify a 30–50 video subset** by source type, duration, segment count, coverage, camera
   quality, and movement density. Practical first target: ~20–30 training rounds, ~10–15 competition
   videos, plus a small number of instructional/study clips **only if rights are unambiguous**.
4. **De-identify** every released sample to the `SCHEMA.md` privacy invariant (A1/A2; opaque
   `sample_id`; no URLs/IDs/names/paths).
5. **Annotate** with the full schema (macro segments → subsegments → transitions → scoring →
   per-athlete roles → coaching refs), using **≥2 annotators** on a subset for inter-annotator
   agreement (kappa/alpha), with expert adjudication of macro positions, transition boundaries, and
   technique labels. Mark occluded/low-confidence intervals **explicitly** rather than forcing false
   precision.
6. **Add the missing v0 GT layers** to the seed match first: annotate transitions (T2) and
   ADCC-style scoring, so the seed becomes fully scorable across all tasks before scaling.
7. **Release** derived annotations + model-output baselines for consent-cleared videos only; for
   non-cleared videos keep them internal-only or report aggregate statistics without video.

### Annotation source provenance
The instructional clip corpus is the upstream source of the production study-library tables; its
clip-level natural-language descriptions and dense technique tags are the candidate pool for
**ontology expansion** (growing the ~91-term vocabulary and discovering rare techniques), distinct
from the position-timeline annotation of full matches. Clip rights and task-fit for a public release
are **unresolved** and must be cleared before any use.

---

## 6. What v0 deliberately does NOT claim

- No multi-model leaderboard (model roster is planned; existing outputs are dev artifacts from one
  pipeline).
- No cross-sport transfer evidence (zero non-BJJ runs; ontology transfer is a hypothesis).
- No inter-rater agreement (single primary reviewer; no second annotator yet).
- No measured accuracy on the production corpus or any 50-video slice (no non-tautological labels).
- No transition-F1 or scoring-accuracy (those ground-truth layers are empty in v0).
- Costs are **estimated** from logged tokens + pricing, not an audited billing export.

v0's contribution is the **measuring stick**: the task definitions, the ontology, the **semantic (not
syntactic) scoring principle** — ontology-normalized equivalence classes plus abstain-aware semantic
adjudication, reported as a strict-syntactic floor vs a semantic standard — the metric vector, a
reproducible single-match seed that already exposes where fluent VLMs break (good span IoU, weak
*strict* position accuracy, coverage ≠ correctness), and a consent-safe protocol to scale it. The
concrete v0 numbers are the strict-syntactic floor; the semantic-correctness rate is the intended
standard and is *not yet measured* (no semantic number is claimed).
