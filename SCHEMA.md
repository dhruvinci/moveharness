# MoveBench v0 — Data & Submission Schema

Formal data-format specification for **MoveBench v0**, the benchmark for fine-grained
vision–language understanding of technical human movement (BJJ / no-gi grappling instantiation).

This file defines: the **sample schema** (a benchmark example with its ground truth), the
**prediction / submission schema** (what a model emits), and the **label ontology binding**
(`ontology.json`). All times are stored both as raw strings (`MM:SS`) and as normalized seconds
(float) to avoid clock-drift bugs across sources.

> Status: this is the **v0 schema definition**. Only Task T1 (macro position timeline) has v0 ground
> truth. T2 (transitions) and the scoring task have empty ground-truth layers in v0 and are
> *planned for v1* — their fields are specified here so submissions and tooling are forward-compatible.

> Privacy invariant (binding on every file in this directory and any release built from it):
> no athlete names, video URLs, YouTube IDs, gym names, user IDs, emails, or absolute/repo-relative
> local paths may appear in any sample, submission, or released ground-truth file. Athletes are
> referred to only as anonymized `A1` / `A2`. The single public match is referenced by an opaque
> `sample_id` only.

---

## 1. Ontology binding

Every label in a sample or submission MUST be drawn from `ontology.json` (`ontology_version` is
recorded in each file). **Correctness is judged SEMANTICALLY, not syntactically** — scoring never
compares raw strings. Surface forms are normalized to canonical ids, and the canonical ids are then
matched as **ontology-normalized equivalence classes** (synonyms, aliases, hierarchical parent/child
equivalence count as matches). The alias-folding below is the **first step** of this semantic
normalization, not the whole of it. Scorers normalize surface forms to canonical ids via:

- `L1A_macro_positions[*].id` and `[*].aliases` (e.g. `"Standing"` → `standing`,
  `"Side Control"` → `side_control`, `"Half Guard"` → `half_guard`).
- `L1B_subpositions`, `L2_roles`, `transition_events`, `scoring_events` for the finer levels.
- `v0_used_labels[*].normalized` documents the exact mapping used for the v0 ground-truth surface
  forms (the v0 GT mixes cased macro names with snake_case technique labels).

Free-text fields (`note`, `coaching`) are **not** string-matched: they are judged by **semantic
adjudication** (expert / LLM-as-judge scoring meaning equivalence against the reference,
abstain-aware — no penalty on genuinely ambiguous / occluded spans). See `README.md §3` and
`../shared/semantic-evaluation.md`.

`unknown` / `occluded` is always a legal L1A value and must be accepted in both GT and predictions.

---

## 2. Sample schema (benchmark example + ground truth)

One JSON object per sample (per video / clip). Filename: `<sample_id>.json`.

```jsonc
{
  "schema_version": "movebench-sample-0.1.0",
  "ontology_version": "0.1.0",
  "sample_id": "mb_v0_0001",            // opaque, anonymized. NO youtube id / url / name.
  "split": "test",                       // one of: dev | test | (v1: train)
  "source_type": "competition",          // competition | training | instructional | study_clip
  "modality": "no_gi",                   // gi | no_gi | unknown
  "duration_sec": 959.0,                 // normalized seconds (float). Authoritative time base.
  "duration_str": "15:59",               // human-readable; may differ slightly from duration_sec by trim.
  "rights_status": "public_seed",        // public_seed | consented_release | internal_only | aggregate_only
  "camera_view": "broadcast",            // broadcast | side | elevated | mat_level | unknown
  "annotation": {
    "annotator_roles": ["expert_reviewer"],   // anonymized roles, NEVER names/ids
    "n_annotators": 1,
    "label_source": "human_review",            // human_review | prediction_review | seed
    "inter_annotator_agreement": null          // populated in v1 (kappa/alpha) when >=2 annotators
  },

  // ---- T1 ground truth: macro position timeline (PRESENT in v0) ----
  "macro_segments": [
    {
      "seg_id": "s0",
      "start_sec": 8.0,
      "end_sec": 50.0,
      "start_str": "00:08",
      "end_str": "00:50",
      "position": "standing",            // L1A id (or v0 reduced field value; see ontology)
      "sub_position": "clinch",          // L1B id, nullable
      "roles": {"A1": "neutral", "A2": "neutral"},   // L2 per-athlete, nullable
      "confidence": 0.9,                 // annotator confidence, nullable
      "rating": 2,                       // optional 1-5 reviewer quality rating of the segment
      "ambiguous": false,                // explicit uncertainty flag (occluded/unclear)
      "subsegments": [                   // finer technique/action spans (T3 ground truth)
        {
          "sub_id": "s0.0",
          "start_sec": 8.0,
          "end_sec": 10.0,
          "labels": ["standing", "grip_fighting"],   // ontology label ids/surface forms
          "note": "free-text technical description (kept SEPARATE from coaching critique)"
        }
      ]
    }
    // ... 32 macro segments in the v0 seed (62 benchmark subsegments, 122 label instances)
  ],

  // ---- T2 ground truth: transitions (EMPTY in v0; planned v1) ----
  "transitions": [
    {
      "time_sec": 50.0,
      "type": "takedown",                // transition_events id
      "actor": "A2",                     // A1 | A2 | null
      "from_position": "standing",
      "to_position": "guard",
      "tolerance_sec": 5.0               // matching window for scoring (default 5s)
    }
  ],

  // ---- Scoring ground truth: ADCC/IBJJF points (EMPTY in v0; planned v1) ----
  "scoring": {
    "ruleset": "adcc",
    "events": [
      {"time_sec": 50.0, "athlete": "A2", "kind": "takedown", "points": 2}
    ],
    "totals": {"A1": 0, "A2": 0}
  },

  // ---- Coaching critique ground truth (T5; expert-rated, planned v1) ----
  "coaching_refs": [
    {"seg_id": "s0", "athlete": "A1", "note": "reference coaching point", "evidence": ["grip_fighting"]}
  ]
}
```

### Field rules
- `start_sec` / `end_sec` are the authoritative time base. `*_str` are advisory; scorers MUST use seconds.
- `macro_segments` must be **non-overlapping** and ordered by `start_sec`. Gaps are allowed
  (uncovered video → counts against coverage, see metrics in README).
- In v0, `position` may take a value from the reduced 7-value working set
  (`standing, guard, sweep, back_control, escape, mount, side_control`) — `sweep`/`escape` are
  event-style states used as segment labels (see `ontology.json → v0_macro_position_field_values`).
- `transitions` and `scoring.events` are **empty arrays / zeroed** in v0. Tooling must treat empty
  GT layers as "task not yet evaluated", NOT as "model predicted nothing wrong".
- Protocol annotations (`pre_match`, `match_start`) are present in the raw seed but excluded from the
  62/122 benchmark counts; released samples may carry them flagged via `ambiguous`/note or omit them.

---

## 3. Prediction / submission schema

One JSON object per sample, mirroring the GT structure. Filename: `<sample_id>.json` under a
submission directory. A submission is a directory (or zip) of per-sample prediction files plus a
top-level `submission_meta.json`.

```jsonc
// submission_meta.json
{
  "schema_version": "movebench-submission-0.1.0",
  "ontology_version": "0.1.0",
  "submission_name": "baseline-measurement-reasoning",   // free text, no PII
  "model_descriptor": "long-context commercial VLM + CV anchors",  // system-level, no product names
  "model_version_locked": "2026-05-30",                  // date models/checkpoints were frozen
  "tasks": ["T1", "T3"],                                 // which tasks this submission targets
  "regime": "end_to_end",                                // oracle_segment | end_to_end | system
  "system_metrics": {                                    // required when regime == "system"
    "cost_usd_total": 0.013, "cost_usd_per_min": 0.0008,
    "latency_sec_total": 92.5, "latency_sec_per_min": 5.8,
    "est_or_billed": "estimated"                          // estimated | billed
  }
}
```

```jsonc
// <sample_id>.json (one prediction file)
{
  "sample_id": "mb_v0_0001",
  "ontology_version": "0.1.0",
  "segments": [                          // T1 + T3 prediction
    {
      "start_sec": 8.0,
      "end_sec": 50.0,
      "position": "standing",            // L1A id (normalized on scoring)
      "sub_position": "clinch",          // nullable
      "roles": {"A1": "neutral", "A2": "neutral"},   // nullable; scored for T4
      "labels": ["standing", "grip_fighting"],       // segment/subsegment labels, scored for T3
      "coaching": "optional short note"              // scored for T5
    }
  ],
  "transitions": [                       // T2 prediction (scored only when GT layer is non-empty)
    {"time_sec": 50.0, "type": "takedown", "actor": "A2"}
  ],
  "scoring": {                           // scoring-task prediction (scored only when GT non-empty)
    "ruleset": "adcc",
    "events": [{"time_sec": 50.0, "athlete": "A2", "kind": "takedown", "points": 2}]
  }
}
```

### Submission rules
- A prediction file MUST exist for every `sample_id` in the targeted split. Missing files are
  scored as **all false-negatives for that sample** (never silently skipped).
- Predicted `segments` need not align to GT boundaries; scoring matches greedily by position + IoU
  (see `score_p1_eval.py` adaptation in README §metrics).
- Unknown / out-of-ontology labels are normalized when possible (via aliases) and otherwise counted
  as an **ontology violation** (a reported sub-metric), then dropped from F1 matching.
- For `regime: "oracle_segment"`, the submission reuses GT `macro_segments` boundaries and only
  supplies `position`/`labels` — this isolates semantic labeling from temporal localization.

---

## 4. Compatibility with the existing scorer

The v0 reference scorer is adapted from `score_p1_eval.py` (the P1 harness). Its core matching
logic is reused unchanged:

- `interval_iou(a_start, a_end, b_start, b_end)` — symmetric temporal IoU.
- `score_segments(gt, pred, iou_threshold=0.3)` — greedy position+IoU matching → per-position
  TP/FP/FN → macro-F1 + mean matched IoU.
- `score_transitions(gt, pred, tolerance_sec=5.0)` — timestamp-tolerant event F1.
- `normalize_position` / `normalize_transition` — alias folding (extended to read `ontology.json`).
  This alias-folding is the **first (syntactic-normalization) step of semantic matching**, not a
  string comparison; the semantic standard extends it to full ontology-normalized equivalence classes
  and to expert / LLM-as-judge adjudication of free-text. The current `metrics.py` implements the
  alias-folding floor; full semantic free-text adjudication is the design standard (pilot/planned).

Differences in MoveBench v0:
- Field names are `start_sec`/`end_sec`/`position` (same as the scorer expects).
- The scorer's hard-coded `POSITIONS` set is **replaced** by `ontology.json → L1A_macro_positions`
  (plus the v0 reduced field values), so macro-F1 is computed over the real label space.
- T3 label F1 (set-level over `labels`) is added alongside T1 segment F1.
- The transition scorer is wired but **inert in v0** (empty GT) — it reports `task_status:
  "no_v0_ground_truth"` rather than a misleading `f1: 0.0` or `1.0`.

### Integrity guard (mandatory)
The scorer MUST refuse to emit a headline accuracy when GT == predictions by construction. Any
sample whose `annotation.label_source == "prediction_review"` and whose GT segments are
byte-identical to the submitted predictions is flagged `degenerate: true` and **excluded** from
aggregate macro-F1 / IoU / transition-F1. (This is why the P1 50-video slice's 1.0 scores are
tautological and must never be reported as accuracy.) The only honest machine-verified baseline
signal in v0 is the stored multi-pass evaluation: recall 0.688 / precision 0.667 / F1 0.677 /
position-accuracy 0.273 / mean-IoU 0.703 over the 32 macro positions of the single match.

These figures (and the ~30% segment acceptance cited elsewhere) are a **strict-syntactic floor / lower
bound**, computed under exact-class matching — **not** a semantic correctness rate. Because VLM output
is non-deterministic in surface form, the **semantic** score (ontology-normalized equivalence classes
plus abstain-aware semantic adjudication) is the **intended standard** and is **expected to be higher**;
it is **not yet rigorously measured**, so no semantic number is asserted. Report **two regimes**
(strict-syntactic floor vs semantic standard); the gap between them is itself a result.
