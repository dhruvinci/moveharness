# MoveBench v0 — Our Label Dataset Characterization

Generated: 2026-05-30. All numbers below are computed deterministically (SQL aggregation over
our label dataset SQLite + ontology string/semantic mapping). No model inference was run.
Counts cross-checked against `shared/evidence-ledger.md` (section 6).

Our label dataset is a self-labeled technique dataset: an expert-curated corpus of instructional and
competition grappling video, segmented into timestamped clips, each carrying a free-text expert
description and a set of free-form technique/position hashtags.

---

## 1. Corpus totals

| Metric | Value |
|---|---|
| Videos | **430** |
| Distinct channels / sources | **189** |
| Timestamped clips | **5,604** |
| Clips with a text label | **5,604** (100%) |
| Video-level hashtag rows | **28,503** |
| Clip-level hashtag rows | **68,925** |
| Distinct hashtags (shared vocab) | **1,653** |

Every clip has a free-text expert description (coaching prose): length **17–1,801 chars, mean 286**.
Video-level and clip-level hashtags draw from the same 1,653-tag vocabulary.

---

## 2. Unique technique/position tags

The corpus uses **1,653 distinct hashtags** across both video- and clip-level annotations. This is the
free-form expert tag vocabulary, roughly **18x larger** than the v0 ontology's ~91-term technique set.
See section 5 for how it maps onto the ontology.

---

## 3. Top clip (timestamp) tags

This was originally a top-30 list; **two athlete-name (person-name) tags were
removed for privacy**, so it now shows the **top 28** technique/position/metadata
tags. The remaining counts are unchanged.

| # | Tag | Count | # | Tag | Count |
|--:|---|--:|--:|---|--:|
| 1 | #nogi | 5,138 | 15 | #takedown | 354 |
| 2 | #footage | 4,809 | 16 | #kuzushi | 349 |
| 3 | #legentanglement | 934 | 17 | #guardretention | 341 |
| 4 | #standing | 849 | 18 | #leglockdefense | 335 |
| 5 | #resource | 798 | 19 | #singleleg | 325 |
| 6 | #adcc | 734 | 20 | #halfguardpass | 318 |
| 7 | #wno | 697 | 21 | #backtake | 298 |
| 8 | #destabilization | 656 | 22 | #turtleentry | 298 |
| 9 | #guardpass | 534 | 23 | #gripfight | 289 |
| 10 | #submission | 439 | 24 | #handfight | 289 |
| 11 | #wrestling | 432 | 25 | #back | 285 |
| 12 | #underhook | 426 | 26 | #sidecontrolentry | 281 |
| 13 | #dilemma | 388 | 27 | #outsidepassing | 280 |
| 14 | #openguard | 370 | 28 | #loosepassing | 276 |

The head is dominated by modality/source markers (`#nogi`, `#footage`, `#resource`) and event tags
(`#adcc`, `#wno`); the rest are technique/position concepts.

---

## 4. Distributions

### Tags per clip
min **4**, max **40**, mean **12.3**, median **11**. Unimodal, peak at 9–11 tags/clip, right tail to 40.

```
 4 |  30
 5 |  83
 6 | 177
 7 | 353
 8 | 507
 9 | 536
10 | 564   <- peak
11 | 563
12 | 505
13 | 417
14 | 382
15 | 304
16 | 257
17 | 233
18 | 153
19 | 110
20 | 114
21+|  ~390 (tail to 40)
```

### Clips per video
min **1**, max **139**, mean **13.0**, median **9**. Zero videos have no clips. Long right tail: most
videos have 1–15 clips; two extreme-length compilation videos hold 135 and 139 clips.

---

## 5. Ontology coverage

The v0 ontology (`benchmark/ontology.json`) defines **18 macro positions** (L1A) and a
**~91-term technique/action vocabulary** (guards, passes, sweeps, takedowns, submissions,
controls/grips, transitions, roles, lifecycle).

### 5a. Exact string match (strict, normalized)

| Measure | Value |
|---|---|
| Distinct tags exact-mapped to ontology | **69 / 1,653 (4.2%)** |
| Tag instances exact-mapped | **7,981 / 68,925 (11.6%)** |
| Of the 18 macro positions, attested by id/alias | **14 / 18** |
| Vocab norm-terms attested | **52** |

Exact match is low because the corpus uses a far richer free-form hashtag vocabulary than the
ontology (e.g. `#legentanglement`, `#guardretention`, `#sidecontrolentry`, `#kuzushi` have no exact
ontology id). This 11.6% is a coverage **floor**.

### 5b. Semantic categorization (concept-word containment — estimate)

Mapping each tag by concept-word containment and metadata class gives a realistic picture:

| Category | Distinct tags | Instances | % of instances |
|---|--:|--:|--:|
| Exact ontology match | 69 | 7,981 | 11.6% |
| Semantic technique/position | 845 | 28,906 | 41.9% |
| Metadata — event | 4 | 1,901 | 2.8% |
| Metadata — source/genre | 5 | 6,093 | 8.8% |
| Metadata — modality/sport | 3 | 5,612 | 8.1% |
| Metadata — athlete | 20 | 2,446 | 3.5% |
| Other free-form concepts | 707 | 15,986 | 23.2% |

- **Technique/position-relevant (exact + semantic): 914 distinct tags, 36,887 instances (53.5%).**
- **Metadata (event + source + modality + athlete): 32 distinct tags, 16,052 instances (23.3%).**

Interpretation: the 18 macro positions are well-attested (14/18 by exact id/alias; the remainder
appear via free-form synonyms). Roughly **53.5% of tag instances are technique/position-relevant**
and map conceptually onto the ontology families. The remaining ~23% are explicit metadata and ~23%
are finer-grained free-form concepts not yet in the v0 vocabulary. **Ontology expansion — folding the
high-frequency free-form technique tags into the canonical vocabulary — is the main data-curation lever.**

---

## 6. Proposed splits

### 6a. Athlete/event-disjoint split (channel-disjoint proxy)

`channel` (the source instructor/athlete/event) is used as the disjointness key so that no
same-athlete or same-event footage leaks across train/val/test. Greedy, clip-balanced assignment of
all **189** channels:

| Split | Channels | Videos | Clips |
|---|--:|--:|--:|
| train | 76 | 259 (60.2%) | 3,923 (70.0%) |
| val | 57 | 88 (20.5%) | 841 (15.0%) |
| test | 56 | 83 (19.3%) | 840 (15.0%) |

Disjointness verified: all 189 channels assigned to exactly one split. Clip counts balanced to a
70/15/15 target; the video split skews slightly because clips-per-video varies by channel.

### 6b. Leave-one-video-out (LOVO)

- **Corpus LOVO**: 430 folds — train on 429 videos, test on the held-out 1, repeat 430x.
  Channel-aware LOVO additionally drops all same-channel videos from the train fold to avoid leakage.
- **v0 LOVO** is degenerate: v0 is a single annotated public match, so LOVO collapses to one held-out
  fold. Reported here as the protocol to apply once more matches are annotated.

---

## 7. v0 seed (single expert-annotated public match)

Public v0 seed = one expert-annotated public no-gi competition match (`benchmark/data/v0/mb_v0_0001.json`).

| Metric | Value |
|---|---|
| Macro segments | **32** |
| Subsegments (raw) | **64** |
| Subsegments (benchmark, excl. 2 protocol) | **62** |
| Label instances (raw) | **124** |
| Label instances (benchmark) | **122** |
| Unique labels (raw) | **44** |
| Unique technique labels | **42** |
| Protocol annotations excluded | `pre_match`, `match_start` |
| Avg subsegments / position | 1.94 |
| Avg labels / subsegment | 1.97 |
| Annotators | 1 (expert_reviewer); inter-annotator agreement: none |

v0 macro-position field uses a reduced 7-value working set: `standing, guard, sweep, back_control,
escape, mount, side_control`.

---

## 8. Production candidate pool (system-scale, private)

The system-scale corpus the seed and candidate clips were triaged from (private, not release-cleared):

| Metric | Value |
|---|---|
| Videos | **313** |
| Completed analyses | **269** |
| Round types | training 228 / competition 84 / instructional 1 |
| Sport | grappling 313 (100%) |
| Eligible triage subset | **213** (training 158 / competition 54 / instructional 1) |
| Eligible subset duration (median) | 206 s |
| Predictions written | 50 |
| Expert ground-truth files | 4 of 50 |
| Timeline segments / analysis | median 17, mean 19.96, aggregate 5,369 |

---

## 9. Caveats

- Exact ontology-to-tag mapping is intentionally strict (normalized string match); the **11.6%**
  instance coverage is a floor. The **~53.5%** semantic coverage is an **estimate** from concept-word
  containment and is labeled as such.
- `channel` is an athlete/event-disjoint **proxy**, not a per-athlete identifier (one channel may
  feature several athletes; one athlete may appear across channels). The split is disjoint by channel,
  the strongest available leakage control without per-athlete labels.
- Median tags/clip (11) and clips/video (9) computed by ordered offset; means are exact.
- v0 LOVO is degenerate (single annotated match); reported as a protocol for the 430-video corpus.
- The 50-video production prediction slice is tautological (GT == prediction) and is excluded from
  any accuracy claim (evidence-ledger 9.6).

---

*Sources: our label dataset SQLite (the project label-dataset `.db`); `benchmark/ontology.json`;
`benchmark/data/v0/mb_v0_0001.json`; `shared/evidence-ledger.md` sections 1, 2, 5, 6, 9.*
