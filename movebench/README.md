# `movebench` — MoveBench v0 reference tool

A small, self-contained, standard-library-only Python package implementing the
MoveBench v0 benchmark (the CAISc movement-benchmark paper). It loads the v0
single-match seed ground truth into the MoveBench schema, enumerates external
candidate-clip pools **by aggregate count only** (no raw redistribution), and
scores predictions with a **metric vector** plus mandatory integrity guards.

See `../README.md`, `../SCHEMA.md`, and `../ontology.json` for the full spec.

## Layout

| File | Role |
|------|------|
| `__init__.py` | package surface (`load_sample`, `Ontology`, `score_segments`, `coverage`, `label_f1`, `score_transitions`, `enumerate_label_dataset_candidates`) |
| `loader.py` | loads v0 samples / predictions; `Ontology` alias-folding from `ontology.json`; counts-only label-dataset enumerator |
| `metrics.py` | metric vector — adapted from `p1-measurement-reasoning/scripts/score_p1_eval.py` (lineage cited in-file): `interval_iou`, `score_segments`, `score_transitions`, plus new `label_f1` (T3) and `coverage` (T5) |
| `evaluate.py` | CLI: `python -m movebench.evaluate --gt <gt.json> --pred <pred.json>` |
| `requirements.txt` | requirements note (stdlib only; Python ≥ 3.9) |
| `_convert_seed.py` | build-time tool: raw GT → anonymized `data/v0/mb_v0_0001.json` (strips names/url/id; not part of the runtime surface) |
| `_smoketest.py` | non-degeneracy smoke test (identity vs perturbed prediction) |

## Requirements

Python ≥ 3.9, **standard library only**. No `pip install`, no external services.

## Run the evaluator

```bash
cd benchmark
python -m movebench.evaluate \
    --gt   data/v0/mb_v0_0001.json \
    --pred data/v0/examples/pred_oracle_identity.json
```

Optional flags: `--ontology <path>` (default: bundled `ontology.json`),
`--iou-threshold 0.3`, `--transition-tolerance-sec 5.0`, `--output report.json`.

The CLI prints a **metric vector** (never one conflated scalar):

- **T1** macro-position segment F1 — segment- and duration-weighted — + temporal IoU
- **T3** ontology-aware label F1 (oracle-segment) + ontology-violation rate
- **T2** transition-F1 → reports `no_v0_ground_truth` (the v0 transition layer is empty)
- **T5** coverage — *completeness, NOT correctness*

## Integrity guards (mandatory)

1. **No degenerate baselines.** If a sample's `annotation.label_source ==
   "prediction_review"` and the GT segments are byte-identical to the
   prediction, the run is flagged `degenerate: true` and headline accuracy is
   suppressed to `null`. (This is exactly why the P1 50-video 1.0 is tautological
   and must never be reported.)
2. **Empty GT layers are not 0/1.** Empty v0 layers (transitions, scoring) report
   `task_status: "no_v0_ground_truth"`, never a misleading F1.

The only honest, machine-verified v0 baseline signal is the stored multi-pass
evaluation over the 32 macro positions of the seed: recall 0.688 / precision
0.667 / F1 0.677 / position-accuracy 0.273 / mean-IoU 0.703 (good span IoU, weak
position accuracy).

## Verify the scorer is not degenerate

```bash
python -m movebench._smoketest
```

Constructs an identity prediction (scores 1.0, **not** flagged degenerate because
the seed is `human_review`) and a perturbed prediction (wrong positions, shifted
spans, dropped + hallucinated labels). The perturbed run scores strictly lower on
T1 macro-F1, IoU, and T3 label-F1, and shows a nonzero ontology-violation rate —
proving the metrics actually respond to errors.

## Candidate-pool enumeration (counts only)

```python
from movebench.loader import enumerate_label_dataset_candidates
enumerate_label_dataset_candidates("<LABEL_DATASET_DB>")  # or os.environ["LABEL_DATASET_DB"]
# -> {n_videos: 430, n_described_clips: 5604, n_distinct_technique_tags: 1653, ...}
```

Returns **only aggregate integer counts**; it never reads or returns raw video
ids, urls, titles, channels, or clip text. The corpus is an external resource,
not bundled and not redistributed.
```
