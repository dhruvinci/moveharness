"""MoveBench v0 reference tool.

A small, self-contained, import-clean Python package implementing the MoveBench
v0 benchmark described in the CAISc movement-benchmark paper. It loads the v0
single-match seed ground truth into the MoveBench schema, enumerates external
candidate-clip pools by aggregate count only (never redistributing raw data),
and scores predictions with a metric VECTOR (macro-F1 over positions, temporal
IoU, ontology-aware label F1, transition-F1, coverage) plus mandatory integrity
guards (no degenerate baselines; empty-GT layers report `no_v0_ground_truth`,
never a misleading 0/1).

Privacy invariant: no athlete names, video URLs/IDs, gym names, user IDs,
emails, or absolute/local paths appear in any released artifact. Athletes are
referred to as A1/A2; the seed match is referenced by an opaque sample_id.
"""

from .metrics import (
    interval_iou,
    score_segments,
    score_transitions,
    coverage,
    label_f1,
)
from .loader import (
    load_sample,
    load_prediction,
    enumerate_label_dataset_candidates,
    Ontology,
)

__version__ = "0.1.0"
ONTOLOGY_VERSION = "0.1.0"
SCHEMA_SAMPLE_VERSION = "movebench-sample-0.1.0"
SCHEMA_SUBMISSION_VERSION = "movebench-submission-0.1.0"

__all__ = [
    "interval_iou",
    "score_segments",
    "score_transitions",
    "coverage",
    "label_f1",
    "load_sample",
    "load_prediction",
    "enumerate_label_dataset_candidates",
    "Ontology",
    "__version__",
    "ONTOLOGY_VERSION",
]
