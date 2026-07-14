"""MoveBench v0 metric vector.

LINEAGE / CITATION
------------------
The core matching logic in this module is ADAPTED from the P1 measurement-
reasoning scorer (`p1-measurement-reasoning/scripts/score_p1_eval.py`). The
following functions are reused with the same semantics:

  * `interval_iou`        -- symmetric temporal IoU of two [start, end] spans.
  * `score_segments`      -- greedy position + IoU matching -> per-position
                             TP/FP/FN -> macro-F1 + mean matched IoU.
  * `score_transitions`   -- timestamp-tolerant event F1 (actor-aware here).

MoveBench differences from the P1 scorer:
  * The hard-coded `POSITIONS` set is REPLACED by the ontology-driven label
    space (positions are passed in / normalized via `ontology.json`), so
    macro-F1 is computed over the real v0 label space rather than a fixed list.
  * A set-level, ontology-aware label F1 (`label_f1`, task T3) is added with an
    ontology-violation sub-metric.
  * `coverage` (task T5) is added: fraction of video duration spanned by valid
    non-overlapping segments. NOTE: coverage measures COMPLETENESS, not
    correctness -- it must always be reported next to T1/T3 accuracy, never as a
    stand-in for it.
  * Empty ground-truth layers (e.g. v0 transitions / scoring) are NOT scored as
    0.0 or 1.0; callers report `task_status: "no_v0_ground_truth"` instead.

This module has NO external dependencies (standard library only) and contacts
no external services.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Optional, Sequence


# --------------------------------------------------------------------------- #
# Temporal IoU (verbatim semantics from score_p1_eval.py).
# --------------------------------------------------------------------------- #
def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Symmetric temporal intersection-over-union of two [start, end] spans."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _identity(x: Optional[str]) -> str:
    return (x or "").strip().lower()


# --------------------------------------------------------------------------- #
# T1: macro-position segment scoring (adapted from score_p1_eval.score_segments)
# --------------------------------------------------------------------------- #
def score_segments(
    gt: Sequence[dict],
    pred: Sequence[dict],
    iou_threshold: float = 0.3,
    normalize_position: Callable[[Optional[str]], str] = _identity,
    label_space: Optional[Iterable[str]] = None,
    duration_weighted: bool = False,
) -> dict:
    """Greedy position + IoU matching -> per-position TP/FP/FN -> macro-F1.

    Adapted from `score_p1_eval.score_segments`. Each GT segment greedily claims
    the best unmatched same-position prediction whose IoU >= `iou_threshold`.

    Args:
        gt / pred: lists of dicts with `start_sec`, `end_sec`, `position`.
        iou_threshold: minimum IoU for a match to count (default 0.3).
        normalize_position: ontology alias-folding function (default lowercase).
        label_space: optional iterable of canonical positions to seed the label
            set (replaces the P1 hard-coded POSITIONS set; safe to leave None).
        duration_weighted: if True, also returns a duration-weighted macro-F1
            where each position's F1 is weighted by its GT duration share.

    Returns a dict with macro_f1, mean_matched_iou, per-position breakdown, and
    (when duration_weighted) duration_weighted_macro_f1.
    """
    matched_pred: set[int] = set()
    true_pos: Counter = Counter()
    false_neg: Counter = Counter()
    false_pos: Counter = Counter()
    ious: list[float] = []
    gt_duration_by_pos: Counter = Counter()

    for g in gt:
        gpos = normalize_position(g.get("position"))
        gt_duration_by_pos[gpos] += max(0.0, float(g["end_sec"]) - float(g["start_sec"]))
        best = None
        best_iou = 0.0
        for idx, p in enumerate(pred):
            if idx in matched_pred:
                continue
            if normalize_position(p.get("position")) != gpos:
                continue
            iou = interval_iou(
                float(g["start_sec"]), float(g["end_sec"]),
                float(p["start_sec"]), float(p["end_sec"]),
            )
            if iou > best_iou:
                best_iou = iou
                best = idx
        if best is not None and best_iou >= iou_threshold:
            matched_pred.add(best)
            true_pos[gpos] += 1
            ious.append(best_iou)
        else:
            false_neg[gpos] += 1

    for idx, p in enumerate(pred):
        if idx not in matched_pred:
            false_pos[normalize_position(p.get("position"))] += 1

    seed = set(label_space or [])
    labels = sorted(set(true_pos) | set(false_pos) | set(false_neg) | seed)
    per_label: dict[str, dict] = {}
    f1s: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    total_gt_duration = sum(gt_duration_by_pos.values())

    for label in labels:
        tp = true_pos[label]
        fp = false_pos[label]
        fn = false_neg[label]
        if tp == fp == fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
        f1s.append(f1)
        w = gt_duration_by_pos.get(label, 0.0)
        weighted_num += f1 * w
        weighted_den += w

    out = {
        "gt_segments": len(gt),
        "pred_segments": len(pred),
        "matched_segments": sum(true_pos.values()),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0,
        "per_label": per_label,
    }
    if duration_weighted:
        out["duration_weighted_macro_f1"] = (
            weighted_num / weighted_den if weighted_den else 0.0
        )
        out["total_gt_duration_sec"] = total_gt_duration
    return out


# --------------------------------------------------------------------------- #
# T2: transition event scoring (adapted from score_p1_eval.score_transitions)
# --------------------------------------------------------------------------- #
def score_transitions(
    gt: Sequence[dict],
    pred: Sequence[dict],
    tolerance_sec: float = 5.0,
    normalize_transition: Callable[[Optional[str]], str] = _identity,
    actor_aware: bool = True,
) -> dict:
    """Timestamp-tolerant transition-event F1.

    Adapted from `score_p1_eval.score_transitions`, extended to optionally
    require actor agreement (A1/A2). A GT event matches the nearest unmatched
    prediction of the same type (and actor, if `actor_aware`) within
    `tolerance_sec`.

    INTEGRITY: this function is INERT in MoveBench v0 because the seed's
    transition ground-truth layer is empty. Callers MUST check whether GT is
    empty and report `task_status: "no_v0_ground_truth"` rather than calling
    this and surfacing the resulting 0.0 F1 as a real result.
    """
    matched_pred: set[int] = set()
    actor_mismatches = 0
    tp = 0
    for g in gt:
        gtype = normalize_transition(g.get("type"))
        gtime = float(g.get("time_sec", 0))
        gactor = g.get("actor")
        best = None
        best_delta = tolerance_sec + 1
        for idx, p in enumerate(pred):
            if idx in matched_pred:
                continue
            if normalize_transition(p.get("type")) != gtype:
                continue
            delta = abs(float(p.get("time_sec", 0)) - gtime)
            if delta <= tolerance_sec and delta < best_delta:
                best = idx
                best_delta = delta
        if best is not None:
            matched_pred.add(best)
            tp += 1
            if actor_aware and gactor is not None and pred[best].get("actor") != gactor:
                actor_mismatches += 1
    fp = len(pred) - len(matched_pred)
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt_transitions": len(gt),
        "pred_transitions": len(pred),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "actor_mismatches": actor_mismatches,
        "false_transition_rate": fp / len(pred) if pred else 0.0,
    }


# --------------------------------------------------------------------------- #
# T3: ontology-aware label F1 (new; set-level over per-segment `labels`).
# --------------------------------------------------------------------------- #
def label_f1(
    gt: Sequence[dict],
    pred: Sequence[dict],
    normalize_label: Callable[[Optional[str]], str] = _identity,
    is_in_ontology: Optional[Callable[[str], bool]] = None,
) -> dict:
    """Micro/macro F1 over per-segment `labels` sets (oracle-segment regime).

    `gt` and `pred` are assumed to be ALIGNED segment lists (same order / GT
    boundaries) — i.e. the oracle-segment regime that isolates semantic labeling
    from temporal localization. Each segment exposes a `labels` list.

    Reports:
      * micro_f1   -- pooled TP/FP/FN over all label instances.
      * macro_f1   -- mean per-label F1 over the union of GT+pred labels.
      * ontology_violation_rate -- fraction of predicted labels that normalize
        to something outside the ontology (counted, then dropped from matching).
    """
    n = min(len(gt), len(pred))
    micro_tp = micro_fp = micro_fn = 0
    per_label_tp: Counter = Counter()
    per_label_fp: Counter = Counter()
    per_label_fn: Counter = Counter()
    pred_label_total = 0
    violations = 0

    for i in range(n):
        gset = {normalize_label(x) for x in (gt[i].get("labels") or [])}
        gset.discard("")
        praw = [normalize_label(x) for x in (pred[i].get("labels") or [])]
        pset = set()
        for lbl in praw:
            if not lbl:
                continue
            pred_label_total += 1
            if is_in_ontology is not None and not is_in_ontology(lbl):
                violations += 1
                continue  # out-of-ontology labels are dropped from matching
            pset.add(lbl)

        tp = gset & pset
        fp = pset - gset
        fn = gset - pset
        micro_tp += len(tp)
        micro_fp += len(fp)
        micro_fn += len(fn)
        for lbl in tp:
            per_label_tp[lbl] += 1
        for lbl in fp:
            per_label_fp[lbl] += 1
        for lbl in fn:
            per_label_fn[lbl] += 1

    def _f1(tp_: int, fp_: int, fn_: int) -> float:
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    micro = _f1(micro_tp, micro_fp, micro_fn)
    all_labels = set(per_label_tp) | set(per_label_fp) | set(per_label_fn)
    macro_vals = [
        _f1(per_label_tp[l], per_label_fp[l], per_label_fn[l]) for l in all_labels
    ]
    macro = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0

    return {
        "aligned_segments": n,
        "micro_f1": micro,
        "macro_f1": macro,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "unique_labels": len(all_labels),
        "ontology_violation_rate": violations / pred_label_total if pred_label_total else 0.0,
        "ontology_violations": violations,
    }


# --------------------------------------------------------------------------- #
# T5: coverage (new). COMPLETENESS, NOT CORRECTNESS.
# --------------------------------------------------------------------------- #
def coverage(
    segments: Sequence[dict],
    duration_sec: float,
    clip_to_duration: bool = True,
) -> dict:
    """Fraction of `duration_sec` spanned by valid non-overlapping segments.

    WARNING (load-bearing): coverage measures whether the timeline is COMPLETE,
    not whether it is ACCURATE. A model can reach ~100% coverage while getting
    positions badly wrong (the development trace hit ~100% coverage with 241
    segments — 7.5x the ground truth — at low position accuracy). Always report
    coverage NEXT TO T1/T3 accuracy, never as a substitute.

    Overlapping intervals are merged so coverage cannot exceed 1.0 by stacking.
    """
    spans: list[tuple[float, float]] = []
    for s in segments:
        a = float(s["start_sec"])
        b = float(s["end_sec"])
        if clip_to_duration:
            a = max(0.0, min(a, duration_sec))
            b = max(0.0, min(b, duration_sec))
        if b > a:
            spans.append((a, b))
    spans.sort()

    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    covered = sum(b - a for a, b in merged)
    frac = covered / duration_sec if duration_sec > 0 else 0.0
    return {
        "duration_sec": duration_sec,
        "covered_sec": covered,
        "coverage_fraction": min(1.0, frac),
        "n_segments": len(segments),
        "n_valid_spans": len(merged),
        "caveat": "completeness_not_correctness",
    }
