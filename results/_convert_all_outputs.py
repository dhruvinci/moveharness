"""Convert EVERY available VLM-pipeline output in repos/vlm-experiments/outputs/
into MoveBench v0 prediction files (D1 metric-vector scoring).

This EXTENDS results/_convert_experiment_timeline.py to cover all available
machine outputs (naive single-pass, multi-pass Exp2, Exp3 stage1/stage3, Exp4
stage1 v1/v2/v3 runs, and pipeline composites), mapping each to the MoveBench
prediction schema.

PRIVACY (same contract as _convert_experiment_timeline.py): we read the
experiment artifacts ONLY to extract anonymous structural fields
(start_sec / end_sec / position / sub_position / technique tokens). We NEVER
copy athlete names, the video id, narratives, coaching free-text, or any URL
into the emitted MoveBench prediction. A privacy scan aborts on any leak.

INTEGRITY: emitted predictions are real machine outputs, NOT hand-edited toward
the GT. Position/sub_position/technique tokens are carried verbatim (lowercased,
token-cleaned). Normalization to the ontology happens ONLY inside the scorer, so
vocabulary mismatches stay VISIBLE in the score, not hidden by the converter.

Run from the benchmark/ directory:
    python3 results/_convert_all_outputs.py
"""

from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
_REPOS = os.path.normpath(os.path.join(_BENCH, os.pardir, "repos", "vlm-experiments"))
_OUTPUTS = os.path.join(_REPOS, "outputs")

sys.path.insert(0, _HERE)
from _privacy import scan as _privacy_scan_terms  # noqa: E402

OUT_DIR = os.path.join(_HERE, "predictions_all")

SAMPLE_ID = "mb_v0_0001"
ONTOLOGY_VERSION = "0.1.0"

# Athlete-name fragments that can appear *inside* snake_case technique tokens
# (e.g. "<name>_single_leg_attempt"). Any token containing one of these is a
# name-bearing free-form label and is DROPPED (not carried into the prediction).
# Loaded from the external (withheld) deny-list; no person names hard-coded here.
from _privacy import name_fragments as _name_fragments  # noqa: E402
_NAME_FRAGMENTS = _name_fragments()


def _parse_ts(ts) -> float:
    """Parse 'M:SS' / 'MM:SS' / 'H:MM:SS' or numeric seconds into float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    ts = str(ts).strip()
    if not ts:
        raise ValueError("empty timestamp")
    parts = [float(p) for p in ts.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unparseable timestamp: {ts!r}")


def _clean_labels(raw_labels) -> list[str]:
    """Keep only short snake_case technique/position tokens; drop free text.

    Identical policy to _convert_experiment_timeline._clean_labels: reject
    multiword / sentence-like strings (could carry names/free text); keep only
    [a-z0-9_]+ tokens up to 40 chars; de-dup preserving order.
    """
    out: list[str] = []
    if not raw_labels:
        return out
    if isinstance(raw_labels, str):
        raw_labels = re.split(r"[,;]", raw_labels)
    if not isinstance(raw_labels, (list, tuple)):
        return out
    for tok in raw_labels:
        if not isinstance(tok, str):
            continue
        t = tok.strip().lower()
        if not t or t in {"-", "n/a", "none", "null"}:
            continue
        if " " in t or len(t) > 40:
            continue
        # Drop any token carrying an athlete-name fragment (privacy).
        if any(frag in t for frag in _NAME_FRAGMENTS):
            continue
        if re.fullmatch(r"[a-z0-9_]+", t):
            out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _seg(start, end, position, *label_sources) -> dict | None:
    try:
        s = _parse_ts(start)
        e = _parse_ts(end)
    except (ValueError, TypeError):
        return None
    if e <= s:
        return None
    labels: list[str] = []
    for src in label_sources:
        labels.extend(_clean_labels(src))
    # de-dup preserving order
    seen = set()
    uniq = []
    for t in labels:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return {
        "start_sec": s,
        "end_sec": e,
        "position": (position or "").strip().lower(),
        "labels": uniq,
    }


def _wrap(system: str, provenance: str, segs: list[dict]) -> dict:
    segs = [x for x in segs if x is not None]
    return {
        "sample_id": SAMPLE_ID,
        "ontology_version": ONTOLOGY_VERSION,
        "system": system,
        "provenance": provenance,
        "segments": segs,
        "transitions": [],
        "scoring": {},
    }


# --------------------------------------------------------------------------- #
# Per-format converters.
# --------------------------------------------------------------------------- #
def conv_position_timeline(path: str, system: str, provenance: str,
                           label_keys=("sub_position", "labels")) -> dict:
    """Exp2 / naive 'position_timeline' or 'unified_timeline' shape."""
    d = json.load(open(path, "r", encoding="utf-8"))
    rows = d.get("position_timeline") or d.get("unified_timeline") or []
    segs = []
    for r in rows:
        srcs = [r.get(k) for k in label_keys]
        segs.append(_seg(r.get("start_time"), r.get("end_time"),
                         r.get("position"), *srcs))
    return _wrap(system, provenance, segs)


def conv_exp3_stage(path: str, system: str, provenance: str) -> dict:
    """Exp3 stage1 / stage3 'segments' shape (start/end + sub_position + key_actions)."""
    d = json.load(open(path, "r", encoding="utf-8"))
    segs = []
    for s in d.get("segments", []):
        segs.append(_seg(s.get("start"), s.get("end"), s.get("position"),
                         s.get("sub_position"), s.get("key_actions")))
    return _wrap(system, provenance, segs)


def conv_exp4_stage(path: str, system: str, provenance: str) -> dict:
    """Exp4 stage1 v1/v2/v3 'segments' shape: time='0:06-0:49', reasons/focus tokens."""
    d = json.load(open(path, "r", encoding="utf-8"))
    segs = []
    for s in d.get("segments", []):
        time = s.get("time", "")
        if "-" not in time:
            continue
        a, b = time.split("-", 1)
        segs.append(_seg(a, b, s.get("position"), s.get("reasons"), s.get("focus")))
    return _wrap(system, provenance, segs)


def _privacy_scan(obj) -> list[str]:
    # Deny-list = generic non-PII terms (incl. "rash guard") + externally-loaded
    # person-name terms (see _privacy.py). No person names are hard-coded here.
    return _privacy_scan_terms(obj, extra=("rash guard",))


# (output_filename, callable, source_path, system_id, provenance)
JOBS = [
    ("pred_naive_pass1.json",
     lambda p, s, pr: conv_position_timeline(p, s, pr,
                                             label_keys=("sub_position",)),
     os.path.join(_OUTPUTS, "pass1_holistic.json"),
     "naive_single_pass_holistic",
     "honest machine output; single-pass holistic VLM (no CV grounding); in-sample"),

    ("pred_exp2_pass1_enhanced.json",
     lambda p, s, pr: conv_position_timeline(p, s, pr,
                                             label_keys=("sub_position", "labels")),
     os.path.join(_OUTPUTS, "experiment2", "pass1_enhanced.json"),
     "exp2_multipass_pass1_enhanced",
     "honest machine output; Exp2 multi-pass, pass1 enhanced; in-sample"),

    ("pred_exp2_pass1_fixed.json",
     lambda p, s, pr: conv_position_timeline(p, s, pr,
                                             label_keys=("sub_position", "labels")),
     os.path.join(_OUTPUTS, "experiment2", "pass1_enhanced_fixed.json"),
     "exp2_multipass_pass1_fixed",
     "honest machine output; Exp2 multi-pass, pass1 fixed (241 micro-segs); in-sample"),

    ("pred_exp2_pass3_synthesis.json",
     lambda p, s, pr: conv_position_timeline(p, s, pr,
                                             label_keys=("sub_position",)),
     os.path.join(_OUTPUTS, "experiment2", "pass3_synthesis.json"),
     "exp2_multipass_pass3_synthesis",
     "honest machine output; Exp2 multi-pass, pass3 synthesized timeline; in-sample"),

    ("pred_exp3_stage1.json", conv_exp3_stage,
     os.path.join(_OUTPUTS, "experiment3", "stage1_timeline.json"),
     "exp3_stage1_cv_cached_timeline",
     "honest machine output; Exp3 stage1 CV-cached single-pass timeline; in-sample"),

    ("pred_exp3_stage1_v33_pass1.json", conv_exp3_stage,
     os.path.join(_OUTPUTS, "experiment3", "stage1_timeline_v3.3-pass1.json"),
     "exp3_stage1_v33_pass1",
     "honest machine output; Exp3 stage1 v3.3 pass1; in-sample"),

    ("pred_exp3_stage1_v33_compressed.json", conv_exp3_stage,
     os.path.join(_OUTPUTS, "experiment3", "stage1_timeline_v3.3-compressed2.json"),
     "exp3_stage1_v33_compressed",
     "honest machine output; Exp3 stage1 v3.3 compressed; in-sample"),

    ("pred_exp3_stage3_v33.json", conv_exp3_stage,
     os.path.join(_OUTPUTS, "experiment3", "stage3_timeline_v3.3.json"),
     "exp3_stage3_merged_timeline",
     "honest machine output; Exp3 stage3 merged/validated timeline; in-sample"),

    ("pred_exp4_v1_run1.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1_skeleton_competition_run1.json"),
     "exp4_v1_skeleton_run1",
     "honest machine output; Exp4 stage1 v1 skeleton, run1; in-sample"),

    ("pred_exp4_v1_run2.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1_skeleton_competition_run2.json"),
     "exp4_v1_skeleton_run2",
     "honest machine output; Exp4 stage1 v1 skeleton, run2; in-sample"),

    ("pred_exp4_v1_run3.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1_skeleton_competition_run3.json"),
     "exp4_v1_skeleton_run3",
     "honest machine output; Exp4 stage1 v1 skeleton, run3; in-sample"),

    ("pred_exp4_v2_run1.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1v2_skeleton_competition_run1.json"),
     "exp4_v2_skeleton_run1",
     "honest machine output; Exp4 stage1 v2 skeleton, run1; in-sample"),

    ("pred_exp4_v2_run2.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1v2_skeleton_competition_run2.json"),
     "exp4_v2_skeleton_run2",
     "honest machine output; Exp4 stage1 v2 skeleton, run2; in-sample"),

    ("pred_exp4_v3_run1.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1v3_skeleton_competition_run1.json"),
     "exp4_v3_skeleton_run1",
     "honest machine output; Exp4 stage1 v3 skeleton (CV-grounded), run1; in-sample"),

    ("pred_exp4_v3_run2.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "stage1v3_skeleton_competition_run2.json"),
     "exp4_v3_skeleton_run2",
     "honest machine output; Exp4 stage1 v3 skeleton (CV-grounded), run2; in-sample"),

    ("pred_exp4_combined.json", conv_exp4_stage,
     os.path.join(_OUTPUTS, "experiment4.0", "combined_Who's Number One.json"),
     "exp4_combined_stage1_stage2",
     "honest machine output; Exp4 combined stage1+stage2 composite; in-sample"),
]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for name, conv, src, system, prov in JOBS:
        if not os.path.exists(src):
            print(f"SKIP {name}: source not found ({src})", file=sys.stderr)
            continue
        pred = conv(src, system, prov)
        leaks = _privacy_scan(pred)
        if leaks:
            print(f"ABORT {name}: privacy leak {leaks}", file=sys.stderr)
            return 2
        out = os.path.join(OUT_DIR, name)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(pred, fh, indent=2)
        written.append((name, system, len(pred["segments"])))
        print(f"wrote {name}: system={system} segments={len(pred['segments'])}")
    print(f"\n{len(written)} prediction files written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
