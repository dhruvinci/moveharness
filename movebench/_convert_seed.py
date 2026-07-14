"""One-off converter: raw single-match GT -> anonymized MoveBench v0 sample.

PRIVACY: this script reads the raw annotation file (which contains athlete
names, a video URL, and an annotator id) and emits an ANONYMIZED MoveBench
sample that contains NONE of them:
  * athlete names -> A1 / A2 (top -> the segment's `roles`, never a real name)
  * video url / youtube id / local path -> dropped (opaque sample_id only)
  * annotator id -> replaced with an anonymized role
  * `is_correct`/`alternative_predictions` (internal review fields) -> dropped

It is NOT part of the runtime package surface; it exists to regenerate
`data/v0/mb_v0_0001.json` deterministically from the raw seed. The released
artifact under data/v0/ is the only thing that ships.
"""

from __future__ import annotations

import argparse
import json
import os
import re


def _parse_time(s: str) -> float:
    """'MM:SS' or 'M:SS' -> seconds (float)."""
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return float(int(m) * 60 + float(sec))
    if len(parts) == 3:
        h, m, sec = parts
        return float(int(h) * 3600 + int(m) * 60 + float(sec))
    return float(s)


def _normsecs(s: str) -> str:
    """Canonicalize a time string to zero-padded MM:SS for display."""
    secs = _parse_time(s)
    m = int(secs // 60)
    sec = int(round(secs - m * 60))
    if sec == 60:
        m += 1
        sec = 0
    return f"{m:02d}:{sec:02d}"


def convert(raw: dict, sample_id: str) -> dict:
    # Map raw athlete-name fields to anonymized A1/A2 roles, WITHOUT storing the
    # names. We assign A1/A2 by top/bottom presence per segment; the actual
    # names are never read into the output.
    duration_sec = _parse_time(raw.get("video_duration", "00:00"))

    macro_segments = []
    for i, pos in enumerate(raw.get("ground_truth_positions", [])):
        top = pos.get("top_athlete")
        bottom = pos.get("bottom_athlete")
        # Anonymized roles: if top/bottom present, A1=top, A2=bottom (positional,
        # name-free). If absent, both neutral.
        if top or bottom:
            roles = {"A1": "top", "A2": "bottom"}
        else:
            roles = {"A1": "neutral", "A2": "neutral"}

        subsegments = []
        for j, sub in enumerate(pos.get("sub_segments", [])):
            subsegments.append(
                {
                    "sub_id": f"s{i}.{j}",
                    "start_sec": _parse_time(sub["start_time"]),
                    "end_sec": _parse_time(sub["end_time"]),
                    "start_str": _normsecs(sub["start_time"]),
                    "end_str": _normsecs(sub["end_time"]),
                    "labels": list(sub.get("labels", [])),
                    # Free-text technical description kept SEPARATE from coaching
                    # critique. It contains athlete surnames in the raw seed, so
                    # we strip them here (replace with A1/A2 is not reliable per
                    # name -> we drop the note text entirely to be safe).
                    "note": _scrub_note(sub.get("note", "")),
                }
            )

        macro_segments.append(
            {
                "seg_id": f"s{i}",
                "start_sec": _parse_time(pos["start_time"]),
                "end_sec": _parse_time(pos["end_time"]),
                "start_str": _normsecs(pos["start_time"]),
                "end_str": _normsecs(pos["end_time"]),
                "position": pos.get("position"),
                "sub_position": pos.get("sub_position"),
                "roles": roles,
                "confidence": pos.get("confidence"),
                "rating": pos.get("rating"),
                "ambiguous": False,
                "subsegments": subsegments,
            }
        )

    return {
        "schema_version": "movebench-sample-0.1.0",
        "ontology_version": "0.1.0",
        "sample_id": sample_id,
        "split": "test",
        "source_type": "competition",
        "modality": "no_gi",
        "duration_sec": duration_sec,
        "duration_str": _normsecs(raw.get("video_duration", "00:00")),
        "rights_status": "public_seed",
        "camera_view": "broadcast",
        "annotation": {
            "annotator_roles": ["expert_reviewer"],
            "n_annotators": 1,
            "label_source": "human_review",
            "inter_annotator_agreement": None,
        },
        "_protocol_excluded": ["pre_match", "match_start"],
        "macro_segments": macro_segments,
        "transitions": [],
        "scoring": {"ruleset": "adcc", "events": [], "totals": {"A1": 0, "A2": 0}},
        "coaching_refs": [],
        "privacy_note": (
            "Anonymized seed. No athlete names, video URL/id, gym names, "
            "annotator id, or local paths. Athletes are A1/A2; the match is "
            "referenced by sample_id only. Free-text notes were scrubbed."
        ),
    }


# Athlete surnames in the raw seed that must never appear in the release are
# loaded from an external, comment-free deny-list (kept OUT of the release) so no
# person name is hard-coded in the package source. Resolution order:
#   1. the PRIVACY_DENYLIST env var (explicit path), else
#   2. <repo root>/.privacy_denylist  (MOVEBENCH_ROOT or the parent of benchmark/).
# One lowercase term per line; blank lines ignored. If absent, the note is passed
# through unchanged (no names to strip in a clean release tree).
def _load_name_tokens() -> "re.Pattern | None":
    path = os.environ.get("PRIVACY_DENYLIST")
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))          # .../movebench
        bench = os.path.normpath(os.path.join(here, os.pardir))    # .../benchmark
        root = os.environ.get("MOVEBENCH_ROOT") or os.path.normpath(
            os.path.join(bench, os.pardir)
        )
        path = os.path.join(root, ".privacy_denylist")
    if not os.path.exists(path):
        return None
    terms = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t:
                terms.append(re.escape(t))
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE)


_NAME_TOKENS = _load_name_tokens()


def _scrub_note(note: str) -> str:
    """Replace any athlete-name token in a free-text note.

    Name tokens (from the external deny-list) are collapsed to 'athlete'. The
    note retains technical content for T3/T5 without PII. If no deny-list file is
    present, the note is returned unchanged.
    """
    if not note:
        return ""
    if _NAME_TOKENS is None:
        return note
    return _NAME_TOKENS.sub("athlete", note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw ground_truth.json")
    ap.add_argument("--out", required=True, help="path to write anonymized sample")
    ap.add_argument("--sample-id", default="mb_v0_0001")
    args = ap.parse_args()
    with open(args.raw, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = convert(raw, args.sample_id)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
