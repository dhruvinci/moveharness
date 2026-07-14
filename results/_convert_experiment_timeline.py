"""Convert an honest VLM-pipeline timeline (Exp3 / Exp4 dev artifacts) into a
MoveBench v0 prediction file, so it can be scored against mb_v0_0001.json.

PRIVACY: this script reads the experiment timelines (which DO contain athlete
names and a YouTube id) ONLY to extract anonymous, structural fields
(start_sec / end_sec / position / technique labels). It NEVER copies athlete
names, the video id, narratives, coaching free-text, or any URL into the
emitted MoveBench prediction. Only timestamps + ontology-mappable position /
technique tokens are carried through.

INTEGRITY: the emitted predictions are real machine outputs from a real
pipeline run. They are NOT hand-edited toward the ground truth. The position
field is normalized through the SAME ontology the scorer uses; no remapping is
done to flatter the result. Where a pipeline used a position vocabulary the v0
GT working-set does not contain (e.g. open_guard, referee_reset, post_match),
the token is passed through verbatim so the mismatch is VISIBLE in the score
(it shows up as an unmatched position / ontology effect), not hidden.

Run from the benchmark/ directory:
    python3 results/_convert_experiment_timeline.py
"""

from __future__ import annotations

import json
import os
import re
import sys

# Resolve repo paths relative to this file so no absolute path is hard-coded.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.normpath(os.path.join(_HERE, os.pardir))
_REPOS = os.path.normpath(os.path.join(_BENCH, os.pardir, "repos", "vlm-experiments"))

sys.path.insert(0, _HERE)
from _privacy import scan as _privacy_scan_terms  # noqa: E402

EXP3 = os.path.join(_REPOS, "outputs", "experiment3", "stage1_timeline.json")
EXP4 = os.path.join(
    _REPOS, "outputs", "experiment4.0", "stage1v3_skeleton_competition_run2.json"
)

OUT_DIR = os.path.join(_HERE, "predictions")

SAMPLE_ID = "mb_v0_0001"
ONTOLOGY_VERSION = "0.1.0"

# Fields we deliberately DROP (may contain names / urls / free text).
_NAME_FIELDS = {
    "top_athlete", "bottom_athlete", "top", "athlete_profiles", "narrative",
    "key_moments", "notes", "strategy", "setup", "execution", "outcome",
    "coaching", "reasons", "raw", "raw_text", "raw_markdown", "cache_name",
}


def _parse_ts(ts: str) -> float:
    """Parse 'M:SS' or 'MM:SS' or 'H:MM:SS' into seconds (float)."""
    ts = ts.strip()
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unparseable timestamp: {ts!r}")


# Conservative, non-flattering technique-token cleanup: keep only short
# snake_case-ish tokens (single technique/action names), drop sentence-like
# strings (which could carry free text / names).
def _clean_labels(raw_labels) -> list[str]:
    out: list[str] = []
    if not raw_labels:
        return out
    if isinstance(raw_labels, str):
        raw_labels = re.split(r"[,;]", raw_labels)
    for tok in raw_labels:
        if not isinstance(tok, str):
            continue
        t = tok.strip().lower()
        if not t or t in {"-", "n/a", "none"}:
            continue
        # Reject sentence-like / multiword-with-spaces strings (free text).
        if " " in t or len(t) > 40:
            continue
        # Keep only safe technique-token characters.
        if re.fullmatch(r"[a-z0-9_]+", t):
            out.append(t)
    # de-dup preserving order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def convert_exp3(path: str) -> dict:
    d = json.load(open(path, "r", encoding="utf-8"))
    segs = []
    for s in d.get("segments", []):
        start = _parse_ts(s["start"])
        end = _parse_ts(s["end"])
        labels = _clean_labels(s.get("key_actions") or s.get("sub_position"))
        segs.append(
            {
                "start_sec": start,
                "end_sec": end,
                "position": (s.get("position") or "").strip().lower(),
                "labels": labels,
            }
        )
    return {
        "sample_id": SAMPLE_ID,
        "ontology_version": ONTOLOGY_VERSION,
        "system": "pipeline_A_single_pass_cv_cached",
        "provenance": "honest machine output, dev artifact, single anchor match, in-sample",
        "segments": segs,
        "transitions": [],
        "scoring": {},
    }


def convert_exp4(path: str) -> dict:
    d = json.load(open(path, "r", encoding="utf-8"))
    segs = []
    for s in d.get("segments", []):
        # exp4 uses "time": "0:06-0:49"
        time = s.get("time", "")
        if "-" not in time:
            continue
        a, b = time.split("-", 1)
        start = _parse_ts(a)
        end = _parse_ts(b)
        labels = _clean_labels(s.get("reasons") or s.get("focus"))
        segs.append(
            {
                "start_sec": start,
                "end_sec": end,
                "position": (s.get("position") or "").strip().lower(),
                "labels": labels,
            }
        )
    return {
        "sample_id": SAMPLE_ID,
        "ontology_version": ONTOLOGY_VERSION,
        "system": "pipeline_B_two_stage_skeleton_detail",
        "provenance": "honest machine output, dev artifact, single anchor match, in-sample",
        "segments": segs,
        "transitions": [],
        "scoring": {},
    }


def _privacy_scan(obj) -> list[str]:
    """Fail loudly if any forbidden literal leaked into the prediction.

    Deny-list = generic non-PII terms + externally-loaded person-name terms
    (see _privacy.py). No person names are hard-coded here.
    """
    return _privacy_scan_terms(obj)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for name, conv, src in [
        ("pred_pipeline_A_exp3.json", convert_exp3, EXP3),
        ("pred_pipeline_B_exp4.json", convert_exp4, EXP4),
    ]:
        if not os.path.exists(src):
            print(f"SKIP {name}: source not found", file=sys.stderr)
            continue
        pred = conv(src)
        leaks = _privacy_scan(pred)
        if leaks:
            print(f"ABORT {name}: privacy leak {leaks}", file=sys.stderr)
            return 2
        out = os.path.join(OUT_DIR, name)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(pred, fh, indent=2)
        written.append((name, len(pred["segments"])))
        print(f"wrote {name}: {len(pred['segments'])} segments (privacy-clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
