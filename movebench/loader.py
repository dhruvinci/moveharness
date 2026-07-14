"""MoveBench v0 loaders.

Provides:
  * `Ontology`         -- reads `ontology.json`; alias-folds surface forms to
                          canonical ids; answers membership / normalization
                          queries used by the scorer.
  * `load_sample`      -- loads a MoveBench v0 sample (ground-truth) file.
  * `load_prediction`  -- loads a MoveBench prediction file.
  * `enumerate_label_dataset_candidates` -- enumerates the external instructional
                          SQLite corpus by AGGREGATE COUNT ONLY. It NEVER reads
                          or returns raw video ids, urls, titles, channels, or
                          clip text — only counts — so no private data is
                          redistributed.

Standard library only; no external services.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional


# Default location of the bundled ontology, resolved relative to this file so
# the package is import-clean regardless of the caller's working directory.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ONTOLOGY = os.path.normpath(os.path.join(_PACKAGE_DIR, os.pardir, "ontology.json"))


class Ontology:
    """Alias-folding ontology helper sourced from `ontology.json`.

    Replaces the P1 scorer's hard-coded alias maps. Builds a single surface ->
    canonical-id table from L1A macro positions (id + aliases), the v0
    used-label normalizations, L1B sub-positions, L2 roles, transition events,
    and the technique vocabulary, so both GT and prediction surface forms fold
    to the same canonical ids.
    """

    def __init__(self, data: dict):
        self.data = data
        self.version = data.get("ontology_version", "unknown")

        # Canonical macro-position label space for T1 macro-F1: the 18 L1A ids
        # PLUS the v0 reduced macro-position field values (which include the
        # event-style states 'sweep'/'escape' used as segment labels in v0).
        self.macro_positions: set[str] = {
            m["id"] for m in data.get("L1A_macro_positions", [])
        }
        v0_field = data.get("v0_macro_position_field_values", {})
        self.v0_position_values: set[str] = set(v0_field.get("values", []))
        self.position_label_space: set[str] = self.macro_positions | self.v0_position_values

        # Build the surface -> canonical alias map.
        self._alias: dict[str, str] = {}

        def _add(surface: Optional[str], canonical: str) -> None:
            if surface is None:
                return
            key = surface.strip().lower()
            if key:
                self._alias.setdefault(key, canonical)

        for m in data.get("L1A_macro_positions", []):
            _add(m["id"], m["id"])
            _add(m["label"], m["id"])
            for a in m.get("aliases", []):
                _add(a, m["id"])

        # v0 used-label normalizations (surface -> canonical).
        for lbl in data.get("v0_used_labels", {}).get("labels", []):
            _add(lbl["surface"], lbl["normalized"])
            _add(lbl["normalized"], lbl["normalized"])

        # Sub-positions, roles, transitions, technique vocab: self-mapping so
        # they register as in-ontology.
        for parent, subs in data.get("L1B_subpositions", {}).items():
            for s in subs:
                _add(s, s)
        for r in data.get("L2_roles", {}).get("values", []):
            _add(r, r)
        for t in data.get("transition_events", {}).get("values", []):
            _add(t, t)
        for cat in data.get("technique_vocabulary", {}).get("categories", {}).values():
            for term in cat:
                _add(term, term)
        for v in self.v0_position_values:
            _add(v, v)

        # The full set of canonical ids that count as "in ontology".
        self._known: set[str] = set(self._alias.values()) | self.position_label_space

    # -- normalization API used by metrics.py ------------------------------- #
    def normalize_position(self, position: Optional[str]) -> str:
        p = (position or "").strip().lower()
        return self._alias.get(p, p)

    def normalize_label(self, label: Optional[str]) -> str:
        l = (label or "").strip().lower()
        return self._alias.get(l, l)

    def normalize_transition(self, event: Optional[str]) -> str:
        return self.normalize_label(event)

    def is_in_ontology(self, canonical: str) -> bool:
        c = (canonical or "").strip().lower()
        return c in self._known

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Ontology":
        path = path or _DEFAULT_ONTOLOGY
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))


def load_sample(path: str) -> dict:
    """Load a MoveBench v0 sample (ground-truth) file as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_prediction(path: str) -> dict:
    """Load a MoveBench prediction file as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def flatten_subsegments(sample: dict) -> list[dict]:
    """Flatten macro-segment subsegments into one ordered list (for T3 oracle).

    Returns subsegment dicts with `start_sec`, `end_sec`, `labels`, carrying the
    parent macro `position`. Protocol-only subsegments (labels are a subset of
    the excluded protocol annotations, e.g. pre_match/match_start) are dropped so
    the benchmark count stays at 62 subsegments / 122 label instances.
    """
    excluded = set(
        sample.get("_protocol_excluded")
        or ["pre_match", "match_start"]
    )
    out: list[dict] = []
    for seg in sample.get("macro_segments", []):
        for sub in seg.get("subsegments", []):
            labels = sub.get("labels") or []
            if labels and all(l in excluded for l in labels):
                continue
            out.append(
                {
                    "sub_id": sub.get("sub_id"),
                    "start_sec": sub["start_sec"],
                    "end_sec": sub["end_sec"],
                    "position": seg.get("position"),
                    "labels": labels,
                }
            )
    return out


def enumerate_label_dataset_candidates(db_path: str) -> dict:
    """Enumerate the external instructional SQLite corpus by COUNTS ONLY.

    PRIVACY: this function deliberately returns ONLY aggregate integer counts.
    It never selects, returns, or writes raw youtube_ids, urls, titles,
    channels, or clip-label text, so no private data leaves the database. It is
    provided so v1 expansion tooling can size the candidate pool without
    redistributing the corpus.

    Returns counts of videos, described clips, and distinct technique tags.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"External label-dataset candidate corpus not found at the provided "
            f"path. This external resource is referenced by counts only and is "
            f"not bundled with the benchmark."
        )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        n_videos = cur.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        n_clips = cur.execute(
            "SELECT COUNT(*) FROM timestamps "
            "WHERE label IS NOT NULL AND TRIM(label) != ''"
        ).fetchone()[0]
        n_tags = cur.execute(
            "SELECT COUNT(DISTINCT hashtag) FROM timestamp_hashtags"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "resource": "external_instructional_clip_corpus",
        "redistributed": False,
        "note": "counts only; no raw ids/urls/titles/labels are returned",
        "n_videos": n_videos,
        "n_described_clips": n_clips,
        "n_distinct_technique_tags": n_tags,
    }
