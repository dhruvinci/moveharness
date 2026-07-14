#!/usr/bin/env python3
"""Privacy deny-list loader for the MoveBench converters/scorers.

The converters run a `_privacy_scan` over every emitted prediction and abort if a
forbidden literal leaked in. Part of that deny-list is a set of *person names*
that must be stripped from any released artifact. Those names are PII, so they
are NOT hard-coded as literals in the released source; instead they are loaded at
runtime from an external, comment-free deny-list file that is kept OUT of the
release (the same way the API key lives in an external `.env`).

Resolution order for the name deny-list file:
  1. the ``PRIVACY_DENYLIST`` environment variable (an explicit path), else
  2. ``<repo root>/.privacy_denylist`` (one term per line, comment-free),
     where the repo root is ``MOVEBENCH_ROOT`` or the parent of ``benchmark/``.

If no file is found the scrubber still works — it falls back to the generic
(non-PII) deny terms below, which is the safe default for a clean release tree.
The file format is one lowercase term per line; blank lines are ignored.
"""
from __future__ import annotations

import os

# Generic, non-PII deny terms (never person names). These are safe to ship.
_GENERIC_DENY = (
    "smrbzebxepa",   # legacy raw video id (must never reappear)
    "youtube",
    "http",
    "/users/",
    ".mp4",
    "rash guard",
)


def _denylist_path() -> str:
    explicit = os.environ.get("PRIVACY_DENYLIST")
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))           # .../results
    bench = os.path.normpath(os.path.join(here, os.pardir))     # .../benchmark
    root = os.environ.get("MOVEBENCH_ROOT") or os.path.normpath(
        os.path.join(bench, os.pardir)
    )
    return os.path.join(root, ".privacy_denylist")


def load_name_denylist() -> list[str]:
    """Load the (private, withheld) person-name deny terms from an external file.

    Returns a list of lowercase terms; empty if the file is absent.
    """
    path = _denylist_path()
    if not os.path.exists(path):
        return []
    terms: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            term = line.strip().lower()
            if term:
                terms.append(term)
    return terms


def forbidden_terms(extra: tuple[str, ...] = ()) -> list[str]:
    """Full deny-list: generic non-PII terms + externally-loaded person names.

    `extra` lets a caller add scope-specific generic terms (e.g. a converter that
    also wants to forbid a non-PII marker) without re-listing the names.
    """
    terms = list(_GENERIC_DENY) + list(extra) + load_name_denylist()
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        t = t.lower()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def scan(obj, extra: tuple[str, ...] = ()) -> list[str]:
    """Return the list of forbidden terms found in `json.dumps(obj).lower()`."""
    import json

    blob = json.dumps(obj).lower()
    return [f for f in forbidden_terms(extra) if f in blob]


def name_fragments() -> tuple[str, ...]:
    """Person-name fragments used to DROP name-bearing free-form tokens.

    Sourced from the same external deny-list as `load_name_denylist()`; empty if
    the file is absent. No person names are hard-coded in the release source.
    """
    return tuple(load_name_denylist())
