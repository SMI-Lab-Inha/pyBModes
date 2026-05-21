# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoorDyn v1 + v2 ``.dat`` section parser.

Used by :meth:`pybmodes.mooring.MooringSystem.from_moordyn`. The helpers
here are pure-Python regex / split utilities — no FEM, no catenary, no
numpy beyond what the calling site needs.

The section layout we recognise:

- **LINE TYPES** / LINE DICTIONARY — material spec rows.
- **POINTS** / CONNECTION PROPERTIES / CONNECTIONS — endpoint geometry.
- **LINES** / LINE PROPERTIES — connectivity + unstretched length.
- **OPTIONS** / SOLVER OPTIONS — depth / rho / g overrides.

Other sections (ROD TYPES, BODIES, OUTPUTS) are recognised as
section dividers but their bodies are not parsed.
"""
from __future__ import annotations

import math
import pathlib
from typing import Optional


def _looks_like_header_row(parts: list[str]) -> bool:
    """Return True if a MoorDyn-section row looks like a column-name
    or units header rather than a data row.

    Two heuristics, taken together:

    1. **Units row** — every token is parenthesised (e.g. ``(m)``,
       ``(kg/m)``, ``(-)``). Common second header line.
    2. **Column-name row** — no token in the first four columns
       parses as a number. Every MoorDyn data row across LINE TYPES /
       POINTS / LINES carries at least one numeric column in its
       first four (diameter, ID, attachment id, X-coordinate, …),
       so this distinguishes the column-name header
       (``Name Diam MassPerLength EA Diff``) from a data row whose
       first column happens to be a string LineType name.

    Used by ``MooringSystem.from_moordyn`` so the section parsers can
    tolerate MoorDyn-deck variants that ship a 1-row header (only
    column names, no units row) or no header at all, without
    accidentally eating real data rows. The previous hardcoded ``pending_skip = 2`` in
    ``_split_sections`` assumed exactly two header rows, which is
    safe on the OC3 / IEA-15 reference decks the suite already
    covers but not on every valid deck in the wild.
    """
    if not parts:
        return True
    if all(p.startswith("(") and p.endswith(")") for p in parts):
        return True
    # MoorDyn-v1 dialects emit a leading "count" row inside each section
    # (e.g. ``1   NTypes  - number of LineTypes``). That row carries a
    # single integer in column 0 and an ``N*`` label in column 1 — every
    # MoorDyn count label follows the ``N<Capital>...`` convention
    # (NTypes, NConnects, NPoints, NLines, NRods, NBodies, ...). Treat
    # this as a header so the strict per-section parsers don't try to
    # interpret it as data. The IFE UPSCALE 25 MW deck (Sandua-Fernández
    # et al. 2023) is the in-tree example.
    if (
        len(parts) >= 2
        and _looks_like_integer(parts[0])
        and len(parts[1]) >= 2
        and parts[1][0] == "N"
        and parts[1][1].isupper()
    ):
        return True
    return not any(_looks_like_number(p) for p in parts[:4])


def _looks_like_number(token: str) -> bool:
    """Return True if ``token`` parses as a finite float. Used to
    distinguish data rows (which carry at least one numeric column
    among their first few) from column-name headers (which don't)."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_integer(token: str) -> bool:
    """Return True if ``token`` parses as a base-10 integer (no leading
    sign, no fractional part). Used by :func:`_looks_like_header_row` to
    spot MoorDyn-v1 ``NTypes`` / ``NConnects`` / ``NLines`` count rows."""
    return token.isdigit()


def _parse_lines_row_v2(
    parts: list[str], points: dict,
) -> Optional[tuple[int, int, float]]:
    """Try to parse a LINES row as MoorDyn v2 (``ID LineType AttachA
    AttachB UnstrLen ...``). Returns ``(attach_a, attach_b, unstr_len)``
    or ``None`` if the columns don't validate against ``points``.
    Non-finite ``unstr_len`` returns ``None`` so the v1 fallback gets
    a fair try; a downstream ``unstr_len <= 0`` check also rejects
    NaN (``nan <= 0`` is False but ``unstr_len > 0`` is also False)."""
    try:
        attach_a = int(parts[2])
        attach_b = int(parts[3])
        unstr_len = float(parts[4])
    except ValueError:
        return None
    if not math.isfinite(unstr_len):
        return None
    if attach_a not in points or attach_b not in points or unstr_len <= 0:
        return None
    return attach_a, attach_b, unstr_len


def _parse_lines_row_v1(
    parts: list[str], points: dict,
) -> Optional[tuple[int, int, float]]:
    """Try to parse a LINES row as MoorDyn v1 (``ID LineType UnstrLen
    NumSegs NodeAnch NodeFair``). Returns ``(attach_a, attach_b,
    unstr_len)`` or ``None`` if the columns don't validate. Non-finite
    ``unstr_len`` returns ``None`` for the same reason as the v2
    helper above."""
    try:
        unstr_len = float(parts[2])
        attach_a = int(parts[4])
        attach_b = int(parts[5])
    except ValueError:
        return None
    if not math.isfinite(unstr_len):
        return None
    if attach_a not in points or attach_b not in points or unstr_len <= 0:
        return None
    return attach_a, attach_b, unstr_len


def _parse_finite_option(value: str, key: str, path: pathlib.Path) -> float:
    """Strict-parse a MoorDyn OPTIONS row value. Raises ``ValueError``
    with the key, path, and offending token on any parse failure or
    non-finite result. Used for the three load-bearing keys
    (``WtrDpth`` / ``rhoW`` / ``g``); unknown OPTIONS keys stay
    permissive (callers don't reach this helper). A bare
    ``try / except: pass`` would silently swallow typos in these
    keys, which directly shift mooring
    stiffness through the wet-weight calculation."""
    try:
        out = float(value)
    except ValueError as err:
        raise ValueError(
            f"Malformed OPTIONS row in {path}: value {value!r} for "
            f"key {key!r} is not a number."
        ) from err
    if not math.isfinite(out):
        raise ValueError(
            f"Malformed OPTIONS row in {path}: value {value!r} for "
            f"key {key!r} parses to {out!r}, which is not finite. "
            f"Physical quantities (depth, density, gravity) must be "
            f"finite."
        )
    return out


# Map of (lowercase keywords that appear in a section header) -> canonical
# section name. We pick the FIRST keyword that matches inside each header
# line so trailing decorations don't trip us up.
_SECTION_KEYWORDS = {
    "line types": "LINE TYPES",
    "line dictionary": "LINE TYPES",
    "rod types": "ROD TYPES",  # reserved (not parsed)
    "rod dictionary": "ROD TYPES",
    "bodies": "BODIES",         # reserved (not parsed)
    "points": "POINTS",
    "connection properties": "POINTS",   # v1 alias
    "connections": "POINTS",             # v1 alias
    "lines": "LINES",
    "line properties": "LINES",
    "options": "OPTIONS",
    "solver options": "OPTIONS",
    "outputs": "OUTPUTS",       # reserved (not parsed)
    "output list": "OUTPUTS",
}


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Group MoorDyn file lines into sections keyed by canonical name.

    Header detection: a line that starts with three dashes (after
    stripping whitespace) is a section divider; its lowercase content
    (with decoration stripped) is matched against
    ``_SECTION_KEYWORDS``. Inside a recognised section, up to two
    column-name / units rows immediately after the divider are skipped
    if they match :func:`_looks_like_header_row`; the moment a row that
    looks like data appears we stop eating header rows. This handles
    both the de-facto MoorDyn convention (column-names + units lines,
    two header rows) and hand-edited variants with one header row or
    none at all. The previous
    fixed ``pending_skip = 2`` ate the first data row on decks shipped
    without a units line.

    Comment lines (``!``, ``#``) and blank lines are dropped.
    """
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    pending_header_skip = 0  # rows of header to inspect-and-maybe-skip
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("!") or stripped.startswith("#"):
            continue
        if stripped.startswith("---"):
            # Section divider. Find a known keyword inside it.
            content = stripped.strip("- ").lower()
            section = None
            for kw, canon in _SECTION_KEYWORDS.items():
                if kw in content:
                    section = canon
                    break
            current = section
            if current is not None:
                sections.setdefault(current, [])
                # Allow up to two consecutive header rows after the
                # divider. OPTIONS has no header / units convention —
                # every row there is ``value label``.
                pending_header_skip = 0 if current == "OPTIONS" else 2
            continue
        # Detect "END" sentinel.
        if stripped.upper() == "END":
            current = None
            continue
        if current is None:
            continue
        if pending_header_skip > 0:
            parts = raw.split()
            if _looks_like_header_row(parts):
                pending_header_skip -= 1
                continue
            # First non-header row marks the start of real data; stop
            # the inspect-and-skip loop so any subsequent rows that
            # incidentally pattern-match the header heuristic still
            # land in the section list (where the strict-parse path
            # raises on them rather than silently dropping).
            pending_header_skip = 0
        sections[current].append(raw)
    return sections
