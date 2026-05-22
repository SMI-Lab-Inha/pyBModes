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

"""Parser for modal analysis output files (.out).

Output format:
  - Header / title
  - "rotating blade frequencies & mode shapes"   OR
    "tower frequencies & mode shapes"
  - For each mode:
      -------- Mode No.  N  (freq = X.XXXE+XX Hz)
      <column-header line>
      <blank line>
      <data rows: span_loc + 5 columns>
      <blank line>

Column order:
  Blade : span_loc  flap_disp  flap_slope  lag_disp  lag_slope  twist
  Tower : span_loc  ss_disp    ss_slope    fa_disp   fa_slope   twist
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

import numpy as np

from pybmodes.io.errors import ParseError as _ParseError


class BModeOutParseError(_ParseError):
    """Raised by :func:`read_out` under ``strict=True`` when a ``.out``
    file is not cleanly parseable.

    The default (``strict=False``) parser is intentionally tolerant —
    it skips malformed rows so a partially-corrupt file still yields
    whatever modes it can. Validation / cross-solver-comparison
    workflows that must trust every value should pass ``strict=True``;
    the message then carries the source file, the 1-based line number,
    and the mode context so the offending location is unambiguous.

    Inherits :class:`pybmodes.io.errors.ParseError` (which in turn
    inherits :class:`ValueError`), so existing
    ``except ValueError`` callers still catch this exception
    unchanged — the inheritance addition is backward-compatible. New
    callers can ``except ParseError`` to get a typed handler that
    works across every ``pybmodes.io.*`` parser.
    """


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModeShape:
    """Mode shape data for a single mode."""
    mode_number: int
    frequency: float             # Hz

    span_loc:   np.ndarray       # normalised span (0–1)
    col1:       np.ndarray       # flap_disp  (blade) or ss_disp  (tower)
    col2:       np.ndarray       # flap_slope (blade) or ss_slope (tower)
    col3:       np.ndarray       # lag_disp   (blade) or fa_disp  (tower)
    col4:       np.ndarray       # lag_slope  (blade) or fa_slope (tower)
    twist:      np.ndarray

    col_names: list[str] = field(default_factory=list)

    # Convenience properties — blade
    @property
    def flap_disp(self) -> np.ndarray:
        return self.col1

    @property
    def flap_slope(self) -> np.ndarray:
        return self.col2

    @property
    def lag_disp(self) -> np.ndarray:
        return self.col3

    @property
    def lag_slope(self) -> np.ndarray:
        return self.col4

    # Convenience properties — tower (fore-aft / side-side)
    @property
    def ss_disp(self) -> np.ndarray:
        return self.col1

    @property
    def ss_slope(self) -> np.ndarray:
        return self.col2

    @property
    def fa_disp(self) -> np.ndarray:
        return self.col3

    @property
    def fa_slope(self) -> np.ndarray:
        return self.col4


@dataclass
class BModeOutput:
    """All mode shapes parsed from a .out file."""
    title: str
    beam_type: str               # 'blade' or 'tower'
    modes: list[ModeShape]
    source_file: pathlib.Path | None = None

    def __len__(self) -> int:
        return len(self.modes)

    def __getitem__(self, index: int) -> ModeShape:
        return self.modes[index]

    def frequencies(self) -> np.ndarray:
        """Return array of natural frequencies (Hz) in mode order."""
        return np.array([m.frequency for m in self.modes])


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_MODE_HEADER = re.compile(
    r'-+\s*Mode\s+No\.\s*(\d+)\s*\(freq\s*=\s*([0-9Ee.+\-]+)\s*Hz\)',
    re.IGNORECASE,
)
_RE_BEAM_TYPE_BLADE = re.compile(r'rotating\s+blade\s+frequencies', re.IGNORECASE)
_RE_BEAM_TYPE_TOWER = re.compile(r'tower\s+frequencies', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_out(
    path: str | pathlib.Path, *, strict: bool = False
) -> BModeOutput:
    """Parse a .out file and return a :class:`BModeOutput`.

    With ``strict=False`` (default, unchanged behaviour) malformed
    data rows are silently skipped so a partially-corrupt file still
    yields whatever modes it can. With ``strict=True`` the parser
    raises :class:`BModeOutParseError` — with the source file, the
    1-based line number, and the mode context — on a short data row,
    an unparseable number, a non-finite (``NaN`` / ``inf``) value, a
    duplicate mode number, or a file that yields no modes at all.
    Use ``strict=True`` for validation / cross-solver comparison.
    """
    path = pathlib.Path(path)
    lines = path.read_text(encoding='latin-1').splitlines()
    return _parse(lines, source_file=path, strict=strict)


def _parse(
    lines: list[str],
    source_file: pathlib.Path | None = None,
    *,
    strict: bool = False,
) -> BModeOutput:
    title = ''
    beam_type = 'blade'
    modes: list[ModeShape] = []

    i = 0
    n = len(lines)

    # --- scan header for title and beam type ---
    while i < n:
        ln = lines[i]
        if _RE_BEAM_TYPE_BLADE.search(ln):
            beam_type = 'blade'
        elif _RE_BEAM_TYPE_TOWER.search(ln):
            beam_type = 'tower'

        m = _RE_MODE_HEADER.search(ln)
        if m:
            break

        # The second non-empty, non-separator line is the title
        stripped = ln.strip()
        if stripped and not stripped.startswith('=') and not title:
            # Generated-by line comes first; the second such line is title
            title = stripped

        i += 1

    src = source_file.name if source_file is not None else "<.out>"
    seen_modes: set[int] = set()

    def _err(lineno: int, mode_no: int, what: str) -> BModeOutParseError:
        return BModeOutParseError(
            f"{src}: line {lineno} (Mode No. {mode_no}): {what}"
        )

    # --- parse mode blocks ---
    while i < n:
        ln = lines[i]
        m = _RE_MODE_HEADER.search(ln)
        if not m:
            i += 1
            continue

        mode_no = int(m.group(1))
        freq    = float(m.group(2))
        hdr_lineno = i + 1                  # 1-based, for error context
        if strict:
            if mode_no in seen_modes:
                raise _err(hdr_lineno, mode_no, "duplicate mode number")
            seen_modes.add(mode_no)

        # consume column-header line (skip blank lines before it)
        i += 1
        while i < n and not lines[i].strip():
            i += 1
        col_names = lines[i].split() if i < n else []
        i += 1

        # collect data rows
        rows: list[list[float]] = []
        while i < n:
            row_line = lines[i].strip()
            i += 1
            if not row_line:
                continue
            # stop at next mode header
            if _RE_MODE_HEADER.search(row_line) or row_line.startswith('='):
                i -= 1   # put the header line back
                break
            tokens = row_line.split()
            if len(tokens) < 6:
                if strict:
                    raise _err(
                        i, mode_no,
                        f"short data row — expected >= 6 columns, got "
                        f"{len(tokens)}: {row_line!r}",
                    )
                continue
            try:
                vals = [float(t) for t in tokens[:6]]
            except ValueError as exc:
                if strict:
                    raise _err(
                        i, mode_no,
                        f"non-numeric value in data row: {row_line!r}",
                    ) from exc
                continue
            # Finite check per row, with the *offending row's* line
            # number — checking the assembled block reports the next
            # header / EOF line instead, which is useless for
            # validation against a reference solver.
            if strict and not all(np.isfinite(vals)):
                raise _err(
                    i, mode_no,
                    f"non-finite (NaN / inf) value in data row: "
                    f"{row_line!r}",
                )
            rows.append(vals)

        if strict and not rows:
            # A detected mode header that yields zero data rows must
            # not silently vanish just because another block parsed —
            # that contradicts the fail-loud strict contract.
            raise _err(
                hdr_lineno, mode_no,
                "mode header has no data rows (empty or malformed "
                "mode block)",
            )

        if rows:
            arr = np.array(rows, dtype=float)
            modes.append(ModeShape(
                mode_number=mode_no,
                frequency=freq,
                span_loc=arr[:, 0],
                col1=arr[:, 1],
                col2=arr[:, 2],
                col3=arr[:, 3],
                col4=arr[:, 4],
                twist=arr[:, 5],
                col_names=col_names,
            ))

    if strict and not modes:
        raise BModeOutParseError(
            f"{src}: no parseable mode blocks found "
            f"(file is empty, truncated, or not a BModes .out)"
        )

    return BModeOutput(
        title=title,
        beam_type=beam_type,
        modes=modes,
        source_file=source_file,
    )
