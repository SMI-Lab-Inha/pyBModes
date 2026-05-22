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

"""WAMIT v7 output file reader for OpenFAST / HydroDyn floating-platform decks.

Parses ``.1`` (added-mass + radiation-damping) and ``.hst`` (hydrostatic
restoring) files written by WAMIT, redimensionalises them per the WAMIT v7
convention (``ρ·L^k`` and ``ρ·g·L^k`` factors), and returns the 6 × 6 SI
matrices a pyBmodes ``PlatformSupport`` block consumes.

The reader is read-only and pure-Python; pairs with a thin
:class:`HydroDynReader` that picks ``WAMITULEN`` and the ``PotFile`` root
from an OpenFAST HydroDyn ``.dat`` so callers can chain
``HydroDynReader(...).read_platform_matrices()`` in a single step.

WAMIT file formats
==================

``.1`` (added mass / radiation damping) — each line carries::

    period  i  j  A(i,j)  [B(i,j)]

* ``period = -1.0`` rows are the infinite-frequency added mass ``A_inf``
  (only the ``A`` column is present; no damping at infinite frequency).
* ``period =  0.0`` rows are the zero-frequency added mass ``A_0`` (also
  ``A`` only).
* ``period > 0`` rows are frequency-dependent ``A(ω) + B(ω)``; this
  reader currently extracts only ``A_inf`` and ``A_0``.

Indices ``i``, ``j`` are 1-indexed over the six rigid-body DOFs
``{1: surge, 2: sway, 3: heave, 4: roll, 5: pitch, 6: yaw}``. The file is
sparse — only non-trivial entries are listed; missing entries are zero.

``.hst`` (hydrostatic restoring) — each line carries::

    i  j  C(i,j)

Same 1-indexed convention. Some WAMIT runs write only the upper triangle;
others write the full 6 × 6 including explicit zeros.

WAMIT v7 nondimensionalisation
==============================

All WAMIT outputs are dimensionless. Redimensionalisation factors depend
on the DOF-pair type:

* **Added mass** (``.1``):
  ``trans-trans → ρ·L³``, ``trans-rot → ρ·L⁴``, ``rot-rot → ρ·L⁵``.
* **Hydrostatic stiffness** (``.hst``):
  ``trans-trans → ρ·g·L²``, ``trans-rot → ρ·g·L³``, ``rot-rot → ρ·g·L⁴``.

Here ``L = WAMITULEN`` (declared in the HydroDyn ``.dat``), ``ρ`` is the
sea-water density, and ``g`` is gravitational acceleration. The exponent
table is captured in :meth:`WamitReader._dim_factor` as
``base + n_rot_dofs`` where ``base ∈ {3, 2}`` and ``n_rot_dofs ∈ {0, 1, 2}``.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

__all__ = ["HydroDynReader", "WamitData", "WamitReader"]


def _parse_fortran_float(value: str) -> float:
    """``float(value)`` after normalising Fortran-style ``D`` / ``d``
    exponent notation (``1.234D+02``) to Python's ``E`` / ``e``.

    Strict-finite: rejects ``nan`` / ``inf`` results. Upstream WAMIT
    and HydroDyn writers occasionally emit Fortran-style exponents
    instead of the C convention; using bare ``float`` on those silently
    fails (``ValueError``). Shared by :class:`HydroDynReader` for the
    one-shot scalar reads where a non-finite value is unambiguously a
    bug. The matrix-row parsers
    (:func:`WamitReader._parse_dot1`, :func:`WamitReader._parse_hst`)
    instead use :func:`_parse_fortran_float_lenient` inside their
    schema-probe try blocks and call :func:`_require_finite` *outside*
    the try so a non-finite numeric in an otherwise-schema-matching
    row raises loudly rather than getting silently skipped as a
    header / comment line.

    The previous ``nan``-tolerance is rejected here.
    """
    import math
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite float in WAMIT / HydroDyn token: {value!r} "
            f"parses to {result!r}. Physical quantities must be "
            f"finite."
        )
    return result


def _is_fortran_default(value: str) -> bool:
    """Return ``True`` if a HydroDyn ``.dat`` value carries Fortran's
    ``DEFAULT`` sentinel (case-insensitive, quoted or unquoted).

    HydroDyn ``v2.03+`` accepts ``"default"`` / ``"DEFAULT"`` /
    ``DEFAULT`` for several scalar inputs (``WtrDens``, ``MSL2SWL``,
    ``RdtnDT``, ``CurrSSDir``, …) — the solver then falls back to a
    documented default. The pyBmodes WAMIT reader needs to recognise
    this sentinel so it doesn't run the Fortran-float normaliser
    (which substitutes ``d → e`` and silently mangles ``default`` to
    ``eefault``). Pre-1.0 review: the IFE UPSCALE 25MW deck uses
    ``"default"`` for ``WtrDens``.
    """
    stripped = value.strip().strip('"').strip("'").upper()
    return stripped == "DEFAULT"


def _parse_fortran_float_lenient(value: str) -> float:
    """Like :func:`_parse_fortran_float` but accepts ``nan`` / ``inf``.

    Used inside the ``.1`` / ``.hst`` row-parse try blocks where a
    ``ValueError`` is interpreted as "this isn't a data row, skip it
    as header / comment". A finite check there would let non-finite
    matrix entries fall through the silent-skip path; instead the
    matrix parsers call :func:`_require_finite` outside the try block
    so a real non-finite value raises.
    """
    return float(value.replace("D", "E").replace("d", "e"))


def _require_finite(value: float, label: str, lineno: int) -> None:
    """Raise ``ValueError`` if ``value`` is non-finite, naming the
    label (e.g. ``"A(1,4)"``) and the WAMIT file line number."""
    import math
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite WAMIT entry on line {lineno}: {label} = "
            f"{value!r}. Physical quantities must be finite; a stray "
            f"``nan`` / ``inf`` in a WAMIT output is almost certainly "
            f"a transcription error or an unconverged hydrodynamic "
            f"solve."
        )


def _symmetrise_in_place(M: np.ndarray) -> None:
    """For any ``(i, j)`` where ``M[i, j] == 0`` and ``M[j, i] != 0``,
    copy ``M[j, i] → M[i, j]``. Some WAMIT runs write only the upper
    (or only the lower) triangle of a symmetric matrix; this fills the
    missing half without disturbing explicit zeros that came from a
    fully-written matrix."""
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] == 0.0 and M[j, i] != 0.0:
                M[i, j] = M[j, i]
            elif M[j, i] == 0.0 and M[i, j] != 0.0:
                M[j, i] = M[i, j]


@dataclass
class WamitData:
    """Dimensionalised WAMIT output for a single floating body.

    All matrices are in SI units. Added-mass entries are ``kg`` for
    trans-trans, ``kg·m`` for trans-rot, and ``kg·m²`` for rot-rot.
    Hydrostatic-stiffness entries are ``N/m`` for trans-trans, ``N`` (or
    ``N·m/m``, same thing) for trans-rot, and ``N·m/rad`` for rot-rot.

    Attributes
    ----------
    A_inf : ndarray, shape (6, 6)
        Infinite-frequency added mass (``period = -1`` rows of ``.1``).
    A_0 : ndarray, shape (6, 6)
        Zero-frequency added mass (``period = 0`` rows of ``.1``).
    C_hst : ndarray, shape (6, 6)
        Hydrostatic restoring matrix (all rows of ``.hst``).
    rho, g, ulen : float
        Re-dimensionalisation constants applied; stored so downstream code
        can audit the SI conversion that was used.
    pot_file_root : pathlib.Path
        Absolute path of the resolved ``PotFile`` (no extension).
    """

    A_inf: np.ndarray
    A_0: np.ndarray
    C_hst: np.ndarray
    rho: float
    g: float
    ulen: float
    pot_file_root: pathlib.Path


class WamitReader:
    """Read a WAMIT v7 ``.1`` / ``.hst`` pair and return SI 6 × 6 matrices.

    The constructor only normalises the inputs; on-disk reads happen inside
    :meth:`read` so callers can catch ``FileNotFoundError`` at a single
    well-defined point.

    Parameters
    ----------
    pot_file_root : str or Path
        ``PotFile`` value taken verbatim from the HydroDyn ``.dat``. May
        carry surrounding double or single quotes, may use backslashes
        on Windows, and may be relative (resolved against ``dat_dir``) or
        absolute.
    dat_dir : pathlib.Path
        Directory of the HydroDyn ``.dat`` whose ``PotFile`` is being
        followed. Used to resolve relative ``PotFile`` values.
    rho, g, ulen : float
        Sea-water density (``kg/m³``), gravitational acceleration
        (``m/s²``), and WAMIT reference length (``WAMITULEN``, m). Default
        to standard sea-water values so simple smoke tests don't need an
        explicit ``HydroDynReader``.
    """

    def __init__(
        self,
        pot_file_root: str | pathlib.Path,
        dat_dir: pathlib.Path,
        rho: float = 1025.0,
        g: float = 9.80665,
        ulen: float = 1.0,
    ) -> None:
        self._raw_pot_file = str(pot_file_root)
        self._dat_dir = pathlib.Path(dat_dir).resolve()
        self.rho = float(rho)
        self.g = float(g)
        self.ulen = float(ulen)
        self.pot_file_root = self._resolve_pot_path(
            self._raw_pot_file, self._dat_dir,
        )

    @staticmethod
    def _resolve_pot_path(
        pot_file: str, dat_dir: pathlib.Path,
    ) -> pathlib.Path:
        """Normalise a HydroDyn ``PotFile`` value and resolve it to a path.

        Handles: surrounding double or single quotes, Windows-style
        backslashes (rewritten to forward slashes so ``pathlib`` treats
        them as separators on Linux too), and relative-vs-absolute path
        prefixes. Returns the root path WITHOUT extension; ``.1`` / ``.hst``
        are appended when each file is opened.
        """
        s = pot_file.strip()
        # Strip a single layer of matching surrounding quotes.
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1].strip()
        # Normalise separators so backslash inputs work on Linux too.
        s = s.replace("\\", "/")
        p = pathlib.Path(s)
        if not p.is_absolute():
            p = pathlib.Path(dat_dir) / p
        # ``strict=False`` (the default) leaves missing path components
        # alone; we only check existence when actually opening files.
        return p.resolve()

    def read(self) -> WamitData:
        """Parse the ``.1`` and ``.hst`` files alongside the resolved root.

        Raises
        ------
        FileNotFoundError
            If either ``<root>.1`` or ``<root>.hst`` is missing; the
            message names the expected absolute path and the verbatim
            ``PotFile`` value that produced it.
        """
        dot1 = pathlib.Path(str(self.pot_file_root) + ".1")
        hst = pathlib.Path(str(self.pot_file_root) + ".hst")
        if not dot1.is_file():
            raise FileNotFoundError(
                f"WAMIT .1 file not found at {dot1}; expected alongside "
                f"the HydroDyn deck at PotFile = {self._raw_pot_file!r}"
            )
        if not hst.is_file():
            raise FileNotFoundError(
                f"WAMIT .hst file not found at {hst}; expected alongside "
                f"the HydroDyn deck at PotFile = {self._raw_pot_file!r}"
            )
        A_inf, A_0 = self._parse_dot1(dot1)
        C_hst = self._parse_hst(hst)
        return WamitData(
            A_inf=A_inf,
            A_0=A_0,
            C_hst=C_hst,
            rho=self.rho,
            g=self.g,
            ulen=self.ulen,
            pot_file_root=self.pot_file_root,
        )

    def _parse_dot1(
        self, path: pathlib.Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse a ``.1`` file and return ``(A_inf, A_0)`` SI 6 × 6 matrices.

        Finite-period rows (``period > 0``) are skipped — this reader
        currently extracts only the two limiting added-mass matrices that
        a quasi-static modal solver needs.
        """
        A_inf = np.zeros((6, 6))
        A_0 = np.zeros((6, 6))
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for ln, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    period = _parse_fortran_float_lenient(parts[0])
                    i = int(parts[1])
                    j = int(parts[2])
                    a_val = _parse_fortran_float_lenient(parts[3])
                except ValueError:
                    # Ignore header / comment lines that don't fit the
                    # numeric schema; WAMIT outputs are sometimes prefaced
                    # by a one-line title.
                    continue
                # Period must be finite to be meaningful. The documented
                # WAMIT period markers are ``-1.0`` (A_inf), ``0.0``
                # (A_0), and any positive finite value (finite-period
                # frequency-dependent row, skipped). ``nan`` / ``inf``
                # in the period column would otherwise fall into the
                # ``else: continue`` branch below as if it were a
                # finite-period row — silently dropping the entry from
                # an otherwise-schema-matching ``A_inf`` / ``A_0`` row.
                _require_finite(period, f"period (row i,j={i},{j})", ln)
                if period == -1.0:
                    target = A_inf
                elif period == 0.0:
                    target = A_0
                else:
                    # Finite-period (frequency-dependent) row; not extracted.
                    continue
                ii, jj = i - 1, j - 1
                if not (0 <= ii < 6 and 0 <= jj < 6):
                    raise ValueError(
                        f"{path.name}:{ln}: WAMIT (i,j)=({i},{j}) outside "
                        f"the 6 rigid-body DOFs"
                    )
                # Finite check OUTSIDE the schema-probe try block — a
                # non-finite value in an otherwise-schema-matching row
                # should raise, not silently get skipped as a header.
                _require_finite(a_val, f"A({i},{j})", ln)
                target[ii, jj] = a_val * self._dim_factor(ii, jj, "added_mass")
        # Fill in any half-mirror gaps so upper-triangle-only WAMIT
        # outputs still produce a full symmetric matrix.
        _symmetrise_in_place(A_inf)
        _symmetrise_in_place(A_0)
        return A_inf, A_0

    def _parse_hst(self, path: pathlib.Path) -> np.ndarray:
        """Parse a ``.hst`` file and return ``C_hst`` as an SI 6 × 6 matrix."""
        C = np.zeros((6, 6))
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for ln, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    c_val = _parse_fortran_float_lenient(parts[2])
                except ValueError:
                    continue
                ii, jj = i - 1, j - 1
                if not (0 <= ii < 6 and 0 <= jj < 6):
                    raise ValueError(
                        f"{path.name}:{ln}: WAMIT (i,j)=({i},{j}) outside "
                        f"the 6 rigid-body DOFs"
                    )
                _require_finite(c_val, f"C({i},{j})", ln)
                C[ii, jj] = c_val * self._dim_factor(ii, jj, "hydrostatic")
        _symmetrise_in_place(C)
        return C

    def _dim_factor(self, i: int, j: int, matrix_type: str) -> float:
        """WAMIT v7 dimensionalisation factor for entry ``(i, j)``.

        Parameters
        ----------
        i, j : int
            0-indexed DOFs. ``0..2`` are translations (surge / sway /
            heave); ``3..5`` are rotations (roll / pitch / yaw).
        matrix_type : {'added_mass', 'hydrostatic'}
            Selects the base exponent: 3 for added mass, 2 for
            hydrostatic. Each rotational DOF in the index pair adds 1 to
            the exponent (so trans-trans = base, trans-rot = base+1,
            rot-rot = base+2).
        """
        n_rot = int(i >= 3) + int(j >= 3)
        if matrix_type == "added_mass":
            k = 3 + n_rot
            return self.rho * (self.ulen ** k)
        if matrix_type == "hydrostatic":
            k = 2 + n_rot
            return self.rho * self.g * (self.ulen ** k)
        raise ValueError(
            f"unknown matrix_type {matrix_type!r}; expected "
            f"'added_mass' or 'hydrostatic'"
        )


class HydroDynReader:
    """Minimal reader for the floating-platform section of a HydroDyn ``.dat``.

    Surfaces ``WAMITULEN``, ``PotMod``, ``PotFile``, and ``PtfmRefzt`` —
    the four values needed to drive :class:`WamitReader`. ``WtrDens`` and
    ``Gravity`` are NOT in the modern HydroDyn ``.dat`` (HydroDyn ≥ v2.03
    delegates those to the paired SeaState input file), so the corresponding
    properties fall back to standard sea-water defaults.

    The parser is deliberately loose: it scans each non-blank line for the
    pattern ``<value> <label> ...`` and stores the first occurrence of
    each label. Matrix continuation rows (e.g. the five extra rows of
    ``AddCLin``) and wrapped comment text get parsed too but produce
    harmless ``_values[<numeric_string>] = ...`` entries that no caller
    asks for.
    """

    def __init__(self, dat_path: str | pathlib.Path) -> None:
        self.dat_path = pathlib.Path(dat_path).resolve()
        if not self.dat_path.is_file():
            raise FileNotFoundError(
                f"HydroDyn .dat not found at {self.dat_path}"
            )
        self._values: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        with self.dat_path.open(
            "r", encoding="utf-8", errors="replace",
        ) as fh:
            for raw in fh:
                # HydroDyn comments use ``-`` after the label, not ``!``.
                # We don't strip them; ``str.split()`` naturally stops at
                # whitespace so the value and label always come out as
                # ``parts[0]`` and ``parts[1]``.
                parts = raw.split()
                if len(parts) < 2:
                    continue
                # Only store the first hit for each label so continuation
                # lines (matrix rows beyond the labelled first row) and
                # wrapped comment text don't overwrite real values.
                self._values.setdefault(parts[1], parts[0])

    def _get(self, key: str, default: str | None = None) -> str:
        if key in self._values:
            return self._values[key]
        if default is None:
            raise KeyError(
                f"{key!r} not found in HydroDyn .dat at {self.dat_path}"
            )
        return default

    @property
    def ulen(self) -> float:
        """``WAMITULEN`` — WAMIT reference length used for re-dimensionalisation."""
        return _parse_fortran_float(self._get("WAMITULEN"))

    @property
    def rho_water(self) -> float:
        """Water density (``kg/m³``).

        HydroDyn v2.03+ moves ``WtrDens`` to the paired SeaState file;
        if the legacy inline value isn't present, or carries Fortran's
        ``"DEFAULT"`` keyword, we fall back to a standard sea-water
        default of 1025 kg/m³.
        """
        if "WtrDens" in self._values and not _is_fortran_default(
            self._values["WtrDens"]
        ):
            return _parse_fortran_float(self._values["WtrDens"])
        return 1025.0

    @property
    def gravity(self) -> float:
        """Gravitational acceleration (``m/s²``).

        OpenFAST stores ``Gravity`` at the top-level ``.fst`` file, not in
        HydroDyn. We fall back to ISO standard gravity (9.80665) so the
        reader is self-contained.
        """
        if "Gravity" in self._values:
            return _parse_fortran_float(self._values["Gravity"])
        return 9.80665

    @property
    def ptfm_ref_zt(self) -> float:
        """``PtfmRefzt`` — vertical offset of the body reference point (m)."""
        return _parse_fortran_float(self._get("PtfmRefzt", "0.0"))

    @property
    def pot_file(self) -> str:
        """``PotFile`` — WAMIT root path (verbatim, may carry quotes)."""
        return self._get("PotFile")

    @property
    def pot_mod(self) -> int:
        """``PotMod`` — 0 = no potential-flow model, 1 = WAMIT."""
        return int(self._get("PotMod", "0"))

    def read_platform_matrices(self) -> WamitData:
        """Resolve ``PotFile`` alongside this deck and read its WAMIT outputs.

        Raises ``ValueError`` if ``PotMod == 0`` (no WAMIT data attached).
        """
        if self.pot_mod == 0:
            raise ValueError(
                f"HydroDyn deck at {self.dat_path} has PotMod=0; no WAMIT "
                f"output to read"
            )
        return WamitReader(
            self.pot_file,
            self.dat_path.parent,
            rho=self.rho_water,
            g=self.gravity,
            ulen=self.ulen,
        ).read()
