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

""":class:`CampbellResult` — typed return of :func:`campbell_sweep`.

Carries the frequency grid, mode labels, participation tensor, and
per-step MAC tracking confidence, plus NPZ + CSV round-trip
serialisation matching :class:`pybmodes.models.result.ModalResult`'s
contract (embedded ``__meta__`` JSON with pyBmodes-version + UTC
timestamp + optional source-file + git hash via
:mod:`pybmodes.io._serialize`).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CampbellResult:
    """Frequencies and labels from a Campbell sweep — blade + tower combined.

    Attributes
    ----------
    omega_rpm : (N,) array of rotor speeds in rpm.
    frequencies : (N, n_total_modes) array of natural frequencies in Hz.
        Columns are ordered *blade modes first, then tower modes*. With
        MAC tracking enabled, blade columns hold the same physical mode
        across all rotor speeds. Tower columns are constant across rows
        (tower frequencies don't depend on rotor speed).
    labels : list of length ``n_total_modes`` with human-readable mode
        names — blade modes look like ``"1st flap"`` / ``"2nd edge"``,
        tower modes are prefixed with ``"tower"`` (e.g.
        ``"1st tower FA"``, ``"1st tower SS"``) so callers can split
        the two by string match if needed.
    participation : (N, n_total_modes, 3) array of energy fractions in
        the FEM's per-mode (flap or FA, edge or SS, torsion) axes.
        Each row sums to 1. Note the axis interpretation is
        beam-type-specific: blade columns use flap/edge/torsion, tower
        columns use FA/SS/torsion.
    mac_to_previous : (N, n_total_modes) array of per-step MAC values
        between each output slot's mode shape at step ``k`` and the
        same slot at step ``k - 1`` (i.e. the tracking confidence).
        Row 0 is filled with NaN (no previous step). Tower columns are
        also NaN (tower modes don't change with rotor speed, so a MAC
        confidence is not meaningful for them).
    n_blade_modes : how many of the leading columns are blade modes.
    n_tower_modes : how many of the trailing columns are tower modes.
    """

    omega_rpm: np.ndarray
    frequencies: np.ndarray
    labels: list[str]
    participation: np.ndarray
    n_blade_modes: int
    n_tower_modes: int
    mac_to_previous: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # ------------------------------------------------------------------
    # Schema validation (shared by save / to_csv)
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Assert the documented array-shape contract before any
        export, so ``save`` / ``to_csv`` can't silently emit a
        malformed archive or CSV that loads back inconsistent.

        One uniform contract — no special-cased "empty sweep"
        exemption (that ad-hoc branch repeatedly leaked edge cases:
        missing arrays, ``.size`` vs ``.shape``). ``frequencies`` is
        ``(n_steps, n_modes)`` (2-D, always); ``omega_rpm`` is
        ``(n_steps,)``; ``labels`` has ``n_modes`` entries;
        ``participation`` is ``(n_steps, n_modes, 3)``;
        ``mac_to_previous`` is **either** the empty ``(0, 0)`` default
        (unset) **or** exactly ``(n_steps, n_modes)``;
        ``n_blade_modes, n_tower_modes`` are non-negative and sum to
        ``n_modes``. A genuinely empty sweep is simply the
        ``n_steps == n_modes == 0`` instance of this contract — every
        check below holds vacuously for the canonical empty shapes
        (``frequencies (0,0)``, ``omega_rpm (0,)``, ``participation
        (0,0,3)``, ``mac (0,0)``) and fails for any malformed
        zero-size variant such as ``(0,3)`` / ``(2,0)`` / ``(0,2)``.
        """
        freqs = np.asarray(self.frequencies)
        if freqs.ndim != 2:
            raise ValueError(
                f"frequencies must be 2-D (n_steps, n_modes); got "
                f"shape {freqs.shape}"
            )
        n_steps, n_modes = freqs.shape
        omega = np.asarray(self.omega_rpm)
        if omega.shape != (n_steps,):
            raise ValueError(
                f"omega_rpm shape {omega.shape} != (n_steps,) = "
                f"({n_steps},)"
            )
        if len(self.labels) != n_modes:
            raise ValueError(
                f"len(labels)={len(self.labels)} != n_modes={n_modes}"
            )
        part = np.asarray(self.participation)
        if part.shape != (n_steps, n_modes, 3):
            raise ValueError(
                f"participation shape {part.shape} != "
                f"(n_steps, n_modes, 3) = ({n_steps}, {n_modes}, 3)"
            )
        mac = np.asarray(self.mac_to_previous)
        # Unset iff the canonical empty default ``(0, 0)`` — *not*
        # merely ``size == 0`` (a ``(2, 0)`` / ``(0, 2)`` array is
        # size-0 but malformed). Otherwise exactly ``(n_steps,
        # n_modes)``. For the empty sweep both collapse to ``(0, 0)``.
        if mac.shape != (0, 0) and mac.shape != (n_steps, n_modes):
            raise ValueError(
                f"mac_to_previous shape {mac.shape} is neither the "
                f"empty (0, 0) default nor (n_steps, n_modes) = "
                f"({n_steps}, {n_modes})"
            )
        if self.n_blade_modes < 0 or self.n_tower_modes < 0:
            raise ValueError(
                f"mode counts must be non-negative; got "
                f"n_blade_modes={self.n_blade_modes}, "
                f"n_tower_modes={self.n_tower_modes}"
            )
        if self.n_blade_modes + self.n_tower_modes != n_modes:
            raise ValueError(
                f"n_blade_modes ({self.n_blade_modes}) + n_tower_modes "
                f"({self.n_tower_modes}) != n_modes ({n_modes})"
            )
        # Physical arrays must be finite. ``mac_to_previous`` is
        # exempt from the finite check — NaN there is the documented
        # "not meaningful" sentinel (row 0 / tower columns) — but
        # ``inf`` is *not* a valid sentinel and is rejected.
        for nm, a in (("frequencies", freqs),
                      ("omega_rpm", omega),
                      ("participation", part)):
            if not np.all(np.isfinite(np.asarray(a, dtype=float))):
                raise ValueError(
                    f"{nm} contains non-finite (NaN / inf) values"
                )
        if mac.size and np.isinf(np.asarray(mac, dtype=float)).any():
            raise ValueError(
                "mac_to_previous contains inf — NaN is the only "
                "permitted non-finite sentinel"
            )
        # participation: documented energy fractions — every row sums
        # to 1, or to 0 for a null mode shape (the documented
        # zero-shape sentinel, mirroring the mac NaN one). Negative
        # entries or any other row sum is corruption.
        if np.any(part < 0.0):
            raise ValueError(
                "participation contains negative values (energy "
                "fractions must be >= 0)"
            )
        rs = part.sum(axis=-1)
        ok = np.isclose(rs, 1.0, atol=1e-6) | np.isclose(
            rs, 0.0, atol=1e-9
        )
        if not np.all(ok):
            raise ValueError(
                "participation rows must each sum to 1 (or 0 for a "
                "null mode); got sums outside that set"
            )

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def save(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the sweep result to a ``.npz`` archive.

        Arrays go in as named keys; labels and the two integer scalars
        ride in via the embedded JSON ``__meta__`` blob alongside the
        standard pyBmodes-version / timestamp / source-file / git-hash
        metadata captured by :func:`pybmodes.io._serialize._capture_metadata`.
        """
        from pybmodes.io._serialize import _capture_metadata, _metadata_to_npz_value

        self._validate()
        meta = _capture_metadata(source_file=source_file)
        meta["labels"] = list(self.labels)
        meta["n_blade_modes"] = int(self.n_blade_modes)
        meta["n_tower_modes"] = int(self.n_tower_modes)

        np.savez_compressed(
            pathlib.Path(path),
            omega_rpm=np.asarray(self.omega_rpm, dtype=float),
            frequencies=np.asarray(self.frequencies, dtype=float),
            participation=np.asarray(self.participation, dtype=float),
            mac_to_previous=np.asarray(self.mac_to_previous, dtype=float),
            __meta__=_metadata_to_npz_value(meta),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CampbellResult":
        """Read a sweep result back from a ``.npz`` archive saved by
        :meth:`save`."""
        from pybmodes.io._serialize import _read_npz_meta

        path = pathlib.Path(path)
        with np.load(path, allow_pickle=False) as npz:
            meta = _read_npz_meta(npz, path)
            inst = cls(
                omega_rpm=np.asarray(npz["omega_rpm"], dtype=float),
                frequencies=np.asarray(npz["frequencies"], dtype=float),
                labels=list(meta["labels"]),
                participation=np.asarray(npz["participation"], dtype=float),
                n_blade_modes=int(meta["n_blade_modes"]),
                n_tower_modes=int(meta["n_tower_modes"]),
                mac_to_previous=np.asarray(npz["mac_to_previous"], dtype=float),
            )
        # Validate on ingest, not only on export — a corrupt /
        # hand-edited archive must fail loudly at load(), not later
        # in plotting / CSV export.
        inst._validate()
        return inst

    # ------------------------------------------------------------------
    # CSV emission
    # ------------------------------------------------------------------

    def to_csv(self, path: str | pathlib.Path) -> None:
        """Write a spreadsheet-friendly CSV with one row per rotor-speed
        step.

        Columns: ``rpm``, then one frequency column per mode (named by
        the mode's label), then one MAC-confidence column per mode
        suffixed with ``_mac``. Tower-mode MAC columns are NaN
        throughout because tower modes don't change with rotor speed —
        kept as columns for shape-stability across blade-only / tower-
        only / mixed sweeps.
        """
        import csv

        self._validate()
        n_steps, n_modes = self.frequencies.shape
        freq_cols = list(self.labels)
        mac_cols = [f"{lbl}_mac" for lbl in self.labels]
        header = ["rpm", *freq_cols, *mac_cols]

        with pathlib.Path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for step in range(n_steps):
                row: list[object] = [float(self.omega_rpm[step])]
                row.extend(float(self.frequencies[step, k]) for k in range(n_modes))
                # Per-mode MAC confidence (NaN where unset / not meaningful).
                if self.mac_to_previous.shape == self.frequencies.shape:
                    row.extend(
                        float(self.mac_to_previous[step, k])
                        for k in range(n_modes)
                    )
                else:
                    row.extend([float("nan")] * n_modes)
                writer.writerow(row)
