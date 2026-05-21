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

"""Rotor-speed sweep drivers and the public :func:`campbell_sweep` entry point.

Two solver paths:

- :func:`_solve_blade_sweep` — re-solves the blade FEM at each
  rotor speed in the input grid, tracks each mode across consecutive
  steps via :func:`_hungarian_assignment` on the per-step
  :func:`_mac_matrix`, and records the per-step MAC confidence.
- :func:`_solve_tower_once` — solves the tower model once at
  ``rot_rpm = 0`` and broadcasts the result across the sweep
  (tower modes are rotor-speed independent in an Earth-fixed frame).

The public :func:`campbell_sweep` is a thin orchestrator that
validates inputs, resolves the input(s) via :func:`_load_models`, runs
each side that's available, and packs the result into a
:class:`~pybmodes.campbell.result.CampbellResult`.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models._pipeline import run_fem

from ._classify import (
    _label_blade_modes,
    _label_tower_modes_with_overrides,
    _participation,
)
from ._mac import _hungarian_assignment, _mac_matrix
from ._models import _load_models, _Model
from .result import CampbellResult

if TYPE_CHECKING:
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.tower import Tower


def _solve_blade_sweep(
    blade: _Model,
    omega_rpm: np.ndarray,
    n_modes: int,
    track_by_mac: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Run the rotor-speed sweep on the blade model.

    Returns ``(frequencies, participation, labels, mac_to_previous)``
    with shapes ``(n_steps, n_modes)``, ``(n_steps, n_modes, 3)``, a
    list of ``n_modes`` labels, and ``(n_steps, n_modes)`` per-step
    MAC values vs the immediately-preceding step (row 0 is NaN).
    Restores the original ``bbmi.rot_rpm`` after the sweep so the
    caller's BMI object is unmutated.
    """
    bbmi, bsp = blade
    original_rpm = float(getattr(bbmi, "rot_rpm", 0.0))
    n_steps = omega_rpm.size
    freqs = np.zeros((n_steps, n_modes))
    parts = np.zeros((n_steps, n_modes, 3))
    mac_to_prev = np.full((n_steps, n_modes), np.nan, dtype=float)
    slot_shapes: list[NodeModeShape] | None = None

    try:
        for step, rpm in enumerate(omega_rpm):
            bbmi.rot_rpm = float(rpm)
            modal = run_fem(bbmi, n_modes=n_modes, sp=bsp)
            # Defensive: the symmetric ``eigh`` solver always returns
            # exactly ``n_modes`` rows, but the rare general-eig
            # fallback (floating platforms with non-symmetric K,
            # ``solve_modes`` → ``scipy.linalg.eig``) can drop NaN
            # rows from a degenerate eigenproblem. Detect and fail
            # loudly rather than silently NaN-padding downstream.
            if len(modal.frequencies) < n_modes:
                raise RuntimeError(
                    f"campbell_sweep: at rot_rpm = {rpm:.3f}, the FEM "
                    f"solver returned only {len(modal.frequencies)} of "
                    f"the requested {n_modes} modes — typically a sign "
                    f"of a near-degenerate eigenproblem (rotating "
                    f"floating platforms with a non-symmetric "
                    f"PlatformSupport block at certain rotor speeds). "
                    f"Reduce ``n_blade_modes`` or use a finer RPM grid "
                    f"that avoids the degeneracy."
                )
            shapes = list(modal.shapes[:n_modes])
            f_step = np.asarray(modal.frequencies[:n_modes], dtype=float)
            p_step = np.array([_participation(s) for s in shapes])

            if step == 0 or not track_by_mac or slot_shapes is None:
                order = np.arange(n_modes, dtype=int)
                mac_row = np.full(n_modes, np.nan, dtype=float)
            else:
                mac = _mac_matrix(shapes, slot_shapes)
                order = _hungarian_assignment(mac)
                free = [s for s in range(n_modes) if s not in order]
                for k in range(n_modes):
                    if order[k] < 0 and free:
                        order[k] = free.pop(0)
                # MAC confidence of the chosen pairing per output slot.
                mac_row = np.empty(n_modes, dtype=float)
                for k in range(n_modes):
                    slot = int(order[k])
                    mac_row[slot] = float(mac[k, slot]) if slot >= 0 else np.nan

            for k in range(n_modes):
                slot = int(order[k])
                freqs[step, slot] = f_step[k]
                parts[step, slot, :] = p_step[k]
            mac_to_prev[step, :] = mac_row

            new_slot_shapes: list[NodeModeShape | None] = [None] * n_modes
            for k in range(n_modes):
                new_slot_shapes[int(order[k])] = shapes[k]
            slot_shapes = [s for s in new_slot_shapes if s is not None]
    finally:
        bbmi.rot_rpm = original_rpm

    labels = _label_blade_modes(parts[0])
    return freqs, parts, labels, mac_to_prev


def _solve_tower_once(
    tower: _Model,
    n_modes: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Solve the tower once at ``rot_rpm = 0`` and broadcast across the sweep.

    Tower modes are rotor-speed-independent (the tower lives in an
    Earth-fixed frame), so a single eigensolve is enough; we tile the
    result across ``n_steps`` rows for shape compatibility with the
    blade-sweep output.
    """
    tbmi, tsp = tower
    # Save and restore the caller's ``rot_rpm`` — tower modes are
    # rotor-speed-independent so we force ``0.0`` for the solve, but
    # we mustn't leave the caller's BMI mutated on the way out
    # (mirrors the try / finally pattern in ``_solve_blade_sweep``;
    # Pre-1.0 review caught this).
    original_rpm = tbmi.rot_rpm
    try:
        tbmi.rot_rpm = 0.0
        modal = run_fem(tbmi, n_modes=n_modes, sp=tsp)
    finally:
        tbmi.rot_rpm = original_rpm
    # Defensive: mirror the blade-sweep "too few modes" guard. The
    # symmetric ``eigh`` path always returns exactly ``n_modes`` rows,
    # but the rare general-eig fallback (floating ``PlatformSupport``
    # with non-symmetric ``hydro_K`` / ``mooring_K``) can drop NaN rows
    # from a degenerate eigenproblem and return fewer. Without this
    # guard the downstream ``np.broadcast_to`` call would raise a
    # cryptic shape error; we want the same friendly diagnostic the
    # blade path emits.
    if len(modal.frequencies) < n_modes:
        raise RuntimeError(
            f"campbell_sweep: tower solve returned only "
            f"{len(modal.frequencies)} of the requested {n_modes} "
            f"modes — typically a sign of a near-degenerate "
            f"eigenproblem (floating tower with a non-symmetric "
            f"PlatformSupport block). Reduce ``n_tower_modes``."
        )
    tshapes = list(modal.shapes[:n_modes])
    tfreqs = np.asarray(modal.frequencies[:n_modes], dtype=float)
    tparts = np.array([_participation(s) for s in tshapes])

    freqs = np.broadcast_to(tfreqs, (n_steps, n_modes)).copy()
    parts = np.broadcast_to(tparts, (n_steps, n_modes, 3)).copy()
    # Prefer the FEM's own platform-mode classification for a floating
    # tower (``ModalResult.mode_labels`` — populated only for
    # ``hub_conn == 2``); fall back to participation argmax for
    # cantilever / monopile towers and for flexible bending modes.
    labels = _label_tower_modes_with_overrides(tparts, modal.mode_labels)
    return freqs, parts, labels


def campbell_sweep(
    input_path: "str | pathlib.Path | RotatingBlade | Tower",
    omega_rpm: np.ndarray,
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
    *,
    tower_input: "str | pathlib.Path | Tower | None" = None,
    track_by_mac: bool = True,
) -> CampbellResult:
    """Build a Campbell-diagram dataset for the given turbine.

    Parameters
    ----------
    input_path :
        Either a path **or an already-loaded model** (issue #51):

        - an OpenFAST ElastoDyn main ``.dat`` file — the function
          loads the blade *and* the tower from the deck and runs both;
        - a blade ``.bmi`` (``beam_type = 1``) — blade-only sweep
          unless ``tower_input`` is also supplied;
        - a tower ``.bmi`` (``beam_type = 2``) — tower-only result
          (frequencies are constant across ``omega_rpm``; the result
          is mostly useful for overlay against the per-rev family);
        - an already-constructed :class:`~pybmodes.models.RotatingBlade`
          or :class:`~pybmodes.models.Tower` (from *any* constructor —
          ``__init__``, ``from_elastodyn``, ``from_windio``,
          ``from_windio_floating``, …). The model is used **verbatim,
          with no disk re-read**, so a single load point feeds both
          ``.run()`` and the sweep, and a ``from_windio`` /
          ``from_elastodyn`` model (whose section properties no path
          can re-read) can finally be swept. Routed to blade/tower by
          its ``beam_type`` so either may be passed here.
    omega_rpm :
        1-D array of rotor speeds in rpm. ``Ω = 0`` is fine and
        produces the parked-rotor frequencies.
    n_blade_modes :
        Number of blade modes to extract per speed and report in
        ``frequencies[:, :n_blade_modes]``. Default 4 covers
        1st/2nd flap and 1st/2nd edge — the modes that actually drive
        resonance design. Pushing this much higher just adds
        high-order flap modes that no realistic per-rev family
        crosses inside the operating envelope; raise it deliberately
        when you need them.
    n_tower_modes :
        Number of tower modes (default 4 — 1st/2nd FA + 1st/2nd SS).
        Drop to 2 to overlay only the 1st FA + 1st SS pair, or push
        higher for offshore decks where 3rd-mode crossings matter.
        Ignored when no tower model is available.
    tower_input :
        Optional explicit tower — a tower ``.bmi`` path **or a loaded
        :class:`~pybmodes.models.Tower`** (keyword-only). Useful when
        ``input_path`` is a blade-only deck or a loaded
        ``RotatingBlade``. Overrides the deck-supplied tower if
        ``input_path`` was an ElastoDyn ``.dat``. So the
        single-load-point form is
        ``campbell_sweep(blade, omega, tower_input=tower)``.
    track_by_mac :
        Whether to use MAC across consecutive rotor speeds to keep
        each blade output column corresponding to the same physical
        mode. ``False`` returns the eigensolver's native order (useful
        for debugging mode re-ordering issues). Tower modes don't
        change with rotor speed and are unaffected by this flag.

    Returns
    -------
    :class:`CampbellResult`.
    """
    # ``_load_models`` accepts a path *or* a loaded RotatingBlade /
    # Tower for either argument (issue #51) — do not coerce to Path
    # here (that would break a model object).
    blade, tower = _load_models(input_path, tower_input)

    omega_rpm = np.asarray(omega_rpm, dtype=float).ravel()
    if omega_rpm.size == 0:
        raise ValueError("omega_rpm must contain at least one rotor speed")
    if not np.all(np.isfinite(omega_rpm)):
        raise ValueError(
            "omega_rpm must be finite; found NaN or inf in "
            f"{omega_rpm.tolist()!r}"
        )
    if np.any(omega_rpm < 0.0):
        raise ValueError(
            "omega_rpm must be non-negative (rotor speeds in rpm); found "
            f"min = {float(omega_rpm.min())!r}"
        )
    if omega_rpm.size >= 2 and np.any(np.diff(omega_rpm) < 0.0):
        raise ValueError(
            "omega_rpm must be sorted ascending so MAC tracking can pair "
            "consecutive steps; got "
            f"{omega_rpm.tolist()!r}"
        )
    if not isinstance(n_blade_modes, int) or n_blade_modes < 0:
        raise ValueError(
            f"n_blade_modes must be a non-negative integer; got {n_blade_modes!r}"
        )
    if not isinstance(n_tower_modes, int) or n_tower_modes < 0:
        raise ValueError(
            f"n_tower_modes must be a non-negative integer; got {n_tower_modes!r}"
        )

    # Silently zero-out mode counts for components that aren't present —
    # easier on the caller than raising for the common "no tower" case.
    if blade is None:
        n_blade_modes = 0
    if tower is None:
        n_tower_modes = 0
    if n_blade_modes + n_tower_modes < 1:
        raise ValueError(
            "no modes to compute: input had neither a blade nor a tower "
            "component, or both n_blade_modes and n_tower_modes were 0"
        )

    n_steps = omega_rpm.size
    blade_freqs = blade_parts = None
    blade_labels: list[str] = []
    blade_mac: np.ndarray | None = None
    if blade is not None and n_blade_modes > 0:
        blade_freqs, blade_parts, blade_labels, blade_mac = _solve_blade_sweep(
            blade, omega_rpm, n_blade_modes, track_by_mac,
        )

    tower_freqs = tower_parts = None
    tower_labels: list[str] = []
    if tower is not None and n_tower_modes > 0:
        tower_freqs, tower_parts, tower_labels = _solve_tower_once(
            tower, n_tower_modes, n_steps,
        )

    parts_pieces = [a for a in (blade_parts, tower_parts) if a is not None]
    freqs_pieces = [a for a in (blade_freqs, tower_freqs) if a is not None]
    frequencies = np.concatenate(freqs_pieces, axis=1)
    participation = np.concatenate(parts_pieces, axis=1)
    labels = blade_labels + tower_labels

    # Build the per-step MAC table: blade columns get the tracked
    # MACs from the sweep; tower columns are NaN (no rotor-speed
    # dependence, so a MAC confidence is not meaningful).
    mac_pieces: list[np.ndarray] = []
    if blade_mac is not None:
        mac_pieces.append(blade_mac)
    if tower_freqs is not None:
        mac_pieces.append(np.full((n_steps, n_tower_modes), np.nan))
    if mac_pieces:
        mac_to_previous = np.concatenate(mac_pieces, axis=1)
    else:
        mac_to_previous = np.empty((n_steps, 0))

    return CampbellResult(
        omega_rpm=omega_rpm,
        frequencies=frequencies,
        labels=labels,
        participation=participation,
        n_blade_modes=n_blade_modes,
        n_tower_modes=n_tower_modes,
        mac_to_previous=mac_to_previous,
    )
