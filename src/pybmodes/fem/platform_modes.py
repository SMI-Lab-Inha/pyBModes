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

"""Name the floating-platform rigid-body modes.

For a free-free tower (``hub_conn = 2``) carrying a
:class:`~pybmodes.io.bmi.PlatformSupport`, the low modes are the
platform rigid-body modes — surge / sway / heave / roll / pitch / yaw.
They are identifiable from the tower-base node's 6-DOF motion in the
eigenvector, weighted by the platform 6×6 inertia (which supplies the
mass / moment-of-inertia metric that makes a translation amplitude
comparable to a rotation amplitude).

The DOF labels are assigned *globally*: the six rigid-body candidates
and the six platform DOFs are matched by a Hungarian optimal
assignment over the per-DOF modal-kinetic-energy fractions, so the
mode that best expresses each DOF wins that label. The earlier greedy
"argmax per mode, drop later duplicates" rule mislabelled asymmetric
platforms — an early mode whose mass-weighted energy was inflated by a
parasitic high-inertia DOF (a small coupled yaw rotation reads as huge
energy) would steal that DOF's label, starve the true owner into
``None``, and cascade onto the remaining modes (issue #93). The global
assignment removes that order/greed sensitivity; on symmetric decks,
where every rigid-body mode is ~98 % single-DOF, it reduces to the
same per-DOF matching as before.

Before the assignment, frequency-degenerate rigid pairs are rotated
onto axis-aligned directions. On a (bi)symmetric platform surge≈sway
and roll≈pitch share an eigenvalue, so the eigensolver may return any
rotation of that 2-D eigenspace — a non-deterministic basis (BLAS
thread-order dependent) that would otherwise leave a 45°-mixed pair
below the dominance threshold and unnamed. Rotating each degenerate
pair onto its platform axes (the rigid-body analog of the FA/SS
resolver in ``pybmodes.elastodyn.params``) makes the labelling
deterministic. Asymmetric platforms break the degeneracy, so this step
is a no-op there and the global assignment carries the work.

The classifier stays deliberately *conservative*: after assignment a
mode is named only when its assigned platform DOF clearly dominates
the base-node modal kinetic energy (``_DOMINANCE_THRESHOLD``). A
flexible tower mode, or a genuinely strongly-coupled pair whose energy
splits across DOFs, is left ``None`` rather than mislabelled —
consistent with the project's "only name what's unambiguous" stance.
Empirically every genuine rigid-body mode on the validated floating
decks (OC3 Hywind, the IEA-15 / IEA-22 / OC4 / UPSCALE samples) is
overwhelmingly single-DOF and is named; the first flexible bending
pair and above stay ``None``.
"""

from __future__ import annotations

import numpy as np

from pybmodes.fem.boundary import NESH

# Platform DOF names in OpenFAST order.
_PLATFORM_DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

# FEM base-node DOF order is [axial, v_disp, v_slope, w_disp, w_slope,
# phi]; the platform (file) order is [surge, sway, heave, roll, pitch,
# yaw]. This index list reorders an FEM-ordered base 6-vector into
# platform order — the inverse of the ``P`` reorder in
# ``pybmodes.fem.nondim._rigid_arm_T``:
#   surge ← v_disp(1) · sway ← w_disp(3) · heave ← axial(0)
#   roll  ← w_slope(4) · pitch ← v_slope(2) · yaw ← phi(5)
_FEM_TO_PLATFORM = np.array([1, 3, 0, 4, 2, 5])

# A mode is named only if its dominant platform DOF carries at least
# this fraction of the base-node modal kinetic energy. 0.6 cleanly
# separates the genuine (single-DOF) rigid-body modes from
# coupled / rotated ones on every validated floating deck.
_DOMINANCE_THRESHOLD = 0.6

# A 6-DOF rigid platform has exactly 6 rigid-body modes, and for any
# real floating wind system they are the 6 lowest-frequency modes:
# the mooring / hydrostatic restoring is orders of magnitude softer
# than the tower bending stiffness, so the rigid-body periods
# (10–100 s) sit far below the first flexible tower mode (1–2 s) —
# a large spectral gap on every validated deck (OC3 0.12→0.48 Hz,
# IEA-15 0.049→0.525 Hz). Only the lowest ``_N_RIGID`` modes are
# rigid-body candidates; a free-free flexible bending mode also moves
# the base and would otherwise be mis-named.
_N_RIGID = 6

# Two rigid-body modes whose relative frequency gap is below this are
# treated as a degenerate pair: on a (bi)symmetric platform surge≈sway
# and roll≈pitch share an eigenvalue, so the eigensolver returns an
# arbitrary rotation of that 2-D eigenspace (non-deterministic across
# BLAS thread orderings — the same hazard commit-fixed for the FA/SS
# tower pair). Matches ``_DEGENERATE_FREQ_RTOL`` in
# ``pybmodes.elastodyn.params``.
_DEGENERATE_FREQ_RTOL = 1e-4


def _platform_dof_energy(b: np.ndarray, Mp: np.ndarray) -> np.ndarray:
    """Per-platform-DOF modal kinetic energy of a base 6-vector ``b``
    (FEM order), reordered to platform DOF order. Not normalised.

    The platform mass matrix is the metric that puts a translation
    amplitude and a rotation amplitude on a comparable footing
    (``b_i·(M_p b)_i``).
    """
    e = np.abs(b * (Mp @ b))
    return np.asarray(e[_FEM_TO_PLATFORM], dtype=float)


def _align_degenerate_rigid_pairs(
    base: np.ndarray, freqs: np.ndarray, Mp: np.ndarray
) -> np.ndarray:
    """Rotate near-degenerate rigid-body base-motion pairs onto axis-
    aligned directions.

    ``base`` is ``(6, n_rigid)`` of tower-base 6-vectors (FEM order),
    ``freqs`` the matching frequencies. Walks consecutive pairs; when a
    pair is frequency-degenerate (``_DEGENERATE_FREQ_RTOL``) it rotates
    the 2-D eigenspace — any in-plane rotation is an equally valid
    eigenpair — to put the pair's dominant platform DOF entirely in the
    first slot and the orthogonal partner in the second. The closed-form
    angle ``θ = ½·arctan2(2ab, a²−b²)`` (``a``, ``b`` the two modes'
    components in the target FEM DOF) maximises that DOF's content in
    the first slot, identical in form to the FA/SS resolver in
    ``pybmodes.elastodyn.params``.

    The rotation is accepted only when it *cleanly* separates the pair —
    both rotated modes dominated (``_DOMINANCE_THRESHOLD``) by *different*
    platform DOFs. Otherwise the pair is genuinely coupled (an asymmetric
    floater whose degeneracy is broken anyway, so this branch rarely
    triggers there) and the originals are kept for the global assignment
    to handle. The input is not mutated; a fresh copy is returned.
    """
    out = base.copy()
    # Walk only pairs covered by BOTH a column and a frequency, so a
    # caller passing fewer frequencies than columns can't index past the
    # end of ``freqs`` (defense-in-depth for the guard in the caller).
    n = min(out.shape[1], int(np.asarray(freqs).shape[0]))
    i = 0
    while i < n - 1:
        denom = max(abs(float(freqs[i])), abs(float(freqs[i + 1])), 1e-12)
        if abs(float(freqs[i]) - float(freqs[i + 1])) / denom <= _DEGENERATE_FREQ_RTOL:
            bi = out[:, i].copy()
            bj = out[:, i + 1].copy()
            # Target the pair's combined-dominant platform DOF; f1 is the
            # FEM DOF that platform DOF maps from.
            combined = _platform_dof_energy(bi, Mp) + _platform_dof_energy(bj, Mp)
            f1 = int(_FEM_TO_PLATFORM[int(np.argmax(combined))])
            a, b = float(bi[f1]), float(bj[f1])
            theta = 0.5 * float(np.arctan2(2.0 * a * b, a * a - b * b))
            c, s = np.cos(theta), np.sin(theta)
            ri = c * bi + s * bj
            rj = -s * bi + c * bj

            ei = _platform_dof_energy(ri, Mp)
            ej = _platform_dof_energy(rj, Mp)
            ti, tj = float(ei.sum()), float(ej.sum())
            if ti > 0.0 and tj > 0.0:
                fi, fj = ei / ti, ej / tj
                ki, kj = int(np.argmax(fi)), int(np.argmax(fj))
                if (
                    ki != kj
                    and fi[ki] >= _DOMINANCE_THRESHOLD
                    and fj[kj] >= _DOMINANCE_THRESHOLD
                ):
                    out[:, i], out[:, i + 1] = ri, rj
                    i += 2
                    continue
        i += 1
    return out


def classify_platform_modes(
    eigvecs: np.ndarray,
    active_dofs: np.ndarray,
    nselt: int,
    platform_mass: np.ndarray,
    frequencies: np.ndarray | None = None,
) -> list[str | None]:
    """Return a per-mode label list naming the platform rigid-body
    modes (``surge`` / … / ``yaw``) or ``None`` where no single
    platform DOF dominates.

    Parameters
    ----------
    eigvecs : (ngd, n_modes) compact (active-DOF) eigenvectors, the
        array :func:`pybmodes.fem.solver.solve_modes` returns.
    active_dofs : (ngd,) sorted global indices of the active DOFs
        (from :func:`pybmodes.fem.boundary.active_dof_indices`), used
        to scatter the compact eigenvector back to full DOF size — the
        same expansion :func:`pybmodes.fem.normalize.extract_mode_shapes`
        performs.
    nselt : number of beam elements.
    platform_mass : the platform 6×6 inertia at the tower base in FEM
        DOF order (``PlatformND.mass`` from
        :func:`pybmodes.fem.nondim.nondim_platform`). Supplies the
        mass / inertia metric for the energy weighting.
    frequencies : (n_modes,) modal frequencies (any unit), ascending.
        When given, frequency-degenerate rigid-body pairs (surge≈sway,
        roll≈pitch on a symmetric platform) are rotated onto axis-
        aligned directions before labelling so the result is
        deterministic regardless of the arbitrary basis the eigensolver
        returns within a degenerate eigenspace. When ``None`` the
        rotation is skipped (the labels then rely on the global
        assignment alone).

    Caller must invoke this only for a floating model
    (``hub_conn == 2`` with a ``PlatformSupport``); for any other
    model there are no rigid-body modes to name.
    """
    from scipy.optimize import linear_sum_assignment

    ndt = NESH * nselt + 6
    n_modes = eigvecs.shape[1]

    ev_full = np.zeros((ndt, n_modes))
    ev_full[active_dofs, :] = eigvecs

    root_base = NESH * nselt          # base-node block start
    base = ev_full[root_base:root_base + 6, :]   # (6, n_modes), FEM order

    Mp = np.asarray(platform_mass, dtype=float)

    # Modes are returned ascending in frequency, so the rigid-body
    # candidates are the first _N_RIGID columns.
    n_rigid = min(_N_RIGID, n_modes)

    labels: list[str | None] = [None] * n_modes
    if n_rigid == 0:
        return labels

    # Resolve symmetric-platform degeneracies first: rotate surge≈sway
    # and roll≈pitch pairs onto axis-aligned directions so the labelling
    # doesn't depend on which (equally valid) rotation the eigensolver
    # happened to return inside the degenerate eigenspace. Only run when
    # ``frequencies`` actually covers the rigid block — a caller passing a
    # truncated frequency array (e.g. an externally-built mode subset)
    # would otherwise index past its end (Codex P2); in that case skip the
    # alignment and let the global assignment carry the labelling.
    work = base[:, :n_rigid].astype(float, copy=True)
    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=float)
        if freqs.shape[0] >= n_rigid:
            work = _align_degenerate_rigid_pairs(work, freqs[:n_rigid], Mp)

    # Per-DOF modal-kinetic-energy fractions for each rigid-body
    # candidate, in platform-DOF order: score[m, k] is the fraction of
    # mode m's base-node modal kinetic energy carried by platform DOF k.
    score = np.zeros((n_rigid, 6))
    for m in range(n_rigid):
        e = _platform_dof_energy(work[:, m], Mp)
        total = float(e.sum())
        if total <= 0.0 or not np.isfinite(total):
            continue                            # inert row → stays zero
        score[m] = e / total

    # Global one-mode-per-DOF matching (Hungarian, maximising total
    # energy fraction), so the mode that best expresses each platform
    # DOF wins that label instead of an earlier mode greedily stealing
    # it and starving the true owner (issue #93). The square 6×6 case
    # gives a perfect matching, so no DOF is ever named twice; a short
    # rigid block (fewer than six modes requested) matches the
    # available rows. Each assignment is then gated by the dominance
    # threshold: a genuinely coupled / rotated pair whose energy splits
    # across DOFs falls below it and is left ``None``.
    rows, cols = linear_sum_assignment(score, maximize=True)
    for m, k in zip(rows, cols):
        if score[m, k] >= _DOMINANCE_THRESHOLD:
            labels[m] = _PLATFORM_DOF_NAMES[k]

    return labels
