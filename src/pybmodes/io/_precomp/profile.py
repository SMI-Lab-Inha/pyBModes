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

"""Airfoil profile geometry for the composite blade reduction
(issue #35).

A WindIO airfoil is a closed polyline in ``coordinates.{x, y}``,
chord-normalised, ordered **trailing edge → suction side → leading
edge → pressure side → trailing edge**. WindIO locates every
structural region (layer / web) by a *normalised arc coordinate*
``nd_arc`` ∈ [0, 1] that runs the perimeter in exactly that order
(``s = 0`` at the TE, ``s ≈ s_LE`` at the LE, ``s = 1`` back at the
TE). This module turns the raw coordinate list into:

* the cumulative ``nd_arc`` ↔ ``(x, y)`` map (``arc_to_xy`` and the
  per-vertex ``s`` table) — the spine the parametric arc resolver
  and the thin-wall reduction walk;
* an upper/lower-surface ``y(x)`` split on a common cosine-spaced
  chord grid — the sector representation the PreComp reduction needs;
* a chord-fraction → ``nd_arc`` map per surface (``arc_of_chord``) —
  used to resolve ``midpoint_nd_arc.fixed: LE`` + ``width`` regions
  and web ``offset_y_pa`` attachments;
* a spanwise :meth:`blend` between two airfoils.

Clean-room reimplementation (pure ``numpy``, WindIO-coordinate-native,
different API) of the algorithm in NREL *PreComp* / WISDEM
``wisdem/precomp/profile.py``; the upstream file is studied as the
reference, not vendored (independence stance; see ``VALIDATION.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalised(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chord-normalise: LE (min x) → origin, chord → 1, no rescale of an
    already-unit polyline (idempotent to round-off)."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    i_le = int(np.argmin(x))
    x -= x[i_le]
    y -= y[i_le]
    chord = x.max() - x.min()
    if chord <= 0.0:
        raise ValueError("degenerate airfoil: zero chord extent")
    return x / chord, y / chord


@dataclass
class Profile:
    """A chord-normalised airfoil with its WindIO ``nd_arc`` spine.

    ``xc, yc`` — the closed perimeter polyline (TE→SS→LE→PS→TE),
    chord-normalised. ``s`` — cumulative normalised arc length at each
    vertex, ``s[0] = 0`` (TE), ``s[-1] = 1`` (TE). ``s_le`` — the
    ``nd_arc`` of the leading edge. ``x_grid, yu, yl`` — upper
    (suction) / lower (pressure) surface on a shared cosine chord grid
    (the PreComp sector representation).
    """

    xc: np.ndarray
    yc: np.ndarray
    s: np.ndarray
    s_le: float
    x_grid: np.ndarray
    yu: np.ndarray
    yl: np.ndarray

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_windio_coords(
        cls, x, y, *, n_chord: int = 200
    ) -> "Profile":
        """Build from a WindIO airfoil ``coordinates.{x, y}`` (TE→TE,
        chord-normalised, closed loop)."""
        xc, yc = _normalised(x, y)
        # Perimeter arc length, normalised to [0, 1].
        seg = np.hypot(np.diff(xc), np.diff(yc))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total <= 0.0:
            raise ValueError("degenerate airfoil: zero perimeter")
        s = s / total

        i_le = int(np.argmin(xc))
        s_le = float(s[i_le])

        # Upper (suction, TE→LE half) and lower (pressure, LE→TE half),
        # re-expressed as y(x) on a shared cosine-spaced chord grid so
        # the reduction can integrate per chordwise sector. The WindIO
        # loop starts at the TE on the suction side, so 0..i_le is the
        # suction half (x decreasing) and i_le..end the pressure half.
        beta = np.linspace(0.0, np.pi, n_chord)
        x_grid = 0.5 * (1.0 - np.cos(beta))            # 0 (LE) → 1 (TE)

        xu = xc[: i_le + 1][::-1]                       # LE→TE, x ascending
        yu = yc[: i_le + 1][::-1]
        xl = xc[i_le:]                                  # LE→TE, x ascending
        yl = yc[i_le:]
        # Guard monotonic x for np.interp (round-off at the blunt TE).
        yu = np.interp(x_grid, _monotone(xu), yu)
        yl = np.interp(x_grid, _monotone(xl), yl)
        # Orient so yu is the suction (upper, larger-y) surface.
        if np.mean(yu) < np.mean(yl):
            yu, yl = yl, yu

        return cls(xc=xc, yc=yc, s=s, s_le=s_le,
                   x_grid=x_grid, yu=yu, yl=yl)

    # ---- arc <-> geometry maps ------------------------------------------

    def arc_to_xy(self, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the perimeter polyline at ``nd_arc`` value(s)."""
        sq = np.asarray(s_query, dtype=float)
        x = np.interp(sq, self.s, self.xc)
        y = np.interp(sq, self.s, self.yc)
        return x, y

    def arc_of_chord(self, x_over_c: float, *, side: str) -> float:
        """``nd_arc`` of a chordwise fraction ``x/c`` on a given surface.

        ``side`` ∈ {``"suction"``, ``"pressure"``}. Used to resolve
        ``fixed: LE`` (x/c = 0) / ``fixed: TE`` (x/c = 1) anchors,
        ``midpoint_nd_arc`` + ``width`` bands, and web ``offset_y_pa``
        attachments. The suction half is the TE→LE run ``s ∈ [0, s_le]``
        (x decreasing); the pressure half ``s ∈ [s_le, 1]`` (x
        increasing).
        """
        xq = float(np.clip(x_over_c, 0.0, 1.0))
        i_le = int(np.argmin(self.xc))
        if side == "suction":
            xs = self.xc[: i_le + 1]                    # TE(x=1)→LE(x=0)
            ss = self.s[: i_le + 1]
            return float(np.interp(xq, xs[::-1], ss[::-1]))
        if side == "pressure":
            xs = self.xc[i_le:]                          # LE(x=0)→TE(x=1)
            ss = self.s[i_le:]
            return float(np.interp(xq, xs, ss))
        raise ValueError(f"side must be 'suction' or 'pressure'; got {side!r}")

    # ---- spanwise blend --------------------------------------------------

    def blend(self, other: "Profile", weight: float) -> "Profile":
        """Linear blend toward ``other`` (``weight`` 0→self, 1→other),
        on this profile's shared chord grid (the spanwise airfoil
        interpolation)."""
        w = float(weight)
        yu = self.yu + w * (other.yu - self.yu)
        yl = self.yl + w * (other.yl - self.yl)
        return _from_surfaces(self.x_grid, yu, yl)

    @property
    def tc(self) -> float:
        """Thickness-to-chord ratio (max of upper − lower)."""
        return float(np.max(self.yu - self.yl))


def _monotone(x: np.ndarray) -> np.ndarray:
    """Nudge a near-monotone chord array strictly increasing so
    ``np.interp`` (which needs increasing xp) is well-posed at the
    blunt-TE coincident points."""
    x = np.asarray(x, dtype=float).copy()
    eps = 1e-12
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x


def _from_surfaces(
    x_grid: np.ndarray, yu: np.ndarray, yl: np.ndarray
) -> Profile:
    """Rebuild a :class:`Profile` (incl. the ``nd_arc`` spine) from an
    upper/lower-surface pair on a shared chord grid — used by
    :meth:`Profile.blend`."""
    # Reconstruct the TE→SS→LE→PS→TE loop: suction TE→LE then pressure
    # LE→TE (drop the duplicated LE point).
    xc = np.concatenate([x_grid[::-1], x_grid[1:]])
    yc = np.concatenate([yu[::-1], yl[1:]])
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / s[-1]
    i_le = int(np.argmin(xc))
    return Profile(xc=xc, yc=yc, s=s, s_le=float(s[i_le]),
                   x_grid=x_grid, yu=yu, yl=yl)
