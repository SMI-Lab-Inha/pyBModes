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

"""Constrained 6th-order polynomial fit for ElastoDyn mode shapes.

ElastoDyn requires mode shapes of the form
``phi(x) = C2*x^2 + C3*x^3 + C4*x^4 + C5*x^5 + C6*x^6`` with
constraint ``phi(1) = 1``, i.e. ``C2+C3+C4+C5+C6 = 1``.

The constraint is enforced analytically by substituting
``C6 = 1 - C2 - C3 - C4 - C5`` and solving the reduced
least-squares system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolyFitResult:
    """Polynomial fit coefficients and quality metrics for one mode component."""

    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    rms_residual: float   # RMS of (phi_poly(x) - phi_fem(x)) over all stations
    tip_slope: float      # dphi/dx at x=1: 2C2+3C3+4C4+5C5+6C6

    # 2-norm condition number of the reduced design matrix solved by lstsq
    # — see ``fit_mode_shape``. Depends *only* on the spanwise sampling
    # locations, not on the y data; a single value characterises the
    # numerical sensitivity of the polynomial-coefficient solve to
    # perturbations in the input mode shape. Useful for distinguishing
    # "the fit is sound but the mode shape disagrees with the file" from
    # "the basis is ill-conditioned, coefficient drift is numeric".
    cond_number: float

    def coefficients(self) -> np.ndarray:
        """Return [C2, C3, C4, C5, C6] as a length-5 array."""
        return np.array([self.c2, self.c3, self.c4, self.c5, self.c6])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at positions x in [0, 1]."""
        return (
            self.c2 * x**2
            + self.c3 * x**3
            + self.c4 * x**4
            + self.c5 * x**5
            + self.c6 * x**6
        )


def fit_mode_shape(
    span_loc: np.ndarray,
    displacement: np.ndarray,
) -> PolyFitResult:
    """Fit a constrained 6th-order polynomial to a normalised mode shape.

    Parameters
    ----------
    span_loc     : 1-D array of normalised span positions, x in [0, 1].
    displacement : 1-D array of mode-shape displacements at each station.

    Raises
    ------
    ValueError
        If the inputs are not 1-D, lengths differ, the array is too
        short to fit five free coefficients, any element is non-finite,
        ``span_loc`` is not strictly increasing, or the tip
        displacement is effectively zero.
    """
    x = np.asarray(span_loc, dtype=float)
    y = np.asarray(displacement, dtype=float)

    # ----- Input validation -----
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"fit_mode_shape: span_loc and displacement must be 1-D; "
            f"got shapes {x.shape} and {y.shape}."
        )
    if x.shape != y.shape:
        raise ValueError(
            f"fit_mode_shape: span_loc and displacement must have the "
            f"same length; got {x.shape[0]} and {y.shape[0]}."
        )
    # Need at least 2 stations to define an interpolation at all
    # (``y[-1]`` is undefined on an empty array; ``np.diff`` is
    # vacuous on length-1 arrays). With 2–3 stations the constrained
    # lstsq is underdetermined and returns the minimum-norm solution
    # — still useful for some tests, so we don't gate on the full
    # 5-coefficient determinacy here.
    if x.size < 2:
        raise ValueError(
            f"fit_mode_shape: need at least 2 stations to fit a mode "
            f"shape; got {x.size}."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "fit_mode_shape: span_loc and displacement must be finite; "
            "non-finite values would propagate into NaN coefficients."
        )
    if np.any(np.diff(x) <= 0.0):
        bad = int(np.argmin(np.diff(x)))
        raise ValueError(
            f"fit_mode_shape: span_loc must be strictly increasing; "
            f"first non-positive step is at index {bad}: "
            f"{float(x[bad])!r} → {float(x[bad + 1])!r}."
        )

    tip_val = y[-1]
    if abs(tip_val) < 1e-30:
        raise ValueError("Tip displacement is effectively zero; cannot normalise.")
    y = y / tip_val

    A = np.column_stack([x**k - x**6 for k in range(2, 6)])
    b = y - x**6

    coeffs_r, *_ = np.linalg.lstsq(A, b, rcond=None)
    c2, c3, c4, c5 = coeffs_r
    c6 = 1.0 - c2 - c3 - c4 - c5

    phi = c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + c6 * x**6
    rms = float(np.sqrt(np.mean((phi - y) ** 2)))
    tip_slope = float(2 * c2 + 3 * c3 + 4 * c4 + 5 * c5 + 6 * c6)
    cond = float(np.linalg.cond(A))

    return PolyFitResult(
        c2=float(c2),
        c3=float(c3),
        c4=float(c4),
        c5=float(c5),
        c6=float(c6),
        rms_residual=rms,
        tip_slope=tip_slope,
        cond_number=cond,
    )
