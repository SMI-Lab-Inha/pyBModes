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

"""Centralised numerical options for pyBmodes.

Three frozen dataclasses gather the magic numbers that used to live as
module-level constants scattered across the FEM solver, the polynomial
fitter, and the pre-solve sanity checker. Each carries the same default
the original constant did, so importing this module changes no
behaviour; the value is that callers (and reviewers) can now find every
numerical threshold in one place.

The dataclasses are :func:`dataclasses.dataclass(frozen=True)` so they
hash, compare structurally, and can't be mutated after construction —
suitable as defaults on public APIs and as keys in caches.

Examples
--------

Read the default thresholds::

    from pybmodes.options import SolverOptions, FitOptions, CheckOptions

    SolverOptions().sparse_ndof_threshold     # 500
    FitOptions().polynomial_rms_threshold     # 0.09
    CheckOptions().stiffness_jump_factor      # 5.0

Override one threshold while keeping the rest at default::

    custom = SolverOptions(sparse_ndof_threshold=2000)

The dataclasses are intentionally **additive**: future fields gain
sensible defaults so existing call sites stay working. Removing or
renaming a field is a semver-major change (see :doc:`api_contract`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverOptions:
    """FEM solver dispatch + matrix-symmetry tolerances.

    Attributes
    ----------
    sparse_ndof_threshold : int, default 500
        Above this many reduced-system DOFs and when a small subset of
        modes is requested, the solver routes through
        :func:`scipy.sparse.linalg.eigsh` shift-invert instead of the
        dense :func:`scipy.linalg.eigh`. 5-18 x speedup on real-tower
        meshes; below the threshold the dense path's LAPACK back-end is
        faster.
    symmetry_rtol : float, default 1e-12
        Relative tolerance (multiplied by ``max|matrix|``) for treating
        an assembled stiffness or mass matrix as symmetric. Asymmetry
        beyond this routes the eigenproblem through the general dense
        :func:`scipy.linalg.eig` instead of the symmetric
        :func:`scipy.linalg.eigh`. The OC3 Hywind cross-coupled
        ``hydro_K + mooring_K`` exercises this branch.
    """

    sparse_ndof_threshold: int = 500
    symmetry_rtol: float = 1.0e-12


@dataclass(frozen=True)
class FitOptions:
    """Polynomial-fit + family-selection thresholds.

    Attributes
    ----------
    polynomial_rms_threshold : float, default 0.09
        Maximum RMS residual a clamped-base polynomial fit is allowed
        to have for its mode to count as a viable FA / SS family
        candidate (after the tangent-line root-rigid subtraction). The
        ``_select_tower_family`` algorithm drops candidates above this.
    torsion_contamination_threshold : float, default 0.10
        Drops candidates whose modal-kinetic-energy torsion fraction
        ``T_tor = sum(phi_tor**2) / sum(phi_total**2)`` exceeds this.
        Hybrid bending + twist modes (T_tor in the 1-3 % range are
        "near-pure-bending") cannot be expressed by the polynomial
        ansatz; the filter keeps them out of the FA / SS selection.
    fit_cond_warn : float, default 1e4
        Condition-number ceiling above which the polynomial-fit
        design-matrix conditioning emits a ``RuntimeWarning``. Suggests
        numerical sensitivity of the polynomial-coefficient solve to
        perturbations in the input mode shape.
    fit_cond_fail : float, default 1e6
        Condition-number ceiling above which the polynomial fit is
        flagged as FAIL by :func:`pybmodes.checks.check_model`. The
        reconstructed shape may still be a good visual fit, but the
        coefficient values are not trustworthy beyond a couple of
        significant figures.
    """

    polynomial_rms_threshold: float = 0.09
    torsion_contamination_threshold: float = 0.10
    fit_cond_warn: float = 1.0e4
    fit_cond_fail: float = 1.0e6


@dataclass(frozen=True)
class CheckOptions:
    """Pre-solve sanity-check thresholds (consumed by
    :func:`pybmodes.checks.check_model`).

    Attributes
    ----------
    stiffness_jump_factor : float, default 5.0
        Maximum allowed ratio of bending stiffness between adjacent
        section nodes (forward and backward). A jump above this flags
        a likely transition-piece discontinuity that the polynomial
        fit will struggle to represent — typically wants extra mesh
        refinement around it.
    ei_ratio_min : float, default 0.1
        Lower bound on ``EI_FA / EI_SS`` at every section node. Below
        this is unphysical for a real tower section; the structural
        inputs have probably swapped FA / SS.
    ei_ratio_max : float, default 10.0
        Upper bound on ``EI_FA / EI_SS``. Symmetric to ``ei_ratio_min``.
    support_asymmetry_rtol : float, default 1e-6
        Tolerance (relative to ``max|support|``) for treating a
        ``PlatformSupport`` 6 x 6 matrix as symmetric. Above this, the
        check warns that the support matrix is asymmetric, which is
        load-bearing for OC3-Hywind-style cross-coupled platforms but
        commonly indicates a deck-assembly bug for axisymmetric ones.
    fit_cond_warn : float, default 1e4
        Polynomial-fit design-matrix condition above which check_model
        emits a WARN. Mirrors :class:`FitOptions.fit_cond_warn` so the
        check can run independently of the fit module.
    fit_cond_fail : float, default 1e6
        Polynomial-fit design-matrix condition above which check_model
        emits a FAIL. Mirrors :class:`FitOptions.fit_cond_fail`.
    platform_cm_offset_gyradius_factor : float, default 1.0
        The horizontal platform CM offset (``cm_pform_x`` / ``cm_pform_y``
        on a ``PlatformSupport``) is flagged when its magnitude exceeds
        this factor times the platform's yaw radius of gyration
        ``√(I_yaw / m)``. ``cm_pform_x`` / ``cm_pform_y`` are the CM
        offset *from the tower axis*; a value comparable to the
        platform's own size is almost always a coordinate-origin error
        leaking into the field, which injects spurious surge/sway↔yaw
        coupling and mislabels the rigid-body modes (issue #95).
    """

    stiffness_jump_factor: float = 5.0
    ei_ratio_min: float = 0.1
    ei_ratio_max: float = 10.0
    support_asymmetry_rtol: float = 1.0e-6
    fit_cond_warn: float = 1.0e4
    fit_cond_fail: float = 1.0e6
    platform_cm_offset_gyradius_factor: float = 1.0


# Module-level default instances. Internal call sites that don't (yet)
# accept an ``options=`` kwarg read from these singletons. Future PRs
# will thread the dataclasses through the public-API call sites so
# users can override per-call.
DEFAULT_SOLVER_OPTIONS = SolverOptions()
DEFAULT_FIT_OPTIONS = FitOptions()
DEFAULT_CHECK_OPTIONS = CheckOptions()


__all__ = [
    "DEFAULT_CHECK_OPTIONS",
    "DEFAULT_FIT_OPTIONS",
    "DEFAULT_SOLVER_OPTIONS",
    "CheckOptions",
    "FitOptions",
    "SolverOptions",
]
