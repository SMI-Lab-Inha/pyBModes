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

"""Pre-solve sanity checks for ``Tower`` and ``RotatingBlade`` models.

The :func:`check_model` entry point runs a small suite of cheap,
deterministic checks on a parsed model and returns a list of
:class:`ModelWarning` records describing any anomalies it found. The
list is empty when the model is clean.

The checks run automatically inside :meth:`Tower.run` and
:meth:`RotatingBlade.run` with ``check_model=True`` (the default).
WARN- and ERROR-severity findings are routed through Python's
``warnings`` module so they surface at the call site without changing
the function's return type. INFO-severity findings are surfaced only
when :func:`check_model` is called directly — they're useful context
but not actionable noise on every solve.

Suppress the auto-run with ``Tower(...).run(n_modes, check_model=False)``
(symmetric for ``RotatingBlade.run``). Suppression is meant for
scripted callers that have already validated their input and want
the solver path to stay quiet.

Checks performed (see :func:`check_model` for the details):

0. Every section-property field is finite (no NaN, no ±Inf). Runs
   first so the per-field checks below don't have to be NaN-aware
   (NaN silently passes every ``<=`` / ``>`` / ratio comparison).
1. Span stations are strictly increasing.
2. Mass density is strictly positive at every station.
3. Bending stiffness (FA + SS) does not jump by more than 5× between
   adjacent stations.
4. EI_FA / EI_SS ratio stays within ``[0.1, 10]`` at every station.
5. Tower-top RNA mass is not larger than the integrated tower mass.
6. PlatformSupport 6×6 inertia / hydro / mooring matrices are well-
   formed: shape (6, 6), all entries finite, and symmetric (within
   ``1e-6 · max|A|``). Rank deficiency is **not** flagged — surge /
   sway / yaw hydrostatic restoring is legitimately zero on most
   floaters and mooring layouts can be low-rank by design.
7. The horizontal platform CM offset (``cm_pform_x`` / ``cm_pform_y``)
   does not exceed the platform's yaw radius of gyration
   ``√(I_yaw / m)`` — a larger value is almost always a coordinate-
   origin offset leaking into a field that means "CM offset from the
   tower axis", which mislabels the rigid-body modes (issue #95).
8. Floating platform inertia is physical: positive ``mass_pform`` and
   strictly-positive ``i_matrix`` diagonal (ERROR otherwise).
9. Floating model carries hydrodynamic added mass (``hydro_M`` not all
   zero) — omitting it biases every rigid-body frequency high (WARN).
10. Floating model has *some* restoring (``hydro_K`` or ``mooring_K``
    non-zero); with neither, the rigid-body modes collapse to ~0 Hz
    (WARN).

   Checks 8–10 are the "floating-model readiness" gates: they catch the
   seakeeping omissions a non-specialist makes when a WindIO ``.yaml``
   (geometry + material only) is treated as sufficient for a floating
   system — it is not, the way it is for a land tower (issue #95).
11. The requested ``n_modes`` does not exceed the model's DOF count.
12. The polynomial-fit design matrix on the mesh stations is not
    ill-conditioned (cond > 1e4 ⇒ WARN, > 1e6 ⇒ ERROR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from pybmodes.options import DEFAULT_CHECK_OPTIONS as _CHECK_OPTIONS

if TYPE_CHECKING:
    from pybmodes.io.sec_props import SectionProperties
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.tower import Tower


Severity = Literal["INFO", "WARN", "ERROR"]


@dataclass(frozen=True)
class ModelWarning:
    """One finding from :func:`check_model`.

    Attributes
    ----------
    severity : ``"INFO"``, ``"WARN"``, or ``"ERROR"``. INFO is
        contextual (e.g. "RNA mass dominates the structure"); WARN
        indicates the solve will probably complete but with degraded
        accuracy; ERROR indicates a non-physical input that will
        produce undefined results.
    message : human-readable description of the finding.
    location : dotted path to the offending data, e.g.
        ``"section_properties.mass_den"`` or
        ``"bmi.support.hydro_K"``.
    """

    severity: Severity
    message: str
    location: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.location}: {self.message}"


_Model = Union["Tower", "RotatingBlade"]

# The ``_check_*`` helpers below read their numerical thresholds from
# the module-level ``_CHECK_OPTIONS`` imported at the top of this
# file. Future PRs will accept a ``CheckOptions`` argument on
# :func:`check_model` so users can override per-call; the defaults
# match the previously-hardcoded literals (5.0 / 0.1 / 10.0 / 1e-6 /
# 1e4 / 1e6) so no behaviour change in this PR.


def check_model(
    model: _Model,
    *,
    n_modes: int | None = None,
) -> list[ModelWarning]:
    """Run the full pre-solve check suite on ``model``.

    Parameters
    ----------
    model : a ``Tower`` or ``RotatingBlade`` instance.
    n_modes : optional. When supplied, additionally checks that
        ``n_modes`` doesn't exceed the model's DOF count. Skipped
        when ``None`` (e.g. when callers want to validate the static
        model definition before deciding how many modes to request).

    Returns
    -------
    list of :class:`ModelWarning`
        Empty when every check passes. The list is ordered roughly
        by check number (see module docstring), which keeps the
        output diffable across runs.
    """
    bmi = model._bmi
    sp = _resolve_section_properties(model)
    out: list[ModelWarning] = []

    # The non-finite check runs FIRST so the per-field checks below
    # don't have to be NaN-aware. ``np.diff(span) <= 0`` returns False
    # for NaN entries, ``m <= 0`` returns False for NaN entries, and
    # the stiffness-jump check masks non-finite ratios as 1.0 — so a
    # model with NaN section properties could silently pass every
    # downstream check and enter the eigensolver.
    _check_section_properties_finite(sp, out)
    _check_span_monotonic(sp, out)
    _check_mass_positive(sp, out)
    _check_stiffness_jumps(sp, out)
    _check_ei_ratio(sp, out)
    _check_rna_vs_tower_mass(bmi, sp, out)
    _check_support_conditioning(bmi, out)
    _check_platform_cm_offset(bmi, out)
    _check_platform_inertia_physical(bmi, out)
    _check_added_mass_present(bmi, out)
    _check_restoring_present(bmi, out)
    if n_modes is not None:
        _check_n_modes_vs_dof(bmi, n_modes, out)
    _check_polyfit_conditioning(bmi, out)

    return out


# ---------------------------------------------------------------------------
# Section-property resolution
# ---------------------------------------------------------------------------

def _resolve_section_properties(model: _Model) -> SectionProperties:
    """Return the model's section-properties record, reading it from
    disk if the model is BMI-only and ``_sp`` has not been set."""
    if model._sp is not None:
        return model._sp
    from pybmodes.io.sec_props import read_sec_props
    return read_sec_props(model._bmi.resolve_sec_props_path())


# ---------------------------------------------------------------------------
# Individual checks (each appends 0 or 1 ModelWarning entries to ``out``)
# ---------------------------------------------------------------------------

_SECTION_PROPERTY_FIELDS = (
    "span_loc", "mass_den",
    "flp_iner", "edge_iner", "flp_stff", "edge_stff",
    "tor_stff", "axial_stff",
    "str_tw", "tw_iner",
    "cg_offst", "sc_offst", "tc_offst",
)


def _check_section_properties_finite(
    sp: SectionProperties, out: list[ModelWarning],
) -> None:
    """Flag any non-finite (NaN / ±Inf) entry in the numeric section-
    property fields. ERROR-severity because every downstream
    consumer (``np.trapezoid``, ``np.linalg.eigh``, the FE assembly)
    silently produces NaN-filled outputs on NaN inputs.

    The per-field checks below this one (``_check_span_monotonic`` /
    ``_check_mass_positive`` / ``_check_stiffness_jumps``) all use
    comparison operators (``<=``, ``>``, ``/``) that return False on
    NaN — i.e. they don't catch the failure mode this guard exists to
    catch.
    """
    for fname in _SECTION_PROPERTY_FIELDS:
        if not hasattr(sp, fname):
            continue  # optional field, not populated on this dataclass
        arr = np.asarray(getattr(sp, fname), dtype=float)
        if arr.size == 0:
            continue
        bad = np.flatnonzero(~np.isfinite(arr))
        if bad.size == 0:
            continue
        first = int(bad[0])
        out.append(ModelWarning(
            "ERROR",
            f"{fname} has {bad.size} non-finite entry(ies) (first "
            f"idx = {first}, value = {float(arr[first])!r}). NaN or "
            f"Inf in section properties will propagate through the "
            f"FE assembly and eigensolve as NaN frequencies. Check "
            f"the upstream section-property table for transcription "
            f"errors.",
            f"section_properties.{fname}",
        ))


def _check_span_monotonic(sp: SectionProperties, out: list[ModelWarning]) -> None:
    span = np.asarray(sp.span_loc, dtype=float)
    if span.size < 2:
        return
    if np.any(np.diff(span) <= 0.0):
        bad = int(np.argmin(np.diff(span)))
        out.append(ModelWarning(
            "WARN",
            f"span_loc is not strictly increasing; first non-positive "
            f"step is span_loc[{bad}]={span[bad]:.4g} → "
            f"span_loc[{bad + 1}]={span[bad + 1]:.4g}. The FEM "
            f"interpolator assumes monotonic stations and will return "
            f"non-physical element properties otherwise.",
            "section_properties.span_loc",
        ))


def _check_mass_positive(sp: SectionProperties, out: list[ModelWarning]) -> None:
    m = np.asarray(sp.mass_den, dtype=float)
    bad_idx = np.flatnonzero(m <= 0.0)
    if bad_idx.size:
        out.append(ModelWarning(
            "ERROR",
            f"mass_den ≤ 0 at {bad_idx.size} station(s) "
            f"(first idx = {int(bad_idx[0])}, value = "
            f"{float(m[bad_idx[0]]):.4g}). The global mass matrix will "
            f"not be positive-definite and the eigensolve will return "
            f"undefined results.",
            "section_properties.mass_den",
        ))


def _check_stiffness_jumps(sp: SectionProperties, out: list[ModelWarning]) -> None:
    for attr, label in (("flp_stff", "EI_FA"), ("edge_stff", "EI_SS")):
        arr = np.asarray(getattr(sp, attr), dtype=float)
        if arr.size < 2:
            continue
        # Element-wise jump factor; guard zero / negative entries to
        # avoid spurious infinities.
        safe = np.where(arr > 0.0, arr, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_fwd = safe[1:] / safe[:-1]
            ratio_bwd = safe[:-1] / safe[1:]
        # The "jump" is the larger of forward and backward ratios so
        # we catch both up-steps and down-steps at the same threshold.
        jump = np.fmax(ratio_fwd, ratio_bwd)
        jump = np.where(np.isfinite(jump), jump, 1.0)
        worst = float(jump.max()) if jump.size else 1.0
        if worst > _CHECK_OPTIONS.stiffness_jump_factor:
            idx = int(np.argmax(jump))
            out.append(ModelWarning(
                "WARN",
                f"{label} jumps by {worst:.1f}× between adjacent stations "
                f"(idx {idx} → {idx + 1}: {arr[idx]:.3e} → "
                f"{arr[idx + 1]:.3e}). Such jumps strain the polynomial "
                f"fit and may need extra mesh refinement around the "
                f"discontinuity.",
                f"section_properties.{attr}",
            ))


def _check_ei_ratio(sp: SectionProperties, out: list[ModelWarning]) -> None:
    fa = np.asarray(sp.flp_stff, dtype=float)
    ss = np.asarray(sp.edge_stff, dtype=float)
    mask = (fa > 0.0) & (ss > 0.0)
    if not mask.any():
        return
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fa[mask] / ss[mask]
    min_r = float(np.min(ratio))
    max_r = float(np.max(ratio))
    if max_r > _CHECK_OPTIONS.ei_ratio_max or min_r < _CHECK_OPTIONS.ei_ratio_min:
        extreme = max_r if max_r > _CHECK_OPTIONS.ei_ratio_max else min_r
        out.append(ModelWarning(
            "INFO",
            f"EI_FA / EI_SS extreme ratio = {extreme:.2g} "
            f"(per-station range = [{min_r:.2g}, {max_r:.2g}]). Strong "
            f"asymmetry separates FA / SS modes by frequency and is "
            f"expected on wind-turbine blades; treat this as context "
            f"rather than an error.",
            "section_properties.{flp_stff,edge_stff}",
        ))


def _check_rna_vs_tower_mass(
    bmi, sp: SectionProperties, out: list[ModelWarning]
) -> None:
    if bmi.beam_type != 2:
        return
    if bmi.tip_mass is None or bmi.tip_mass.mass <= 0.0:
        return
    span_phys = np.asarray(sp.span_loc, dtype=float) * float(bmi.radius)
    mass_den = np.asarray(sp.mass_den, dtype=float)
    if span_phys.size < 2 or not np.all(np.diff(span_phys) > 0):
        return  # other checks will flag the bad span; skip cleanly here
    tower_mass = float(np.trapezoid(mass_den, span_phys))
    if tower_mass <= 0.0:
        return  # mass-density check will catch this
    rna = float(bmi.tip_mass.mass)
    if rna > tower_mass:
        out.append(ModelWarning(
            "INFO",
            f"RNA tip mass ({rna:,.0f} kg) exceeds the integrated tower "
            f"mass ({tower_mass:,.0f} kg). The 1st-FA tower-bending "
            f"frequency will be dominated by the RNA inertia rather "
            f"than the tower's distributed mass; this is normal for "
            f"slender towers but worth a sanity check on the upstream "
            f"deck.",
            "bmi.tip_mass.mass / section_properties.mass_den",
        ))


def _check_support_conditioning(bmi, out: list[ModelWarning]) -> None:
    from pybmodes.io.bmi import PlatformSupport

    if not isinstance(bmi.support, PlatformSupport):
        return
    sup = bmi.support
    # i_matrix / hydro_M / hydro_K / mooring_K are all symmetric 6x6 by
    # physics (Newton's 3rd law / Maxwell-Betti reciprocity). The
    # matrices are allowed to be rank-deficient — surge/sway/yaw
    # hydrostatic restoring is legitimately zero on most floaters, and
    # mooring stiffness can be low-rank depending on layout — so we
    # only flag non-physical structure: non-finite entries or
    # asymmetric coupling that the FEM assembly would silently mis-
    # interpret.
    for name, mat in (
        ("i_matrix", sup.i_matrix),
        ("hydro_M", sup.hydro_M),
        ("hydro_K", sup.hydro_K),
        ("mooring_K", sup.mooring_K),
    ):
        arr = np.asarray(mat, dtype=float)
        if arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)):
            out.append(ModelWarning(
                "ERROR",
                f"{name} 6×6 matrix contains non-finite entries "
                f"(NaN or Inf). Verify the matrix in the BMI deck.",
                f"bmi.support.{name}",
            ))
            continue
        if arr.shape != (6, 6):
            out.append(ModelWarning(
                "ERROR",
                f"{name} matrix has shape {arr.shape}, expected (6, 6). "
                f"Verify the matrix block in the BMI deck.",
                f"bmi.support.{name}",
            ))
            continue
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0:
            continue
        asym = float(np.max(np.abs(arr - arr.T)))
        if asym > _CHECK_OPTIONS.support_asymmetry_rtol * scale:
            out.append(ModelWarning(
                "WARN",
                f"{name} 6×6 matrix is not symmetric "
                f"(max|A - Aᵀ| = {asym:.3e}, scale = {scale:.3e}). "
                f"i_matrix / hydro_M / hydro_K / mooring_K are all "
                f"symmetric by physics; an asymmetric matrix will "
                f"flow through to the FEM assembly unchanged and "
                f"trigger the general-eig fallback in "
                f"``pybmodes.fem.solver``, but the upstream deck "
                f"likely has a transcription error worth fixing.",
                f"bmi.support.{name}",
            ))


def _check_platform_cm_offset(bmi, out: list[ModelWarning]) -> None:
    """Flag an implausibly large horizontal platform CM offset.

    ``cm_pform_x`` / ``cm_pform_y`` are the platform CM offset *from the
    tower axis* — for a standard floater the tower sits at or near the
    platform centroid, so these are small. A horizontal offset that
    rivals the platform's own size (its yaw radius of gyration
    ``√(I_yaw / m)``) is almost always a coordinate-origin value leaking
    into the field; it injects spurious surge/sway↔yaw coupling through
    the rigid-arm transform, shifts the rigid-body frequencies, and
    mislabels the modes (issue #95). Emitted as WARN, not ERROR — a
    genuinely large offset is physically representable, just unusual.
    """
    from pybmodes.io.bmi import PlatformSupport

    if not isinstance(bmi.support, PlatformSupport):
        return
    sup = bmi.support
    cm_x = float(getattr(sup, "cm_pform_x", 0.0) or 0.0)
    cm_y = float(getattr(sup, "cm_pform_y", 0.0) or 0.0)
    offset = float(np.hypot(cm_x, cm_y))
    if offset == 0.0:
        return

    # Yaw radius of gyration r_g = √(I_yaw / m). Use the platform mass
    # and the 6×6 inertia's yaw (5,5) term; bail quietly if either is
    # unavailable / non-positive (can't form a meaningful scale).
    mass = float(getattr(sup, "mass_pform", 0.0) or 0.0)
    i_mat = np.asarray(sup.i_matrix, dtype=float)
    if mass <= 0.0 and i_mat.shape == (6, 6) and np.isfinite(i_mat[0, 0]):
        mass = float(i_mat[0, 0])      # i_matrix[0,0] is the platform mass
    if mass <= 0.0 or i_mat.shape != (6, 6):
        return
    i_yaw = float(i_mat[5, 5])
    if not np.isfinite(i_yaw) or i_yaw <= 0.0:
        return
    r_g = float(np.sqrt(i_yaw / mass))
    if r_g <= 0.0:
        return

    if offset > _CHECK_OPTIONS.platform_cm_offset_gyradius_factor * r_g:
        out.append(ModelWarning(
            "WARN",
            f"Horizontal platform CM offset (cm_pform_x={cm_x:.3g} m, "
            f"cm_pform_y={cm_y:.3g} m, magnitude {offset:.3g} m) exceeds the "
            f"platform's yaw radius of gyration √(I_yaw/m) = {r_g:.3g} m. "
            f"cm_pform_x / cm_pform_y are the CM offset FROM THE TOWER AXIS — "
            f"a value this large is usually a coordinate-origin offset leaking "
            f"into the field; it injects spurious surge/sway↔yaw coupling that "
            f"shifts the rigid-body frequencies and mislabels the modes. Set "
            f"them to the CM offset relative to the tower axis (≈0 for a tower "
            f"on the platform centroid).",
            "bmi.support.cm_pform_x",
        ))


_FLOATING_FIX_HINT = (
    "A floating model needs more than the WindIO geometry + material: the "
    "rigid-body behaviour is set by hydrodynamics (added mass + hydrostatic "
    "restoring) and mooring, none of which live in the .yaml. Supply them via "
    "the companion decks — Tower.from_windio_floating(yaml, hydrodyn_dat=…, "
    "moordyn_dat=…, elastodyn_dat=…) — or an explicit, correctly-assembled "
    "PlatformSupport. The deck path reads the WAMIT A_inf / C_hst and the "
    "MoorDyn system in their correct reference frames and is the validated "
    "(BModes-JJ ≈ 0.0003 %) route."
)


def _check_platform_inertia_physical(bmi, out: list[ModelWarning]) -> None:
    """Flag a non-physical platform inertia (zero / negative mass or
    diagonal moment of inertia).

    A real floating body has strictly positive mass and rotational
    inertia about every axis; a zero or negative diagonal in the 6×6
    ``i_matrix`` (or a non-positive ``mass_pform``) is a transcription
    error that produces meaningless rigid-body modes. ERROR severity.
    """
    from pybmodes.io.bmi import PlatformSupport

    if not isinstance(bmi.support, PlatformSupport):
        return
    sup = bmi.support
    i_mat = np.asarray(sup.i_matrix, dtype=float)
    mass = float(getattr(sup, "mass_pform", 0.0) or 0.0)
    if mass <= 0.0 and i_mat.size and i_mat.ndim == 2 \
            and np.isfinite(i_mat[0, 0]):
        mass = float(i_mat[0, 0])      # i_matrix[0,0] is the platform mass
    if mass <= 0.0:
        out.append(ModelWarning(
            "ERROR",
            f"Platform mass is not positive (mass_pform = {mass:.3g} kg). A "
            f"floating body must have a strictly positive mass; verify the "
            f"PlatformSupport / ElastoDyn PtfmMass.",
            "bmi.support.mass_pform",
        ))
    if i_mat.ndim == 2 and i_mat.shape[0] == i_mat.shape[1] and i_mat.size:
        diag = np.diag(i_mat)
        bad = [int(k) for k in range(diag.size)
               if np.isfinite(diag[k]) and diag[k] <= 0.0]
        if bad:
            out.append(ModelWarning(
                "ERROR",
                f"Platform inertia matrix has non-positive diagonal entries at "
                f"DOF index {bad} (value(s) {[float(diag[k]) for k in bad]}). "
                f"Every translational mass and rotational moment of inertia on "
                f"the i_matrix diagonal must be > 0 for a physical body.",
                "bmi.support.i_matrix",
            ))


def _check_added_mass_present(bmi, out: list[ModelWarning]) -> None:
    """Warn when a floating model carries no hydrodynamic added mass.

    ``hydro_M`` (the infinite-frequency added-mass matrix ``A_inf``) is
    typically large for a floating platform — often comparable to the
    structural mass in surge / sway / heave — so omitting it biases
    *every* rigid-body frequency high. An all-zero ``hydro_M`` is the
    single most common seakeeping omission for a non-specialist hand-
    assembling a PlatformSupport (issue #95). WARN severity (a zero
    added mass is occasionally a deliberate screening simplification).
    """
    from pybmodes.io.bmi import PlatformSupport

    if not isinstance(bmi.support, PlatformSupport):
        return
    h_m = np.asarray(bmi.support.hydro_M, dtype=float)
    has_added_mass = bool(h_m.size and np.any(np.isfinite(h_m) & (h_m != 0.0)))
    if not has_added_mass:
        out.append(ModelWarning(
            "WARN",
            "Floating model has no hydrodynamic added mass (hydro_M / A_inf is "
            "zero). Added mass is typically large for a floating platform "
            "(often comparable to the structural mass in surge / sway / heave); "
            "omitting it biases all rigid-body frequencies high (commonly by "
            "10–30 %). " + _FLOATING_FIX_HINT,
            "bmi.support.hydro_M",
        ))


def _check_restoring_present(bmi, out: list[ModelWarning]) -> None:
    """Warn when a floating model has no restoring at all.

    A floating platform's rigid-body modes are set by hydrostatic
    restoring (``hydro_K`` / ``C_hst``, the waterplane + buoyancy) plus
    mooring stiffness (``mooring_K``). With neither, surge / sway /
    heave / roll / pitch / yaw have no restoring and collapse to ~0 Hz —
    a non-physical "free body in vacuum" rather than a station-kept
    floater. WARN severity (the solve still completes, just meaningless).
    """
    from pybmodes.io.bmi import PlatformSupport

    if not isinstance(bmi.support, PlatformSupport):
        return
    sup = bmi.support
    any_restoring = False
    for mat in (sup.hydro_K, sup.mooring_K):
        arr = np.asarray(mat, dtype=float)
        if arr.size and np.any(np.isfinite(arr) & (arr != 0.0)):
            any_restoring = True
            break
    if not any_restoring:
        out.append(ModelWarning(
            "WARN",
            "Floating model has no restoring: both the hydrostatic restoring "
            "(hydro_K / C_hst) and the mooring stiffness (mooring_K) are zero. "
            "The platform's rigid-body modes are entirely set by this "
            "restoring; with none, surge / sway / heave / roll / pitch / yaw "
            "collapse to ~0 Hz (a free body, not a station-kept floater). "
            + _FLOATING_FIX_HINT,
            "bmi.support.mooring_K",
        ))


def _check_n_modes_vs_dof(
    bmi, n_modes: int, out: list[ModelWarning]
) -> None:
    # Use the FEM's *exact* post-constraint solvable DOF count rather
    # than a hand-rolled per-node estimate. The element carries 9 DOFs
    # per global node (see ``fem.assembly``), so a ``6 × n_nodes``
    # estimate undercounts the true free-DOF count for any non-trivial
    # mesh and would fire a false ERROR on perfectly valid n_modes
    # requests. ``boundary.n_free_dof`` already encodes ``9·nselt``
    # minus the exact constraint count for each ``hub_conn``.
    from pybmodes.fem.boundary import n_free_dof

    ngd = int(n_free_dof(int(bmi.n_elements), int(bmi.hub_conn)))
    if n_modes > ngd:
        out.append(ModelWarning(
            "ERROR",
            f"requested n_modes={n_modes} exceeds the model's solvable "
            f"DOF count ({ngd} free DOFs for nselt={int(bmi.n_elements)},"
            f" hub_conn={int(bmi.hub_conn)}). Reduce n_modes or refine "
            f"the mesh (increase ``nselt`` in the BMI).",
            f"run(n_modes={n_modes}) / bmi.n_elements",
        ))


def _check_polyfit_conditioning(bmi, out: list[ModelWarning]) -> None:
    el_loc = np.asarray(bmi.el_loc, dtype=float)
    if el_loc.size < 6:
        return  # not enough rows to even fit a 5-coefficient polynomial
    # Mirror the design-matrix construction in
    # ``pybmodes.fitting.poly_fit.fit_mode_shape``: columns are
    # ``x**k - x**6`` for k = 2..5 (with the C6 = 1 - sum constraint
    # substituted). The cond number depends only on the sampling
    # locations, so we can compute it pre-solve.
    A = np.column_stack([el_loc ** k - el_loc ** 6 for k in range(2, 6)])
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = np.inf
    if not np.isfinite(cond) or cond > _CHECK_OPTIONS.fit_cond_fail:
        out.append(ModelWarning(
            "ERROR",
            f"polynomial-fit design matrix condition number = "
            f"{cond:.2e} > {_CHECK_OPTIONS.fit_cond_fail:.0e} (FAIL "
            f"threshold). The polynomial coefficient solve will be "
            f"numerically unreliable; refine or re-space the mesh.",
            "bmi.el_loc",
        ))
    elif cond > _CHECK_OPTIONS.fit_cond_warn:
        out.append(ModelWarning(
            "WARN",
            f"polynomial-fit design matrix condition number = "
            f"{cond:.2e} > {_CHECK_OPTIONS.fit_cond_warn:.0e} (WARN "
            f"threshold). Polynomial coefficient "
            f"sensitivity to mode-shape perturbations is elevated; "
            f"results are usable but minor input changes may produce "
            f"larger coefficient shifts than the mode shapes warrant.",
            "bmi.el_loc",
        ))
