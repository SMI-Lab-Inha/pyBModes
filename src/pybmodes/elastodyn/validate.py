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

"""Coefficient-consistency validation for OpenFAST ElastoDyn decks.

The polynomial coefficient blocks shipped in industry ElastoDyn ``.dat``
files (NREL 5MW Reference Turbine, IEA-3.4-130-RWT, and others) are
demonstrably inconsistent with the structural-property blocks in the
same files — see ``cases/ECOSYSTEM_FINDING.md`` for the longer write-up.
This module exposes a programmatic way to surface that inconsistency:

- :func:`validate_dat_coefficients` parses the deck, runs pyBmodes on
  the same structural inputs, fits its own polynomials, and computes
  per-block RMS residuals for both the file's polynomial and pyBmodes'
  polynomial against the FEM-computed mode shape.
- :class:`ValidationResult` and :class:`CoeffBlockResult` carry the
  results in a form the CLI (``pybmodes validate``) and the
  ``Tower.from_elastodyn(validate_coeffs=True)`` path both consume.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pybmodes.elastodyn.params import (
    _remove_root_rigid_motion,
    compute_blade_params,
    compute_tower_params_report,
)
from pybmodes.fitting.poly_fit import PolyFitResult

# RMS thresholds (in unit-tip-normalised displacement).
_PASS_RMS = 0.01    # below this: file polynomial fits the FEM shape well
_FAIL_RMS = 0.10    # at or above this: file polynomial does not represent
                    # the mode shape from these structural inputs

VerdictStr = Literal["PASS", "WARN", "FAIL"]


@dataclass
class CoeffBlockResult:
    """Validation result for a single ElastoDyn coefficient block.

    Both ``file_rms`` and ``pybmodes_rms`` are RMS residuals of the
    respective polynomial evaluated at the FEM stations against the
    same tip-normalised mode-shape data. They are directly comparable.
    ``ratio = file_rms / pybmodes_rms`` quantifies how much worse the
    file polynomial fits the structural model than pyBmodes' fit does;
    a ratio above ~50× indicates the file polynomial was generated
    against a different structural model than the one in the deck.

    Tower blocks (``TwFAM1Sh`` / ``TwFAM2Sh`` / ``TwSSM1Sh`` /
    ``TwSSM2Sh``) carry four extra fields populated from the family-
    selection report:

    * ``fa_participation`` / ``ss_participation`` /
      ``torsion_participation`` — modal kinetic-energy fractions
      (unit-mass approximation) of the *selected* mode for this
      block. Sum to 1.
    * ``rejected_modes`` — mode numbers that were dropped from this
      block's family during selection because their torsion fraction
      exceeded 10 %. Empty when every candidate passed the gate.

    Blade blocks leave all four fields at their NaN / empty defaults
    because the blade adapter doesn't run a torsion-contamination
    filter (blade torsion is part of the structural model and not
    handled the same way as tower torsion).
    """

    name: str
    file_rms: float
    pybmodes_rms: float
    ratio: float
    verdict: VerdictStr
    file_coeffs: list[float]
    pybmodes_coeffs: list[float]
    fa_participation: float = float("nan")
    ss_participation: float = float("nan")
    torsion_participation: float = float("nan")
    rejected_modes: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ValidationResult:
    """Aggregated coefficient-validation result for an ElastoDyn deck."""

    dat_path: pathlib.Path
    tower_results: dict[str, CoeffBlockResult] = field(default_factory=dict)
    blade_results: dict[str, CoeffBlockResult] = field(default_factory=dict)
    overall: VerdictStr = "PASS"
    summary: str = ""

    def all_blocks(self) -> dict[str, CoeffBlockResult]:
        """Iterate tower-then-blade blocks in canonical order."""
        merged: dict[str, CoeffBlockResult] = {}
        merged.update(self.tower_results)
        merged.update(self.blade_results)
        return merged

    def failing_blocks(self) -> list[CoeffBlockResult]:
        return [b for b in self.all_blocks().values() if b.verdict == "FAIL"]

    def warning_blocks(self) -> list[CoeffBlockResult]:
        return [b for b in self.all_blocks().values() if b.verdict == "WARN"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(file_rms: float) -> VerdictStr:
    if not np.isfinite(file_rms):
        return "FAIL"
    if file_rms < _PASS_RMS:
        return "PASS"
    if file_rms < _FAIL_RMS:
        return "WARN"
    return "FAIL"


def _eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the ElastoDyn 6th-order polynomial at *x*.

    ``coeffs`` is the length-5 array ``[C2, C3, C4, C5, C6]``; the
    implied ``C0 = C1 = 0`` clamped-base constraint is enforced by the
    polynomial form.
    """
    c2, c3, c4, c5, c6 = (float(c) for c in coeffs)
    return c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4 + c5 * x ** 5 + c6 * x ** 6


def _build_block_result(
    name: str,
    x: np.ndarray,
    y_normalized: np.ndarray,
    pybmodes_fit: PolyFitResult,
    file_coeffs: np.ndarray,
) -> CoeffBlockResult:
    """Compute RMS residuals for one block and assign a verdict."""
    file_y = _eval_poly(file_coeffs, x)
    file_rms = float(np.sqrt(np.mean((file_y - y_normalized) ** 2)))
    pybmodes_rms = float(pybmodes_fit.rms_residual)

    if pybmodes_rms > 1.0e-12:
        ratio = file_rms / pybmodes_rms
    elif file_rms < 1.0e-12:
        ratio = 1.0
    else:
        ratio = float("inf")

    return CoeffBlockResult(
        name=name,
        file_rms=file_rms,
        pybmodes_rms=pybmodes_rms,
        ratio=ratio,
        verdict=_classify(file_rms),
        file_coeffs=[float(c) for c in file_coeffs],
        pybmodes_coeffs=[float(c) for c in pybmodes_fit.coefficients()],
    )


def _worst(verdicts: list[VerdictStr]) -> VerdictStr:
    """Return the worst verdict across a list (PASS < WARN < FAIL)."""
    if "FAIL" in verdicts:
        return "FAIL"
    if "WARN" in verdicts:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dat_coefficients(
    dat_path: pathlib.Path | str,
    *,
    verbose: bool = False,
    n_modes: int = 10,
) -> ValidationResult:
    """Validate the polynomial coefficient blocks in an ElastoDyn deck.

    Parses the main ``.dat`` file plus the tower and blade files it
    references, builds pyBmodes Tower and RotatingBlade models from the
    structural-property blocks (NOT the polynomial blocks), runs the
    eigensolver, fits 6th-order polynomials, and compares those fits
    against the polynomial coefficients embedded in the deck.

    Parameters
    ----------
    dat_path :
        Path to the ElastoDyn main ``.dat`` file.
    verbose :
        Reserved for future diagnostic output. The function currently
        emits no prints regardless of this flag — the CLI front-end
        is responsible for stdout formatting.
    n_modes :
        Number of FEM modes to extract per model (default 10; must be
        large enough to cover the four tower bending modes plus the
        three blade bending modes — pyBmodes' family selectors warn
        if any required mode falls outside the requested range).

    Returns
    -------
    :class:`ValidationResult`
        Carries seven :class:`CoeffBlockResult` entries (4 tower + 3
        blade) and an overall PASS/WARN/FAIL verdict (worst across all
        blocks).
    """
    # Imported here to avoid a circular import (models -> elastodyn).
    from pybmodes.io.elastodyn_reader import (
        read_elastodyn_blade,
        read_elastodyn_main,
        read_elastodyn_tower,
    )
    from pybmodes.models import RotatingBlade, Tower

    dat_path = pathlib.Path(dat_path).resolve()
    if not dat_path.is_file():
        raise FileNotFoundError(f"ElastoDyn main file not found: {dat_path}")

    main = read_elastodyn_main(dat_path)
    tower = read_elastodyn_tower(dat_path.parent / main.twr_file)

    blade_path = dat_path.parent / main.bld_file[0]
    if not blade_path.is_file():
        raise FileNotFoundError(
            f"Blade file referenced as BldFile(1)={main.bld_file[0]} not "
            f"found at {blade_path}"
        )
    blade_dat = read_elastodyn_blade(blade_path)

    # --- Tower ---------------------------------------------------------
    tower_model = Tower.from_elastodyn(dat_path)
    # The validator runs as a service to other workflows (CLI ``validate``,
    # ``patch``, ``batch``) — pre-solve model checks belong to user-driven
    # ``.run()`` calls, not to this internal solve path. Suppress them
    # here so callers don't see duplicate warnings on every validate.
    tower_modal = tower_model.run(n_modes=n_modes, check_model=False)
    tower_params, tower_report = compute_tower_params_report(tower_modal)

    by_mode = {s.mode_number: s for s in tower_modal.shapes}
    fa1 = by_mode[tower_report.selected_fa_modes[0]]
    fa2 = by_mode[tower_report.selected_fa_modes[1]]
    ss1 = by_mode[tower_report.selected_ss_modes[0]]
    ss2 = by_mode[tower_report.selected_ss_modes[1]]

    # Build a per-mode lookup of the four kinetic-energy fields the
    # family selection has already computed. Used by ``_tower_block``
    # to attach them to each per-block result.
    participation_by_mode: dict[int, tuple[float, float, float]] = {
        member.mode_number: (
            member.fa_participation,
            member.ss_participation,
            member.torsion_participation,
        )
        for member in (
            *tower_report.fa_family,
            *tower_report.ss_family,
        )
    }

    def _tower_block(
        name: str,
        shape,
        is_fa: bool,
        fit: PolyFitResult,
        file_coeffs: np.ndarray,
    ) -> CoeffBlockResult:
        tfa, tss, ttor = participation_by_mode.get(
            shape.mode_number, (float("nan"), float("nan"), float("nan")),
        )
        # For the family this block lives in, the rejected modes are
        # the ones the selection dropped due to torsion contamination.
        rejected = (
            tower_report.rejected_fa_modes if is_fa
            else tower_report.rejected_ss_modes
        )
        if is_fa:
            disp = _remove_root_rigid_motion(
                shape.span_loc, shape.flap_disp, shape.flap_slope
            )
        else:
            disp = _remove_root_rigid_motion(
                shape.span_loc, shape.lag_disp, shape.lag_slope
            )
        tip = disp[-1]
        if abs(tip) < 1e-30:
            y_norm = np.zeros_like(disp)
            file_rms = float("nan")
            pybmodes_rms = float(fit.rms_residual)
            return CoeffBlockResult(
                name=name,
                file_rms=file_rms,
                pybmodes_rms=pybmodes_rms,
                ratio=float("nan"),
                verdict="FAIL",
                file_coeffs=[float(c) for c in file_coeffs],
                pybmodes_coeffs=[float(c) for c in fit.coefficients()],
                fa_participation=tfa,
                ss_participation=tss,
                torsion_participation=ttor,
                rejected_modes=rejected,
            )
        y_norm = disp / tip
        block = _build_block_result(name, shape.span_loc, y_norm, fit,
                                    file_coeffs)
        # _build_block_result doesn't know about the tower family;
        # attach the participation fields here.
        block.fa_participation = tfa
        block.ss_participation = tss
        block.torsion_participation = ttor
        block.rejected_modes = rejected
        return block

    tower_results: dict[str, CoeffBlockResult] = {
        "TwFAM1Sh": _tower_block("TwFAM1Sh", fa1, True, tower_params.TwFAM1Sh,
                                 tower.tw_fa_m1_sh),
        "TwFAM2Sh": _tower_block("TwFAM2Sh", fa2, True, tower_params.TwFAM2Sh,
                                 tower.tw_fa_m2_sh),
        "TwSSM1Sh": _tower_block("TwSSM1Sh", ss1, False, tower_params.TwSSM1Sh,
                                 tower.tw_ss_m1_sh),
        "TwSSM2Sh": _tower_block("TwSSM2Sh", ss2, False, tower_params.TwSSM2Sh,
                                 tower.tw_ss_m2_sh),
    }

    # --- Blade ---------------------------------------------------------
    blade_model = RotatingBlade.from_elastodyn(dat_path)
    # See note above for the tower side: validator-internal solves
    # don't emit pre-solve check warnings; callers see them on direct
    # ``.run()`` invocations.
    blade_modal = blade_model.run(n_modes=n_modes, check_model=False)
    blade_params = compute_blade_params(blade_modal)

    # compute_blade_params already picks 1st flap, 2nd flap, 1st edge by
    # FA-vs-edge classification — re-walk the same logic here so we know
    # which physical mode is which fit.
    from pybmodes.elastodyn.params import _is_fa_dominated

    flap_shapes = [s for s in blade_modal.shapes if _is_fa_dominated(s)]
    edge_shapes = [s for s in blade_modal.shapes if not _is_fa_dominated(s)]
    if len(flap_shapes) < 2 or not edge_shapes:
        raise RuntimeError(
            "Blade modal solve did not return enough flap/edge modes "
            f"(flap={len(flap_shapes)}, edge={len(edge_shapes)}); "
            f"increase n_modes (currently {n_modes})."
        )

    def _blade_block(
        name: str,
        shape,
        is_flap: bool,
        fit: PolyFitResult,
        file_coeffs: np.ndarray,
    ) -> CoeffBlockResult:
        disp = shape.flap_disp if is_flap else shape.lag_disp
        tip = disp[-1]
        if abs(tip) < 1e-30:
            return CoeffBlockResult(
                name=name,
                file_rms=float("nan"),
                pybmodes_rms=float(fit.rms_residual),
                ratio=float("nan"),
                verdict="FAIL",
                file_coeffs=[float(c) for c in file_coeffs],
                pybmodes_coeffs=[float(c) for c in fit.coefficients()],
            )
        y_norm = disp / tip
        return _build_block_result(name, shape.span_loc, y_norm, fit,
                                   file_coeffs)

    blade_results: dict[str, CoeffBlockResult] = {
        "BldFl1Sh": _blade_block(
            "BldFl1Sh", flap_shapes[0], True,
            blade_params.BldFl1Sh, blade_dat.bld_fl1_sh,
        ),
        "BldFl2Sh": _blade_block(
            "BldFl2Sh", flap_shapes[1], True,
            blade_params.BldFl2Sh, blade_dat.bld_fl2_sh,
        ),
        "BldEdgSh": _blade_block(
            "BldEdgSh", edge_shapes[0], False,
            blade_params.BldEdgSh, blade_dat.bld_edg_sh,
        ),
    }

    # --- Aggregate ----------------------------------------------------
    all_blocks = list(tower_results.values()) + list(blade_results.values())
    overall = _worst([b.verdict for b in all_blocks])
    n_warn = sum(1 for b in all_blocks if b.verdict == "WARN")
    n_fail = sum(1 for b in all_blocks if b.verdict == "FAIL")
    if overall == "PASS":
        summary = (
            f"All {len(all_blocks)} coefficient blocks in {dat_path.name} "
            "fit pyBmodes' mode shapes within 1 % RMS."
        )
    elif overall == "WARN":
        summary = (
            f"{n_warn} of {len(all_blocks)} blocks in {dat_path.name} "
            "show a noticeable shape mismatch (RMS 1-10 %)."
        )
    else:
        summary = (
            f"{n_fail} of {len(all_blocks)} blocks in {dat_path.name} "
            "do not represent the mode shape implied by the deck's "
            "structural inputs (RMS ≥ 10 %); run `pybmodes patch` to "
            "regenerate them."
        )

    return ValidationResult(
        dat_path=dat_path,
        tower_results=tower_results,
        blade_results=blade_results,
        overall=overall,
        summary=summary,
    )
