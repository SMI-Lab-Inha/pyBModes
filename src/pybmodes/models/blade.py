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

"""RotatingBlade: high-level model for a rotating wind-turbine blade."""

from __future__ import annotations

import logging
import pathlib
import warnings
from typing import TYPE_CHECKING

from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem
from pybmodes.models.result import ModalResult

if TYPE_CHECKING:
    from pybmodes.elastodyn.validate import ValidationResult

_log = logging.getLogger(__name__)


def _apply_elastodyn_compatibility(sp: SectionProperties) -> None:
    """In-place override of section properties to match ElastoDyn's simplified
    structural model.

    Per Jonkman (NREL forum, March 2015): the polynomial mode shapes
    ElastoDyn consumes must come from a model that shares ElastoDyn's
    structural assumptions — pure flapwise + edgewise bending of a
    straight isotropic beam, no axial or torsional DOFs, no
    mass / shear-centre / tension-centre offsets, no inertial-vs-
    structural twist split. If the BModes (or pyBmodes) model carries
    any of those extra effects, the polynomial fit represents a
    physically different beam and the resulting ElastoDyn time-domain
    response is biased.

    Translation to the pyBmodes section-properties dataclass:

      * ``str_tw  = 0``  zeroes structural twist so flap and edge
                         bending don't cross-couple. ElastoDyn applies
                         pre-twist on top of the polynomial mode shape
                         itself — leaving twist non-zero in the FEM
                         double-counts it.
      * ``tw_iner = 0``  no separate inertial-twist axis; ElastoDyn's
                         model has none.
      * ``cg_offst = sc_offst = tc_offst = 0``
                         no mass / shear-centre / tension-centre
                         offsets.

    Two of Jonkman's BModes recommendations — "very small flp_iner /
    edge_iner" and "very large tor_stff / axial_stff" — are already
    covered by :func:`pybmodes.io.elastodyn_reader._stack_blade_section_props`
    (rotary-inertia floor at ``1e-6 · char² · ρA``; ``axial_stff =
    1e6 · EI``). The third — ``tor_stff = 1e6 · EI`` — would be
    numerically unstable in pyBmodes' dense LAPACK solver (sees ghost
    eigenvalues from the ill-conditioned ``M``), so the adapter's
    ``tor_stff = 100 · EI`` default is left in place. That value
    already pushes the lowest torsion mode well above any blade mode
    of interest (multi-MHz on the NREL 5MW), which is functionally
    indistinguishable from "rigid" for ElastoDyn-mode-shape extraction.
    """
    sp.str_tw[:] = 0.0
    sp.tw_iner[:] = 0.0
    sp.cg_offst[:] = 0.0
    sp.sc_offst[:] = 0.0
    sp.tc_offst[:] = 0.0


class RotatingBlade:
    """Compute natural frequencies and mode shapes for a rotating blade.

    Parameters
    ----------
    bmi_path : path to the .bmi input file (beam_type must be 1).
    """

    # See Tower.coeff_validation for the rationale; populated only on
    # the from_elastodyn(..., validate_coeffs=True) path.
    coeff_validation: "ValidationResult | None" = None

    def __init__(self, bmi_path: str | pathlib.Path) -> None:
        self._bmi = read_bmi(bmi_path)
        self._sp: SectionProperties | None = None
        if self._bmi.beam_type != 1:
            raise ValueError(
                f"RotatingBlade requires beam_type=1, got {self._bmi.beam_type}"
            )

    @classmethod
    def from_elastodyn(
        cls,
        main_dat_path: str | pathlib.Path,
        *,
        elastodyn_compatible: bool = True,
        validate_coeffs: bool = False,
    ) -> "RotatingBlade":
        """Build a blade model from an OpenFAST ElastoDyn main ``.dat``.

        The blade .dat is resolved relative to the main file via
        ``BldFile(1)`` (or ``BldFile1`` in the IEA-RWT convention).
        Centrifugal stiffening uses ``RotSpeed`` from the main file.

        Parameters
        ----------
        main_dat_path :
            Path to the ElastoDyn main ``.dat`` file.
        elastodyn_compatible :
            When ``True`` (default) the blade section properties are
            overridden in place to match ElastoDyn's simplified
            structural model — pure flap/edge bending, no axial /
            torsional DOFs, no offsets — per Jonkman's March 2015 NREL
            forum guidance for generating ElastoDyn polynomial inputs.
            See :func:`_apply_elastodyn_compatibility` for the exact
            settings. Set to ``False`` to keep the structural-property
            blocks exactly as parsed; a :class:`UserWarning` is emitted
            in that case because the resulting mode shapes will not
            match ElastoDyn's expectations and any polynomial fit
            generated from them is unsafe to feed back into ElastoDyn.
        validate_coeffs :
            If ``True``, run
            :func:`pybmodes.elastodyn.validate_dat_coefficients` after
            building the model and attach the result as
            ``self.coeff_validation``. Emits a ``UserWarning`` if any
            block fails or warns. Default ``False``.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            to_pybmodes_blade,
        )
        from pybmodes.models._shared import _run_validation_and_warn

        main_dat_path = pathlib.Path(main_dat_path)
        main = read_elastodyn_main(main_dat_path)
        bld_path = main_dat_path.parent / main.bld_file[0]
        blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_blade(main, blade)

        if elastodyn_compatible:
            _apply_elastodyn_compatibility(sp)
            _log.info(
                "Building ElastoDyn-compatible blade model: structural "
                "twist, torsion, axial, and offset DOFs suppressed per "
                "Jonkman (2015). Use elastodyn_compatible=False for "
                "physical accuracy."
            )
        else:
            warnings.warn(
                "elastodyn_compatible=False: mode shapes may not match "
                "ElastoDyn expectations. Use True when generating "
                "ElastoDyn polynomial coefficients.",
                UserWarning,
                stacklevel=2,
            )

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None

        if validate_coeffs:
            obj.coeff_validation = _run_validation_and_warn(main_dat_path)

        return obj

    @classmethod
    def from_windio(
        cls,
        yaml_path: str | pathlib.Path,
        *,
        component: str = "blade",
        n_span: int = 30,
        rot_rpm: float = 0.0,
        n_perim: int = 300,
        elastic: str = "auto",
    ) -> "RotatingBlade":
        """Build a blade model from a WindIO ontology ``.yaml`` (issue #35).

        ``elastic`` (issue #48) selects the beam-property source:
        ``"auto"`` (default) uses the WindIO *published* distributed
        properties (``elastic_properties`` /
        ``elastic_properties_mb``) when present so the model matches
        the reference exactly, and reduces the composite layup via the
        PreComp-class thin-wall cross-section reduction
        (:mod:`pybmodes.io.windio_blade`) only when they are absent;
        ``"precomp"`` always reduces the layup (the pre-1.5
        behaviour); ``"file"`` requires the published properties. The
        blade is clamped at the root (``hub_conn = 1``); ``rot_rpm``
        sets the centrifugal stiffening (default 0 = parked). Needs
        the optional ``[windio]`` extra (PyYAML).

        Unlike :meth:`from_elastodyn`, this keeps the *physical*
        section properties (twist, offsets, torsion / axial) — the
        WindIO path's purpose is structural fidelity, not regenerating
        ElastoDyn polynomial coefficients.
        """
        from pybmodes.io._elastodyn.adapter import (
            _build_bmi_skeleton,
            _tower_element_boundaries,
        )
        from pybmodes.io.bmi import TipMassProps
        from pybmodes.io.windio_blade import (
            read_windio_blade,
            windio_blade_section_props,
        )

        blade = read_windio_blade(yaml_path, component=component,
                                  n_span=n_span)
        sp = windio_blade_section_props(blade, n_perim=n_perim,
                                        elastic=elastic)

        el_loc = _tower_element_boundaries(blade.span_grid)
        bmi = _build_bmi_skeleton(
            title=f"WindIO composite blade ({component})",
            beam_type=1,
            radius=float(blade.flexible_length),
            hub_rad=0.0,
            rot_rpm=float(rot_rpm),
            precone=0.0,
            n_elements=max(el_loc.size - 1, 1),
            el_loc=el_loc,
            tip_mass_props=TipMassProps(
                mass=0.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            ),
        )

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None
        return obj

    def run(
        self, n_modes: int = 20, *, check_model: bool = True
    ) -> ModalResult:
        """Solve the eigenvalue problem and return frequencies + mode shapes.

        Parameters
        ----------
        n_modes : number of modes to extract (must be >= 1; default 20).
        check_model : run :func:`pybmodes.checks.check_model` before the
            solve (default ``True``). WARN and ERROR findings are
            emitted as ``UserWarning``; INFO findings are silent (call
            ``pybmodes.checks.check_model(model)`` explicitly to see
            those). Pass ``check_model=False`` to skip the pre-solve
            checks for scripted callers that have already validated
            their inputs.
        """
        if not isinstance(n_modes, int) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer; got {n_modes!r}")
        if check_model:
            from pybmodes.checks import check_model as _check_model
            for w in _check_model(self, n_modes=n_modes):
                if w.severity != "INFO":
                    warnings.warn(str(w), UserWarning, stacklevel=2)
        return run_fem(self._bmi, n_modes=n_modes, sp=self._sp)
