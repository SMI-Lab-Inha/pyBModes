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

"""Tower: high-level model for a wind-turbine tower.

Phase 3 PR C3 of the v1.x architecture refactor pulled three module-
private helpers out of this file into siblings under
:mod:`pybmodes.models`:

- :func:`pybmodes.models._shared._run_validation_and_warn` —
  cross-model (also used by ``RotatingBlade.from_elastodyn``).
- :func:`pybmodes.models._platform._scan_platform_fields`,
  :func:`pybmodes.models._platform._platform_inertia_matrix` — tower-
  side platform-scalar parsers + inertia assembler.

All three are re-exported below so callers / tests that still import
via ``from pybmodes.models.tower import _scan_platform_fields`` (etc.)
keep working unchanged.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

from pybmodes.io.bmi import read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem
from pybmodes.models._platform import (
    _platform_inertia_matrix,
    _scan_platform_fields,
)
from pybmodes.models._shared import (
    _run_validation_and_warn,
)
from pybmodes.models.result import ModalResult

if TYPE_CHECKING:
    import numpy as np

    from pybmodes.elastodyn.validate import ValidationResult
    from pybmodes.io.bmi import PlatformSupport, TipMassProps


class Tower:
    """Compute natural frequencies and mode shapes for a tower.

    Parameters
    ----------
    bmi_path : path to the .bmi input file (beam_type must be 2).
    """

    # Populated by ``from_elastodyn(..., validate_coeffs=True)``;
    # ``None`` when the constructor didn't run validation. Declared
    # at class scope so mypy sees the attribute on instances built
    # via ``cls.__new__(cls)`` (the from_elastodyn path bypasses
    # ``__init__``).
    coeff_validation: ValidationResult | None = None

    def __init__(self, bmi_path: str | pathlib.Path) -> None:
        self._bmi = read_bmi(bmi_path)
        self._sp: SectionProperties | None = None
        if self._bmi.beam_type != 2:
            raise ValueError(
                f"Tower requires beam_type=2, got {self._bmi.beam_type}"
            )

    @classmethod
    def from_bmi(cls, bmi_path: str | pathlib.Path) -> Tower:
        """Build a tower model from a BModes-format ``.bmi`` deck.

        Equivalent to ``Tower(bmi_path)`` — exposed as an explicit
        classmethod so callers can pick the constructor by source format
        symmetrically with :meth:`from_elastodyn` and
        :meth:`from_elastodyn_with_subdyn`.

        The BMI parser already covers all four certtest configurations
        (cantilever blade, blade + tip mass, cantilever tower, tension-
        wire-supported tower) plus the offshore platform-support paths
        (``hub_conn`` ∈ {1, 2, 3}, ``tow_support`` ∈ {0, 1, 2}, with
        ``PlatformSupport`` carrying hydro / mooring / platform-inertia
        6×6 matrices). All of those flow through the standard FEM
        pipeline; this constructor is a thin handle.
        """
        return cls(bmi_path)

    @classmethod
    def from_elastodyn(
        cls,
        main_dat_path: str | pathlib.Path,
        *,
        validate_coeffs: bool = False,
    ) -> Tower:
        """Build a tower model from an OpenFAST ElastoDyn main ``.dat``.

        The main file is parsed plus the tower file referenced via
        ``TwrFile`` and (when the path is resolvable) the first blade file
        referenced via ``BldFile(1)`` — the latter is read only to compute
        the rotor-mass contribution to the lumped tower-top assembly.

        Parameters
        ----------
        main_dat_path :
            Path to the ElastoDyn main ``.dat`` file.
        validate_coeffs :
            If ``True``, run
            :func:`pybmodes.elastodyn.validate_dat_coefficients` after
            building the model and attach the result as
            ``self.coeff_validation``. Emits a ``UserWarning`` if any
            block fails or warns. Default ``False`` so the standard
            constructor stays cheap.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
            to_pybmodes_tower,
        )

        main_dat_path = pathlib.Path(main_dat_path)
        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)

        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_tower(main, tower, blade=blade)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None

        if validate_coeffs:
            obj.coeff_validation = _run_validation_and_warn(main_dat_path)

        return obj

    @classmethod
    def from_geometry(
        cls,
        station_grid: np.ndarray | list[float],
        outer_diameter: np.ndarray | list[float],
        wall_thickness: np.ndarray | list[float],
        *,
        flexible_length: float,
        E: float = 2.0e11,
        rho: float = 7850.0,
        nu: float = 0.3,
        outfitting_factor: float = 1.0,
        hub_conn: int = 1,
        tip_mass: TipMassProps | float | None = None,
        n_nodes: int | None = None,
    ) -> Tower:
        """Build a tower model from tubular **geometry** instead of
        pre-computed structural properties (issue #35).

        The user supplies only what they actually know — the circular
        tube's outer diameter and wall thickness per station, plus the
        material — and pyBmodes derives mass / EI / GJ / EA from the
        exact closed-form tube relations
        (:func:`pybmodes.io.geometry.tubular_section_props`),
        eliminating the hand-computed-properties error class.

        Parameters
        ----------
        station_grid : (n,) normalised station locations ``[0, 1]``
            from tower base (0) to top (1). WindIO grids are already
            in this form. Duplicate-pair stations encoding a property
            step are handled exactly as the ElastoDyn path does.
        outer_diameter, wall_thickness : (n,) metres, per station.
        flexible_length : physical flexible tower length (m), i.e.
            ``TowerHt - TowerBsHt``. Sets the FEM beam length.
        E, rho, nu : isotropic material (default ASTM-A572 steel:
            200 GPa, 7850 kg/m^3, 0.3).
        outfitting_factor : non-structural mass multiplier (internals
            / flanges / paint / bolts). Scales the distributed mass
            density only — *not* rotary inertia (a structural section
            property) and never stiffness. This is the WindIO-native
            way to "account for internals/flanges"; for a single
            discrete tower-top mass pass ``tip_mass``.
        hub_conn : root BC — 1 cantilever (default; the basis
            ElastoDyn polynomial coefficients require), 3 soft
            monopile, etc.
        tip_mass : optional RNA / tower-top lump — a
            :class:`pybmodes.io.bmi.TipMassProps`, **or a bare float**
            (the RNA mass in kg; the inertia / offset terms default to
            zero — the common case, issue #35). ``None`` -> zero tip
            mass.
        n_nodes : optional FE-mesh refinement (issue #35). When given,
            the geometry is **re-gridded onto ``n_nodes`` evenly-
            spaced stations over the normalised span** and the outer
            diameter / wall thickness are linearly interpolated onto
            it before the closed-form tube reduction (so each refined
            station still gets *exact* tube properties, not
            interpolated ones). A finer mesh resolves the higher
            tower-bending mode shapes; frequencies are unbiased
            (convergent) — see ``tests/test_geometry_windio.py``.
            ``None`` keeps the supplied grid verbatim. Note: a
            uniform resample linearly smooths a deliberately *stepped*
            geometry (e.g. a wall-thickness jump); omit ``n_nodes`` to
            preserve such steps exactly.

        Notes
        -----
        Arbitrary *discrete mid-span* point masses are not yet
        modelled (a separate FEM-assembly extension with its own
        validation track — see issue #35); ``outfitting_factor``
        covers the dominant distributed non-structural mass and
        ``tip_mass`` the tower-top lump.
        """
        import numpy as _np

        from pybmodes.io._elastodyn.adapter import (
            _build_bmi_skeleton,
            _tower_element_boundaries,
        )
        from pybmodes.io.bmi import TipMassProps
        from pybmodes.io.geometry import tubular_section_props

        grid = _np.asarray(station_grid, dtype=float)
        od = _np.asarray(outer_diameter, dtype=float)
        wt = _np.asarray(wall_thickness, dtype=float)

        if n_nodes is not None:
            if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) \
                    or n_nodes < 2:
                raise ValueError(
                    f"n_nodes must be an integer >= 2; got {n_nodes!r}"
                )
            # Re-grid onto a uniform span and linearly interpolate the
            # *geometry* (not the derived props) so each refined
            # station still gets exact closed-form tube properties.
            fine = _np.linspace(float(grid[0]), float(grid[-1]), n_nodes)
            od = _np.interp(fine, grid, od)
            wt = _np.interp(fine, grid, wt)
            grid = fine

        sp = tubular_section_props(
            grid, od, wt,
            E=E, rho=rho, nu=nu, outfitting_factor=outfitting_factor,
        )
        if tip_mass is None:
            tip_mass = TipMassProps(
                mass=0.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            )
        elif not isinstance(tip_mass, TipMassProps):
            # Bare-scalar convenience: a tower-top RNA *mass* in kg.
            try:
                m = float(tip_mass)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "tip_mass must be a TipMassProps or a float "
                    f"(RNA mass in kg); got {tip_mass!r}"
                ) from exc
            if not _np.isfinite(m) or m < 0.0:
                raise ValueError(
                    f"tip_mass (kg) must be finite and >= 0; got {m!r}"
                )
            tip_mass = TipMassProps(
                mass=m, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            )
        el_loc = _tower_element_boundaries(grid)
        bmi = _build_bmi_skeleton(
            title="geometry-derived tower",
            beam_type=2,
            radius=float(flexible_length),
            hub_rad=0.0,
            rot_rpm=0.0,
            precone=0.0,
            n_elements=max(el_loc.size - 1, 1),
            el_loc=el_loc,
            tip_mass_props=tip_mass,
        )
        bmi.hub_conn = int(hub_conn)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None
        return obj

    @classmethod
    def from_windio(
        cls,
        yaml_path: str | pathlib.Path,
        *,
        component: str = "tower",
        thickness_interp: str = "linear",
        hub_conn: int = 1,
        tip_mass: TipMassProps | float | None = None,
        n_nodes: int | None = None,
    ) -> Tower:
        """Build a tower (or monopile) model directly from a **WindIO**
        ontology ``.yaml`` (issue #35).

        Parses the structural subset — ``components.<component>``'s
        ``outer_shape.outer_diameter``, ``structure.layers`` wall
        thickness, ``structure.outfitting_factor``, ``reference_axis``
        — plus the referenced entry in the top-level ``materials``
        list, and feeds it to :meth:`from_geometry`.

        Parameters
        ----------
        yaml_path : path to a WindIO ontology file.
        component : ``"tower"`` (default) or ``"monopile"``.
        thickness_interp : how a layer thickness grid maps onto the
            FEM stations — ``"linear"`` (WindIO-native piecewise-
            linear, default) or ``"piecewise_constant"`` (WISDEM-style
            constant-per-segment). The choice measurably moves the
            2nd tower-bending polynomial coefficients; see
            ``tests/test_windio.py``.
        hub_conn : root BC (default 1 cantilever; use 3 for a
            soil-flexible monopile).
        tip_mass : optional tower-top RNA lump (issue #35) — a
            :class:`pybmodes.io.bmi.TipMassProps` **or a bare float**
            (RNA mass in kg). Replaces the
            ``tower._bmi.tip_mass = …`` workaround and mirrors
            :meth:`from_windio_floating`'s ``rna_tip``. ``None`` ->
            zero tip mass.
        n_nodes : optional FE-mesh refinement (issue #35) — re-grid
            the tower onto ``n_nodes`` evenly-spaced stations
            (geometry linearly interpolated, properties recomputed
            exactly), to resolve higher tower-bending mode shapes.
            ``None`` keeps the WindIO grid. The WindIO blade path
            has the analogous ``n_span``.

        Notes
        -----
        Requires the optional ``[windio]`` extra (PyYAML). This is the
        tubular tower / monopile path; for a WindIO blade composite
        layup use :meth:`pybmodes.models.RotatingBlade.from_windio`
        (PreComp-class thin-wall cross-section reduction).
        """
        from pybmodes.io.windio import read_windio_tubular

        g = read_windio_tubular(
            yaml_path, component=component, thickness_interp=thickness_interp,
        )
        return cls.from_geometry(
            g.station_grid,
            g.outer_diameter,
            g.wall_thickness,
            flexible_length=g.flexible_length,
            E=g.E, rho=g.rho, nu=g.nu,
            outfitting_factor=g.outfitting_factor,
            hub_conn=hub_conn,
            tip_mass=tip_mass,
            n_nodes=n_nodes,
        )

    @classmethod
    def from_elastodyn_with_mooring(
        cls,
        main_dat_path: str | pathlib.Path,
        moordyn_dat_path: str | pathlib.Path,
        hydrodyn_dat_path: str | pathlib.Path | None = None,
    ) -> Tower:
        """Build a free-free floating tower model with a populated
        :class:`~pybmodes.io.bmi.PlatformSupport` block.

        Assembles the platform-support 6 × 6 matrices from three OpenFAST
        decks:

        - **Mooring stiffness** ``K_moor`` from a MoorDyn ``.dat`` (parsed
          via :class:`pybmodes.mooring.MooringSystem.from_moordyn` and
          linearised at zero offset).
        - **Hydrodynamic added mass** ``A_inf`` and **hydrostatic
          restoring** ``C_hst`` from a HydroDyn ``.dat`` (parsed via
          :class:`pybmodes.io.HydroDynReader`, which follows ``PotFile``
          to the WAMIT ``.1`` and ``.hst`` files). Optional — if
          ``hydrodyn_dat_path`` is omitted, both default to zero, so
          the resulting model couples only mooring + platform inertia.
        - **Platform inertia** from the ``PtfmMass`` / ``PtfmRIner`` /
          ``PtfmPIner`` / ``PtfmYIner`` / ``PtfmCM*`` / ``PtfmRefzt``
          scalars in the ElastoDyn main file. The 6 × 6 ``i_matrix`` is
          stored AT THE CM (no parallel-axis transfer); the downstream
          ``pybmodes.fem.nondim.nondim_platform`` applies the rigid-arm
          transform from CM to tower base using ``cm_pform - draft``.
          ``cm_pform`` and ``draft`` are written in BModes file
          convention (positive distance below MSL; signed draft with
          negative = base above MSL).

        Sets ``hub_conn = 2`` (free-free floating base) and
        ``tow_support = 1`` (inline platform-support block).

        Notes
        -----
        For ElastoDyn polynomial-coefficient generation use the standard
        cantilever :meth:`Tower.from_elastodyn` instead — the polynomial
        ansatz lives in a clamped-base frame regardless of platform
        configuration (see ``cases/ECOSYSTEM_FINDING.md``). This method
        is for coupled-frequency prediction only.
        """
        import numpy as np

        from pybmodes.io._elastodyn.adapter import to_pybmodes_tower
        from pybmodes.io.bmi import PlatformSupport
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
        )
        from pybmodes.mooring import MooringSystem

        main_dat_path = pathlib.Path(main_dat_path)
        moordyn_dat_path = pathlib.Path(moordyn_dat_path)

        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)
        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)
        # Free-base floating: use physically-scaled section properties.
        # The cantilever proxies (EA ≈ 1e6·EI) wreck the conditioning
        # of the global matrices and, on an asymmetric spar/semi
        # platform, collapse the soft rigid-body modes into an
        # n_modes-dependent degenerate cluster (v1.1.1; the bundled-
        # sample fix, extended here to the in-memory path).
        bmi, sp = to_pybmodes_tower(main, tower, blade, physical_sec_props=True)

        # The cantilever adapter sets ``bmi.radius`` to the flexible
        # tower length (``TowerHt − TowerBsHt``). The floating BMI
        # convention (matching the bundled OC3Hywind.bmi) is
        # ``radius = TowerHt`` paired with ``draft = -TowerBsHt`` so
        # ``radius + draft = flexible length`` after the nondim step
        # in :func:`pybmodes.fem.nondim.make_params`. Overriding the
        # radius here keeps the FEM beam length consistent with the
        # ``draft = -TowerBsHt`` assignment below. Pre-1.0 review.
        # caught this — without the override the flexible length came
        # out as ``TowerHt - 2·TowerBsHt`` (e.g. 67.6 m for OC3
        # instead of 77.6 m).
        bmi.radius = float(main.tower_ht)

        ptfm = _scan_platform_fields(main_dat_path)

        moor_sys = MooringSystem.from_moordyn(moordyn_dat_path)
        K_moor = moor_sys.stiffness_matrix(np.zeros(6))
        # ElastoDyn carries six scalar springs (``PtfmSurgeStiff``,
        # ``PtfmSwayStiff``, ``PtfmHeaveStiff``, ``PtfmRollStiff``,
        # ``PtfmPitchStiff``, ``PtfmYawStiff``) that act *in addition*
        # to whatever HydroDyn / MoorDyn provide at runtime. The OC3
        # delta-line crowfoot is conventionally folded into
        # ``PtfmYawStiff`` (~ 9.83e7 N·m/rad); without including these
        # the coupled-yaw frequency for an OC3-style deck would land
        # an order of magnitude low.
        #
        # DOF order assumption — verified via the canonical OC3Hywind.bmi
        # ``mooring_K[0,4] = -2.821e6`` matching Jonkman (2010) NREL/TP-
        # 500-47535 K_15 surge→pitch coupling: BMI rigid-body matrices
        # are in standard OpenFAST DOF order [surge, sway, heave, roll,
        # pitch, yaw] — the same order as ``MooringSystem.stiffness_matrix()``
        # and as the ElastoDyn ``Ptfm*Stiff`` enumeration below.
        # ``test_oc3hywind_mooring_K_cross_coupling_sign`` pins this
        # invariant so any future DOF-order regression fails loudly
        # rather than silently producing wrong physics on asymmetric
        # platforms.
        for axis, key in enumerate((
            "PtfmSurgeStiff", "PtfmSwayStiff", "PtfmHeaveStiff",
            "PtfmRollStiff", "PtfmPitchStiff", "PtfmYawStiff",
        )):
            K_moor[axis, axis] += ptfm[key]

        A_inf = np.zeros((6, 6))
        C_hst = np.zeros((6, 6))
        if hydrodyn_dat_path is not None:
            from pybmodes.io.wamit_reader import HydroDynReader
            wamit = HydroDynReader(hydrodyn_dat_path).read_platform_matrices()
            A_inf = wamit.A_inf
            C_hst = wamit.C_hst

        M = ptfm["PtfmMass"]
        i_mat = _platform_inertia_matrix(ptfm)

        # BModes file convention for these scalars (see the OC3 Hywind
        # sample BMI in ``src/pybmodes/_examples/sample_inputs/
        # reference_turbines/07_nrel5mw_oc3hywind_spar/``):
        #   ``draft``    — signed depth of the flexible-tower base
        #                  *below* MSL (positive = below; negative =
        #                  above). For OC3 the TP sits at +10 m above
        #                  MSL so ``draft = -10``.
        #   ``cm_pform`` — POSITIVE distance from MSL down to the
        #                  platform CM. For OC3 ``cm_pform = 89.9155``
        #                  (CM at z = −89.9155 in MSL frame).
        #   ``ref_msl``  — positive distance below MSL of the platform
        #                  reference point (usually 0).
        # ElastoDyn stores all three as signed z (positive = above MSL
        # via ``TowerBsHt``; negative = below MSL via ``PtfmCMzt``).
        # The sign flips below translate ElastoDyn → BModes convention.
        # Horizontal CM offset (asymmetric floating substructure). The
        # vertical ``cm_pform`` flips sign (ElastoDyn signed-z "below
        # MSL" → BModes positive-down); the horizontal components carry
        # straight through — ``PtfmCMxt`` is downwind (surge-aligned,
        # = the FEM v axis) and ``PtfmCMyt`` lateral (sway-aligned,
        # = the FEM w axis), the same frame the rigid-arm transform in
        # ``nondim_platform`` expects. Both are 0 for an axisymmetric
        # spar / symmetric semi, so those decks are unchanged.
        platform_support = PlatformSupport(
            draft=-float(main.tower_bs_ht),
            cm_pform=-ptfm["PtfmCMzt"],
            mass_pform=M,
            i_matrix=i_mat,
            ref_msl=-ptfm["PtfmRefzt"],
            hydro_M=A_inf,
            hydro_K=C_hst,
            mooring_K=K_moor,
            distr_m_z=np.zeros(0),
            distr_m=np.zeros(0),
            distr_k_z=np.zeros(0),
            distr_k=np.zeros(0),
            cm_pform_x=ptfm["PtfmCMxt"],
            cm_pform_y=ptfm["PtfmCMyt"],
        )

        bmi.hub_conn = 2
        bmi.tow_support = 1
        bmi.support = platform_support

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        return obj

    @classmethod
    def from_windio_floating(
        cls,
        yaml_path: str | pathlib.Path,
        *,
        component_tower: str = "tower",
        water_depth: float | None = None,
        hydrodyn_dat: str | pathlib.Path | None = None,
        moordyn_dat: str | pathlib.Path | None = None,
        elastodyn_dat: str | pathlib.Path | None = None,
        platform_support: PlatformSupport | None = None,
        rna_tip: TipMassProps | None = None,
        rho: float = 1025.0,
        g: float = 9.80665,
    ) -> Tower:
        """Coupled floating tower+platform model from a WindIO ``.yaml``
        (issue #35).

        The WindIO-native analogue of
        :meth:`from_elastodyn_with_mooring`. The tower beam always
        comes from ``components.tower`` (the validated Phase-1 tubular
        path — machine-exact vs the ElastoDyn tower). The platform has
        **two fidelity tiers**:

        * **Industry-grade (companion decks present).** When a
          ``hydrodyn_dat`` / ``moordyn_dat`` / ``elastodyn_dat`` is
          supplied, that leg uses the *complete* deck model — WAMIT
          ``A_inf`` + ``C_hst``, the full MoorDyn system (its own
          anchor/fairlead geometry *and* line properties), and the
          ElastoDyn ``PtfmMass``/``RIner`` (incl. trim ballast) +
          lumped RNA + draft convention. With all three present this
          is byte-identical to the BModes-JJ-validated
          :meth:`from_elastodyn_with_mooring` except the tower is the
          WindIO one (≈ 0.0003 % reference grade).
        * **Screening preview (yaml-only legs).** Any leg without a
          deck falls to the WindIO model — member-waterplane
          ``C_hst`` (geometry-exact, ≈ 1.6 %), Morison + end-cap
          ``A_inf`` (heave is screening-only — Morison ≠ potential
          flow, as RAFT/WISDEM also find), WindIO catenary mooring
          geometry, and structural + fixed-ballast inertia (no trim
          ballast). A :class:`UserWarning` names every screening leg;
          it is **not** industry-grade for the platform and is for
          fast pre-deck previewing only.

        * **Injected platform (``platform_support=``).** When a
          :class:`pybmodes.io.bmi.PlatformSupport` is passed, the
          floater is taken *verbatim* from it — its own ``A_inf`` /
          ``C_hst`` / ``mooring_K`` / 6×6 inertia / ``draft`` /
          ``ref_msl``. The tower geometry still comes from the WindIO
          ``.yaml``; nothing about the floating substructure is read
          from the yaml or any deck. This is the "floater designed
          separately" workflow — a frequency-domain tool / WAMIT
          export / published 6×6 set feeding the *same*
          BModes-JJ-validated free-free FEM that reproduces OC3 Hywind
          to ≈ 0.0003 %. The tower beam length stays the WindIO
          ``flexible_length`` independent of the supplied ``draft``
          (the ``radius + draft`` cancellation the deck path also
          relies on — see ``make_params``). No screening warning (the
          caller owns the platform fidelity). Mutually exclusive with
          the companion decks; optionally pass ``rna_tip`` for the
          tower-top RNA lump (default: bare tower top, no RNA). The
          frequencies inherit the validated PlatformSupport assembly;
          this path adds no new numerics.

        Sets ``hub_conn = 2`` / ``tow_support = 1`` and reuses the
        existing BModes-JJ-validated free-free ``PlatformSupport`` FEM
        unchanged. Needs the optional ``[windio]`` extra. For
        ElastoDyn polynomial generation use the cantilever
        :meth:`from_windio` regardless of platform (see
        ``cases/ECOSYSTEM_FINDING.md``)."""
        import warnings

        import numpy as np

        from pybmodes.io._elastodyn.adapter import (
            _build_bmi_skeleton,
            _tower_element_boundaries,
            _tower_top_assembly_mass,
        )
        from pybmodes.io.bmi import PlatformSupport, TipMassProps
        from pybmodes.io.geometry import tubular_section_props
        from pybmodes.io.windio import read_windio_tubular
        from pybmodes.io.windio_floating import (
            added_mass,
            hydrostatic_restoring,
            read_windio_floating,
            rigid_body_inertia,
        )
        from pybmodes.mooring import MooringSystem

        # Fail fast (with one clear message) on an explicitly-supplied
        # deck path that does not exist — a deep FileNotFoundError from
        # inside from_moordyn / read_elastodyn_main is opaque, and
        # silently degrading to the screening tier would hide a typo
        # and hand back wrong-fidelity results. (The CLI auto-discovery
        # only ever passes existing paths or None, so this guards the
        # explicit-API caller.)
        _missing = [
            f"{name}={p}"
            for name, p in (("hydrodyn_dat", hydrodyn_dat),
                            ("moordyn_dat", moordyn_dat),
                            ("elastodyn_dat", elastodyn_dat))
            if p is not None and not pathlib.Path(p).is_file()
        ]
        if _missing:
            raise FileNotFoundError(
                "from_windio_floating: companion deck(s) not found — "
                + "; ".join(_missing)
                + ". Pass an existing path, or omit the argument to "
                "use the labelled screening preview for that leg."
            )

        # --- injected-platform tier: WindIO tower geometry + a
        #     caller-supplied PlatformSupport (floater designed
        #     separately). Bypasses every yaml/deck platform
        #     derivation; feeds the supplied 6x6s straight into the
        #     validated free-free FEM. (issue #35)
        if platform_support is not None:
            _decks = [
                n for n, p in (("hydrodyn_dat", hydrodyn_dat),
                               ("moordyn_dat", moordyn_dat),
                               ("elastodyn_dat", elastodyn_dat))
                if p is not None
            ]
            if _decks:
                raise ValueError(
                    "from_windio_floating: platform_support is mutually "
                    "exclusive with companion decks ("
                    + ", ".join(_decks)
                    + ") — supply the platform either as decks OR as a "
                    "PlatformSupport, not both."
                )
            gt = read_windio_tubular(yaml_path, component=component_tower)
            sp = tubular_section_props(
                gt.station_grid, gt.outer_diameter, gt.wall_thickness,
                E=gt.E, rho=gt.rho, nu=gt.nu,
                outfitting_factor=gt.outfitting_factor,
                title="WindIO floating tower (injected platform)",
            )
            tip = rna_tip if rna_tip is not None else TipMassProps(
                mass=0.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            )
            el_loc = _tower_element_boundaries(gt.station_grid)
            # The FEM beam length is ``radius + draft - hub_rad``
            # (``make_params``). The deck path makes this cancel to
            # ``flexible_length`` by passing ``radius = tower_top =
            # flexible_length - draft``; do the same here so a
            # supplied floater's non-zero ``draft`` does not silently
            # shorten/lengthen the tower and shift every modal
            # frequency. (Static-review P1 on v1.4.2; fixed in 1.4.3.)
            bmi = _build_bmi_skeleton(
                title="WindIO floating tower + injected platform",
                beam_type=2,
                radius=float(gt.flexible_length)
                - float(platform_support.draft),
                hub_rad=0.0,
                rot_rpm=0.0,
                precone=0.0,
                n_elements=max(el_loc.size - 1, 1),
                el_loc=el_loc,
                tip_mass_props=tip,
            )
            bmi.hub_conn = 2
            bmi.tow_support = 1
            bmi.support = platform_support

            obj = cls.__new__(cls)
            obj._bmi = bmi
            obj._sp = sp
            obj.coeff_validation = None
            return obj

        # --- tower beam (validated Phase-1 tubular path) ---------------
        gt = read_windio_tubular(yaml_path, component=component_tower)
        sp = tubular_section_props(
            gt.station_grid, gt.outer_diameter, gt.wall_thickness,
            E=gt.E, rho=gt.rho, nu=gt.nu,
            outfitting_factor=gt.outfitting_factor,
            title="WindIO floating tower",
        )
        fl = read_windio_floating(yaml_path)
        if fl.transition_joint is None:
            raise KeyError(
                "WindIO floating_platform has no joint flagged "
                "`transition: true` — cannot locate the tower foot."
            )
        z_base = float(fl.joints[fl.transition_joint][2])   # MSL datum

        preview: list[str] = []        # legs without a validating deck

        # --- mooring: full MoorDyn deck model (geometry + props) when
        #     present — the BModes-JJ-validated path; else the WindIO
        #     catenary preview (its own geometry, screening fidelity).
        if moordyn_dat is not None:
            K_moor = (MooringSystem.from_moordyn(moordyn_dat, rho, g)
                      .stiffness_matrix(np.zeros(6)))
        else:
            if water_depth is None or water_depth <= 0.0:
                raise ValueError(
                    "water_depth (m) is required for the yaml-only "
                    "mooring preview (the WindIO component file does "
                    "not carry the site depth); or pass moordyn_dat "
                    "for the validated mooring model."
                )
            K_moor = MooringSystem.from_windio_mooring(
                fl, depth=float(water_depth), rho=rho, g=g,
            ).stiffness_matrix(np.zeros(6))
            preview.append("mooring (WindIO catenary geometry)")

        # --- hydro: WAMIT C_hst + A_inf when a HydroDyn deck is
        #     present; else the geometry-exact member C_hst (≈1.6 %,
        #     not flagged) + the Morison/end-cap A_inf (heave is
        #     screening-only — flagged).
        wamit = None
        if hydrodyn_dat is not None:
            from pybmodes.io.wamit_reader import HydroDynReader
            try:
                wamit = HydroDynReader(
                    hydrodyn_dat).read_platform_matrices()
            except (ValueError, FileNotFoundError) as exc:
                # e.g. PotMod=0 (no WAMIT output) — degrade gracefully
                # to the screening hydro for this leg rather than crash.
                preview.append(
                    f"added mass (HydroDyn deck has no WAMIT — "
                    f"{exc.__class__.__name__}; Morison fallback)"
                )
        if wamit is not None:
            C_hst = np.asarray(wamit.C_hst, float)
            A_inf = np.asarray(wamit.A_inf, float)
        else:
            C_hst = hydrostatic_restoring(fl, rho=rho, g=g)
            A_inf = added_mass(fl, rho=rho)
            if hydrodyn_dat is None:
                preview.append("added mass (Morison strip+end-cap; "
                                "heave is screening-only)")

        # --- inertia / RNA / draft framing: full ElastoDyn (incl.
        #     trim ballast + lumped RNA + the validated draft
        #     convention) when present; else WindIO struct+fixed
        #     inertia, bare tower top (no RNA), yaml geometry framing.
        rna_tip = TipMassProps(
            mass=0.0, cm_offset=0.0, cm_axial=0.0,
            ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
        )
        if elastodyn_dat is not None:
            from pybmodes.io.elastodyn_reader import (
                read_elastodyn_blade,
                read_elastodyn_main,
            )
            ed_path = pathlib.Path(elastodyn_dat)
            main = read_elastodyn_main(ed_path)
            blade = None
            if main.bld_file[0]:
                bp = ed_path.parent / main.bld_file[0]
                if bp.is_file():
                    blade = read_elastodyn_blade(bp)
            rna_tip = _tower_top_assembly_mass(main, blade)
            ptfm = _scan_platform_fields(ed_path)
            M = ptfm["PtfmMass"]
            i_mat = _platform_inertia_matrix(ptfm)
            cm_pform = -ptfm["PtfmCMzt"]
            cm_x, cm_y = ptfm["PtfmCMxt"], ptfm["PtfmCMyt"]
            ref_msl = -ptfm["PtfmRefzt"]
            for axis, key in enumerate((
                "PtfmSurgeStiff", "PtfmSwayStiff", "PtfmHeaveStiff",
                "PtfmRollStiff", "PtfmPitchStiff", "PtfmYawStiff",
            )):
                K_moor[axis, axis] += ptfm[key]
            # Match the validated from_elastodyn_with_mooring framing
            # exactly: radius = TowerHt, draft = -TowerBsHt.
            draft = -float(main.tower_bs_ht)
            tower_top = float(main.tower_ht)
        else:
            M, _M6_ref, cg = rigid_body_inertia(fl)
            # PlatformSupport.i_matrix is stored AT THE CM (the
            # downstream nondim_platform applies the CM→base rigid arm).
            _, i_mat, _ = rigid_body_inertia(fl, ref_point=cg)
            cm_pform = -float(cg[2])
            cm_x, cm_y = float(cg[0]), float(cg[1])
            ref_msl = 0.0
            draft = -z_base
            tower_top = z_base + float(gt.flexible_length)
            preview.append("platform inertia (struct+fixed ballast, "
                            "no trim ballast) + no RNA")

        if preview:
            warnings.warn(
                "Tower.from_windio_floating: SCREENING-fidelity "
                "(NOT industry-grade) for the platform leg(s): "
                + "; ".join(preview)
                + ". Supply the companion HydroDyn / MoorDyn / "
                "ElastoDyn decks for the BModes-JJ-validated coupled "
                "model.",
                UserWarning,
                stacklevel=2,
            )

        platform_support = PlatformSupport(
            draft=draft,
            cm_pform=cm_pform,
            mass_pform=float(M),
            i_matrix=i_mat,
            ref_msl=ref_msl,
            hydro_M=A_inf,
            hydro_K=C_hst,
            mooring_K=K_moor,
            distr_m_z=np.zeros(0), distr_m=np.zeros(0),
            distr_k_z=np.zeros(0), distr_k=np.zeros(0),
            cm_pform_x=cm_x, cm_pform_y=cm_y,
        )

        el_loc = _tower_element_boundaries(gt.station_grid)
        bmi = _build_bmi_skeleton(
            title="WindIO floating tower + platform",
            beam_type=2,
            radius=tower_top,                            # MSL datum
            hub_rad=0.0,
            rot_rpm=0.0,
            precone=0.0,
            n_elements=max(el_loc.size - 1, 1),
            el_loc=el_loc,
            tip_mass_props=rna_tip,
        )
        bmi.hub_conn = 2
        bmi.tow_support = 1
        bmi.support = platform_support

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
        obj.coeff_validation = None
        return obj

    @classmethod
    def from_elastodyn_with_subdyn(
        cls,
        main_dat_path: str | pathlib.Path,
        subdyn_dat_path: str | pathlib.Path,
    ) -> Tower:
        """Build a combined pile + tower cantilever from an ElastoDyn deck
        plus a SubDyn substructure file.

        The pile geometry comes from the SubDyn file (joints + members +
        circular cross-section properties); the tower above the transition
        piece comes from the ElastoDyn main + tower files. The two are
        spliced into a single cantilever with a clamped base at the
        SubDyn reaction joint (no soil flexibility).

        Designed for OC3-style fixed-base monopiles. Does not handle soil
        springs, hydrodynamic added mass, or non-circular substructure
        members. See :func:`pybmodes.io.subdyn_reader.to_pybmodes_pile_tower`
        for the assembly details.
        """
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
        )
        from pybmodes.io.subdyn_reader import read_subdyn, to_pybmodes_pile_tower

        main_dat_path = pathlib.Path(main_dat_path)
        subdyn_dat_path = pathlib.Path(subdyn_dat_path)

        main = read_elastodyn_main(main_dat_path)
        tower = read_elastodyn_tower(main_dat_path.parent / main.twr_file)
        subdyn = read_subdyn(subdyn_dat_path)

        blade = None
        if main.bld_file[0]:
            bld_path = main_dat_path.parent / main.bld_file[0]
            if bld_path.is_file():
                blade = read_elastodyn_blade(bld_path)

        bmi, sp = to_pybmodes_pile_tower(main, tower, subdyn, blade=blade)

        obj = cls.__new__(cls)
        obj._bmi = bmi
        obj._sp = sp
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

        Warning
        -------
        ``n_modes`` affects the LAPACK solver path. For symmetric or
        nearly-symmetric towers (``EI_FA ≈ EI_SS`` and small RNA c.m.
        offset), use ``n_modes >= 6``. With ``n_modes <= 4``,
        ``scipy.linalg.eigh`` invokes a subset eigenvalue routine that
        can artificially lift the degeneracy of the 1st FA / SS bending
        pair — the modes come back at slightly different frequencies
        and pre-separated, which prevents the degenerate-pair resolver
        in :mod:`pybmodes.elastodyn.params` from triggering. The
        polynomial fits still succeed, but the FA / SS classification
        may flip relative to a full solve, and downstream
        ``compute_tower_params_report`` may select different modes for
        ``TwFAM1Sh`` / ``TwSSM1Sh`` between runs at different
        ``n_modes``.

        Minimum recommended: ``n_modes >= 6`` for reliable FA / SS
        classification on symmetric structures. The default of 20 is
        safely above this threshold.
        """
        if not isinstance(n_modes, int) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer; got {n_modes!r}")
        if check_model:
            from pybmodes.checks import check_model as _check_model
            for w in _check_model(self, n_modes=n_modes):
                if w.severity != "INFO":
                    warnings.warn(str(w), UserWarning, stacklevel=2)
        return run_fem(self._bmi, n_modes=n_modes, sp=self._sp)
