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

"""pybmodes — Python finite-element library for wind turbine modal analysis.

Public API
==========

The following subpackage entry points are the stable, semver-protected
1.x surface (see the *Compatibility policy* section in the README).
Anything else in the package tree is internal and may change between
minor releases.

    from pybmodes.models    import RotatingBlade, Tower, ModalResult
    from pybmodes.elastodyn import (
        compute_blade_params,
        compute_tower_params,
        compute_tower_params_report,
        patch_dat,
        validate_dat_coefficients,
        BladeElastoDynParams,
        TowerElastoDynParams,
        ValidationResult,
        CoeffBlockResult,           # tower blocks carry fa/ss/torsion
                                    # participation + rejected_modes
        FloatingFrequencyGap,        # cantilever-vs-coupled tower
                                     # bending frequency reconciler
        report_floating_frequency_gap,
    )
    from pybmodes import MudlineFoundation   # soft-monopile mudline
                                              # coupled-spring SSI
    from pybmodes.fitting   import PolyFitResult, fit_mode_shape
    from pybmodes.campbell  import (
        CampbellResult,             # save / load / to_csv
        campbell_sweep,
        plot_campbell,
    )
    from pybmodes.checks    import check_model, ModelWarning
    from pybmodes.mac       import (
        mac_matrix,
        compare_modes,
        ModeComparison,
        plot_mac,
        shape_to_vector,
    )
    from pybmodes.report    import generate_report
    from pybmodes.mooring   import LineType, Point, Line, MooringSystem
    from pybmodes.io        import (
        HydroDynReader, WamitReader, WamitData,
        PlatformSupport, TipMassProps,
        read_out, BModeOutParseError,   # read_out(..., strict=True)
    )
    from pybmodes.coords    import DOF_NAMES, DOF_INDEX  # 6-DOF convention
    from pybmodes.plots     import (
        apply_style,
        plot_mode_shapes,
        plot_fit_quality,
        bir_mode_shape_plot,
        bir_mode_shape_subplot,
        plot_environmental_spectra,   # wind/wave + 1P/3P vs tower
        kaimal_spectrum,
        jonswap_spectrum,
    )

    # On Tower:
    #   Tower.from_bmi(bmi_path)
    #   Tower.from_elastodyn(main_dat)
    #   Tower.from_elastodyn_with_subdyn(main_dat, subdyn_dat)
    #   Tower.from_elastodyn_with_mooring(main_dat, moordyn_dat,
    #                                     hydrodyn_dat=None)
    #   Tower.from_geometry(station_grid, outer_diameter,
    #                       wall_thickness, *, flexible_length,
    #                       E, rho, nu, outfitting_factor)
    #   Tower.from_windio(yaml_path, *, component, thickness_interp,
    #       hub_conn, tip_mass, n_nodes, lumped_rna_cal)
    #     tip_mass: TipMassProps | float (RNA mass kg); n_nodes:
    #     refine the FE mesh onto N even stations (issue #35);
    #     lumped_rna_cal=True auto-derives the tower-top RNA from the
    #     ontology's elastic_properties_mb blocks (issue #82; hub_conn=1
    #     only)
    #   Tower.from_windio_with_monopile(yaml_path, *,
    #       component_tower, component_monopile, thickness_interp,
    #       tip_mass, n_nodes, water_depth, lumped_rna_cal)  # splice
    #       monopile + tower into one fixed-bottom cantilever clamped at
    #       the mudline (issue #92); water_depth clamps at the true
    #       seabed, dropping the embedded pile (issue #121);
    #       lumped_rna_cal auto-derives the RNA (issue #82)
    #   Tower.from_windio_floating(yaml_path, *, water_depth,
    #                              hydrodyn_dat, moordyn_dat,
    #                              elastodyn_dat)  # coupled FOWT

    # On RotatingBlade:
    #   RotatingBlade.from_bmi(bmi_path)
    #   RotatingBlade.from_elastodyn(main_dat)
    #   RotatingBlade.from_windio(yaml_path, *, component, n_span,
    #                             rot_rpm, elastic)
    #     elastic="auto" (default) uses the WindIO *published*
    #     distributed elastic properties when present — the full 6×6
    #     decoupled at the elastic/shear centre + principal elastic
    #     axes (issue #50; pybmodes.io._precomp.decouple), not the
    #     raw diagonal — else the PreComp-class layup reduction;
    #     "precomp" forces the reduction; "file" requires the
    #     published properties.

    from pybmodes.io.geometry import tubular_section_props
    # WindIO .yaml input needs the optional [windio] extra (PyYAML);
    # the runtime core stays numpy+scipy only — same stance as
    # [plots]/[notebook]. Tower (tubular tower/monopile):
    # read_windio_rna: assemble the tower-top RNA lump (hub + nacelle +
    # blades) from the elastic_properties_mb blocks (issue #82; also the
    # engine behind ``lumped_rna_cal=True``).
    from pybmodes.io.windio  import (
        read_windio_rna,
        read_windio_tubular,
        WindIOTubular,
    )
    # Blade (composite layup -> PreComp-class thin-wall reduction):
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
        WindIOBlade,
    )
    # Floating substructure (member-Morison hydro + catenary mooring;
    # used by Tower.from_windio_floating, yaml-first + deck-fallback):
    from pybmodes.io.windio_floating import (
        read_windio_floating,
        hydrostatic_restoring,
        added_mass,
        rigid_body_inertia,
        WindIOFloating,
    )

``ModalResult`` ships ``save(.npz)`` / ``load(.npz)`` /
``to_json(.json)`` / ``from_json(.json)`` with metadata (pyBmodes
version, source file, timestamp, git hash) and optional
``participation`` + ``fit_residuals`` + ``mode_labels`` fields
(``mode_labels`` names the floating-platform rigid-body modes —
surge / sway / heave / roll / pitch / yaw — for a free-free model;
``None`` otherwise). ``CampbellResult`` ships ``save(.npz)`` /
``load(.npz)`` / ``to_csv(.csv)``.

Known limitations of the 1.0 surface:

- ``pybmodes.mooring`` is catenary-only quasi-static. No seabed
  friction (``CB > 0``), no sloped seabed, no U-shape lines, no
  time-domain dynamics, no line drag or added mass.
- ``pybmodes.io.WamitReader`` extracts ``A_inf`` (infinite-frequency
  added mass), ``A_0`` (zero-frequency), and ``C_hst`` (hydrostatic
  restoring); finite-period frequency-dependent ``A(ω)`` / ``B(ω)``
  are skipped.
- ``Tower.from_elastodyn_with_mooring`` assembles a free-free floating
  BMI for coupled-frequency prediction; ElastoDyn polynomial-
  coefficient generation continues to use the cantilever
  ``Tower.from_elastodyn`` regardless of platform configuration (see
  ``cases/ECOSYSTEM_FINDING.md`` and
  ``src/pybmodes/_examples/reference_decks/FLOATING_CASES.md`` for
  the source-code citations). Use
  :func:`pybmodes.elastodyn.report_floating_frequency_gap` to
  reconcile the polynomial-basis cantilever frequency against the
  coupled-system frequency an OpenFAST linearisation will report.
- :class:`pybmodes.MudlineFoundation` computes the mudline coupled-
  spring stiffness for a soft monopile (Yu and Amdahl 2023) and
  emits a 6x6 block that drops into ``PlatformSupport.mooring_K``
  for a ``hub_conn = 3`` BMI. The mudline stiffness affects the
  coupled-system frequency only; ElastoDyn polynomial generation
  remains on the cantilever path regardless of soil flexibility.
- ``BMIFile.support.distr_m`` (distributed hydrodynamic added mass
  per unit tower length) is parsed by ``pybmodes.io.bmi.read_bmi``
  but NOT wired into the FEM mass matrix; ``distr_k`` (distributed
  soil stiffness) IS consumed. A ``UserWarning`` fires at parse time
  if a deck specifies non-empty ``distr_m`` so the gap is not
  silent.

Internal modules (``pybmodes.fem.*``, the underscore-prefixed
``pybmodes.models._pipeline``, and the private sub-package
``pybmodes.io._elastodyn``) carry the implementation and should not
be imported directly by user code; their signatures may change
between minor releases. The per-format submodules under
``pybmodes.io`` (``pybmodes.io.bmi``, ``elastodyn_reader``,
``subdyn_reader``, ``wamit_reader``) are reachable directly but the
public-freeze contract covers only the re-exports listed above.

The CLI is exposed via ``pybmodes`` (see ``pybmodes --help``) and is
declared in ``[project.scripts]``.
"""

from importlib.metadata import PackageNotFoundError, version

# Numerical-options dataclasses (Phase 1 of the v1.x architecture
# refactor) — centralised thresholds for the FEM solver, polynomial
# fit, and pre-solve sanity checker. Defaults preserve every
# previously-embedded magic number, so importing this module changes
# no behaviour; the public-API value is that callers can now find
# every numerical threshold in one place. Future PRs will accept
# instances on ``Tower.run()`` / ``RotatingBlade.run()`` /
# ``check_model()`` for per-call override.
from pybmodes.foundation import MudlineFoundation
from pybmodes.options import CheckOptions, FitOptions, SolverOptions

try:
    __version__ = version("pybmodes")
except PackageNotFoundError:
    # Fallback for an uninstalled source tree (no package metadata).
    # Keep in step with ``pyproject.toml`` ``[project] version``.
    __version__ = "1.16.0"

__all__ = [
    "CheckOptions",
    "FitOptions",
    "MudlineFoundation",
    "SolverOptions",
    "__version__",
]
