Limitations
===========

What ``pybmodes`` deliberately does *not* attempt to do, where
the modelling fidelity stops, and the specific edge cases
where the published numerics legitimately disagree with a
naive expectation.

Polynomial representation limits
--------------------------------

ElastoDyn's polynomial ansatz
:math:`\mathrm{SHP}(s) = \sum_{i=2}^{6} c_i s^{\,i}` is a
constrained 6th-order form with ``SHP(0) = SHP'(0) = 0`` baked
in. Three real-world consequences:

The IEA-15-UMaine ``TwSSM2Sh`` block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constrained 6th-order polynomial **cannot** resolve the
section-property gradient of the IEA-15-240-RWT UMaine
VolturnUS-S tower for that specific mode. Even a fresh patch
from the structural inputs ends at ``Overall: WARN`` with
ratio > 100 on that block; every other block on every other
deck reaches PASS.

This is a representation limit of ElastoDyn's polynomial form,
not a pyBmodes bug. The shipped
``src/pybmodes/_examples/reference_decks/iea15mw_umainesemi/validation_report.txt``
carries an auto-emitted footer documenting this.

**Workaround**: accept the WARN. The downstream OpenFAST
simulation runs without complaint; the WARN is a quality flag
for the polynomial form, not for the underlying FEM modes
(which match BModes JJ to < 0.01 %).

Hybrid bending + twist modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some FEM modes are *hybrid* — a bending mode with appreciable
twist content from elastic-centre / shear-centre offset, or
twist modes with bending coupling near a 2-fold degenerate
eigenspace.

The polynomial form **cannot express twist** at all (it's a
scalar single-axis form), so including hybrid modes in the FA
/ SS family selection produces silently-wrong coefficients.

``_select_tower_family`` drops candidates whose modal-kinetic-
energy torsion fraction is :math:`\geq 0.10`. Rejected modes
travel through
``TowerSelectionReport.rejected_fa_modes`` /
``rejected_ss_modes`` so the user sees which were dropped.

**Workaround**: if you have a *deliberately* hybrid tower
design (rare), accept that ElastoDyn's polynomial form can't
represent it and use BeamDyn for the runtime simulation
instead. pyBmodes' FEM modes themselves are accurate; only the
polynomial *projection* is constrained.

Free-free floating polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ElastoDyn's polynomial ansatz forces clamped-base modes
(``SHP(0) = SHP'(0) = 0``); the runtime adds rigid-body
platform motion as a separate sum.

So a floating-deck workflow that wants polynomials uses
``Tower.from_elastodyn(...)`` (cantilever, ``hub_conn = 1``),
**not** ``Tower.from_bmi(...)`` with ``hub_conn = 2``. The
OpenFAST source-code audit behind this is in
``cases/ECOSYSTEM_FINDING.md`` (the "Floating-deck
polynomials" section) and the per-deck
``FLOATING_CASES.md``.

For *coupled frequency prediction* (the 6 platform rigid-body
modes plus the tower bending modes referenced to the
platform), use ``Tower.from_elastodyn_with_mooring(...)``.
That's a different workflow from polynomial generation; the
two coexist by design.

Floating-platform fidelity
--------------------------

``Tower.from_windio_floating`` is a **two-tier** API:

- **Industry-grade**: with companion HydroDyn + MoorDyn +
  ElastoDyn decks supplied (or auto-discovered by the
  ``pybmodes windio`` CLI scoped to the turbine root), the
  path is byte-identical to ``from_elastodyn_with_mooring``,
  which reproduces OC3 Hywind to ≈ 0.0003 % across the first
  9 modes.

- **Screening preview**: yaml-only — member-Morison
  hydrodynamics + RAFT-style end-cap added mass + catenary
  mooring from the yaml. Useful for design-space exploration,
  **not** for final-design certification numbers. Always
  emits one ``UserWarning`` naming the result as
  ``SCREENING-fidelity (NOT industry-grade)``.

This is intentional. Industry decks carry decades of WAMIT
panel-mesh refinement that a member-Morison reduction can't
match for second-order quantities. Use the right tier for the
right question.

Mooring physics
---------------

:class:`pybmodes.mooring.MooringSystem` is a **quasi-static
elastic catenary** (Jonkman 2007 Appendix B). What it includes:

- B-1 / B-2 fully-suspended branch
- B-7 / B-8 with :math:`C_B = 0` for the anchor-on-seabed branch
- Per-line elastic stretch
- Damped Newton on :math:`(H, V_F)` with analytical 2×2 Jacobian
- Cross-coupling between bodies through shared anchors

What it does **not** include:

- **Dynamic effects** — drag, added mass, vortex-induced
  vibration. Use MoorDyn in time-domain for those.
- **Bending stiffness** — lines are assumed to bend freely.
- **Seabed friction** — ``C_B = 0`` by design (Jonkman's B-7 /
  B-8 with friction can be added but isn't standard).
- **Soil-pile interaction** below the anchor — the anchor is
  treated as a fixed reaction point.

The output is a **linearised 6 × 6 stiffness matrix** suitable
for feeding into the FEM platform-support block. If your
analysis needs dynamic mooring, drive MoorDyn in OpenFAST and
extract the linearised stiffness at the operating point, then
feed *that* into ``Tower.from_elastodyn_with_mooring`` via the
deck.

Numerical scope
---------------

- **Beam element only.** Plates / shells / volume elements are
  out of scope. Section properties come in as a distributed
  1-D table.
- **Linear modal analysis only.** Geometric nonlinearity
  (large deflections), material nonlinearity (yield), and
  contact / impact are out of scope.
- **Centrifugal stiffening only on the blade.** Tower
  centrifugal effects (negligible for fixed-base towers) are
  not modelled.
- **Rotor aerodynamics are not modelled.** A Campbell sweep
  does **not** include aeroelastic damping or unsteady
  aerodynamics — those belong in OpenFAST. ``pybmodes``
  answers the structural-frequency question; resonance checks
  against the per-rev family are read off the diagram.
- **Single 1.x public API.** Names listed in
  :doc:`api_contract` are semver-frozen across 1.x minor
  releases. Numerical outputs may shift between minor releases
  when validation tightens or a modelling correction lands;
  every such shift is called out in :doc:`changelog` under
  *Fixed* / *Changed* with magnitude and affected case.

Specific edge cases the validation matrix surfaces
--------------------------------------------------

NREL 5MW r-test polynomial drift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The polynomial blocks shipped in industry ``_ElastoDyn.dat``
files are **not reproducible** from the structural-property
blocks in the same files. Worst block: ``TwSSM2Sh`` on the
NREL 5MW land deck (file_rms 5.90 vs pyBmodes 0.0023 → 2,529×
ratio).

This is documented as the **coefficient-consistency story**
(``cases/ECOSYSTEM_FINDING.md``). ``pybmodes validate``
surfaces it as a per-block PASS / WARN / FAIL verdict;
``pybmodes patch`` rewrites the blocks from the structural
inputs.

This is not a pyBmodes limitation — it's an artefact of the
industry decks themselves. pyBmodes' role is to **surface and
fix** the drift, not pretend it doesn't exist.

OC3 Hywind asymmetric platform support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OC3 Hywind cross-coupled
``hydro_K + mooring_K`` is genuinely asymmetric after the
rigid-arm transformation. Symmetrising it (the pre-1.0 bug
that the validation matrix surfaced) biased the platform
rigid-body modes by ~ 3.7 % — fixed by routing asymmetric ``K``
through :func:`scipy.linalg.eig` instead of
:func:`scipy.linalg.eigh`.

The symmetry check now lives in
:func:`pybmodes.fem.solver.solve_modes` and the dispatch is
automatic. If you build a model that produces a symmetric
``K`` you get the cheaper ``eigh`` path; the asymmetric path
is only taken when needed.

Pitch / roll coupling on IEA-15 UMaineSemi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyBmodes' coupled pitch / roll natural frequency lands ~ 9 %
below the Allen 2020 free-decay value (Table 11). This is
**expected and not a bug** — the two answer different
questions:

- **Allen's free-decay :math:`f_d`** measures the dominant
  period observed in a single-DOF excitation of pitch with
  the other DOFs damped out by nonlinear drag; close to the
  uncoupled diagonal estimate (0.0364 Hz).
- **pyBmodes' eigenvalue** is the true coupled linearised
  mode of the full 6-DOF rigid-body system; surge–pitch
  off-diagonal added mass
  :math:`A_\infty[0,4] = -1.20\times 10^8` kg·m (and the
  symmetric sway–roll term) lowers the coupled eigenvalue.

See ``cases/iea15_deep_diagnostic.md`` for the full numerical
walk-through and bug audit (zero confirmed bugs in pyBmodes'
eigenproblem assembly).

NREL 5MW monopile soil compliance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bundled monopile samples use ``hub_conn = 1`` (cantilever
clamped at the SubDyn reaction joint, no soil flexibility) —
matching ElastoDyn's clamped-base assumption for polynomial
generation. The resulting 1st-FA tower frequency is **~4 %
stiffer** than Jonkman 2010's published 0.275 Hz on OC3
monopile (pyBmodes reports 0.286 Hz).

For quantitative monopile modal analysis with soil included,
use ``CS_Monopile.bmi`` (``hub_conn = 3`` with populated
``mooring_K`` and distributed Winkler ``distr_k``) — bundled
at ``sample_inputs/reference_turbines/02_nrel5mw_oc3monopile/``.

What this is *not*
------------------

- **Not a multi-body dynamics solver.** Use OpenFAST +
  ElastoDyn for time-domain simulation.
- **Not a CFD code.** Hydrodynamics come in as 6 × 6 matrices
  from WAMIT / HydroDyn potential-flow output.
- **Not a structural design tool.** The supported workflow is
  *analysis of a defined structure*, not *optimisation of one*.
  Use WISDEM for design.
- **Not a validation-as-a-service product.** Numerical
  accuracy claims are documented per case in :doc:`validation`;
  deltas vs published references are surfaced, not hidden.
- **Not a Fortran wrapper.** ``pybmodes`` is pure-Python,
  ``numpy + scipy`` runtime; it doesn't call a BModes binary
  or any other compiled solver under the hood.

When to reach for a different tool
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - You need
     - Use
   * - Time-domain aero-servo-hydro-elastic simulation
     - OpenFAST
   * - High-fidelity 1-D beam with full nonlinearity
     - BeamDyn (inside OpenFAST)
   * - 3-D shell / volume element analysis
     - A general FEA package (Abaqus, Ansys, etc.)
   * - Design optimisation
     - WISDEM
   * - Time-domain dynamic mooring
     - MoorDyn (inside OpenFAST)
   * - WAMIT panel-mesh generation / hydrodynamic database
     - WAMIT / NEMOH / Capytaine
   * - First-order coupled modal frequencies + ElastoDyn
       polynomial round-trip + validation matrix
     - **pyBmodes** (you are here)
