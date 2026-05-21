Theory
======

What ``pybmodes`` models, what assumptions it carries, and what
references it cross-validates against.

FEM element
-----------

A **15-DOF Bernoulli-Euler beam element** with four-axis coupling
(flap, lag, twist, axial). Each node carries five physical DOFs —
two transverse deflections, two slope DOFs, and one twist — giving
15 DOFs per element after the cubic-Hermite shape-function basis.

The mass and stiffness matrices are assembled element-by-element via
the standard congruence transform, with gravity (centrifugal
stiffening for a rotating blade) folded in through a geometric-
stiffness term. Section properties — mass density, axial stiffness
``EA``, flapwise / edgewise bending stiffness ``EI_flap`` /
``EI_edge``, torsional stiffness ``GJ``, plus the centre-of-mass
offsets and pre-twist — come from the ``.bmi`` section-properties
table (or its ElastoDyn / WindIO equivalent).

Boundary conditions
-------------------

Four ``hub_conn`` modes are supported, each cross-verified against
either the BModes Fortran reference solver or an analytical closed
form:

.. list-table::
   :header-rows: 1

   * - ``hub_conn``
     - Meaning
     - Verified against
     - Worst-case error
   * - 1
     - Cantilever (clamped: deflections + slopes + twist)
     - BModes CertTest Test03 / Test04
     - < 0.005 %
   * - 2
     - Free-free (all six base DOFs released; reactions through ``PlatformSupport``)
     - OC3 Hywind floating spar (Jonkman 2010)
     - ≤ 0.0003 % across the first 9 modes
   * - 3
     - Soft monopile (axial + torsion locked; lateral + rocking free)
     - CS_Monopile (Jonkman & Musial 2010)
     - < 0.005 % across 10 modes
   * - 4
     - Pinned-free cable (axial + deflections + twist locked; slopes free)
     - Bir 2009 Eq. 8 (analytical Legendre solution)
     - < 0.5 % across flap modes 1–3

See :doc:`validation` for the full matrix.

Polynomial ansatz (ElastoDyn compatibility)
-------------------------------------------

ElastoDyn represents tower and blade mode shapes as constrained
6th-order polynomials in the dimensionless span coordinate
:math:`s = h / H`:

.. math::

   \mathrm{SHP}(s) = \sum_{i=2}^{6} c_i \, s^{\,i}

This form algebraically enforces ``SHP(0) = SHP'(0) = 0`` — a
clamped-base condition. ``pybmodes`` fits FEM mode shapes to this
ansatz under a least-squares constraint with design-matrix
condition-number reporting; the public entry points are
:func:`pybmodes.elastodyn.compute_blade_params` and
:func:`pybmodes.elastodyn.compute_tower_params`.

Two non-obvious consequences:

* **Floating tower polynomials must come from the cantilever
  basis.** ElastoDyn's polynomial ansatz can only express
  clamped-base modes; the runtime adds rigid-body platform motion
  separately. So a floating-deck workflow that wants polynomials
  uses ``Tower.from_elastodyn(...)`` (cantilever), not
  ``Tower.from_bmi(...)`` with ``hub_conn = 2``. The source-code
  audit behind this is at
  ``src/pybmodes/_examples/reference_decks/FLOATING_CASES.md``.
* **The torsion-contamination filter.** Some FEM modes are hybrid
  bending + twist; the constrained polynomial form cannot represent
  twist content. Candidates with torsion-energy fraction
  ``T_tor ≥ 0.10`` are dropped from FA / SS family selection.
  Rejected modes travel through
  ``TowerSelectionReport.rejected_fa_modes`` / ``rejected_ss_modes``.

Solver dispatch
---------------

``pybmodes`` dispatches between three SciPy eigensolvers based on the
problem size and structure:

* :func:`scipy.linalg.eigh` for small symmetric problems (default).
* :func:`scipy.sparse.linalg.eigsh` (shift-invert) for symmetric
  problems with ``ngd > 500`` DOFs — 5–18× faster on real towers.
  ``mode='normal'`` (not ``'buckling'``, which degenerates to
  ``OP = K⁻¹ K = I`` at ``sigma = 0``).
* :func:`scipy.linalg.eig` (general dense) when the platform-support
  contribution introduces genuine asymmetry — e.g. OC3 Hywind's
  cross-coupled ``hydro_K + mooring_K``. Routing decided by an
  automatic symmetry check on the assembled ``K``.

Pre-solve sanity checks
-----------------------

:func:`pybmodes.checks.check_model` runs eight cheap, deterministic
gates before a solve — non-monotonic span, zero / negative mass,
stiffness jumps > 5×, FA / SS ratio outside ``[0.1, 10]``, RNA mass
> tower mass, singular support matrix, ``n_modes`` above DOF cap,
and polynomial-fit design-matrix condition number > 1e4 / 1e6.
Auto-runs in ``.run()`` (suppress with ``check_model=False``);
WARN and ERROR findings route through ``UserWarning``.

Citable references
------------------

The reference set ``pybmodes`` validates against:

- **NREL 5MW Reference Turbine** — Jonkman, Butterfield, Musial, Scott
  (2009), *Definition of a 5-MW Reference Wind Turbine for Offshore
  System Development*, NREL/TP-500-38060.
- **OC3 Monopile** and **OC3 Hywind (floating spar)** — Jonkman &
  Musial (2010), NREL/TP-5000-48191; Jonkman (2010), NREL/TP-500-47535.
- **IEA-3.4-130-RWT**, IEA-10-198-RWT, IEA-15-240-RWT, IEA-22-280-RWT
  — Bortolotti, Tarrés, Dykes et al. (2019), NREL/TP-5000-73492 and
  the follow-on IEA Wind Task 37 reports.
- **BModes** — Bir (2010), NREL/CP-500-47953.
- **Rotating-blade closed forms** — Wright (1982); Bir (2009);
  Bir (2010) Table 5 (tip-mass rotating blade); Bir 2009 Eq. 8
  (rotating cable).
- **Beam tip-mass formulas** — Blevins (1979 / 2016).
