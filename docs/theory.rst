Theory
======

What ``pybmodes`` models, what assumptions it carries, what
references it cross-validates against, and the non-obvious
implementation decisions that shape the public API.

The structural model
--------------------

``pybmodes`` solves the linearised modal eigenproblem for a
**slender beam** with coupled flap, lag, torsion, and axial
deformation. The element is a 15-DOF Bernoulli-Euler beam with
cubic-Hermite shape functions on the bending DOFs and linear
shape functions on the axial and torsion DOFs.

Discrete eigenproblem
^^^^^^^^^^^^^^^^^^^^^

After assembly + boundary-condition reduction the generalised
eigenproblem is:

.. math::

   \bigl[\, \mathbf{K} - \omega^2\, \mathbf{M} \,\bigr]\,
   \boldsymbol{\varphi} \;=\; \mathbf{0}

with :math:`\mathbf{K}` the global stiffness matrix (elastic +
geometric, including centrifugal stiffening on a rotating blade),
:math:`\mathbf{M}` the global mass matrix (including added mass
and parallel-axis contributions for a floating platform),
:math:`\boldsymbol{\varphi}` the mode-shape eigenvector, and
:math:`\omega` the natural circular frequency. pyBmodes reports
frequencies in **Hz**: :math:`f = \omega / (2\pi)`.

Element matrices
^^^^^^^^^^^^^^^^

The element-level stiffness and mass matrices are built from the
distributed section properties:

- **mass density** :math:`\rho A(s)` (kg / m)
- **axial stiffness** :math:`EA(s)` (N)
- **bending stiffness** :math:`EI_\text{flap}(s)`,
  :math:`EI_\text{edge}(s)` (N · m²)
- **torsional stiffness** :math:`GJ(s)` (N · m²)

with :math:`s` the dimensionless span coordinate. Mass and
elastic-centre offsets, pre-twist, and concentrated tip mass
are added through the same element framework.

Element matrix assembly is **vectorised over Gauss points and
over elements** via ``numpy.einsum`` — the original per-element
loop became a single tensor contraction with ~ 2–3× speedup on
representative towers without changing the numerics.

Sign convention + normalisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mode shapes are **mass-normalised**:

.. math::

   \boldsymbol{\varphi}_i^{\mathrm{T}}\, \mathbf{M}\,
   \boldsymbol{\varphi}_i \;=\; 1 \qquad \forall\, i

The sign of each mode shape is canonicalised by the rule
"maximum-amplitude DOF is positive", so two independent solves
of the same problem produce sign-stable shapes that can be MAC-
compared without a sign-flip ambiguity.

Boundary conditions
-------------------

Four ``hub_conn`` modes are supported. Each is cross-verified
against either the BModes Fortran reference solver or an
analytical closed form:

.. list-table::
   :header-rows: 1

   * - ``hub_conn``
     - Meaning
     - Verified against
     - Worst-case error
   * - 1
     - **Cantilever**. All six base DOFs locked.
     - BModes CertTest Test03 / Test04
     - < 0.005 %
   * - 2
     - **Free-free**. All six base DOFs released; reactions
       through the ``PlatformSupport`` block (6 × 6 mooring +
       hydrodynamic + inertial matrices).
     - OC3 Hywind floating spar (Jonkman 2010)
     - ≤ 0.0003 % across the first 9 modes
   * - 3
     - **Soft monopile**. Axial + torsion locked; lateral +
       rocking free. Optional distributed Winkler soil
       stiffness along the embedded section.
     - CS_Monopile (Jonkman & Musial 2010)
     - < 0.005 % across 10 modes
   * - 4
     - **Pinned-free cable**. Axial + transverse deflections +
       twist locked; bending **slopes free** — Bir 2009's cable
       BC.
     - Bir 2009 Eq. 8 (analytical Legendre polynomial)
     - < 0.5 % across flap modes 1–3, Ω ∈ {2..30} rad/s

See :doc:`validation` for the per-case test-file references.

Platform-attached frame (``hub_conn = 2``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``hub_conn = 2`` the tower lives in the **platform-attached
frame**: the rigid-arm transform brings ``PlatformSupport``'s
6 × 6 mass and stiffness matrices to the tower base before
assembly. The structural-inertia transform uses the full 3-D
arm ``r = (PtfmCMxt, PtfmCMyt, cm_pform − draft)``; the hydro
and mooring matrices are referenced at the platform reference
point (``PtfmRefxt = PtfmRefyt = 0`` for every standard
HydroDyn / WAMIT deck), so a zero horizontal arm applies to
those — see :func:`pybmodes.fem.nondim._rigid_arm_T`.

Asymmetric platform support routes the eigenproblem through
:func:`scipy.linalg.eig` (general dense) instead of
:func:`scipy.linalg.eigh` (symmetric) — see *Solver dispatch*
below. The OC3 Hywind cross-coupled ``hydro_K + mooring_K``
exercises this exact branch and is the benchmark behind the
0.0003 % regression.

ElastoDyn-compatible polynomial ansatz
--------------------------------------

ElastoDyn represents tower and blade mode shapes as a
constrained 6th-order polynomial in the dimensionless span
coordinate :math:`s = h / H`:

.. math::

   \mathrm{SHP}(s) \;=\; \sum_{i=2}^{6} c_i\, s^{\,i}

The constraint :math:`i \geq 2` enforces
:math:`\mathrm{SHP}(0) = \mathrm{SHP}'(0) = 0` algebraically —
a clamped-base condition. ``pybmodes`` fits FEM mode shapes to
this ansatz under a least-squares constraint with design-matrix
condition-number reporting. Public entry points:

- :func:`pybmodes.elastodyn.compute_blade_params`
- :func:`pybmodes.elastodyn.compute_tower_params`
- :func:`pybmodes.elastodyn.compute_tower_params_report` —
  same plus a ``TowerSelectionReport`` exposing FA / SS family
  scoring and rejected-mode lists.

Two non-obvious consequences shape every floating workflow:

Floating-deck polynomials use the cantilever basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The polynomial form can only express clamped-base modes. A
floating-deck workflow that wants ElastoDyn-compatible
polynomials must therefore use ``Tower.from_elastodyn(...)``
(``hub_conn = 1``), not ``Tower.from_bmi(...)`` with
``hub_conn = 2``.

The audit behind this — OpenFAST source-code citations showing
that ElastoDyn integrates only the tower beam plus
``TwrTpMass`` (no platform / hydro / mooring matrices) and adds
platform 6-DOF motion at runtime via the rigid-body sum — lives
in ``cases/ECOSYSTEM_FINDING.md`` and the
``FLOATING_CASES.md`` next to each bundled reference deck.

Torsion-contamination filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some FEM modes are hybrid bending + twist; the polynomial form
cannot represent twist content. ``_select_tower_family``
applies a two-gate filter:

1. **Polynomial-fit residual gate.** Candidates whose
   clamped-base polynomial fit has
   ``rms_residual <= 0.09`` survive.
2. **Torsion-contamination gate.** Candidates whose modal-
   kinetic-energy torsion fraction
   :math:`T_\text{tor} = \sum \varphi_\text{tor}^2 /
   \sum \varphi_\text{total}^2` is :math:`\geq 0.10` are dropped.

Per-mode :math:`(T_\text{FA}, T_\text{SS}, T_\text{tor})`
participations travel through
``TowerSelectionReport.rejected_fa_modes`` /
``rejected_ss_modes`` and
``CoeffBlockResult.rejected_modes`` so the user sees what was
dropped.

Improved Direct Method (root rigid-body subtraction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a free-free floating tower the FEM mode shape carries
non-zero deflection and slope at the root (rigid-body platform
motion). The polynomial form can't represent this directly;
``pybmodes`` subtracts the tangent line at the root —
:math:`\widetilde{y}(x) = y(x) - y(0) - y'(0) \cdot x` — before
fitting. That's BModes' *Improved Direct Method* (Bir 2010 §3);
the residual is what the polynomial sees. The alternative
*Projection Method* (rotate by :math:`-\arctan(\text{slope} \cdot
\text{scale})`) is not implemented — for small base slopes the
two are equivalent, and the validation matrix shows the IDM
suffices.

Solver dispatch
---------------

``pybmodes`` chooses between three SciPy eigensolvers based on
problem size + structure:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Solver
     - When used
     - Why
   * - :func:`scipy.linalg.eigh`
     - Small symmetric problems (default)
     - Stable + cheap; returns eigenvalues sorted and
       orthonormal eigenvectors.
   * - :func:`scipy.sparse.linalg.eigsh` (shift-invert)
     - Symmetric, ``ngd > 500``, subset of modes requested
     - 5–18× faster than dense ``eigh`` on real-tower meshes.
       ``mode='normal'`` with ``sigma=0`` (not ``'buckling'``,
       which degenerates to ``OP = K⁻¹ K = I`` there).
   * - :func:`scipy.linalg.eig`
     - Genuinely asymmetric :math:`\mathbf{K}` (after platform-
       support assembly)
     - Cross-coupled ``hydro_K + mooring_K`` on OC3 Hywind is
       not symmetric. Symmetrising would bias the platform
       rigid-body modes; the general dense path matches BModes.

The benchmark
(``scripts/benchmark_sparse_solver.py``) reports speedups at
n_elements ∈ {20, 50, 100, 200, 500} and asserts sparse beats
dense above 100 elements.

Pre-solve sanity checks
-----------------------

:func:`pybmodes.checks.check_model` runs **eight cheap,
deterministic gates** before a solve:

1. Span monotonicity (no out-of-order nodes)
2. Mass non-negativity at every node
3. Stiffness-jump detection (> 5× between adjacent stations)
4. FA / SS bending-ratio sanity (inside ``[0.1, 10]``)
5. RNA mass vs tower mass ratio (warn when RNA > tower)
6. Support-matrix singularity (for free-free without
   ``PlatformSupport``)
7. ``n_modes`` exceeding the DOF cap
8. Polynomial-fit design-matrix condition number > 1e4 (WARN)
   / > 1e6 (FAIL)

Auto-runs in ``.run()`` (suppress with ``check_model=False``);
WARN and ERROR findings route through ``UserWarning``. INFO
findings are suppressed on the auto-path — they're contextual,
not actionable, and noisy at scale — but
``check_model(model)`` called directly surfaces every gate.

Mooring + hydrodynamics
-----------------------

Quasi-static elastic catenary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`pybmodes.mooring.MooringSystem` implements Jonkman 2007
Appendix B:

- **B-1 / B-2** for the fully-suspended branch (line entirely
  off the seabed).
- **B-7 / B-8** with :math:`C_B = 0` for the anchor-on-seabed
  branch (one end resting on the bottom).

The non-linear catenary :math:`(H, V_F)` system is solved by a
damped Newton iteration with an analytical 2 × 2 Jacobian.
``MooringSystem.stiffness_matrix(body_r6=None)`` returns the
6 × 6 linearised stiffness in OpenFAST DOF order
``[surge, sway, heave, roll, pitch, yaw]``. The OC3 Hywind
cross-coupling sign convention is pinned against Jonkman 2010
Table 5-1 by
:func:`tests.test_mooring.test_oc3hywind_bmi_dof_order_matches_jonkman_2010`.

WAMIT / HydroDyn
^^^^^^^^^^^^^^^^

:class:`pybmodes.io.wamit_reader.HydroDynReader` follows the
``PotFile`` pointer in a HydroDyn ``.dat`` to the WAMIT
``.1`` / ``.hst`` outputs and returns a ``WamitData`` with SI
``A_inf`` (infinite-frequency added mass), ``A_0``
(zero-frequency added mass), and ``C_hst`` (hydrostatic
restoring) 6 × 6 matrices.

WAMIT's v7 re-dimensionalisation (``ρ · L^k`` / ``ρ · g · L^k``
per DOF-pair type, :math:`k \in \{2..5\}`) is handled by the
reader; Fortran-style ``D`` / ``d`` exponent notation is
normalised before parsing; upper-triangle-only outputs are
mirrored after read.

Citable references
------------------

The reference set ``pybmodes`` validates against (NREL reports link to
the canonical ``docs.nrel.gov`` PDF or the OSTI.GOV record; papers by
DOI; textbooks have no DOI):

- **NREL 5MW Reference Turbine** — Jonkman, Butterfield, Musial,
  Scott (2009), *Definition of a 5-MW Reference Wind Turbine
  for Offshore System Development*, NREL/TP-500-38060
  (`PDF <https://docs.nrel.gov/docs/fy09osti/38060.pdf>`__).
- **OC3 Monopile** and **OC3 Hywind (floating spar)** —
  Jonkman & Musial (2010), NREL/TP-5000-48191
  (`PDF <https://docs.nrel.gov/docs/fy11osti/48191.pdf>`__);
  Jonkman (2010), *Definition of the Floating System for
  Phase IV of OC3*, NREL/TP-500-47535
  (`PDF <https://docs.nrel.gov/docs/fy10osti/47535.pdf>`__).
- **IEA-3.4-130-RWT**, IEA-10-198-RWT, IEA-15-240-RWT,
  IEA-22-280-RWT — Bortolotti, Tarrés, Dykes et al. (2019),
  *IEA Wind TCP Task 37: Systems Engineering in Wind Energy —
  WP2.1 Reference Wind Turbines*, NREL/TP-5000-73492
  (`PDF <https://docs.nrel.gov/docs/fy19osti/73492.pdf>`__), and the
  follow-on IEA Wind Task 37 reports for the larger sizes — notably
  the IEA-15-240-RWT in Gaertner et al. (2020), NREL/TP-5000-75698
  (`PDF <https://docs.nrel.gov/docs/fy20osti/75698.pdf>`__), on the
  UMaine VolturnUS-S platform of Allen et al. (2020),
  NREL/TP-5000-76773
  (`PDF <https://docs.nrel.gov/docs/fy20osti/76773.pdf>`__).
- **BModes** — Bir (2010), NREL/CP-500-47953
  (`OSTI 975394 <https://www.osti.gov/biblio/975394>`__).
- **Rotating-blade closed forms** — Wright (1982),
  *J. Appl. Mech.* 49(1), 197–202
  (DOI `10.1115/1.3161966 <https://doi.org/10.1115/1.3161966>`__);
  Bir (2009), AIAA 2009-1035
  (`OSTI 982266 <https://www.osti.gov/biblio/982266>`__);
  Bir (2010) Table 5 (tip-mass rotating blade); Bir 2009 Eq. 8
  (rotating cable, analytical Legendre solution).
- **Beam tip-mass formulas** — Blevins (1979 / 2016); Karnovsky &
  Lebed (2001). (Textbooks; no DOI.)
- **Catenary mooring** — Jonkman (2007), *Dynamics Modeling and
  Loads Analysis of an Offshore Floating Wind Turbine*,
  NREL/TP-500-41958, Appendix B
  (`OSTI 921803 <https://www.osti.gov/biblio/921803>`__).
- **WAMIT** — Lee & Newman (1991/2006), *WAMIT User Manual*.
- **Cable structures** — Irvine (1981), *Cable Structures*,
  MIT Press, §2.4.

See :doc:`validation` for the per-case mapping of
reference → quantity-being-checked → test file → worst-observed
margin.
