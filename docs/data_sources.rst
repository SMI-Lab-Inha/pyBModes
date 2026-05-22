Data sources
============

Every input format ``pybmodes`` reads, the bundled examples
that ship with the wheel, and the convention for staging
upstream third-party data under ``external/``.

Supported input formats
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Format
     - Read by
   * - BModes ``.bmi`` main + section-properties ``.dat``
     - :func:`pybmodes.io.bmi.read_bmi`; also ``Tower(bmi_path)``
       and ``RotatingBlade(bmi_path)`` for the high-level path.
   * - OpenFAST ElastoDyn main + tower + blade ``.dat``
     - ``pybmodes.io.elastodyn_reader``;
       :meth:`pybmodes.models.Tower.from_elastodyn` and
       :meth:`pybmodes.models.RotatingBlade.from_elastodyn`.
   * - OpenFAST SubDyn ``.dat`` (monopile pile geometry)
     - :mod:`pybmodes.io.subdyn_reader`;
       :meth:`pybmodes.models.Tower.from_elastodyn_with_subdyn`.
   * - OpenFAST HydroDyn ``.dat`` + WAMIT ``.1`` / ``.hst``
       potential-flow output
     - :class:`pybmodes.io.wamit_reader.HydroDynReader`
       (``A_inf``, ``A_0``, ``C_hst`` 6 × 6 matrices, SI
       re-dimensionalised).
   * - OpenFAST MoorDyn ``.dat`` (v1 ``CONNECTION`` and v2
       ``POINT`` layouts)
     - :meth:`pybmodes.mooring.MooringSystem.from_moordyn`.
   * - BModes ``.out`` reference output
     - :func:`pybmodes.io.out_parser.read_out` — tolerant by
       default; ``strict=True`` raises with file / line / mode
       context on short, non-numeric, non-finite, duplicate, or
       empty content.
   * - WISDEM / WindIO ontology ``.yaml`` — tubular tower /
       monopile
     - :func:`pybmodes.io.windio.read_windio_tubular`;
       :meth:`pybmodes.models.Tower.from_windio`.
   * - WISDEM / WindIO ontology ``.yaml`` — composite blade
     - :func:`pybmodes.io.windio_blade.read_windio_blade`;
       :meth:`pybmodes.models.RotatingBlade.from_windio`.
   * - WISDEM / WindIO ontology ``.yaml`` — floating substructure
     - :func:`pybmodes.io.windio_floating.read_windio_floating`;
       :meth:`pybmodes.models.Tower.from_windio_floating`.

BModes ``.bmi`` — the canonical input
-------------------------------------

A ``.bmi`` file is line-ordered ASCII with comment lines
starting at column 1. The library is tolerant of trailing
comments after data lines. Header fields drive the FEM
configuration; the section-properties table comes from a
separate ``_sec_props.dat`` referenced by the ``.bmi`` header.

A minimal header (rotating uniform blade, ``beam_type = 1``,
``hub_conn = 1``, no platform support):

.. code-block:: text

   ======================  pyBmodes test deck  ======================
   Title:    rotating uniform blade

   beam_type      1     1 = blade, 2 = tower
   romg_g       0.5     gravity acceleration (g)
   rot_rpm       7.5    rotor speed (rpm)
   precone       0.0
   bl_p_ang      0.0    blade pitch angle (deg)
   bl_length    61.5
   radius       63.0    radius to blade tip (m)
   hub_rad       1.5    radius to blade root (m)
   n_secs       50      number of FE section nodes

   hub_conn      1      1 = cantilever, 2 = free-free, 3 = monopile, 4 = pinned-free
   ...

For tower decks (``beam_type = 2``) the floating-platform block
follows the structural section. See the bundled
``sample_inputs/`` reference turbines for full canonical
layouts.

OpenFAST ElastoDyn — the industry path
--------------------------------------

``Tower.from_elastodyn`` reads three files reached from one
path:

1. The main ``.dat`` (tower height, RNA mass, platform 6-DOF
   parameters, polynomial blocks).
2. The ``TwrFile`` it points at (distributed tower properties).
3. The ``BldFile(1)`` it points at — read **only** to lump the
   rotor mass into the tower-top assembly, not solved as a
   blade.

The polynomial blocks the validator and patcher work on:

.. code-block:: text

   --- TOWER FORE-AFT MODE 1 SHAPE COEFFICIENTS ---
        0.7004     TwFAM1Sh(2) - Mode 1, coefficient of x^2 term
        2.1963     TwFAM1Sh(3) - Mode 1, coefficient of x^3 term
       -5.6202     TwFAM1Sh(4) - Mode 1, coefficient of x^4 term
        6.1209     TwFAM1Sh(5) - Mode 1, coefficient of x^5 term
       -2.3974     TwFAM1Sh(6) - Mode 1, coefficient of x^6 term

Same shape for ``TwFAM2Sh`` (FA mode 2), ``TwSSM1Sh``,
``TwSSM2Sh``, and the four blade ``Bld...`` blocks. The
coefficients are :math:`c_2, c_3, c_4, c_5, c_6` of
:math:`\mathrm{SHP}(s) = \sum_{i=2}^{6} c_i\, s^{\,i}` with the
constraint :math:`\sum c_i = 1` (mode shape unit at tip in the
same nondim). pyBmodes's writer re-fits and rewrites these
blocks in place; the validator scores the file's blocks against
pyBmodes's own fit and emits a per-block PASS / WARN / FAIL
verdict with the file-RMS / pyBmodes-RMS ratio.

OpenFAST SubDyn — monopile pile geometry
----------------------------------------

The SubDyn parser handles the joints + members + base-reaction-
joint layout of an OC3-style monopile. The supported model is a
**fixed-base** combined cantilever (clamped at the SubDyn
reaction joint, no soil flexibility): the pile is structurally
spliced below the ElastoDyn tower into a single beam.

For soft-pile soil compliance ( ``hub_conn = 3`` with a
populated ``mooring_K`` and a distributed ``distr_k`` along the
embedded section), use the ``CS_Monopile.bmi`` reference deck
pattern directly — the corresponding BMI is bundled in
``sample_inputs/reference_turbines/02_nrel5mw_oc3monopile``.

WAMIT / HydroDyn output
-----------------------

HydroDyn ``.dat`` carries a ``PotFile`` pointer to a WAMIT
output filename **stem**:

.. code-block:: text

   "IEA-15-240-RWT-UMaineSemi"   PotFile

The reader chains through to ``<stem>.1`` (frequency-domain
added-mass + radiation damping) and ``<stem>.hst`` (hydrostatic
restoring), parses both, applies the WAMIT v7
re-dimensionalisation (``ρ · L^k`` / ``ρ · g · L^k``,
:math:`k \in \{2..5\}` per DOF-pair type), and returns a
:class:`~pybmodes.io.wamit_reader.WamitData` with SI ``A_inf``,
``A_0``, and ``C_hst`` 6 × 6 matrices ready for
``Tower.from_elastodyn_with_mooring``.

Fortran-style ``D`` / ``d`` exponent notation is handled;
upper-triangle-only output files are mirrored symmetric.

OpenFAST MoorDyn — both layouts
-------------------------------

The MoorDyn parser handles **both** version layouts:

- **v1** uses ``CONNECTION`` blocks with positional columns.
- **v2** uses ``POINT`` blocks (renamed); the parser detects
  the version by header keyword.

The point-ID-validated column-order detection prevents a silent
swap when v1 and v2 deck conventions differ in column ordering.
``MooringSystem.from_moordyn(dat_path)`` returns a fully-formed
mooring system; ``MooringSystem.stiffness_matrix(body_r6=None)``
returns the linearised 6 × 6 stiffness in OpenFAST DOF order.

WISDEM / WindIO ontology
------------------------

WindIO ``.yaml`` ontologies are the WISDEM-side schema for
reference wind-turbine definitions. ``pybmodes`` reads three
sub-trees:

Tubular tower / monopile
^^^^^^^^^^^^^^^^^^^^^^^^

A pure circular-tube reduction is applied to the
``outer_shape`` + ``structure`` (modern dialect) or
``outer_shape_bem`` + ``internal_structure_2d_fem`` (older
dialect) blocks. Both are tolerated; WISDEM's duplicate-anchor
files (IEA-10) parse via a custom ``SafeLoader``.

Composite blade
^^^^^^^^^^^^^^^

A PreComp-class **thin-wall multi-cell Bredt–Batho** reduction
of the layup, with classical-lamination theory at each station:

- Airfoil ``nd_arc`` spine + anchor-registry-resolved sectoral
  coordinates per ply.
- Per-ply ``Qbar`` from material orientation + the lamina
  ``Q``-matrix.
- Cell-by-cell ``A``, ``B``, ``D`` matrices and the
  Bredt–Batho torsion compliance.
- Decoupled output: per-station ``EA``, ``EI_flap``,
  ``EI_edge``, ``GJ``, plus mass distribution.

Validated against the turbine's own BeamDyn 6 × 6 on IEA-3.4 /
10 / 15 / 22 ontologies (mass and ``EA`` PreComp-class; ``GJ``
and ``EI`` diagonal-reduction approximate, documented).

When a ``elastic_properties`` or ``elastic_properties_mb`` block
is present, the published distributed properties are preferred
by default (``elastic="auto"``) — this minimises deltas against
the reference model. ``elastic="precomp"`` forces the layup
reduction; ``elastic="file"`` requires the published properties
and raises if absent.

Floating substructure
^^^^^^^^^^^^^^^^^^^^^

For a ``components.floating_platform`` block,
``Tower.from_windio_floating`` is **two-tier**:

- **Industry-grade** when companion HydroDyn + MoorDyn +
  ElastoDyn decks are supplied (or auto-discovered by the
  ``pybmodes windio`` CLI scoped to the turbine root) —
  byte-identical to ``Tower.from_elastodyn_with_mooring``.
- **Screening preview** when only the yaml is supplied —
  member-Morison hydrodynamics, RAFT-style end-cap added mass,
  catenary mooring from the yaml. Emits one ``UserWarning``
  naming the result as ``SCREENING-fidelity (NOT
  industry-grade)``.

BModes ``.out`` — the reference output format
---------------------------------------------

For validation comparisons against BModes JJ runs, the
``.out`` reader is **tolerant by default** — it returns NaN for
unparseable rows so a partial output doesn't crash the whole
read. For automated validation in CI use ``strict=True``:

.. code-block:: python

   from pybmodes.io.out_parser import read_out

   modes = read_out("ref.out", strict=True)

Strict mode raises with file / line / mode context on short
rows, non-numeric fields, non-finite values, duplicate mode
numbers, or empty content.

Bundled examples (ship in every wheel)
--------------------------------------

The ``_examples`` package vendors two redistributable trees as
``setuptools.package-data``:

``src/pybmodes/_examples/sample_inputs/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyBmodes-authored, Apache 2.0-licensed ``.bmi`` and section-
properties ``.dat`` files. Four **analytical-reference cases**
at the top level:

- ``01_uniform_blade/`` — uniform isotropic cantilever blade
  vs. Bernoulli-Euler closed form.
- ``02_tower_topmass/`` — uniform tower with concentrated top
  mass vs. Blevins.
- ``03_rotating_uniform_blade/`` — rotating uniform blade vs.
  Wright 1982 / Bir 2009 Table 3a.
- ``04_pinned_free_cable/`` — pinned-free rotating cable vs.
  Bir 2009 Eq. 8 (the ``hub_conn = 4`` BC).

Plus eleven **reference-turbine samples** under
``reference_turbines/``:

- 01_nrel5mw_land, 02_nrel5mw_oc3monopile,
  03_iea34_land, 04_iea10_monopile, 05_iea15_monopile,
  06_iea22_monopile, 07_nrel5mw_oc3hywind_spar,
  08_nrel5mw_oc4semi, 09_iea15_umainesemi, 10_iea22_semi,
  11_upscale25_centraltower

``verify.py`` runs all four analytical references at <1 % RMS
vs. their closed form; ``reference_turbines/build.py``
regenerates each sample from upstream ElastoDyn decks.

``src/pybmodes/_examples/reference_decks/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Six **pre-patched ElastoDyn decks** — every coefficient block
reaches PASS or WARN; no FAIL after patching:

- Fixed-base: ``nrel5mw_land/``, ``nrel5mw_oc3monopile/``,
  ``iea34_land/``.
- Floating: ``nrel5mw_oc3spar/`` (OC3 Hywind),
  ``nrel5mw_oc4semi/`` (OC4 DeepCwind semi),
  ``iea15mw_umainesemi/`` (UMaine VolturnUS-S).

The floating decks use the cantilever path
(``Tower.from_elastodyn``, not ``Tower.from_bmi`` with
``hub_conn=2``) because ElastoDyn's polynomial ansatz assumes
clamped-base modes regardless of platform configuration —
see :doc:`theory` and :doc:`limitations`.

Vendor either tree out:

.. code-block:: bash

   pybmodes examples --copy ./my_examples            # both trees
   pybmodes examples --copy ./decks --kind decks     # decks only
   pybmodes examples --copy ./samples --kind samples # samples only

Works whether you installed from PyPI or in editable mode from
source — the resolver finds ``_examples/`` relative to
``pybmodes.__file__``.

Local-only upstream data (``external/``)
----------------------------------------

The ``integration``-marked tests and several ``cases/``
walk-throughs read upstream OpenFAST / BModes / WindIO data
that is **not** bundled — typically because it lives in a
different GitHub repository whose licence does not permit
redistribution. The convention is to clone each under
``external/``:

.. code-block:: text

   external/
     OpenFAST_files/
       r-test/                 OpenFAST regression-test suite
       IEA-3.4-130-RWT/        github.com/IEAWindTask37/IEA-3.4-130-RWT
       IEA-10.0-198-RWT/       github.com/IEAWindTask37/IEA-10.0-198-RWT
       IEA-15-240-RWT/         github.com/IEAWindTask37/IEA-15-240-RWT
       IEA-22-280-RWT/         github.com/IEAWindTask37/IEA-22-280-RWT
       WISDEM/                 github.com/WISDEM/WISDEM
     BModes/
       CertTest/               BModes CertTest reference outputs
       docs/examples/          CS_Monopile.bmi + OC3Hywind.bmi
     MoorPy/                   github.com/NREL/MoorPy (cross-checks)
     RAFT/                     github.com/WISDEM/RAFT (cross-checks)
     references/               PDF / TR references (Jonkman 2010 etc.)

All ``external/`` paths are gitignored. A fresh clone runs the
self-contained test suite without any of this data; the
``integration`` marker gates the data-dependent subset and
skips cleanly when the data is absent.

Independence stance
-------------------

The default ``pytest`` run is **self-contained** — every test
in it runs from numbers either constructed inline in the test
or validated against published closed-form formulas. No
third-party reference data is bundled in the repo for this
default run. Any commit that re-introduces a ``.bmi`` /
``.dat`` / ``.out`` file under ``tests/data/`` for a
default-run test should be questioned.

CI runs both the default and the integration suite; the
integration step tolerates ``pytest`` exit code 5 ("no tests
collected") so the job stays green on a runner without the
upstream data, but fails on any other non-zero exit so a custom
workflow run that *does* have the data surfaces real failures
immediately.
