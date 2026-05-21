Data sources
============

Every input format ``pybmodes`` reads, and where the corresponding
data lives.

Supported input formats
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Read by
   * - BModes ``.bmi`` main + section-properties ``.dat``
     - :func:`pybmodes.io.bmi.read_bmi` (also ``Tower(bmi_path)`` /
       ``RotatingBlade(bmi_path)`` for the high-level path).
   * - OpenFAST ElastoDyn main + tower + blade ``.dat``
     - :class:`pybmodes.io.elastodyn_reader` /
       ``Tower.from_elastodyn`` / ``RotatingBlade.from_elastodyn``.
   * - OpenFAST SubDyn ``.dat`` (monopile pile geometry)
     - :mod:`pybmodes.io.subdyn_reader` /
       ``Tower.from_elastodyn_with_subdyn``.
   * - OpenFAST HydroDyn ``.dat`` + WAMIT ``.1`` / ``.hst``
       potential-flow output
     - :class:`pybmodes.io.wamit_reader.HydroDynReader` (``A_inf``,
       ``A_0``, ``C_hst`` 6 × 6 matrices, SI re-dimensionalised).
   * - OpenFAST MoorDyn ``.dat`` (v1 and v2)
     - :meth:`pybmodes.mooring.MooringSystem.from_moordyn` (catenary
       quasi-static, Jonkman 2007 Appendix B).
   * - BModes ``.out`` reference output
     - :func:`pybmodes.io.out_parser.read_out` (tolerant by default;
       ``strict=True`` raises with file/line/mode context).
   * - WISDEM / WindIO ontology ``.yaml`` — tubular tower / monopile
     - :func:`pybmodes.io.windio.read_windio_tubular` /
       ``Tower.from_windio`` (closed-form circular-tube reduction).
   * - WISDEM / WindIO ontology ``.yaml`` — composite blade
     - :func:`pybmodes.io.windio_blade.read_windio_blade` /
       ``RotatingBlade.from_windio`` (PreComp-class thin-wall
       multi-cell Bredt–Batho reduction of the layup, validated
       against BeamDyn 6 × 6).
   * - WISDEM / WindIO ontology ``.yaml`` — floating substructure
     - :func:`pybmodes.io.windio_floating.read_windio_floating` /
       ``Tower.from_windio_floating`` (Morison + RAFT end-cap added
       mass, hydrostatic restoring, catenary mooring).

Bundled examples (ship in every wheel)
--------------------------------------

The ``_examples`` package vendors two redistributable trees:

- ``src/pybmodes/_examples/sample_inputs/`` — pyBmodes-authored,
  Apache 2.0-licensed ``.bmi`` and section-properties ``.dat``
  files. Four analytical-reference cases at the top level (uniform
  blade, tower with concentrated top mass, rotating uniform blade,
  pinned-free cable) plus seven RWT samples under
  ``reference_turbines/``.

- ``src/pybmodes/_examples/reference_decks/`` — six patched
  ElastoDyn decks: three fixed-base (NREL 5MW land, OC3 monopile,
  IEA-3.4-130-RWT) and three floating (NREL 5MW OC3 Hywind spar,
  NREL 5MW OC4 DeepCwind semi, IEA-15-240-RWT UMaine VolturnUS-S).
  Every block reaches PASS or WARN; no FAIL after patching.

Copy a tree out of the wheel with:

.. code-block:: bash

   pybmodes examples --copy ./mydir            # both trees
   pybmodes examples --copy ./mydir --kind decks  # reference decks only

Local-only upstream data (``external/``)
----------------------------------------

The ``integration``-marked tests and several ``cases/`` walk-throughs
read upstream OpenFAST / BModes / WindIO data that is **not**
bundled — typically because it lives in a different GitHub repository
whose licence does not permit redistribution. The convention is:

* ``external/OpenFAST_files/`` — OpenFAST ``r-test`` regression-test
  decks plus the IEA-N-RWT GitHub releases.
* ``external/BModes/`` — the BModes Fortran reference solver's
  CertTest and ``docs/examples`` decks.
* ``external/MoorPy/`` and ``external/RAFT/`` — cross-comparison
  references for the mooring and floating-platform paths.
* ``external/references/`` — papers and technical reports cited from
  the validation matrix.

All ``external/`` paths are ``.gitignore``-d. A fresh clone runs the
**self-contained** test suite without any of this data; the
``integration`` marker gates the data-dependent subset and skips
cleanly when the data is absent.

Independence stance
-------------------

The default ``pytest`` run is self-contained — every test in it
runs from numbers either constructed inline in the test or
validated against published closed-form formulas. No third-party
reference data is bundled in the repo for this default run. CI
runs both the default and the integration suite; the integration
step tolerates ``pytest`` exit code 5 ("no tests collected") so the
job stays green on a runner without the upstream data, but fails on
any other non-zero exit so a custom workflow run that **does** have
the data surfaces real failures immediately.
