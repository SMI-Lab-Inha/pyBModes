Validation matrix
=================

``VALIDATION.md`` at the repository root is the **single
structured source of truth** for cross-checked numerics. Every
validated case carries:

- the **reference** being matched (citable publication, BModes
  Fortran solver output, closed-form analytical formula);
- the **quantity** being compared (mode frequency, mode-shape
  MAC, polynomial coefficient, derived quantity);
- the **tolerance** (relative or absolute, with units);
- the **worst-observed margin** at the time the matrix was
  last regenerated;
- the **test file** that enforces the tolerance in CI;
- the **external-data flag** — whether the test runs on a
  fresh clone (self-contained) or only with upstream data
  staged under ``external/``.

Mechanical audit
----------------

``scripts/audit_validation_claims.py`` parses every test-file
link in ``VALIDATION.md``, asserts the path exists, and
asserts the file (or directory glob) contains at least one
``def test_…`` method. Runs as a required CI step alongside
ruff and mypy, plus step 4.5 of :doc:`release_checklist`.

**Claims cannot drift ahead of tests.** If you remove a test,
you have to remove (or replace) its row in the matrix in the
same PR; if you add a row with no test, the audit blocks the
PR.

Tracks
------

The matrix splits validation work into three tracks:

Track A — frequency accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct mode-frequency comparisons against a reference. The
golden cases:

- **NREL 5MW** on **OC3 Monopile** vs BModes JJ (CS_Monopile.bmi)
  — ≤ 0.005 % across 10 modes.
- **NREL 5MW** on **OC3 Hywind** floating spar vs BModes JJ
  (OC3Hywind.bmi) — ≤ 0.0003 % across 9 modes.
- **BModes CertTest Test03 + Test04** (82.4 m tower with top
  mass + tension-wire support) — < 0.005 %.
- **Rotating uniform blade** vs Wright 1982 / Bir 2009 Table
  3a closed form — synthetic, self-contained.
- **Rotating blade with tip mass** vs Bir 2010 Table 5 — same.
- **Rotating pinned-free cable** vs Bir 2009 Eq. 8
  (analytical Legendre polynomial) — same.

Track B — coefficient consistency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whether the polynomial blocks shipped in industry
ElastoDyn ``.dat`` files are reproducible from the
structural-property blocks in the same files.

**They are not** — see ``cases/ECOSYSTEM_FINDING.md``. The
worst observation: ``TwSSM2Sh`` on NREL 5MW land deck,
file-RMS 5.90 vs pyBmodes-RMS 0.0023 (ratio 2,529×).
``pybmodes validate`` surfaces this as a per-block PASS /
WARN / FAIL verdict; ``pybmodes patch`` rewrites the blocks
from the structural inputs.

Six **patched reference decks** ship in the wheel under
``src/pybmodes/_examples/reference_decks/`` — every coefficient
block reaches PASS or WARN; no FAIL after patching. The
``before_patch.txt`` snapshots preserve the original drift for
reference.

Track C — supporting pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Component-level regressions for everything that supports
the two main tracks:

- BMI / ElastoDyn / SubDyn / WAMIT / MoorDyn parser
  round-trips
- Mooring catenary closed forms (Jonkman 2007 Appendix B)
- Hydrostatic restoring vs cylinder closed form
- Mode-shape classifier (FA / SS / twist labelling)
- Degenerate-pair resolver (symmetric-tower 2-fold
  eigenspaces)
- Pre-solve sanity checks
- Serialisation round-trips
- CLI smoke tests (every subcommand, every flag)

What "self-contained" means
---------------------------

The default ``pytest`` run is **self-contained**: every test
in it runs from numbers either constructed inline in the test
or validated against published closed-form formulas. No
third-party reference data is bundled in the repo for this
default run.

Any commit that re-introduces a ``.bmi`` / ``.dat`` / ``.out``
file under ``tests/data/`` for a default-run test should be
questioned. See :doc:`data_sources` for how external data is
staged under ``external/`` and gated behind the
``integration`` pytest marker.

The full matrix
---------------

.. include:: ../VALIDATION.md
   :parser: myst_parser.sphinx_
