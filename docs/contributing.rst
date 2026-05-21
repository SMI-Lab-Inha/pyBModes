Contributing
============

Thanks for considering a contribution. ``pybmodes`` is a
research-grade engineering library — most contributions will
involve **numerical validation**, **structural-dynamics
references**, or **input-format support**. The guidance below
is what we've learned keeps the project shippable.

Who this is for
---------------

- **End users** filing an issue or feature request — see
  :doc:`installation` for setup, then the
  `issue templates <https://github.com/SMI-Lab-Inha/pyBModes/issues/new/choose>`_
  on GitHub.
- **Contributors** opening a pull request — read on through
  ``CONTRIBUTING.md`` below.
- **Maintainers** preparing a release — see
  :doc:`release_checklist`.

Scope of welcome contributions
------------------------------

In rough order of "easiest to land":

1. **Documentation fixes** — typos, clarifications, missing
   examples in the docs you're reading.
2. **Validation matrix additions** — new cross-check against
   a citable reference (paper, BModes JJ run, BeamDyn run).
   Add the test + the matrix row in the same PR; the
   mechanical audit blocks claims without tests.
3. **Input-format support** — new dialect of an existing
   reader, a new keyword on an existing constructor, a new
   ``Tower.from_X`` for an industry-standard input that
   pyBmodes doesn't yet read.
4. **Bug fixes** — anything where a published reference and
   pyBmodes disagree by more than the documented tolerance.
   Use the
   `validation discrepancy issue template <https://github.com/SMI-Lab-Inha/pyBModes/issues/new?template=validation_discrepancy.yml>`_
   to anchor the report.
5. **Performance improvements** — speedups that come with a
   benchmark script; the existing
   ``scripts/benchmark_sparse_solver.py`` is the model.

Out of scope (will be politely closed):

- Wholesale rewrites of the FEM core that change validation
  numbers without a corresponding tightening of the
  tolerance and an updated matrix.
- New runtime dependencies for the core (the
  ``numpy + scipy``-only stance is a contract; new deps go
  in an extra).
- Refactor-only PRs touching the public API without a
  user-visible benefit.

The contribution guide
----------------------

.. include:: ../CONTRIBUTING.md
   :parser: myst_parser.sphinx_
