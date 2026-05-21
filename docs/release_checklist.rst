Release checklist
=================

Run every item before tagging a new ``pybmodes`` version. The goal
is not "the build looks green" — it's "nothing about the release
is unverified". Each step is a quick local command plus an explicit
expected outcome.

0. Prerequisites
----------------

Activate the dev environment (Windows + conda example; adapt for
your shell):

.. code-block:: bat

   call C:\Users\<you>\miniconda3\Scripts\activate.bat pybmodes
   set PYTHONPATH=%CD%\src

Working tree should be clean before starting:

.. code-block:: bash

   git status
   # expected: "nothing to commit, working tree clean"

1. Default test suite (self-contained, no external data)
--------------------------------------------------------

.. code-block:: bash

   pytest -q

Expected: every collected test passes. The default run skips
integration-marked tests cleanly (see :doc:`validation` for the
list of what needs external data).

2. Integration test suite (needs local OpenFAST + BModes decks)
---------------------------------------------------------------

.. code-block:: bash

   pytest -m integration -q

Expected: every collected test passes. If you don't have the
upstream decks cloned under ``external/``, this step exits with
code 5 ("no tests collected") — that's acceptable for a local
pre-tag pass **only** if you've separately verified the integration
track on another machine that does have the data. CI runs both
steps; the integration job's exit-5 path is allowed but every
other failure mode is a hard fail.

3. Linting + type checking
--------------------------

.. code-block:: bash

   python -m ruff check src/ tests/ scripts/
   python -m mypy src/pybmodes

Expected: both clean. ``scripts/`` is gated because user-facing
workflows (``build_reference_decks``, ``campbell``,
``visualise_*``) live there and any regression in them is
user-visible.

4. Sample-input verifier
------------------------

.. code-block:: bash

   python src/pybmodes/_examples/sample_inputs/verify.py

Expected: every analytical-reference sample passes at < 1 % RMS
against its closed-form reference. Output ends with a summary line
like ``Result: 4/4 sample case(s) passed.``.

4.5. Validation-matrix audit
----------------------------

.. code-block:: bash

   python scripts/audit_validation_claims.py

Expected: ``OK: every VALIDATION.md test-file reference exists and
contains at least one test method``. The script parses every
``tests/...`` link in ``VALIDATION.md``, asserts the path exists,
and asserts the file (or directory glob) contains at least one
``def test_…`` method — catching "claim ahead of test" drift where
the matrix advertises behaviour with no enforcing test. A non-zero
exit is a release blocker; either add the missing test or remove
the row from the matrix before tagging.

5. Reference-deck regeneration + validator
------------------------------------------

.. code-block:: bash

   python scripts/build_reference_decks.py

Expected: every case in the manifest builds successfully; the
post-patch validation report ends in ``Overall: PASS`` or
``Overall: WARN``. A FAIL verdict on any case is a release
blocker. The IEA-15 UMaine VolturnUS-S case is expected to end in
WARN on ``TwSSM2Sh`` — that's documented in
``src/pybmodes/_examples/reference_decks/FLOATING_CASES.md`` and
``src/pybmodes/_examples/reference_decks/iea15mw_umainesemi/validation_report.txt``'s
footer; treat any other WARN as new and investigate before
shipping.

6. Walkthrough notebook smoke-check
-----------------------------------

.. code-block:: bash

   jupyter nbconvert --to notebook --execute notebooks/walkthrough.ipynb --output _smoke.ipynb

Expected: every cell executes without error. Inspect the rendered
notebook briefly to confirm the figures aren't empty / clipped.
Delete ``notebooks/_smoke.ipynb`` afterwards (it's a transient
artefact).

7. Case scripts (optional — produce PNGs under ``outputs/``)
------------------------------------------------------------

.. code-block:: bash

   for case in cases/bir_2010_land_tower cases/bir_2010_monopile \
               cases/bir_2010_floating cases/nrel5mw_land \
               cases/iea3mw_land cases/nrel5mw_monopile; do
       python "$case/run.py"
   done

Expected: each writes its PNGs without raising. These are
local-data-dependent for the BModes case-test decks; the cases
under ``cases/nrel5mw_*/`` need ``external/OpenFAST_files/r-test/``
and the IEA-3.4 case needs
``external/OpenFAST_files/IEA-3.4-130-RWT/``. Missing-data exits
should be obvious from the per-case error message.

8. Version + CHANGELOG promotion
--------------------------------

- ``pyproject.toml``: bump ``version = "X.Y.Z"`` from the previous
  tag's value.
- ``src/pybmodes/__init__.py``: bump the dev fallback string
  ``__version__ = "X.Y.Z-dev"``.
- ``CHANGELOG.md``: promote the ``## [Unreleased]`` block to
  ``## [X.Y.Z] — YYYY-MM-DD``; reset ``[Unreleased]`` to
  ``(nothing yet)``.

Commit with a stand-alone message like
``chore: bump version to X.Y.Z, promote CHANGELOG``. Verify the
commit's stat shows only those three files changed.

9. Tag + push
-------------

.. code-block:: bash

   git push origin master
   git tag -a vX.Y.Z -m "pyBmodes X.Y.Z — <one-line release headline>"
   git push origin vX.Y.Z

The ``v`` prefix is the standard convention PyPI, GitHub Releases,
and conda-forge all expect. Push the master branch *before* the
tag so the tag refers to a commit that's on the remote.

10. GitHub Release
------------------

On https://github.com/SMI-Lab-Inha/pyBModes/releases/new :

1. Choose tag: ``vX.Y.Z``.
2. Release title: ``pyBmodes X.Y.Z``.
3. Paste the relevant ``## [X.Y.Z]`` section from ``CHANGELOG.md``
   as the release notes body. Add a brief *Highlights* section
   above the detailed changelog if the changeset is large enough
   to warrant one (the X.Y.0 minor-bumps usually do; patch-only
   bumps usually don't).
4. Set as the latest release: ✓ (unless this is a back-port).
5. Publish.

The GitHub Actions CI badge in README will repaint to green on the
new tag's commit automatically.

11. Post-release sanity
-----------------------

.. code-block:: bash

   git fetch --tags
   git tag -l "v*" | tail -5

Expected: the new tag is in the list and matches what's on origin.

.. code-block:: bash

   pip install -e . --quiet
   python -c "import pybmodes; print(pybmodes.__version__)"

Expected: the version reported matches the tag exactly (no
``-dev`` suffix — the install picks up the value from
``pyproject.toml``).

If any step fails, **do not push the tag**. Fix the underlying
issue and re-run from the point of failure. The checklist exists
because the cost of a botched public tag (deleting it, retagging,
re-publishing) is much higher than the cost of running through ten
local verifications first.
