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

First, fetch (if needed) and verify the local ``external/`` tree
against the pinned manifest:

.. code-block:: bash

   python scripts/verify_external_data.py --clone    # fetch missing required clones
   python scripts/verify_external_data.py --strict

Expected from ``--strict``: ``0 WARN, 0 FAIL`` with PASS on every
*required* clone. ``SKIP`` is fine for the ``optional = true``
entries (MoorPy / RAFT / BModes) when they're absent. A WARN means
a SHA pin is ``TBD`` — bump those before release. A FAIL means a
required clone is missing or has drifted from its pinned SHA / file
hash; ``--clone`` fetches the missing ones, or
``git -C external/<clone> checkout <sha>`` fixes a drift.

Then run the integration suite:

.. code-block:: bash

   pytest -m integration -q

Expected: every collected test passes. If you don't have the
upstream decks cloned under ``external/`` and pinned per
``external/MANIFEST.toml``, this step exits with code 5 ("no
tests collected") — that's acceptable for a local pre-tag pass
**only** if you've separately verified the integration track on
another machine that does have the data. CI runs both steps;
the integration job's exit-5 path is allowed but every other
failure mode is a hard fail.

.. note::

   The pinned manifest is the published reproducibility
   contract for the validation matrix's external-data tracks:
   manifest-pinned commit SHAs plus line-ending-normalized
   SHA-256 hashes for the text validation decks. The public
   required set (r-test + IEA-3.4 / 10 / 15 / 22 + WISDEM) is
   **CI-gated** by ``validation.yml`` and reproducible by anyone
   with the manifest; only the BModes archive (a NREL download,
   not publicly clonable) and the optional cross-comparison
   clones (MoorPy / RAFT) stay maintainer-local. Treat the
   manifest pins as part of the API contract — bumping a pin is
   a deliberate maintainer action documented in the corresponding
   ``CHANGELOG.md`` entry under *Changed*.

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

The walkthrough notebooks ship **source-only** — committed without
executed cell outputs (no stored figures). CI executes every cell
headlessly via ``tests/test_notebooks.py``; reproduce that here:

.. code-block:: bash

   pytest tests/test_notebooks.py                  # synthetic walkthrough (default)
   pytest -m integration tests/test_notebooks.py   # the two IEA-15 notebooks (needs external/ data)

Expected: ``notebooks/walkthrough.ipynb`` executes in the default run;
the two ``cases/`` IEA-15 notebooks execute under the integration
marker once the upstream decks are present (and assert the friendly
``FileNotFoundError`` contract when the data is absent).

To eyeball the rendered figures (optional — outputs aren't committed),
execute one to a transient copy and open it, then delete it:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute notebooks/walkthrough.ipynb --output _smoke.ipynb
   # ...inspect notebooks/_smoke.ipynb, then remove it (transient artefact)

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
- ``CITATION.cff``: bump the software ``version:`` to ``X.Y.Z`` and
  ``date-released:`` to the release date (this is the ``version:``
  field, not the ``cff-version:`` schema field). ``tests/test_version.py``
  gates this against ``pyproject`` so a forgotten bump fails CI.
- ``CHANGELOG.md``: promote the ``## [Unreleased]`` block to
  ``## [X.Y.Z] — YYYY-MM-DD``; reset ``[Unreleased]`` to
  ``(nothing yet)``.

Commit with a stand-alone message like
``chore: bump version to X.Y.Z, promote CHANGELOG``. Verify the
commit's stat shows only those four files changed.

9. Tag + push
-------------

.. code-block:: bash

   git push origin master
   git tag -a vX.Y.Z -m "pyBmodes X.Y.Z — <one-line release headline>"
   git push origin vX.Y.Z

The ``v`` prefix is the standard convention PyPI, GitHub Releases,
and conda-forge all expect. Push the master branch *before* the
tag so the tag refers to a commit that's on the remote.

.. important::

   **Run the Validation (external data) workflow on the release
   commit and confirm it is green before pushing the tag.** The
   publish workflow's ``validation-gate`` job refuses to publish
   unless a successful ``validation.yml`` run exists for the exact
   commit the tag points at. After ``git push origin master``,
   dispatch it from the Actions tab (Validation (external data) →
   Run workflow → ``master``), wait for green, *then* push the tag.

The tag push fires the **PyPI publish workflow**
(``.github/workflows/publish.yml``) — see step 10 below. Don't
push the tag until you're ready for PyPI to receive the artefact.

10. PyPI publish (automatic, but verify)
----------------------------------------

Pushing the tag fires ``Publish to PyPI`` via Trusted
Publishing. Watch the workflow on the Actions tab:

1. ``build-and-smoke`` — builds sdist + wheel, asserts the tag
   matches ``pyproject.toml`` version, smoke-installs both into
   fresh venvs. A failure here usually means a forgotten
   ``MANIFEST.in`` entry or a ``[project] version`` line that
   doesn't match the tag.
2. ``validation-gate`` — asserts a green *Validation (external
   data)* run exists for this commit. If you skipped the dispatch
   in step 9 this fails closed; run the validation workflow on the
   commit, wait for green, and re-run this job (or re-tag).
3. ``publish`` — pulls the built artefacts and uploads to PyPI
   via the OIDC handshake (no API token). The ``pypi``
   environment may have required-reviewer protection turned
   on; approve the deployment on the Actions UI if so.

After the workflow goes green, verify on PyPI:

.. code-block:: bash

   # in a throwaway venv
   python -m venv /tmp/pypi-check
   /tmp/pypi-check/bin/pip install --upgrade pip
   /tmp/pypi-check/bin/pip install pybmodes==X.Y.Z
   /tmp/pypi-check/bin/python -c "import pybmodes; print(pybmodes.__version__)"

Expected: prints ``X.Y.Z`` exactly. If pip can't find the
version, give PyPI a few minutes to propagate to its CDN.

.. note::

   **Trusted Publishing must be configured PyPI-side first.**
   The maintainer registers ``pybmodes`` on PyPI, adds a
   Trusted Publisher entry pointing at this repository +
   ``publish.yml`` + the ``pypi`` environment, and creates a
   matching environment under the repo settings. The
   ``publish.yml`` header carries the exact configuration.
   First-time setup takes ~5 min in the PyPI UI; subsequent
   releases are token-less and automatic.

11. GitHub Release
------------------

On https://github.com/SMI-Lab-Inha/pyBModes/releases/new :

1. Choose tag: ``vX.Y.Z``.
2. Release title: ``pyBmodes X.Y.Z``.
3. Paste the relevant ``## [X.Y.Z]`` section from ``CHANGELOG.md``
   as the release notes body. Add a brief *Highlights* section
   above the detailed changelog if the changeset is large enough
   to warrant one (the X.Y.0 minor-bumps usually do; patch-only
   bumps usually don't).
4. **Attach the verifier report** as a release asset: from the
   release-checklist machine, run
   ``python scripts/verify_external_data.py --strict > verify-vX.Y.Z.txt``
   and upload that file. Anyone reproducing the integration
   tolerance against the BModes Fortran reference can re-run
   the same verifier to confirm the manifest pins.
5. Set as the latest release: ✓ (unless this is a back-port).
6. Publish.

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
