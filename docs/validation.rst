Validation
==========

``VALIDATION.md`` at the repository root is the single structured
source of truth for cross-checked numerics. Every validated case
(track A frequency-accuracy, track B coefficient-consistency, track C
supporting-pipeline) carries source / quantity / tolerance /
worst-observed margin / test file / external-data flag. It is
mechanically audited by ``scripts/audit_validation_claims.py`` in CI
to prevent the matrix drifting ahead of the tests.

The full matrix:

.. include:: ../VALIDATION.md
   :parser: myst_parser.sphinx_
