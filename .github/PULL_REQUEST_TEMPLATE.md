<!--
Thanks for opening a pull request to pyBmodes.

For non-trivial changes please open an issue first to discuss
scope — see ``CONTRIBUTING.md`` for the welcome-contributions
list and what's out of scope. Trivial typo / docs PRs don't
need a pre-issue.
-->

## Summary

<!--
1-3 bullet points naming the user-visible change. Reference
issue numbers (Fixes #123, Closes #456) so GitHub auto-links
and auto-closes on merge.
-->

-
-

## Why

<!--
The motivation. If this fixes a bug, the bug description; if
it adds a feature, the use case that demands it; if it tightens
validation, the citable reference being matched.
-->

## How

<!--
A short tour of the implementation. File paths + the key
function names. If the PR touches the FEM core or the
polynomial-fit path, also call out which validation-matrix row
will move and by how much.
-->

## Test plan

<!--
Tick what you've already verified locally. CI re-runs the same
plus the integration suite. A PR with no checked boxes will be
politely bounced.
-->

- [ ] `pytest` (default — self-contained suite)
- [ ] `ruff check src/ tests/ scripts/`
- [ ] `mypy src/pybmodes`
- [ ] `python scripts/audit_validation_claims.py`
- [ ] `pre-commit run --all-files` (or hooks installed via `pre-commit install`)
- [ ] `pytest -m integration` *(if applicable — needs `external/` data)*
- [ ] `sphinx-build -b html docs docs/_build/html` *(if docs touched)*

## Numerical impact

<!--
Required if you touched FEM matrices, polynomial fitting,
modal solving, or any validation-track code. Quote the
worst-case delta against the relevant reference. If "no
numerical change", say so explicitly.

Examples:
- Worst-case delta vs BModes JJ on OC3 Hywind: 0.0003 % → 0.0002 % (improvement)
- No numerical change — refactor only
- IEA-15 UMaineSemi TwSSM2Sh stays at WARN (representation limit, not regressed)
-->

## Public API impact

<!--
Tick one:
-->

- [ ] No public-API change (internal / docs / tests / CI only)
- [ ] **Minor**: new entry point / kwarg / dataclass field with default
- [ ] **Major**: a name on `docs/api_contract.rst` is renamed or removed (must coincide with a 2.x bump)

## Provenance

<!--
Tracked source files must NOT carry per-reviewer / per-tool
attribution (AI assistants, consulting firm names, individual
reviewer names). Use neutral wording in CHANGELOG and code
comments: "static review", "follow-up", "regression". See
CONTRIBUTING.md for the rule and the precedent.
-->

- [ ] CHANGELOG entry uses neutral provenance (no AI / vendor / individual-reviewer names)
- [ ] Code comments use neutral provenance
- [ ] No `Co-Authored-By:` trailer on commits
