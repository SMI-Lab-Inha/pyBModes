# Contributing to pyBmodes

Thanks for considering a contribution. pyBmodes is a research-grade engineering library — most contributions will involve numerical validation, structural-dynamics references, or input-format support. The guidance below is what we've learned keeps the project shippable.

## Quick start

```bash
git clone https://github.com/SMI-Lab-Inha/pyBModes.git
cd pyBModes
python -m venv .venv && source .venv/bin/activate   # or conda env
pip install -e ".[dev,plots]"
pytest
```

The default `pytest` run uses only self-contained tests (synthetic decks + closed-form references). The `integration` marker gates tests that need upstream OpenFAST / BModes data; place those clones under `external/` (gitignored). See [installation](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/docs/installation.rst) for the full Windows + conda quickstart.

## Ground rules

### 1. Stage explicit paths

Never `git add -A` or `git add .`. Untracked upstream-data directories under `external/` would leak in as embedded-repo gitlinks. Stage by name:

```bash
git add src/pybmodes/campbell.py tests/test_campbell.py
```

### 2. Author = committer = Jae Hoon Seo

Every commit must have Jae Hoon Seo as both author and committer. **Do not** add `Co-Authored-By` trailers — they appear in GitHub's Contributors sidebar via the trailer's email address and require a Support ticket to remove. If you used a tool that auto-injects them, strip them before pushing.

### 3. No tool / vendor attribution in tracked content

Code comments, CHANGELOG entries, test docstrings, and documentation must not name AI assistants, external review consultancies, or individual reviewers. Use neutral engineering provenance: "static review", "follow-up", "regression". Issue numbers (`issue #54`) are fine and encouraged.

### 4. Independence stance

The default test suite must run on a fresh clone with no external data. Don't introduce a default-run test that requires data outside `external/`. Mark data-dependent tests with `@pytest.mark.integration` (or `pytestmark = pytest.mark.integration` for whole modules).

### 5. Validate against citable references

When adding a new test that exercises FEM behaviour, cite the reference (textbook, peer-reviewed paper, NREL technical report). The validation matrix in `VALIDATION.md` is the single source of truth and is mechanically audited by `scripts/audit_validation_claims.py` in CI.

## Development workflow

### Pre-commit

Install hooks once per clone:

```bash
pip install pre-commit
pre-commit install
```

This runs ruff (auto-fix), the standard hygiene hooks (trailing-whitespace, end-of-file-fixer, YAML / TOML validation, large-file guard, merge-conflict marker check), `codespell`, and `insert-license` — which inserts the Apache 2.0 boilerplate header into any new `.py` file under `src/pybmodes/`, `scripts/`, `cases/`, or `noxfile.py` that doesn't already have it (template at [`.license_header.txt`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/.license_header.txt)). To run manually:

```bash
pre-commit run --all-files
```

### Lint + type-check + audit

```bash
ruff check src/ tests/ scripts/
mypy src/pybmodes
python scripts/audit_validation_claims.py
```

CI matches this exact scope. The `audit_validation_claims.py` script gates "claim ahead of test" drift — every test-file link in `VALIDATION.md` must point at a file containing at least one `def test_…` method.

### Running tests

```bash
pytest                          # self-contained (default)
pytest -m integration           # needs external/ data
pytest tests/test_campbell.py   # one module
pytest -k "blade_label"         # by keyword
```

### Building the docs locally

```bash
pip install -e ".[docs]"
make -C docs strict   # treats warnings as errors (mirrors Read the Docs + CI)
open docs/_build/html/index.html
```

### Using nox

A `noxfile.py` ships pre-built sessions for the common loops:

```bash
nox -s lint          # ruff
nox -s type          # mypy
nox -s tests         # pytest -m 'not integration'
nox -s docs          # sphinx-build -W
nox -s build         # python -m build (sdist + wheel)
```

## Pull-request checklist

Before opening a PR:

- [ ] Tests pass: `pytest` (default) and `pytest -m integration` if you touched data-dependent code.
- [ ] Lint clean: `ruff check src/ tests/ scripts/`.
- [ ] Type-check clean: `mypy src/pybmodes`.
- [ ] Validation audit clean: `python scripts/audit_validation_claims.py`.
- [ ] If you added a public name, it's listed in `src/pybmodes/__init__.py`'s docstring and (if applicable) the README's *Public API* section.
- [ ] If you changed numerical behaviour, `CHANGELOG.md` calls out the magnitude under *Fixed* / *Changed* and a regression test pins the new behaviour.
- [ ] Commit message follows the project's conventional-ish style (look at `git log --oneline -20` for examples).

## Release process

For maintainers, the full pre-tag sequence is in [`https://pybmodes.readthedocs.io/en/latest/release_checklist.html`](https://pybmodes.readthedocs.io/en/latest/release_checklist.html).

## Reporting issues

Open a [GitHub issue](https://github.com/SMI-Lab-Inha/pyBModes/issues). For security-sensitive reports, see [`SECURITY.md`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/CODE_OF_CONDUCT.md).
