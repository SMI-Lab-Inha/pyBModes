# pyBmodes

[![CI](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml)
[![Validation](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/validation.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/validation.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis. It reads OpenFAST (ElastoDyn / SubDyn / HydroDyn / MoorDyn), BModes `.bmi`, and WISDEM / WindIO ontology YAML inputs; solves the coupled flap–lag–torsion–axial vibration modes with a 15-DOF Bernoulli-Euler beam element; and emits ElastoDyn-compatible mode-shape polynomials, MAC-tracked Campbell diagrams, and bundled Markdown / HTML / CSV reports.

Validated against the BModes Fortran reference solver on six benchmark cases (NREL 5MW land + OC3 monopile + OC3 Hywind floating spar, IEA-3.4-130-RWT, BModes CertTest 03 / 04) to **better than 0.01 %** on every comparison. As of 1.8.0 the strict tolerance is enforced in CI by the dedicated [Validation workflow](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/validation.yml) (weekly cron + `workflow_dispatch`), which clones the upstream OpenFAST / IEA-Task-37 repositories on the fly and runs `pytest -m integration` hard-fail — independent of maintainer-local state. The per-PR `ci.yml` continues to run the self-contained suite (synthetic + closed-form-referenced) and tolerates "no tests collected" on the integration step when the runner has no upstream data. See [`VALIDATION.md`](VALIDATION.md) for the full per-case matrix with external-data flags and [`external/MANIFEST.toml`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/external/MANIFEST.toml) for the pinned SHAs + file hashes you can verify against.

## Documentation

> 📖 **The rendered documentation lives at [pybmodes.readthedocs.io](https://pybmodes.readthedocs.io/en/latest/).** Sphinx-specific roles (`:math:`, `:doc:`, `:func:`, `:class:`) inside the `.rst` source files only render correctly through the deployed site; browsing the raw source on GitHub will show them as literal text. Always link readers to the RTD URL, not the source.
>
> *(If the URL 404s, the Read the Docs project hasn't been imported yet — see [`docs/deployment.rst`](https://pybmodes.readthedocs.io/en/latest/deployment.html) for the one-time maintainer setup.)*

| Page | What's there |
| --- | --- |
| [Installation](https://pybmodes.readthedocs.io/en/latest/installation.html) | PyPI / source install, extras matrix, Windows + conda quickstart, troubleshooting |
| [Quickstart](https://pybmodes.readthedocs.io/en/latest/quickstart.html) | Nine worked recipes — synthetic tower, OpenFAST deck, monopile + SubDyn, floating coupled, Campbell sweep, WindIO one-click, MAC, batch, persistence |
| [Theory](https://pybmodes.readthedocs.io/en/latest/theory.html) | Eigenproblem maths, 15-DOF beam element, four boundary conditions, polynomial ansatz, solver dispatch, citable references |
| [Data sources](https://pybmodes.readthedocs.io/en/latest/data_sources.html) | Every input format — BModes `.bmi`, ElastoDyn / SubDyn / HydroDyn / MoorDyn `.dat`, WAMIT `.1` / `.hst`, WindIO `.yaml` — with snippet examples |
| [Units](https://pybmodes.readthedocs.io/en/latest/units.html) | SI conventions, conversion tables, mode-shape normalisation, OpenFAST DOF order, common pitfalls |
| [Limitations](https://pybmodes.readthedocs.io/en/latest/limitations.html) | Polynomial-representation limits, four specific validation-matrix edge cases, "when to reach for a different tool" |
| [Validation matrix](https://pybmodes.readthedocs.io/en/latest/validation.html) | Per-case cross-checks against published references (cross-references [`VALIDATION.md`](VALIDATION.md)) |
| [API reference](https://pybmodes.readthedocs.io/en/latest/api.html) | Autodoc-generated module reference |
| [API contract](https://pybmodes.readthedocs.io/en/latest/api_contract.html) | Semver-frozen public surface + deprecation policy + stability tiers |
| [Changelog](https://pybmodes.readthedocs.io/en/latest/changelog.html) | Versioning policy + full release history (cross-references [`CHANGELOG.md`](CHANGELOG.md)) |
| [Contributing](https://pybmodes.readthedocs.io/en/latest/contributing.html) | Welcome scope, pre-commit, PR checklist, no-AI-attribution rule |
| [Release checklist](https://pybmodes.readthedocs.io/en/latest/release_checklist.html) | 11-step pre-tag sequence (maintainer) |
| [Deployment](https://pybmodes.readthedocs.io/en/latest/deployment.html) | One-time RTD setup + versioning policy (maintainer) |

To build locally:

```bash
pip install -e ".[docs]"
make -C docs html
# then open docs/_build/html/index.html through a real web server
# (file:// blocks MathJax CDN in some browsers):
python -m http.server -d docs/_build/html
```

## Install

> **PyPI status: pre-release.** The `pybmodes` distribution is not yet published to PyPI — that's tracked as a 1.x release-gate item. Until the first PyPI release lands, install from source:

```bash
git clone https://github.com/SMI-Lab-Inha/pyBModes.git
cd pyBModes
pip install -e ".[dev,plots]"
```

Once published, the canonical install will be the standard one:

```bash
pip install pybmodes        # (post-PyPI-release; not available yet)
```

Take care that **`pybmodes` is a different project from `pyModeS`** (an ADS-B / Mode-S decoder). When the PyPI release lands the project name on PyPI will be `pybmodes` (lowercase, no S); double-check the package name + the GitHub `SMI-Lab-Inha/pyBModes` repo URL before installing.

See [`https://pybmodes.readthedocs.io/en/latest/installation.html`](https://pybmodes.readthedocs.io/en/latest/installation.html) for the full Windows + conda quickstart and the optional-extras matrix (`[plots]`, `[windio]`, `[notebook]`, `[docs]`).

## Quick example

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat

# Reads ElastoDyn main + tower from one path; lumps the rotor mass.
tower = Tower.from_elastodyn("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat")
modal = tower.run(n_modes=4)

# Constrained 6th-order fit, FA/SS family selection with torsion-contamination filter.
params = compute_tower_params(modal)

# Rewrite the polynomial blocks (use --dry-run / --diff via the CLI for safety).
patch_dat("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat", params)
```

More — Campbell sweeps, WindIO one-click, mode-by-mode MAC comparison, bundled reports — in [`https://pybmodes.readthedocs.io/en/latest/quickstart.html`](https://pybmodes.readthedocs.io/en/latest/quickstart.html).

## CLI

Seven subcommands surfaced as the `pybmodes` console script:

| Subcommand | Purpose |
| --- | --- |
| `pybmodes validate <main.dat>` | Coefficient-consistency report on one ElastoDyn deck. |
| `pybmodes patch <main.dat>` | Regenerate polynomial blocks. `--dry-run` / `--diff` / `--backup` / `--output-dir` for safety. |
| `pybmodes campbell <input>` | Rotor-speed sweep → Campbell diagram PNG + CSV. |
| `pybmodes batch ROOT` | Walk a directory of decks; per-deck validate + patch + summary CSV. |
| `pybmodes report <main.dat>` | Bundled Markdown / HTML / CSV analysis report. |
| `pybmodes windio <yaml \| dir>` | One-click WISDEM / WindIO → composite blade + tubular tower + coupled platform + Campbell. |
| `pybmodes examples --copy DIR` | Vendor `sample_inputs/` and / or `reference_decks/` out of the installed wheel. |

## Development

```bash
pytest                                       # default — self-contained, no external data
pytest -m integration                        # integration — needs upstream decks under external/
ruff check src/ tests/ scripts/
mypy src/pybmodes
python scripts/audit_validation_claims.py    # gates "claim ahead of test" drift
```

The default `pytest` run is **self-contained** and works on a fresh clone with no external data. Tests that need locally-checked-out OpenFAST `r-test`, BModes CertTest, or IEA-RWT decks are gated behind the `integration` marker; CI tolerates exit code 5 ("no tests collected") on a runner without the data. Full developer guide in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use pyBmodes in academic work, please cite it via the [`CITATION.cff`](CITATION.cff) file. GitHub's *Cite this repository* widget reads it automatically; Zenodo and most reference managers pick it up too.

## License

Released under the [Apache License 2.0](LICENSE).

Copyright 2024-2026 Jae Hoon Seo, Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University.
