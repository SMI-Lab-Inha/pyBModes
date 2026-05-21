# pyBmodes

[![CI](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis. It reads OpenFAST (ElastoDyn / SubDyn / HydroDyn / MoorDyn), BModes `.bmi`, and WISDEM / WindIO ontology YAML inputs; solves the coupled flap–lag–torsion–axial vibration modes with a 15-DOF Bernoulli-Euler beam element; and emits ElastoDyn-compatible mode-shape polynomials, MAC-tracked Campbell diagrams, and bundled Markdown / HTML / CSV reports.

Validated against the BModes Fortran reference solver on six benchmark cases (NREL 5MW land + OC3 monopile + OC3 Hywind floating spar, IEA-3.4-130-RWT, BModes CertTest 03 / 04) to **better than 0.01 %** on every comparison — the strict tolerance is enforced by the `pytest -m integration` suite (which needs the upstream OpenFAST / BModes decks staged under `external/`; see [`docs/data_sources.rst`](docs/data_sources.rst) for the layout and [`external/MANIFEST.toml`](external/MANIFEST.toml) for the pinned SHAs + file hashes you can verify against). Public CI runs the self-contained suite (synthetic + closed-form-referenced) and tolerates "no tests collected" on the integration step when the runner has no upstream data — see [`VALIDATION.md`](VALIDATION.md) for the full per-case matrix with external-data flags.

## Documentation

The full user guide, theory notes, API reference, and validation matrix live in the Sphinx site under [`docs/`](docs/) — built on [Read the Docs](https://pybmodes.readthedocs.io/) and reproducible locally with:

```bash
pip install -e ".[docs]"
make -C docs html
```

Direct links into the source tree:

- [Installation](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [Theory](docs/theory.rst)
- [Data sources](docs/data_sources.rst)
- [Limitations](docs/limitations.rst)
- [Validation matrix](docs/validation.rst) (cross-references [`VALIDATION.md`](VALIDATION.md))
- [API reference](docs/api.rst)
- [API contract](docs/api_contract.rst) (semver-frozen public surface)
- [Changelog](docs/changelog.rst) (cross-references [`CHANGELOG.md`](CHANGELOG.md))

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

See [`docs/installation.rst`](docs/installation.rst) for the full Windows + conda quickstart and the optional-extras matrix (`[plots]`, `[windio]`, `[notebook]`, `[docs]`).

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

More — Campbell sweeps, WindIO one-click, mode-by-mode MAC comparison, bundled reports — in [`docs/quickstart.rst`](docs/quickstart.rst).

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
