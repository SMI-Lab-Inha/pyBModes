# pyBModes

[![CI](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml/badge.svg)](https://github.com/SMI-Lab-Inha/pyBModes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pybmodes` is a pure-Python finite-element library for wind-turbine blade and tower modal analysis.

## Overview

`pybmodes` solves coupled flap–lag–torsion–axial vibration modes of slender beams using a 15-DOF Bernoulli-Euler beam element formulation, dense `scipy.linalg.eigh` for small / medium systems, and sparse `scipy.sparse.linalg.eigsh` shift-invert for systems above 500 DOFs (5–18× faster on real towers). It can:

- read line-ordered `.bmi` main-input files and tabulated section-property `.dat` files;
- read OpenFAST input decks directly (`Tower.from_elastodyn(...)`, `Tower.from_elastodyn_with_subdyn(...)`, `RotatingBlade.from_elastodyn(...)`) — parses ElastoDyn main + tower + blade files, with optional SubDyn pile geometry spliced below the tower for monopile decks;
- build a tower / monopile from **geometry alone** — `Tower.from_geometry(...)` takes outer diameter + wall thickness + isotropic steel `(E, ρ, ν)` + an `outfitting_factor` and derives every distributed structural property (mass, EI, GJ, EA) by the exact closed-form circular-tube identities, eliminating hand-computation error; `Tower.from_windio(...)` reads a [WindIO](https://windio.readthedocs.io/) ontology `.yaml` (both the modern `outer_shape`/`structure` and the older `outer_shape_bem`/`internal_structure_2d_fem` dialects, tolerant of WISDEM's duplicate-anchor files) and feeds it through the same path — needs the optional `[windio]` extra (PyYAML);
- solve rotating blade modal problems with centrifugal stiffening, tip masses, and pre-twist;
- solve onshore and offshore tower modal problems with eight pre-solve sanity checks via `pybmodes.checks.check_model` (non-monotonic span, zero / negative mass, stiffness jumps, FA/SS ratio, RNA mass dominance, singular support matrix, `n_modes` overrun, polynomial-fit conditioning) — runs automatically on `.run()`, suppress with `check_model=False`;
- go **one-click from a WISDEM/WindIO ontology** `.yaml` (or an RWT directory) to the full modal picture — `pybmodes windio <yaml>` discovers the ontology and any companion HydroDyn/MoorDyn/ElastoDyn decks (scoped to that turbine), then solves the composite-layup blade, the tubular tower, and — for a `floating_platform` — the coupled platform rigid-body modes, with an optional Campbell sweep and a bundled report. The blade defaults to its WindIO **published distributed elastic properties** when the ontology carries them (the common case — minimises deltas against the reference model) and falls back to a PreComp-class thin-wall multi-cell classical-lamination reduction of the **composite layup** only when they are absent (`RotatingBlade.from_windio(..., elastic="auto")`, issue #48); `elastic="precomp"` forces the layup reduction, `"file"` requires the published properties. The floating path is **two-tier**: with the companion decks present `Tower.from_windio_floating(...)` is the BModes-JJ-validated industry-grade coupled model (all six platform rigid-body modes + 1st tower bending within 0.0–0.3 % of `from_elastodyn_with_mooring`); without them it degrades to a member-Morison + catenary screening preview that says so via a `UserWarning`. Needs the optional `[windio]` extra (PyYAML);
- fit ElastoDyn-compatible 6th-order blade and tower mode-shape polynomials, with design-matrix condition-number reporting, automatic resolution of degenerate FA/SS eigenpairs on symmetric structures, and a torsion-contamination filter that drops candidates with `T_tor ≥ 10 %` from the family selection;
- patch OpenFAST ElastoDyn input files with fitted coefficients in three modes: in-place (with optional `.bak` backup), `--dry-run` (compute + summarise, write nothing), `--diff` (PR-ready coefficient-only diff with per-block RMS-improvement ratios), or `--output-dir DIR` (write to a separate directory, originals untouched);
- assemble a Campbell diagram from a single OpenFAST deck — blade modes swept across rotor speed with Hungarian MAC-based tracking, tower modes overlaid as horizontal lines, plus the per-rev (1P, 3P, 6P, …) excitation family — for resonance checks like NREL 5MW's *3P × 1st-tower-FA at ~6–7 rpm*; per-step MAC confidence is exposed as `CampbellResult.mac_to_previous` for tracking-quality audits. For a coupled floating tower the six lowest tower columns come out natively named `surge` / `sway` / `heave` / `roll` / `pitch` / `yaw` (the FEM's own BModes-cross-validated classification carried through `CampbellResult.labels`) and `plot_campbell` auto-draws them in the navy platform family with frequency-and-period labels — no need to pass `platform_modes` by hand (it is still honoured for the screening path). `log_freq=True` renders correctly, including the per-rev rays (issue #47); the `pybmodes windio` floating path wires all of this automatically;
- compare two modal results mode-by-mode via `pybmodes.mac.compare_modes` — full MAC matrix, Hungarian-optimal pairing, per-pair % frequency shift, heatmap plotting via `plot_mac`;
- serialise results to disk: `ModalResult.save(.npz)` / `to_json(.json)` round-trips frequencies + mode shapes + optional participation + fit residuals + pyBmodes-version / timestamp / source-file / git-hash metadata; `CampbellResult.save(.npz)` / `to_csv(.csv)` similarly;
- emit bundled reports via `pybmodes.report.generate_report` — Markdown / HTML / CSV summary covering model assumptions, frequencies, mode classification, polynomial coefficients with fit residuals, validation verdict, `check_model` warnings, and optional Campbell-sweep first/last frequencies per mode;
- walk a directory of decks with `pybmodes batch ROOT --validate --patch` — discovers ElastoDyn mains, runs validate / patch per deck, writes per-deck reports and a summary CSV;
- plot FEM mode shapes, polynomial-fit quality, MAC heatmaps, Campbell diagrams, and the environmental-loading frequency-placement diagram (Kaimal wind + JONSWAP wave spectra with the 1P/3P design/constraint bands against the tower fore-aft / side-side frequencies — the soft-stiff separation figure for floating turbines) via the optional `[plots]` extra (standard black/red/blue/green engineering-paper defaults via `apply_style()`);
- parse BModes `.out` reference output tolerantly by default, or with `read_out(path, strict=True)` for fail-loud validation (raises with file / line / mode context on short, non-numeric, non-finite, duplicate, or empty content).

Supported tower configurations (all cross-verified against the BModes Fortran reference solver):

- cantilevered onshore towers (with optional concentrated tip mass)
- tension-wire-supported towers
- floating-platform-supported towers (free-free root, 6×6 platform mass / hydro / mooring matrices, including hydrostatically-unstable spar configurations)
- monopile-supported towers (axial + torsion-fixed root, with mooring stiffness or distributed soil stiffness along the embedded section)

## Installation

Requires Python `>= 3.11`.

### Recommended: Windows + conda (newbie-friendly)

If you don't already have a Python environment set up, this is the path of least resistance on Windows. It uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create an isolated environment, then installs `pybmodes` with `pip` from inside it.

1. **Install Miniconda.** Download and run the Miniconda installer for Windows from the link above. Accept the defaults.
2. **Open the Anaconda Prompt** from the Start menu (not regular CMD or PowerShell — the Anaconda Prompt has `conda` already on `PATH`).
3. **Create and activate a dedicated environment.** Naming it `pybmodes` keeps your base environment clean:

   ```cmd
   conda create -n pybmodes python=3.11 -y
   conda activate pybmodes
   ```

4. **Clone and install in editable mode** with the development and plotting extras:

   ```cmd
   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"
   ```

   `pip install -e ".[dev,plots]"` pulls runtime dependencies (`numpy`, `scipy`), development tools (`pytest`, `ruff`, `mypy`), and `matplotlib` for the plotting helpers.

5. **Verify the install** by running the test suite:

   ```cmd
   pytest
   ```

   You should see all tests pass in a few seconds.

After the first install, you only need `conda activate pybmodes` in a new shell to start working.

### Quick install (existing Python environment)

If you already manage your own virtualenv / conda env / Poetry / uv setup, install straight from GitHub:

```bash
pip install git+https://github.com/SMI-Lab-Inha/pyBModes.git
```

### Development install

```bash
git clone https://github.com/SMI-Lab-Inha/pyBModes.git
cd pyBModes
pip install -e ".[dev]"
```

Add the `plots` extra if you want `matplotlib`-based plotting helpers,
and/or the `windio` extra (PyYAML) for `Tower.from_windio(...)` WindIO
ontology `.yaml` input:

```bash
pip install -e ".[dev,plots]"
pip install -e ".[dev,windio]"        # or .[dev,plots,windio]
```

The runtime core stays `numpy + scipy` only; `[plots]` (matplotlib)
and `[windio]` (PyYAML) are opt-in, and an absent extra raises a
friendly install hint rather than a bare `ModuleNotFoundError`.

### Updating an existing install

If you cloned the repo earlier and installed in editable mode, pull
the latest `master` and your install is automatically up-to-date —
editable installs follow the working tree, so you don't need to
re-run `pip install`:

```bash
cd pyBModes
git pull origin master
```

If the changelog notes a new optional dependency (e.g. a new
plotting helper appears), refresh the extras:

```bash
pip install -e ".[dev,plots]" --upgrade
```

For a non-editable install (`pip install git+https://...`), force a
re-install from the latest tag:

```bash
pip install --upgrade --force-reinstall git+https://github.com/SMI-Lab-Inha/pyBModes.git
```

After any update, verify with `pytest` (a few seconds) so a
breaking change to your downstream code surfaces immediately rather
than at next analysis.

To pin a specific release rather than tracking `master`, install
from a tag:

```bash
pip install git+https://github.com/SMI-Lab-Inha/pyBModes.git@v1.0.0
```

## Quick Start

### Blade analysis

```python
from pybmodes.models import RotatingBlade
from pybmodes.elastodyn import compute_blade_params

blade = RotatingBlade("my_blade.bmi")
result = blade.run(n_modes=10)
params = compute_blade_params(result)

print(result.frequencies[:3])
print(params.BldFl1Sh)
```

### Tower analysis

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, compute_tower_params_report

tower = Tower("my_tower.bmi")          # or Tower.from_bmi("my_tower.bmi")
result = tower.run(n_modes=10)

params = compute_tower_params(result)
params2, report = compute_tower_params_report(result)

print(result.frequencies[:4])
print(report.selected_fa_modes)
print(report.selected_ss_modes)
```

### Reading OpenFAST decks directly

```python
from pybmodes.models import Tower, RotatingBlade

# Onshore tower from ElastoDyn — parses main + tower + blade files,
# lumps the rotor mass into a tower-top tip-mass assembly automatically.
tower = Tower.from_elastodyn("MyTurbine_ElastoDyn.dat")

# Monopile tower with SubDyn pile geometry spliced below the ElastoDyn
# tower into a single combined cantilever (clamped at the seabed).
tower_mp = Tower.from_elastodyn_with_subdyn(
    "MyTurbine_ElastoDyn.dat",
    "MyTurbine_SubDyn.dat",
)

# Rotating blade from the same ElastoDyn deck. RotSpeed is taken from
# the deck; override on the BMI for off-rated analyses.
blade = RotatingBlade.from_elastodyn("MyTurbine_ElastoDyn.dat")
result = blade.run(n_modes=8)
```

### WISDEM/WindIO one-click (FOWT pipeline)

Point the `windio` subcommand at a WindIO ontology `.yaml` — or at a
reference-turbine directory and let it find the yaml — and pyBmodes
discovers the companion OpenFAST decks, solves the composite blade +
tower + (floating) coupled platform, and writes a bundled report:

```bash
# Floating: an IEA-15 UMaine VolturnUS-S RWT tree. The companion
# HydroDyn/MoorDyn/ElastoDyn decks are auto-discovered (scoped to
# that turbine root), so the platform is the industry-grade
# deck-backed coupled model — no screening warning.
pybmodes windio docs/.../IEA-15-240-RWT_VolturnUS-S.yaml \
    --n-modes 12 --campbell --max-rpm 8 \
    --out iea15_volturnus.md

# Fixed tower straight from the ontology (no companion decks needed):
pybmodes windio my_turbine.yaml --out tower.md --n-modes 6
```

Discovery is deliberately conservative: a bare yaml in a scratch
directory yields **no** companion decks (a labelled screening
preview) — it never recursively scans an arbitrary parent and never
picks a different turbine's decks. A proper `<root>/OpenFAST/` layout
**is** discovered, scoped to that turbine root and matched to the
configuration (floating vs fixed).

The same three building blocks are available as constructors. The
`[windio]` extra (PyYAML) is required for all of them:

```python
from pybmodes.models import RotatingBlade, Tower

# Composite blade: a PreComp-class thin-wall multi-cell
# classical-lamination reduction of the layup -> distributed beam
# properties (NOT a deck shortcut). Both WindIO key dialects + the
# WISDEM parametric layer forms are resolved.
blade = RotatingBlade.from_windio("IEA-15-240-RWT.yaml")
bl = blade.run(n_modes=8)

# Tubular tower / monopile straight from the ontology geometry.
tower = Tower.from_windio("IEA-15-240-RWT.yaml")
tw = tower.run(n_modes=6)

# Coupled FOWT. With the companion decks the result is the
# BModes-JJ-validated industry-grade coupled model; without them
# it is a screening preview and a UserWarning says so explicitly.
fowt = Tower.from_windio_floating(
    "IEA-15-240-RWT_VolturnUS-S.yaml",
    hydrodyn_dat="IEA-15-240-RWT_HydroDyn.dat",
    moordyn_dat="IEA-15-240-RWT_MoorDyn.dat",
    elastodyn_dat="IEA-15-240-RWT_ElastoDyn.dat",
)
res = fowt.run(n_modes=12)
print(res.frequencies[:6])
print(res.mode_labels[:6])   # surge / sway / heave / roll / pitch / yaw
```

A worked end-to-end tour with engineering-paper-styled plots lives
at [`cases/iea15_volturnus_windio_walkthrough.ipynb`](cases/iea15_volturnus_windio_walkthrough.ipynb).

### Patch ElastoDyn files

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat

result = Tower("my_tower.bmi").run(n_modes=10)
params = compute_tower_params(result)

patch_dat("ElastoDyn_tower.dat", params)
```

### Campbell diagram

```python
import numpy as np
from pybmodes.campbell import campbell_sweep, plot_campbell
from pybmodes.plots import apply_style

apply_style()  # MATLAB-styled defaults

rpm = np.linspace(0.0, 15.0, 16)
# An ElastoDyn .dat input auto-loads the blade *and* the tower from
# the same deck; the tower frequencies overlay as horizontal lines
# (rotor-speed-independent, since the tower lives in an Earth-fixed
# frame). For a .bmi input the sweep covers whichever component the
# file describes; pair a blade .bmi with a tower .bmi via tower_input.
result = campbell_sweep(
    "MyTurbine_ElastoDyn.dat",
    rpm,
    n_blade_modes=4,    # 1st/2nd flap + 1st/2nd edge (default)
    n_tower_modes=4,    # 1st/2nd FA + 1st/2nd SS (default)
)

# Or pass already-loaded models — a single point of load-in, no disk
# re-read, and the only way to sweep a from_windio / from_elastodyn
# model whose section properties no path can re-read (issue #51):
#   blade = RotatingBlade.from_windio("IEA-15-240-RWT.yaml")
#   tower = Tower.from_windio("IEA-15-240-RWT.yaml")
#   result = campbell_sweep(blade, rpm, tower_input=tower)

# Blade modes are MAC-tracked across rotor speeds, so
# result.frequencies[:, k] is the same physical mode at every speed.
print(result.labels)
# ['1st flap', '1st edge', '2nd flap', '2nd edge',
#  '1st tower SS', '1st tower FA', '2nd tower SS', '2nd tower FA']
print(result.frequencies.shape)  # (16, 8)
print(result.n_blade_modes, result.n_tower_modes)  # 4 4

fig = plot_campbell(
    result,
    excitation_orders=[1, 2, 3, 6, 9],
    rated_rpm=12.1,
)
fig.savefig("campbell.png")
```

The plot draws blade modes as solid coloured lines with markers, tower
modes as horizontal dashed dark-grey lines, and the per-rev family
(1P, 2P, …) as red dotted rays from the origin (shaded medium-to-dark
red so the order of crossings is readable at a glance) — exactly the
layout needed for the canonical *3P × 1st-tower-FA* resonance check at
~6–7 rpm on the NREL 5MW.

The same sweep is also available as a CLI subcommand:

```bash
pybmodes campbell MyTurbine_ElastoDyn.dat \
    --rated-rpm 12.1 --max-rpm 15 \
    --n-blade-modes 4 --n-tower-modes 4 \
    --orders 1,2,3,6,9 --out campbell.png
```

### Mode-by-mode comparison

```python
from pybmodes.mac import compare_modes, plot_mac

# Confirm a coefficient patch didn't change the underlying mode shapes
baseline = Tower.from_elastodyn("upstream.dat").run(n_modes=10)
patched  = Tower.from_elastodyn("patched.dat").run(n_modes=10)
cmp = compare_modes(baseline, patched, label_A="upstream", label_B="patched")

print(cmp.paired_modes)        # Hungarian-optimal pairing
print(cmp.frequency_shift)     # per-pair % change

fig = plot_mac(cmp)            # heatmap with paired cells outlined
fig.savefig("mac.png")
```

### Save / load results

```python
# ModalResult
modal.save("modes.npz")
loaded = ModalResult.load("modes.npz")

modal.to_json("modes.json")
loaded = ModalResult.from_json("modes.json")

# CampbellResult
result.save("campbell.npz")
result.to_csv("campbell.csv")
```

Both serialisations embed pyBmodes version, UTC timestamp, source-file
path, and best-effort git hash so the artefact is self-identifying.

### Bundled report (Markdown / HTML / CSV)

```bash
pybmodes report MyTurbine_ElastoDyn.dat \
    --format html --out report.html \
    --campbell --max-rpm 15
```

Eight sections: model summary, assumptions (BC type, RNA assembly,
ElastoDyn-compat flag), natural frequencies, mode classification,
polynomial coefficients with fit residuals, validation verdict,
`check_model` warnings, optional Campbell-sweep first/last
frequencies per mode.

### Batch validate + patch a tree of decks

```bash
pybmodes batch ./models/ \
    --kind elastodyn \
    --out ./reports/ \
    --n-modes 10 \
    --validate --patch
```

Walks `./models/` for ElastoDyn main `.dat` files, runs validate /
patch on each, writes per-deck reports plus a `summary.csv` with
columns `filename, overall_verdict, TwFAM2Sh_ratio, TwSSM2Sh_ratio,
n_fail, n_warn`. Exit 0 when every deck reaches PASS or WARN; exit
1 if any FAIL or ERROR remains.

## Walkthrough notebook

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) is a self-contained end-to-end tour of the public API.  It builds two synthetic cases inline (a uniform Euler-Bernoulli blade and a uniform tower with a concentrated top mass), runs the FEM solver, fits ElastoDyn polynomials, and validates the FEM frequencies against published closed-form formulas — all without bundling any external data.

Two data-dependent walkthroughs live under [`cases/`](cases/) (they read upstream IEA-15 decks under the gitignored `docs/`, so they need the upstream RWT tree cloned — same rule as the `run.py` case studies):

- [`cases/iea15_umainesemi_walkthrough.ipynb`](cases/iea15_umainesemi_walkthrough.ipynb) — the coupled IEA-15 UMaine VolturnUS-S floating case via `from_elastodyn_with_mooring` + cantilever-basis polynomial generation + Campbell.
- [`cases/iea15_volturnus_windio_walkthrough.ipynb`](cases/iea15_volturnus_windio_walkthrough.ipynb) — the **WISDEM/WindIO one-click** pipeline: the `pybmodes windio` orchestrator plus the individual `RotatingBlade.from_windio` / `Tower.from_windio` / `Tower.from_windio_floating` constructors, with mode-shape, Campbell, and MAC plots.

## Inputs and Outputs

### Inputs

- `.bmi` main input files (line-ordered text, values precede labels) plus the section-property `.dat` they reference
- OpenFAST ElastoDyn main `.dat` plus the tower and blade files referenced via `TwrFile` / `BldFile(1)`
- OpenFAST SubDyn `.dat` (minimal subset: joints, members, circular cross-section properties, base reaction joint, interface joint — sufficient for OC3-style fixed-base monopiles)
- WISDEM/WindIO ontology `.yaml` (both the modern `outer_shape`/`structure` and older `outer_shape_bem`/`internal_structure_2d_fem` dialects; tubular tower geometry, composite blade layup, and `floating_platform` member/mooring topology) — needs the optional `[windio]` extra
- OpenFAST HydroDyn `.dat` + WAMIT `.1`/`.hst` and MoorDyn `.dat` (the companion decks the `windio` one-click discovers to make a floating platform industry-grade; also consumable directly via `Tower.from_elastodyn_with_mooring`)
- BModes `.out` reference output (parsed for cross-solver verification, not used as a primary input)

### Outputs

The high-level model APIs return:

- natural frequencies
- nodal mode shapes
- ElastoDyn-ready polynomial coefficients for blades and towers (with per-fit RMS-residual and design-matrix condition-number diagnostics)

For tower fitting, the companion reporting API exposes which modal candidates were considered and why specific FA/SS family members were selected, and warns when the polynomial-basis condition number exceeds a configurable threshold.

## Validation

The codebase is validated against two complementary sources of truth.

### Closed-form analytical references

| Case | Reference | Tolerance |
| --- | --- | --- |
| Uniform Euler-Bernoulli cantilever (first 5 flap modes) | Analytical: $\beta_n L = [1.875, 4.694, 7.855, 10.996, 14.137]$ | < 0.5 % |
| Uniform cantilever with concentrated tip mass | Frequency equation in Blevins (1979), *Formulas for Natural Frequency and Mode Shape*; Karnovsky & Lebed (2001), *Formulas for Structural Dynamics* | < 0.5 % |
| Hermite-cubic mesh-convergence | $h^4$ convergence rate for first five frequencies | confirmed |
| Uniform steel tube cantilever via `Tower.from_geometry` (D, t, L → derived EI / mass) | Analytical Euler-Bernoulli, $\beta_1 L = 1.875104$ | < 0.1 % |

All analytical test cases are constructed in-test from numbers that come from peer-reviewed textbooks or analytical formulas. Section properties for the synthetic validation cases are generated programmatically by the test suite.

### Cross-solver certification against BModes

Six certification cases compare the pyBmodes pipeline against the BModes Fortran reference solver (Bir 2010), with paired `.bmi` inputs and `.out` reference frequencies. All six pass at strict tolerances on the local development machine:

| Case (BModes deck) | Reference turbine / configuration | Boundary condition | Tolerance | Worst-mode error |
| --- | --- | --- | --- | --- |
| Test01, Test02 | non-uniform rotating blade (with and without tip mass) — BModes v3.00 CertTest | cantilever | < 1 % on modes 1-6, < 3 % on modes 7+ | < 0.005 % across 20 modes |
| Test03 | 82.4 m tower with top mass and offsets — BModes v3.00 CertTest | cantilever | < 1 % / < 3 % | < 0.005 % across 20 modes |
| Test04 | Test03 + tension-wire support — BModes v3.00 CertTest | cantilever + wires | < 1 % / < 3 % | < 0.005 % across 20 modes |
| `CS_Monopile.bmi` | *NREL 5MW Reference Turbine* (Jonkman et al. 2009) on the *OC3 Monopile* configuration (Jonkman & Musial 2010) | `hub_conn = 3` (soft monopile, mooring stiffness, lateral + rocking free) | 0.01 % | < 0.005 % across 10 modes |
| `OC3Hywind.bmi` | *NREL 5MW* on the *OC3 Hywind* floating spar (Jonkman 2010) — full hydro + mooring + 6×6 platform inertia | `hub_conn = 2` (free-free) | 0.01 % | ≤ 0.0003 % across 9 modes (surge, sway, pitch, roll, heave, yaw, 1st-2nd tower bending) |

Citations:

- Jonkman, J., Butterfield, S., Musial, W., & Scott, G. (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Jonkman, J., & Musial, W. (2010). *Offshore Code Comparison Collaboration (OC3) for IEA Wind Task 23 Offshore Wind Technology and Deployment*. NREL/TP-5000-48191.
- Jonkman, J. (2010). *Definition of the Floating System for Phase IV of OC3*. NREL/TP-500-47535.

These certification inputs and reference outputs are local-only — the test module skips at module level when they are absent, so contributors who don't have them can still run the rest of the suite.

#### What "BModes reference" means in the table above

For all six rows the **reference is the BModes Fortran solver `.out`
output run locally on the same `.bmi` input** that pyBmodes consumes.
This is **not** the same as the published frequency tables in
Bir (2010, NREL/CP-500-47953):

- Bir 2010 Tables 1 (land tower), 2 (barge), 3 (monopile DS) and
  4 (spar-buoy) report frequencies from a specific BModes version
  with a specific gravity convention; comparing pyBmodes against
  *those published numbers* on the bundled decks shows 4–9 %
  deviation on platform rigid-body modes.
- Comparing pyBmodes against **BModes JJ `.out` on the same deck and
  the same physics** is a direct solver comparison with no version or
  convention ambiguity, and is reproducibly within 0.01 %.

The one Bir 2010 table where pyBmodes *is* validated directly against
the published numbers is **Table 5** (rotating uniform blade with tip
mass) — those values come from the Wright et al. (1982) analytical
solution and are version-independent. That direct comparison lives at
[`tests/fem/test_rotating_blade_with_tip_mass.py`](tests/fem/test_rotating_blade_with_tip_mass.py)
and passes at ≤ 0.1 %. Wright et al.'s rotating-uniform-blade values
(transcribed from Bir 2009 / AIAA 2009-1035 Table 3a) are similarly
validated at ≤ 0.5 % by
[`tests/fem/test_rotating_uniform_blade.py`](tests/fem/test_rotating_uniform_blade.py).

### WindIO geometry — cross-code structural-property reproduction

`Tower.from_windio(...)` is anchored by a like-for-like check that
touches none of the polynomial machinery: the IEA-15-240-RWT base
WindIO ontology, run forward through the closed-form circular-tube
reduction, reproduces the distributed mass / EI tabulated in the
IEA-15 *Monopile* OpenFAST ElastoDyn tower deck — the deck WISDEM
generated from that same geometry — to **7.5 × 10⁻¹² (machine
precision)**. The full upstream RWT corpus (IEA-3.4 / 10 / 15 / 22 +
WISDEM examples, spanning both WindIO key dialects and IEA-10's
duplicate-anchor file) is exercised for parse-sanity and modal smoke.
This sharpens the ecosystem finding in
[`cases/ECOSYSTEM_FINDING.md`](cases/ECOSYSTEM_FINDING.md): the
*structural* blocks of an RWT deck faithfully encode the published
geometry — the documented coefficient drift lives entirely in the
*polynomial* blocks.

### WindIO composite blade — cross-code beam-property reproduction

`RotatingBlade.from_windio(..., elastic="auto")` (the default,
issue #48) prefers the WindIO **published distributed elastic
properties**
when the ontology carries them — both dialects: the modern
`elastic_properties` (`K33/K44/K55/K66`, `mass`, `i_flap/i_edge`,
`cm_x`) and the `elastic_properties_mb.six_x_six` 21-element
upper-triangular flatten of the symmetric sectional 6×6 — both
ship the **full** 6×6 (the modern `stiffness_matrix` carries the
`K11..K66` upper triangle). The 6×6 is **decoupled** to the
Euler–Bernoulli `EA / EI_flap / EI_edge / GJ` at the elastic /
shear centres and principal elastic axes
(`pybmodes.io._precomp.decouple`, issue #50) — *not* read off the
raw reference-axis diagonal, which is not the physical `EI` for an
offset / pre-twisted blade. Validated *decoupled-vs-decoupled*
against the turbine's own **BeamDyn 6×6** (also reference-axis, so
decoupled the same way): IEA-15 mass / EA median < 3 %, EI < 5 %,
GJ < 8 % — far tighter than the layup reduction, so the model
matches the reference where the reference exists. Only when the published properties are absent does
it fall back to the PreComp-class thin-wall multi-cell
classical-lamination reduction (`elastic="precomp"` forces it; the
pre-1.5 behaviour). The PreComp path is anchored against each
turbine's own WISDEM-PreComp-generated **BeamDyn 6×6**
stiffness/inertia matrices across IEA-3.4 / 10 / 15 / 22 over span
0.15–0.90: distributed `mass` and `EA` reproduce the BeamDyn
reference to PreComp class (mass ≈ 1.5–4 % median, EA ≈ 1–8 %
median); `GJ` and `EI` are diagonal-reduction approximate (≈ 3–18 %
/ 2–27 % median — a documented limitation, since the membrane
reduction omits spar-cap-offset and bend-twist coupling). The
closed-form CLT primitives, the airfoil `nd_arc` profile spine, and
the single- / multi-cell thin-wall reductions are independently
gated against textbook Bredt–Batho / Jones *Mechanics of Composite
Materials* forms. (An honest caveat: a turbine's published
`elastic_properties_mb` is not always mutually consistent with its
own companion BeamDyn deck — IEA-10's `_mb` diverges ≈ 50 % from its
BeamDyn 6×6, the same ecosystem-drift pattern documented in
[`cases/ECOSYSTEM_FINDING.md`](cases/ECOSYSTEM_FINDING.md). `auto`
treats the WindIO ontology as the canonical reference, per #48's
intent of minimising deltas against the WindIO model the user
supplied.)

### WindIO floating platform — two-tier fidelity

The coupled `Tower.from_windio_floating(...)` path is validated at
both tiers against the IEA-15 UMaine VolturnUS-S RWT:

- **Hydrostatics** — the WindIO-geometry `C_hst` reproduces the
  turbine's own potential-flow WAMIT `.hst` to heave 0.8 %,
  roll/pitch 1.6 % (geometry-exact anchor).
- **Industry-grade tier (companion decks present)** — vs the
  BModes-JJ-validated `from_elastodyn_with_mooring`, all six platform
  rigid-body modes + 1st tower bending land at **0.0–0.3 %**; 2nd+
  tower harmonics ≤ 6 % (the Phase-1 WindIO-vs-ElastoDyn tower
  discretisation residual, orthogonal to the platform).
- **Screening tier (yaml only)** — honestly labelled by a
  `UserWarning`: member-Morison added mass differs from BEM as
  RAFT/WISDEM also find (surge ≈ 22 %, heave ≈ 53 %), and the
  structural+fixed mass is a deliberate lower bound (trim ballast
  excluded). Supply a HydroDyn deck for industry grade.

The closed-form hydrostatic, Morison + RAFT end-cap, rigid-body
inertia, and catenary-mooring primitives are gated independently
against analytic references.

### Test suite

Every validation case — what's compared against what, at what tolerance, with what worst observed margin, and which test enforces it — is enumerated in [`VALIDATION.md`](VALIDATION.md). That file is the source of truth and stays in sync with the test suite; the CI badge at the top of this README shows the current public pass status.

The tests cover:

- input parsing and path resolution (parser primitives + inline-fixture round-trips)
- FEM building blocks (boundary conditions, generalised eigensolver — both symmetric `eigh` and asymmetric `eig` paths — non-dimensionalisation, mode-shape extraction)
- model pipelines for blades and towers, including OpenFAST deck adapters
- degenerate FA/SS eigenpair resolution for symmetric tower models
- polynomial fitting and tower FA/SS family classification, with conditioning diagnostics
- ElastoDyn parameter generation and file patching
- closed-form / analytical validation of representative blade and tower configurations
- cross-solver certification against BModes on the six cases above

## Sample inputs

[`src/pybmodes/_examples/sample_inputs/`](src/pybmodes/_examples/sample_inputs/) ships pyBmodes-authored, MIT-licensed `.bmi` and section-properties `.dat` files committed to the repo. Use them as a starting point to copy / adapt when authoring your own decks, or as a self-checking validation kit. Nothing here depends on local-only upstream data.

> **Vendored into the wheel.** As of 0.4.0 the two example trees live *inside* the `pybmodes` package at [`src/pybmodes/_examples/sample_inputs/`](src/pybmodes/_examples/sample_inputs/) and [`src/pybmodes/_examples/reference_decks/`](src/pybmodes/_examples/reference_decks/), and are declared as `package-data` so they ship in every wheel and editable install. To vendor them out to a working directory of your choosing, run `pybmodes examples --copy <dir> [--kind all|samples|decks]`; the command resolves the bundles relative to `pybmodes.__file__` and copies the requested tree out, whether you installed from a wheel or from a source checkout.

### Analytical-reference cases

Four hand-written cases that exercise pyBmodes' four boundary conditions (`hub_conn ∈ {1, 4}`), the tower + blade beam-type split, and the rotating + non-rotating + tip-mass + no-tip-mass splits. Every numeric value is reproducible from a peer-reviewed analytical formula. [`src/pybmodes/_examples/sample_inputs/verify.py`](src/pybmodes/_examples/sample_inputs/verify.py) runs pyBmodes on all four and asserts that the lowest few computed frequencies match the analytical reference to within 1 % relative error.

| #  | Title                                          | Beam   | Ω (rad/s) | Tip mass | BC          | Reference              |
| -- | ---------------------------------------------- | ------ | --------: | -------- | ----------- | ---------------------- |
| 01 | Uniform isotropic cantilever blade             | blade  |       0   | none     | cantilever  | Euler-Bernoulli        |
| 02 | Uniform tower with concentrated top mass       | tower  |       0   | μ = 1.0  | cantilever  | Blevins (1979)         |
| 03 | Rotating uniform blade                         | blade  |       6   | none     | cantilever  | Wright 1982 / Bir 2009 |
| 04 | Rotating pinned-free cable                     | blade  |      10   | none     | pinned-free | Bir 2009 Eq. 8         |

### Reference-wind-turbine library

[`src/pybmodes/_examples/sample_inputs/reference_turbines/`](src/pybmodes/_examples/sample_inputs/reference_turbines/) ships **tower + blade** BMI samples for seven open-literature reference wind turbines, regenerable from the published structural inputs via [`build.py`](src/pybmodes/_examples/sample_inputs/reference_turbines/build.py). Use these as starting points for your own RWT-based modal analysis, or as redistributable test fixtures that don't depend on the upstream `docs/` clone.

| #  | Sub-case                              | Publication             | Tower BMI structure              |
| -- | ------------------------------------- | ----------------------- | -------------------------------- |
| 01 | NREL 5MW — land-based                 | Jonkman 2009            | cantilever                       |
| 02 | NREL 5MW — OC3 monopile               | Jonkman & Musial 2010   | combined pile + tower            |
| 03 | IEA-3.4-130-RWT — land                | Bortolotti et al. 2019  | cantilever                       |
| 04 | IEA-10.0-198-RWT — monopile           | Bortolotti et al. 2019  | combined pile + tower            |
| 05 | IEA-15-240-RWT — monopile             | Gaertner et al. 2020    | combined pile + tower            |
| 06 | IEA-22-280-RWT — monopile             | Bortolotti et al. 2024  | combined pile + tower            |
| 07 | NREL 5MW — OC3 Hywind floating spar   | Jonkman 2010            | floating with PlatformSupport    |

The land-based and monopile sample BMIs use a rigidly-clamped tower base (`hub_conn = 1`), matching ElastoDyn's clamped-base assumption for tower mode-shape polynomials. For monopile sub-cases the pile geometry is **structurally spliced** below the ElastoDyn tower into a single combined cantilever via `Tower.from_elastodyn_with_subdyn`; soil flexibility is not modelled, so the resulting 1st-FA frequency is several percent stiffer than the soft-soil + lateral-pile reference (e.g. 0.286 Hz vs Jonkman 2010's published 0.275 Hz on OC3 monopile). The floating sample matches BModes' canonical `OC3Hywind.bmi` deck byte-for-byte (`hub_conn = 2` + populated platform / hydro / mooring 6 × 6 matrices); pyBmodes' solve of it lands within ~ 0.1 % of Jonkman 2010's published 0.4816 Hz.

Reference-wind-turbine structural definitions are iteratively revised across releases — the same RWT designation at git-tag v1.0.0 may have a few-percent different section-property distribution than at v2.0.0. The pyBmodes frequencies in each per-turbine README are derived from the deck-as-distributed at the time `build.py` was last run; the published reference frequencies are historical anchors, not regression targets.

## Case studies

The [`cases/`](cases/) directory contains exploratory studies that exercise the full pipeline (parse → solve → fit → diagnose) against publicly available reference turbines. Unlike the sample inputs above, several of these depend on locally-cloned upstream data (under gitignored `docs/`):

- [`cases/nrel5mw_land/`](cases/nrel5mw_land/) — *NREL 5MW Reference Turbine* (Jonkman et al. 2009) on the OpenFAST land-based deck from the OpenFAST regression-test corpus.
- [`cases/iea3mw_land/`](cases/iea3mw_land/) — *IEA-3.4-130-RWT* (Bortolotti et al. 2019, IEA Wind Task 37) land-based deck.
- [`cases/nrel5mw_monopile/`](cases/nrel5mw_monopile/) — *NREL 5MW* on a rigid monopile substructure (the OC3 Monopile configuration without soil flexibility) using the SubDyn pile geometry spliced below the ElastoDyn tower.
- [`cases/bir_2010_land_tower/`](cases/bir_2010_land_tower/), [`cases/bir_2010_monopile/`](cases/bir_2010_monopile/), [`cases/bir_2010_floating/`](cases/bir_2010_floating/) — Bir 2010 (NREL/CP-500-47953) Figure 6 / 7 / 8 mode-shape reproductions on the canonical BModes example decks (`CS_Monopile.bmi`, `OC3Hywind.bmi`); these load the `hub_conn = 3` + soil-Winkler monopile model and the `hub_conn = 2` + populated-platform-matrix floating model directly.

Each case has a `run.py` that prints a coefficient comparison table and writes outputs (mode-shape PNGs, full diagnostic text). The cross-deck summary in [`cases/ECOSYSTEM_FINDING.md`](cases/ECOSYSTEM_FINDING.md) documents a recurring pattern: the polynomial-coefficient blocks shipped in industry `_ElastoDyn.dat` files are not always reproducible from the structural-property blocks in the same files — the coefficients in many decks were generated against an earlier revision of the structural model and have not been regenerated.

References for the case-study turbines:

- Jonkman, Butterfield, Musial, & Scott (2009). *Definition of a 5-MW Reference Wind Turbine for Offshore System Development*. NREL/TP-500-38060.
- Bortolotti, Tarrés, Dykes, Merz, Sethuraman, Verelst, & Zahle (2019). *IEA Wind TCP Task 37: Systems Engineering in Wind Energy — WP2.1 Reference Wind Turbines*. NREL/TP-5000-73492.

## Project Layout

```text
src/pybmodes/
  io/         input/output parsers (.bmi, section-properties .dat, .out);
              _elastodyn/ sub-package (types / lex / parser / writer /
              adapter) re-exported via elastodyn_reader.py; WindIO
              ontology readers — windio.py (tubular tower),
              _precomp/ + windio_blade.py (composite-layup blade
              reduction), windio_floating.py (floating substructure)
  fem/        finite-element core (assembly + boundary + dispatch
              between dense eigh / eigsh shift-invert / general eig)
  models/     high-level blade and tower APIs; ModalResult with
              save / load / to_json / from_json + optional
              participation / fit_residuals / metadata
  fitting/    mode-shape polynomial fitting
  elastodyn/  ElastoDyn parameter generation and file patching;
              torsion-contamination filter on tower family selection
  campbell.py rotor-speed sweep + Hungarian MAC tracking; CampbellResult
              with save / load / to_csv + per-step mac_to_previous
  checks.py   8 pre-solve sanity checks; auto-runs in .run()
  mac.py      MAC matrix + ModeComparison + plot_mac
  report.py   Markdown / HTML / CSV bundled analysis report
  plots/      plotting helpers + standard engineering-paper defaults
  cli.py      pybmodes CLI: validate / patch / campbell / batch /
              report / windio / examples
  mooring.py  quasi-static catenary mooring; from_moordyn +
              from_windio_mooring
  _examples/  vendored package-data: sample_inputs/ (4 analytical
              references + 7 RWT samples) + reference_decks/ (6
              patched ElastoDyn decks, 3 fixed + 3 floating) — ships
              in every wheel; reachable via `pybmodes examples --copy`
notebooks/    walkthrough.ipynb — end-to-end usage tour
scripts/      project-maintenance scripts: build_reference_decks,
              audit_validation_claims, benchmark_sparse_solver,
              campbell, visualise_polynomial_comparison_*
tests/        unit + closed-form-analytical validation
cases/        exploratory case studies (bir_2010_*, nrel5mw_*,
              iea3mw_*) + two data-dependent walkthrough notebooks
              (iea15_umainesemi_walkthrough.ipynb,
              iea15_volturnus_windio_walkthrough.ipynb) — NOT the
              sample_inputs library (which now lives under
              src/pybmodes/_examples/)
docs/         RELEASE_CHECKLIST.md — 11-step pre-tag verification
VALIDATION.md     single structured matrix of every validated case
```

## Development

```bash
# Default test run — self-contained, no external decks
pytest

# Integration tests — requires upstream OpenFAST / BModes data under docs/
pytest -m integration

# Both
pytest -m ""

# Lint (matches CI scope: src + tests + scripts)
ruff check src/ tests/ scripts/

# Type check
mypy src/pybmodes

# Validation-matrix audit (gates "claim ahead of test" drift)
python scripts/audit_validation_claims.py
```

The default `pytest` run is **self-contained** and works on a fresh
clone with no external data. The `integration` marker gates the
subset of tests that need locally-checked-out OpenFAST `r-test`
decks, BModes CertTest data, or BModes `.bmi` / `.out` reference
outputs under `docs/` (none of which are bundled — see the
*Compatibility policy* below for why). CI runs both steps on every
commit; the integration step is allowed to skip when data isn't
present. See [`VALIDATION.md`](VALIDATION.md) for the per-case
breakdown of which tests need external data and which don't.

## Public API

These imports are the stable, semver-protected 1.x surface (subject
to the *Compatibility policy* below):

```python
from pybmodes.models    import RotatingBlade, Tower, ModalResult
from pybmodes.elastodyn import (
    compute_blade_params,
    compute_tower_params,
    compute_tower_params_report,
    patch_dat,
    validate_dat_coefficients,
    BladeElastoDynParams,
    TowerElastoDynParams,
    ValidationResult,
    CoeffBlockResult,                # carries fa/ss/torsion_participation
                                     # + rejected_modes for tower blocks
)
from pybmodes.fitting   import PolyFitResult, fit_mode_shape
from pybmodes.campbell  import (
    CampbellResult,                  # save / load / to_csv
    campbell_sweep,
    plot_campbell,
)
from pybmodes.checks    import check_model, ModelWarning
from pybmodes.mac       import (
    mac_matrix,
    compare_modes,
    ModeComparison,
    plot_mac,
    shape_to_vector,
)
from pybmodes.report    import generate_report
from pybmodes.mooring   import LineType, Point, Line, MooringSystem
from pybmodes.io        import (
    HydroDynReader, WamitReader, WamitData,
    PlatformSupport, TipMassProps,   # inject a separately-designed floater
    read_out, BModeOutParseError,    # read_out(path, strict=True)
)
from pybmodes.io.geometry import tubular_section_props
from pybmodes.io.windio   import read_windio_tubular, WindIOTubular
from pybmodes.io.windio_blade import (         # [windio] extra
    read_windio_blade,
    windio_blade_section_props,
    WindIOBlade,
)
from pybmodes.io.windio_floating import (      # [windio] extra
    read_windio_floating,
    hydrostatic_restoring,
    added_mass,
    rigid_body_inertia,
    WindIOFloating,
)
from pybmodes.plots     import (
    apply_style,
    plot_mode_shapes,
    plot_fit_quality,
    bir_mode_shape_plot,
    bir_mode_shape_subplot,
    plot_environmental_spectra,      # wind/wave + 1P/3P vs tower
    kaimal_spectrum,
    jonswap_spectrum,
)

# Tower constructors:
#   Tower.from_bmi(bmi_path)
#   Tower.from_elastodyn(main_dat)
#   Tower.from_elastodyn_with_subdyn(main_dat, subdyn_dat)
#   Tower.from_elastodyn_with_mooring(main_dat, moordyn_dat,
#                                     hydrodyn_dat=None)
#   Tower.from_geometry(station_grid, outer_diameter, wall_thickness,
#                       *, flexible_length, E, rho, nu,
#                       outfitting_factor)
#   Tower.from_windio(yaml_path, *, component, thickness_interp,
#       hub_conn, tip_mass, n_nodes)  # tip_mass: TipMassProps|float
#       # (RNA kg); n_nodes: refine the FE mesh (issue #35)
#       (read_windio_tubular / WindIOTubular need the [windio] extra)
#   Tower.from_windio_floating(yaml_path, *, component_tower,
#       water_depth, hydrodyn_dat, moordyn_dat, elastodyn_dat,
#       platform_support, rna_tip, rho, g)
#                          # coupled FOWT. Platform from: companion
#                          # decks (industry-grade) | yaml screening |
#                          # an injected PlatformSupport (floater
#                          # designed separately — verbatim 6x6s)
#
# Blade constructors:
#   RotatingBlade.from_bmi(bmi_path)
#   RotatingBlade.from_elastodyn(main_dat)
#   RotatingBlade.from_windio(yaml_path, *, component, n_span,
#       rot_rpm, n_perim, elastic)  # elastic="auto" prefers the
#       # WindIO published distributed elastic properties when
#       # present (minimises deltas vs the reference model), else
#       # the PreComp-class layup reduction; "precomp" forces the
#       # reduction; "file" requires the published properties
#
# Mooring:
#   MooringSystem.from_moordyn(dat_path)
#   MooringSystem.from_windio_mooring(floating, *, depth,
#       moordyn_fallback=None, rho, g)
```

Known limitations of the 1.0 surface: `pybmodes.mooring` is catenary-
only quasi-static (no seabed friction, no sloped seabed, no U-shape
lines, no time-domain dynamics); `pybmodes.io.WamitReader` extracts
`A_inf` / `A_0` / `C_hst` only (no frequency-dependent `A(ω)` /
`B(ω)`); `Tower.from_elastodyn_with_mooring` is for coupled-frequency
prediction, not ElastoDyn polynomial-coefficient generation (use
`Tower.from_elastodyn` for the latter regardless of platform
configuration — see `cases/ECOSYSTEM_FINDING.md` for the source-code
citation); `RotatingBlade.from_windio`'s PreComp **fallback**
(`elastic="precomp"`, or `"auto"` when the ontology has no published
distributed elastic properties) is PreComp-class on mass / EA but
diagonal-reduction approximate on GJ / EI (it omits spar-cap-offset
and bend-twist coupling — see the *WindIO composite blade*
validation note); the default `elastic="auto"` avoids this entirely
whenever the WindIO file publishes its own elastic properties; and
the yaml-only tier of
`Tower.from_windio_floating` is a screening estimate, not
industry-grade — it emits a `UserWarning` saying so, and you supply
the companion HydroDyn / MoorDyn / ElastoDyn decks (or let
`pybmodes windio` auto-discover them) to reach the validated
deck-backed coupled model.

The CLI is exposed as `pybmodes` (see `pybmodes --help`); the
`pybmodes windio` subcommand is the one-click WindIO entry point.

Internal modules — `pybmodes.fem.*`, the underscore-prefixed
`pybmodes.models._pipeline`, and `pybmodes.io._elastodyn` — carry
the implementation; user code should not import from them directly.
The per-format submodules under `pybmodes.io` (`pybmodes.io.bmi`,
`elastodyn_reader`, `subdyn_reader`, `wamit_reader`) are reachable
directly but the public-freeze contract covers only the re-exports
listed above.

## Compatibility policy

1.x semver discipline:

- **Public API frozen.** Anything in the *Public API* list above is
  source-compatible across 1.x minor releases. Renaming or removing
  an exported name requires a major-version bump (2.x). Adding new
  keyword arguments with defaults, new optional fields on dataclasses,
  and new entirely-new entry points is non-breaking.
- **Numerical outputs may shift between minor releases** when
  validation tightens or a modelling correction lands. Every such
  shift is called out in `CHANGELOG.md` under *Fixed* / *Changed*
  with the magnitude and the affected case (e.g. the May 2026 OC3
  Hywind 1st-tower-bending fix that closed a 3.7 % gap). The
  ``VALIDATION.md`` matrix records the canonical expected value per
  case so the regression direction is tracked.
- **CLI contract.** Every `pybmodes <subcommand>` exit-code schema,
  output-format header, and required-flag set is locked. New
  subcommands and new optional flags may still be added under the
  additive rule.
- **CI gates are required.** The default-pytest, ruff, mypy, and
  validation-matrix-audit steps in `.github/workflows/ci.yml` block
  merges to `master` via a GitHub branch-protection ruleset.
- **Parser behaviour is best-effort across OpenFAST versions.** The
  ElastoDyn / SubDyn / HydroDyn / MoorDyn readers handle FAST v8 and
  OpenFAST v3+ format drift via label-based scanning, but new
  upstream file-format changes may need tracking patches; if a deck
  parses on one pyBmodes release and not on the next, that's a bug
  worth reporting.

Pre-1.0 history (0.1 → 0.4) is preserved in `CHANGELOG.md` with the
breaking-change boundary at each minor bump documented inline.

## License

Released under the [MIT License](LICENSE).
