# pyBmodes reference-wind-turbine BMI library

Seven pyBmodes-authored BMI samples for the open-literature **reference
wind turbines** (RWTs) cited routinely in the wind-energy literature,
spanning **2009 – 2024** in publication date and **3.4 – 22 MW** in
rated power. Every numerical value is sourced from the named
publication or its public companion deck; the file format,
commentary, and citations are pyBmodes-authored and the files are
Apache 2.0-licensed alongside the rest of the project.

Each turbine ships **two BMI samples** — one for the tower, one for
the blade — so the sample library exercises both halves of the
pyBmodes modelling pipeline (cantilever tower + rotating cantilever
blade with centrifugal stiffening).

These complement the four analytical-reference BMIs in the parent
[`sample_inputs/`](../) directory — those validate pyBmodes'
FEM core against textbook formulas, while these demonstrate the
modelling pipeline against the wind-turbine designs cited routinely
in the literature.

## Index

| #  | Sub-case                                                          | Publication                           | Tower-BMI structure              | Tower 1st-FA | Blade flap-1 | Blade edge-1 | Blade flap-2 |
| -- | ----------------------------------------------------------------- | ------------------------------------- | -------------------------------- | -----------: | -----------: | -----------: | -----------: |
| 01 | [NREL 5MW — land-based](01_nrel5mw_land/)                         | Jonkman 2009                          | cantilever                       |     0.333 Hz |     0.734 Hz |     1.108 Hz |     2.018 Hz |
| 02 | [NREL 5MW — OC3 monopile](02_nrel5mw_oc3monopile/)                | Jonkman & Musial 2010                 | combined pile + tower            |     0.286 Hz |     0.734 Hz |     1.108 Hz |     2.018 Hz |
| 03 | [IEA-3.4-130-RWT — land](03_iea34_land/)                          | Bortolotti et al. 2019                | cantilever                       |     0.417 Hz |     0.796 Hz |     1.029 Hz |     2.359 Hz |
| 04 | [IEA-10.0-198-RWT — monopile](04_iea10_monopile/)                 | Bortolotti et al. 2019                | combined pile + tower            |     0.301 Hz |     0.404 Hz |     0.650 Hz |     1.043 Hz |
| 05 | [IEA-15-240-RWT — monopile](05_iea15_monopile/)                   | Gaertner et al. 2020                  | combined pile + tower            |     0.191 Hz |     0.558 Hz |     0.728 Hz |     1.592 Hz |
| 06 | [IEA-22-280-RWT — monopile](06_iea22_monopile/)                   | Bortolotti et al. 2024                 | combined pile + tower            |     0.182 Hz |     0.400 Hz |     0.538 Hz |     1.103 Hz |
| 07 | [NREL 5MW — OC3 Hywind spar](07_nrel5mw_oc3hywind_spar/)          | Jonkman 2010                          | floating with PlatformSupport    |     0.482 Hz |     0.734 Hz |     1.108 Hz |     2.018 Hz |

Frequencies above are pyBmodes' result on the deck-as-distributed at
the time `build.py` was last run; the deck's RotSpeed is used for the
blade FEM so the flap-mode values include the centrifugal-stiffening
contribution.

## Modelling assumption

**Tower BMI**: `hub_conn = 1` (cantilevered base) clamped at TowerBsHt
with the RNA lumped at the tower top. RNA mass is computed as
`Hub + Nacelle + 3·BladeMass` (the blade mass is the
trapezoidal-rule integral of the blade `BMassDen` column over the
blade length). For monopile sub-cases this treats the substructure
below TowerBsHt as a rigid extension — the same simplification BModes
uses for ElastoDyn-compatible mode-shape generation. The 1st-FA
frequency that comes out is therefore **higher** than the actual
flexible-pile + tower combined-cantilever system frequency, by
5 – 80 % depending on pile length and foundation stiffness. For the
combined-cantilever physics use:

```python
from pybmodes.models import Tower
tower = Tower.from_elastodyn_with_subdyn(
    "path/to/<turbine>_ElastoDyn.dat",
    "path/to/<turbine>_SubDyn.dat",
)
```

That path is validated against BModes JJ on the OC3 monopile
(`tests/test_certtest.py::test_certtest_cs_monopile`) and is the right
choice for any quantitative monopile modal analysis.

**Blade BMI**: `hub_conn = 1` (cantilevered root) spun at the deck's
`RotSpeed`. No tip mass — these RWTs don't ship with tip-brakes. The
blade section-properties file preserves the deck's structural twist
(`StrcTwst` column), uses the standard pyBmodes adapter conventions
for torsion stiffness (`100 · EI`), axial stiffness (`10⁶ · EI`), and
rotary inertia floor (`10⁻⁶ · char² · m`). Centrifugal stiffening
lifts the lowest flap mode by a few percent at rated rotor speed;
users wanting a Campbell sweep across rotor speeds can drive the same
BMI with
[`pybmodes.campbell.campbell_sweep`](../../../src/pybmodes/campbell.py).

## A note on RWT-revision drift

Reference-wind-turbine structural definitions are **iteratively
revised** across releases. The same RWT designation (e.g.
`IEA-15-240-RWT`) at git-tag `v1.0.0` may have a few-percent
different section-property distribution than at `v2.0.0`, and the
companion publication's printed mode-shape frequencies were valid
*for the deck version at publication time* — not necessarily for
whatever revision is current in the upstream repository today.

The **pyBmodes frequencies** in the table above are derived from the
**deck-as-distributed** at the time `build.py` was last run. The
**published reference frequencies** quoted in each per-turbine README
are the values the original publication printed at its publication
date. A drift between them typically reflects deck-revision evolution
since publication, **not a pyBmodes error**. Treat the published
values as historical anchors, not regression targets.

## Floating cases

Sub-case **07 — NREL 5MW OC3 Hywind spar** ships as a full
`PlatformSupport`-block BMI in this library (see
[`07_nrel5mw_oc3hywind_spar/`](07_nrel5mw_oc3hywind_spar/)). The
platform 6 × 6 matrices (hydrostatic restoring, hydrodynamic added
mass, mooring stiffness, inertia tensor) are populated from
Jonkman (2010) NREL/TP-500-47535 Table 9-1; pyBmodes' solve of this
sample lands at 0.4821 Hz vs Jonkman 2010's published 0.4816 Hz
(~ 0.1 % match) on the 1st tower-bending mode.

The IEA-15 / IEA-22 floating sub-cases (UMaine VolturnUS-S semi,
etc.) are **not** generated by `build.py` — those need HydroDyn +
WAMIT + MoorDyn integration that `build.py` doesn't drive. The
downstream `Tower.from_elastodyn_with_mooring(...)` classmethod
handles the OpenFAST side end-to-end for coupled-frequency
prediction directly from the upstream decks; a pre-baked sample BMI
for these floaters can be authored by a downstream user using the
same workflow `build.py` uses for the OC3 sample. For the OC3
Hywind, the canonical `OC3Hywind.bmi` (gitignored under
`external/BModes/docs/examples/`) is also a valid input — the bundled
sample at [`07_nrel5mw_oc3hywind_spar/`](07_nrel5mw_oc3hywind_spar/)
is a re-emission with citations to Jonkman (2010) for
redistributability.

## Reproducibility

All BMI files in this directory are generated by the
[`build.py`](build.py) script. Re-running the script reads each
turbine's source ElastoDyn `.dat` decks and re-emits the four files
per turbine (tower BMI + tower section-properties + blade BMI + blade
section-properties) plus the per-turbine README. Sources for three of
the seven turbines (NREL 5MW land, NREL 5MW OC3 monopile, IEA-3.4) are
in the bundled [`reference_decks/`](../../reference_decks/); sources
for the other monopile sub-cases (IEA-10, IEA-15, IEA-22 monopiles)
live under `external/OpenFAST_files/<turbine>/` which is gitignored under
the project's "Independence stance" — `build.py` reads them when
present and skips otherwise. The OC3 Hywind floating sample (07) is
authored inline in `build.py` from Jonkman (2010) Table 9-1 values
and doesn't read an upstream deck.

To regenerate after ``pip install pybmodes`` (or ``pip install -e .``
from a source checkout)::

    python -m pybmodes._examples.sample_inputs.reference_turbines.build

Or, from a source checkout's repo root without installing::

    python src/pybmodes/_examples/sample_inputs/reference_turbines/build.py

The script discovers each turbine's upstream ElastoDyn ``.dat`` decks
under ``external/OpenFAST_files/`` (gitignored) and re-emits the bundled
samples; turbines whose upstream sources aren't present are skipped
silently, so a fresh clone that just wants the four NREL / IEA-3.4
sub-cases works without cloning the larger OpenFAST repositories.

## Citations

- **Jonkman 2009** — Jonkman, J., Butterfield, S., Musial, W., &
  Scott, G. (2009). *Definition of a 5-MW Reference Wind Turbine for
  Offshore System Development*. NREL/TP-500-38060.
- **Jonkman & Musial 2010** — Jonkman, J., & Musial, W. (2010).
  *Offshore Code Comparison Collaboration (OC3) for IEA Wind Task 23
  Offshore Wind Technology and Deployment*. NREL/TP-5000-48191.
- **Jonkman 2010** (OC3 floating-spar definition, used by sub-case
  07) — Jonkman, J. (2010). *Definition of the Floating System for
  Phase IV of OC3*. NREL/TP-500-47535.
- **Bortolotti et al. 2019** — Bortolotti, P., Tarrés, H. C., Dykes,
  K., Merz, K., Sethuraman, L., Verelst, D., & Zahle, F. (2019).
  *IEA Wind TCP Task 37: Systems Engineering in Wind Energy —
  WP2.1 Reference Wind Turbines*. NREL/TP-5000-73492.
- **Gaertner et al. 2020** — Gaertner, E., Rinker, J., Sethuraman,
  L., Zahle, F., Anderson, B., Barter, G., et al. (2020). *Definition
  of the IEA 15-Megawatt Offshore Reference Wind Turbine*.
  NREL/TP-5000-75698.
- **Bortolotti et al. 2024** (IEA-22, technical report) — IEA Wind TCP Task 55 / NREL technical report,
  repository `IEAWindSystems/IEA-22-280-RWT`.
