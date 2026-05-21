# pyBmodes sample input files

Public-facing, pyBmodes-authored `.bmi` and section-properties `.dat`
files corresponding to validated analytical reference cases. Every
file in this tree was hand-written for the project — no third-party
data is bundled, and every numeric value is reproducible from the
cited textbook formula or peer-reviewed paper named in the case's
own README.

These are the input files to copy / adapt as a starting point when
you want to:

- understand what a minimal, well-formed `.bmi` deck looks like;
- spot-check that pyBmodes parses your local install correctly;
- benchmark a structural FEM change against an analytical reference;
- carve out a small reproducer when filing a bug or a feature
  request.

The four cases together exercise the four boundary conditions
pyBmodes supports (`hub_conn ∈ {1, 4}` here; `hub_conn ∈ {2, 3}` are
covered by the bundled offshore certtest decks under
`external/BModes/docs/examples/` when you have those locally), the tower
+ blade beam-type split, the rotating + non-rotating split, and the
tip-mass + no-tip-mass split.

## Index — analytical-reference cases (this directory)

| Case | Title                                          | Beam   | Ω (rad/s) | Tip mass | BC          | Ref              |
| ---- | ---------------------------------------------- | ------ | --------: | -------- | ----------- | ---------------- |
| [01](01_uniform_blade/) | Uniform isotropic cantilever blade  | blade  |       0   | none     | cantilever  | Euler-Bernoulli  |
| [02](02_tower_topmass/) | Uniform tower with concentrated top mass | tower  |       0   | μ = 1.0  | cantilever  | Blevins (1979)   |
| [03](03_rotating_uniform_blade/) | Rotating uniform blade           | blade  |       6   | none     | cantilever  | Wright 1982 / Bir 2009 Table 3a |
| [04](04_pinned_free_cable/) | Rotating pinned-free cable       | blade  |      10   | none     | pinned-free | Bir 2009 Eq. 8   |

## Index — reference-wind-turbine cases

The [`reference_turbines/`](reference_turbines/) sub-directory ships
**tower + blade** BMI samples for seven widely-cited open-literature
reference wind turbines (RWTs): NREL 5MW (land + OC3 monopile + OC3
Hywind floating spar), IEA-3.4, IEA-10, IEA-15, IEA-22 (monopile
sub-cases), generated from the published structural data via
[`reference_turbines/build.py`](reference_turbines/build.py). See
[`reference_turbines/README.md`](reference_turbines/README.md) for
the full list, the modelling assumptions (cantilever tower from
TowerBsHt with RNA lumped at the top; rotating-cantilever blade
spinning at the deck's RotSpeed), and the per-RWT pyBmodes
frequencies vs the originally-published reference values.

A note on RWT references: their structural definitions are
**iteratively revised** across releases — the same designation at
git-tag v1.0.0 may have a few-percent different section-property
distribution than at v2.0.0. Each per-turbine README reports both the
pyBmodes frequency derived from the deck-as-distributed at the time
the BMI sample was last built *and* the original publication's printed
value; the two need not match exactly, and any drift between them
typically reflects deck-revision evolution rather than a pyBmodes
error.

Each case directory contains:

- a `.bmi` main input;
- a companion section-properties `.dat`;
- a `README.md` documenting the physical parameters, the analytical
  reference value(s) the FEM is expected to match, and a one-line
  copy-paste Python invocation that runs pyBmodes on the case.

## Verification

`verify.py` (in this directory) runs pyBmodes on every sample case
and asserts that the lowest few computed frequencies match the
analytical reference to within 1 % relative error. After ``pip
install pybmodes`` (or ``pip install -e .`` from a source checkout)::

    python -m pybmodes._examples.sample_inputs.verify

Or, if you cloned the repo and want to run from the source tree
directly without installing, from the repo root::

    python src/pybmodes/_examples/sample_inputs/verify.py

Output is one PASS / FAIL line per case plus a one-line summary at
the end. No external data is required — the verification uses only
the closed-form formulas cited in each case's README.

## Conventions

All sample `.bmi` files follow the canonical line-ordered format —
section headers, value-then-label data lines, two-line block
separators — that pyBmodes' parser shares with BModes JJ (Fortran),
so the same files can also be fed to BModes JJ for cross-solver
verification. They differ from BModes' own example decks under
`external/BModes/docs/examples/` only in that the parameter values here
come from peer-reviewed analytical references rather than industry
reference turbines, so they're safe to redistribute under pyBmodes'
Apache 2.0 licence.

## See also

- [`tests/fem/`](https://github.com/SMI-Lab-Inha/pyBModes/tree/master/tests/fem) — the project's whitebox FEM
  validation tests, which build the FEM matrices directly without
  going through the BMI parser. Each case here mirrors one of the
  whitebox tests with the same physical parameters.
- `external/BModes/docs/examples/` (local-only, gitignored under the
  project's "Independence stance") — BModes JJ's own example decks
  `OC3Hywind.bmi` and `CS_Monopile.bmi` for offshore + floating +
  rigid-monopile configurations. Clone the upstream BModes
  repository into your local checkout to populate this directory;
  the path is intentionally not linked because the target isn't
  tracked in git and the previous relative link
  (`../../external/BModes/docs/examples/`) didn't resolve from inside
  the packaged wheel anyway.
- [`tests/_synthetic_bmi.py`](https://github.com/SMI-Lab-Inha/pyBModes/blob/master/tests/_synthetic_bmi.py) — the
  programmatic `.bmi` writer used by tests for in-tmp_path round-trip
  fixtures. Useful as a starting point if you need to generate sample
  decks with parameter sweeps rather than by hand.
