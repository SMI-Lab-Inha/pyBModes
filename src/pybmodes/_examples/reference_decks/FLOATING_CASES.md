<!-- markdownlint-disable MD013 -->
# Floating cases — the correct ElastoDyn polynomial basis is *cantilever*

For floating platforms, ElastoDyn polynomial coefficients
(`TwFAM1Sh`, `TwFAM2Sh`, `TwSSM1Sh`, `TwSSM2Sh`) must be derived
from a **cantilever** (`hub_conn = 1`) tower model with the RNA
lumped at the tower top — **NOT** from a platform-coupled floating
model.

## Why

ElastoDyn represents floating tower dynamics as a clamped-base
cantilever **in the platform-attached frame**, with platform 6-DOF
motion handled separately as independent generalised coordinates
(`Sg / Sw / Hv / R / P / Y`). Three independent code-level evidences
in OpenFAST `modules/elastodyn/src/ElastoDyn.f90` (main branch):

1. The polynomial ansatz in `SHP` evaluates
   `Σ_{i=1..PolyOrd-1} c_i · (h/H)^(i+1)` (lines 2486–2495). The
   lowest power is `Fract²`, so `SHP(0) = SHP'(0) = 0`
   *identically*. A free-free or pinned-pinned mode shape with
   non-zero base slope cannot be represented in this format.
2. The base node is hard-coded zero: `p%TwrFASF(:,0,0:1) = 0`,
   `p%TwrSSSF(:,0,0:1) = 0` (lines 5147–5148).
3. The internal tower modal eigenproblem (`Coeff` subroutine,
   lines 5141–5267) integrates `MTFA = TwrTpMass + ∫ ρA φ² dh`
   and `KTFA = ∫ EI φ'' φ'' dh + KTFAGrav`. **No** `PlatformMass`,
   **no** `hydro_K`, **no** `mooring_K`, **no** `i_matrix` enter
   this assembly. The only tip-end inertia is the scalar
   `TwrTpMass` (lumped RNA mass).

Platform 6-DOF motion enters the absolute tower kinematics via the
**rigid-body sum** (lines 7485–7540):

```text
v_T(J) = v_Z + ω_X × rZT(J) + Σ_k φ_k(h_J) · q̇_k
```

`Sg/Sw/Hv/R/P/Y` and the tower modal coordinates `q_TFA1 / q_TFA2 /
q_TSS1 / q_TSS2` are **independent** generalised coordinates;
platform motion does NOT appear as forcing on `q_TFA1`. Feeding
ElastoDyn polynomials that already encode platform-coupling
**double-counts** the platform restoring forces because ElastoDyn
re-derives those effects independently through the platform DOFs.

Same BC for land and floating — only the runtime treatment of the
clamp point differs (locked in Earth for land; rigidly attached to
the moving platform for floating).

## How

Floating-platform polynomial coefficients are generated with the
**existing** pyBmodes path — `Tower.from_elastodyn(...)` is
*already* the cantilever path. It clamps at `TowerBsHt` with the RNA
lumped at the top, ignores any platform / hydro / mooring matrices,
and produces exactly the basis ElastoDyn assumes. No flag is needed.

```python
from pybmodes.models import Tower
from pybmodes.elastodyn import compute_tower_params, patch_dat
from pybmodes.io.elastodyn_reader import read_elastodyn_main

main_path = "Floating_ElastoDyn.dat"
tower = Tower.from_elastodyn(main_path)        # cantilever, RNA at top, no platform
result = tower.run(n_modes=10)
params = compute_tower_params(result)

# patch_dat rewrites the *tower* .dat file (where the polynomial
# blocks live), not the main ElastoDyn .dat.
main = read_elastodyn_main(main_path)
patch_dat(main_path.replace("ElastoDyn.dat", main.twr_file), params)
```

No WAMIT files, no HydroDyn parsing, and no MoorDyn parsing are
required. The cantilever path is correct and self-contained — the
ElastoDyn `.dat` (plus the tower file it references) carries every
input needed.

## What about `Tower.from_bmi()` with `hub_conn = 2`?

`Tower.from_bmi("OC3Hywind.bmi")` and similar BModes-format decks
with a populated `PlatformSupport` block solve the **coupled**
tower-and-platform eigenproblem (free-free root, full 6×6 hydro /
mooring / inertia matrices). That path:

- **Correctly predicts coupled-system frequencies** for validation
  against BModes JJ. pyBmodes matches BModes JJ to ~ 0.0003 % across
  the first nine OC3 Hywind modes (`test_certtest_oc3hywind`). If
  the goal is "what does the floating tower vibrate at when coupled
  to its platform?", this is the right path.
- **Produces eigenvectors that include platform rigid-body motion**
  — i.e. the modes have non-zero base displacement and non-zero
  base slope, which the ElastoDyn `SHP` ansatz cannot represent.
  Feeding these eigenvectors into a polynomial fit produces
  coefficients ElastoDyn cannot consume without double-counting the
  platform.

The two paths answer different questions; both are correct for
their intended use:

| Goal | Use | BC |
| --- | --- | --- |
| ElastoDyn polynomial coefficients (any floating deck) | `Tower.from_elastodyn(...)` | `hub_conn = 1`, RNA at top |
| Coupled-system frequency validation against BModes JJ | `Tower.from_bmi("OC3Hywind.bmi")` | `hub_conn = 2`, full PlatformSupport |

## Configurations included in `reference_decks/`

This directory now ships pre-patched ElastoDyn decks for three
floating configurations alongside the original three fixed-base
decks:

- [`nrel5mw_oc3spar/`](nrel5mw_oc3spar/) — *NREL 5MW* on the OC3
  Hywind spar (Jonkman 2010). Source: OpenFAST `r-test`
  `5MW_OC3Spar_DLL_WTurb_WavesIrr/`.
- [`nrel5mw_oc4semi/`](nrel5mw_oc4semi/) — *NREL 5MW* on the OC4
  DeepCwind semi-submersible (Robertson et al. 2014). Source:
  OpenFAST `r-test` `5MW_OC4Semi_WSt_WavesWN/`.
- [`iea15mw_umainesemi/`](iea15mw_umainesemi/) — *IEA-15-240-RWT*
  on the UMaine VolturnUS-S semi (Allen et al. 2020). Source:
  upstream `IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/`.

Each deck is built by `scripts/build_reference_decks.py` using the
cantilever path documented above; the validator passes on all four
tower coefficient blocks after patching.

## FAQ — why does my OpenFAST linearisation report a different frequency than pyBmodes?

This is the question that keeps coming back. The short answer is
"by design, here is the gap". The long answer is what the table
above tries to say.

A floating ElastoDyn deck has **two** natural tower bending
frequencies. They differ by 20-30 percent on a typical floating
platform, and the gap is not a bug.

- **Cantilever 1st FA / SS** is the modal basis ElastoDyn uses
  internally. The polynomial blocks `TwFAM1Sh` / `TwSSM1Sh` (and the
  higher-order partners) describe the *cantilever* mode shape because
  the `SHP` ansatz at `ElastoDyn.f90:2486-2495` algebraically forces
  `SHP(0) = SHP'(0) = 0`. ElastoDyn's `Coeff` subroutine
  (`ElastoDyn.f90:5141-5267`) integrates a modal mass that adds the
  tip-end `TwrTpMass` to the spanwise integral of `rho_A` weighted
  by the squared mode shape, and a modal stiffness that integrates
  `EI` weighted by the squared mode curvature plus the `KTFAGrav`
  destiffening term, with **no** platform / hydro / mooring
  contributions. At runtime the augmented system row for `q_TFA1`
  (`ElastoDyn.f90:8426-8445`) carries only the elastic and damping
  restoring `-KTFA q - CTFA qdot`, with no `phi(tip)` weighted
  platform-stiffness coupling, so the polynomial really does describe
  an uncoupled cantilever tower.
- **Coupled 1st FA / SS** is what an OpenFAST linearisation reports
  when platform 6-DOF, mooring restoring, and hydrostatic restoring
  are all engaged. Platform restoring can shift the apparent tower
  bending frequency substantially (typically stiffening it on a spar
  with negative pitch hydrostatic, softening it on a TLP, varying on
  a semi). This is also what pyBmodes'
  `Tower.from_elastodyn_with_mooring(...).run(...)` produces.

The two numbers answer different questions. The cantilever number is
what ElastoDyn's internal modal eigenvalue equation evaluates to and
is the right reference for "is my polynomial block consistent with
my structural blocks?". The coupled number is the actual eigenfrequency
of the closed-loop floating system and is the right reference for "what
frequency will I see in my time-domain output and in my linearisation
table?".

To surface the gap directly, call:

```python
from pybmodes.elastodyn import report_floating_frequency_gap

gap = report_floating_frequency_gap(
    "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat",
    "NRELOffshrBsline5MW_OC3Hywind_MoorDyn.dat",
    "NRELOffshrBsline5MW_OC3Hywind_HydroDyn.dat",
)
print(gap.format_report())
```

Sample output (illustrative; the exact numbers depend on the deck):

```text
Cantilever 1st FA: 0.385 Hz (ElastoDyn polynomial basis)
Coupled 1st FA:    0.493 Hz (actual floating system frequency)
Gap: +28.1% (platform restoring shifts apparent tower bending)

Cantilever 1st SS: 0.385 Hz
Coupled 1st SS:    0.493 Hz
Gap: +28.1%
```

### What about generating polynomials *from* the coupled solve?

A previous proposal (dropped) was to fit ElastoDyn polynomial
coefficients to the coupled eigenvector after subtracting the root
tangent line, i.e. `phi_elastic(x) = phi(x) - phi(0) - phi'(0) * x`.
The proposal would have made `SHP(0) = SHP'(0) = 0` hold by
construction.

This was reviewed against the OpenFAST source and rejected on three
grounds.

1. **Wrong identification.** The transformation `phi(x) - phi(0) -
   phi'(0) * x` is BModes' Improved Direct Method, not the Projection
   Method (which is a 2-D rotation by `-atan(slope * scale)`).
   pyBmodes already applies the Improved Direct Method byte-identically
   inside every tower polynomial fit, at
   `pybmodes.elastodyn.params._remove_root_rigid_motion`.
2. **Rayleigh-quotient bias.** ElastoDyn computes the tower bending
   frequency as the square root of the modal stiffness over the modal
   mass, evaluating both integrals from the polynomial mode shape.
   The cantilever shape is the variational minimiser of that Rayleigh
   quotient subject to `SHP(0) = SHP'(0) = 0`. A projected coupled
   shape does **not** minimise it, so feeding the projected polynomial
   would shift `FreqTFA` away from the cantilever optimum without
   landing on the coupled-system frequency either.
3. **Double-counting platform restoring.** The projected coupled
   eigenvector carries platform-restoring information in its interior
   curvature. ElastoDyn then re-applies platform restoring at runtime
   through the independent `Sg/Sw/Hv/R/P/Y` DOFs in the kinematic
   rigid-body sum (`ElastoDyn.f90:7485-7544`). The result is the same
   class of double-counting flagged in `cases/ECOSYSTEM_FINDING.md`
   for the WISDEM free-free path.

If you want to surface the coupled-vs-cantilever gap to users, use
the diagnostic above. If you want to attack the true BModes
Projection Method gap (the 2-D rotation), that is a separate scoped
piece of work and would not move the needle on OC3 Hywind because
the existing coupled-vs-BModes-JJ match is already 0.0003 percent.

## Citations

- Jonkman, J., Butterfield, S., Musial, W., & Scott, G. (2009).
  *Definition of a 5-MW Reference Wind Turbine for Offshore System
  Development*. NREL/TP-500-38060.
- Jonkman, J. (2010). *Definition of the Floating System for Phase
  IV of OC3*. NREL/TP-500-47535.
- Robertson, A., Jonkman, J., Masciola, M., Song, H., Goupee, A.,
  Coulling, A., & Luan, C. (2014). *Definition of the
  Semisubmersible Floating System for Phase II of OC4*.
  NREL/TP-5000-60601.
- Allen, C., Viselli, A., Dagher, H., Goupee, A., Gaertner, E.,
  Abbas, N., Hall, M., & Barter, G. (2020). *Definition of the
  UMaine VolturnUS-S Reference Platform Developed for the
  IEA Wind 15-Megawatt Offshore Reference Wind Turbine*.
  NREL/TP-5000-76773.
- Gaertner, E., Rinker, J., Sethuraman, L., Zahle, F., Anderson, B.,
  Barter, G., et al. (2020). *Definition of the IEA 15-Megawatt
  Offshore Reference Wind Turbine*. NREL/TP-5000-75698.
