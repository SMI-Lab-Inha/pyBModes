# Case 01 — Uniform isotropic cantilever blade (Euler-Bernoulli reference)

A uniform-section cantilevered beam — the textbook Euler-Bernoulli
modal-analysis case. No rotation, no tip mass, no offsets, no
pre-twist. The lowest few flapwise bending frequencies match the
classical eigenvalue equation `cos(βL) · cosh(βL) + 1 = 0`, whose
roots are the dimensionless wavenumbers
`βₙL = 1.87510, 4.69409, 7.85476, 10.99554, 14.13717, …`.

Frequencies are then `ωₙ = (βₙL)² · √(EI / (ρA · L⁴))` rad/s.

## Files

| File                              | Purpose                          |
| --------------------------------- | -------------------------------- |
| `uniform_blade.bmi`               | Main BModes input                |
| `uniform_blade_sec_props.dat`     | Distributed section properties   |

## Physical parameters

| Parameter   | Value      | Units    | Notes                           |
| ----------- | ---------: | -------- | ------------------------------- |
| `radius`    | 31.623     | m        | blade length (no hub offset)    |
| `hub_rad`   | 0.0        | m        | flexible blade from r = 0       |
| `rot_rpm`   | 0          | rpm      | non-rotating                    |
| `hub_conn`  | 1          | —        | cantilevered root               |
| `mass_den`  | 100.0      | kg/m     | uniform mass per length         |
| `EI_flap`   | 1.0 × 10⁸  | N·m²     | flapwise bending stiffness      |
| `EI_edge`   | 1.0 × 10⁹  | N·m²     | edgewise bending stiffness      |
| `GJ`        | 1.0 × 10⁵  | N·m²     | torsional stiffness             |

The choice `L = 31.623 m`, `ρA = 100 kg/m`, `EI_flap = 10⁸ N·m²`
makes `√(EI / (ρA · L⁴)) = 1.0` rad/s exactly, so the FEM-computed
flap frequencies in rad/s equal `(βₙL)²` directly — convenient for
inspecting the FEM output by eye against the reference table.

`EI_edge = 10·EI_flap` and `GJ = 10⁻³·EI_flap` separate the flap, lag,
and torsion families in frequency so each lowest mode is unambiguous.

## Expected frequencies (pyBmodes vs analytical)

| Mode | Type           | ω_FEM (rad/s) | f_FEM (Hz) | βₙL_FEM | βₙL ref  |
| ---: | -------------- | ------------: | ---------: | ------: | -------: |
|    1 | 1st flap       |        3.5160 |     0.5596 |  1.8751 |  1.87510 |
|    2 | 1st lag (edge) |       11.1185 |     1.7696 |    —    |  (1.875·√10) |
|    3 | 2nd flap       |       22.0342 |     3.5069 |  4.6941 |  4.69409 |
|    4 | 3rd flap       |       61.6974 |     9.8194 |  7.8548 |  7.85476 |

Mode 2 is the 1st lag (edgewise) mode — at `ω = 1.875² · √(EI_edge /
(ρA · L⁴)) = 3.516 · √10 = 11.12 rad/s`. The flap-vs-lag frequency
gap of `√10 ≈ 3.16` is exactly what the chosen stiffness ratio
produces.

## How to run

```python
from pybmodes.models import RotatingBlade
blade = RotatingBlade("sample_inputs/01_uniform_blade/uniform_blade.bmi")
modal = blade.run(n_modes=8)
print(modal.frequencies[:4])   # [0.5596, 1.7696, 3.5069, 9.8194]
```

## Validation reference

This case mirrors the parameters of the Bir 2009 / Wright 1982 Table
3a non-rotating row (Ω = 0). The same analytical reference is used in
the project's whitebox test
[`tests/fem/test_cantilever.py`](https://github.com/SMI-Lab-Inha/pyBModes/blob/main/tests/fem/test_cantilever.py).

Citation: Wright, A. D.; Smith, C. E.; Thresher, R. W.; Wang, J. C.
*Vibration Modes of Centrifugally Stiffened Beams*, ASME *Journal of
Applied Mechanics*, Vol. 49, March 1982. The Ω = 0 row reduces to the
classical Euler-Bernoulli cantilever (Karnovsky & Lebed (2001),
*Formulas for Structural Dynamics*; Blevins (1979), *Formulas for
Natural Frequency and Mode Shape*).
