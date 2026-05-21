# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run pyBmodes on every sample input deck in this directory and assert
that the lowest few computed frequencies match the analytical
reference cited in the corresponding case README.

No external data is needed — every reference value below is the
closed-form analytical answer (Euler-Bernoulli wavenumbers, the
Blevins / Karnovsky cantilever-with-tip-mass implicit equation, Bir
(2009) Table 3a / Eq. 8).

This script ships inside the ``pybmodes._examples.sample_inputs``
package tree. Two ways to invoke it:

1. After ``pybmodes examples --copy DIR --kind samples``, run
   ``python DIR/sample_inputs/verify.py`` (the copy-out path).
2. From a source checkout, run it directly from the install location
   alongside the ``.bmi`` decks it consumes.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# When invoked from inside a source checkout the ``src/`` directory
# four levels up is the un-installed package root; prepending it lets
# the script run without ``pip install -e .`` first. When invoked
# from a vendored copy after ``pybmodes examples --copy``, the four-
# level walk lands somewhere unrelated and the prepend is a no-op
# because the user already has a working ``pybmodes`` install.
_SRC_DIR = pathlib.Path(__file__).resolve().parents[4] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pybmodes.fem.normalize import NodeModeShape  # noqa: E402
from pybmodes.models import RotatingBlade, Tower  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOL = 0.01  # 1 % relative error budget on every reference frequency


def _flap_dominated(shape: NodeModeShape) -> bool:
    fl = float(np.dot(shape.flap_disp, shape.flap_disp))
    ed = float(np.dot(shape.lag_disp, shape.lag_disp))
    tw = float(np.dot(shape.twist, shape.twist))
    return fl > 4.0 * (ed + tw) and shape.freq_hz > 1.0e-6


def _fail(case: str, msg: str) -> None:
    print(f"  FAIL  {case}: {msg}")


def _pass(case: str, msg: str) -> None:
    print(f"  PASS  {case}: {msg}")


def _check(case: str, value: float, ref: float, label: str) -> bool:
    rel = abs(value - ref) / abs(ref)
    if rel < TOL:
        _pass(case, f"{label} = {value:.4f} (ref {ref:.4f}, rel err {rel*100:.3f} %)")
        return True
    _fail(case, f"{label} = {value:.4f} (ref {ref:.4f}, rel err {rel*100:.3f} %)")
    return False


def case_01_uniform_blade() -> bool:
    """Euler-Bernoulli cantilever — flap modes match (βₙL)² · √(EI/(ρA L⁴))."""
    case = "01_uniform_blade"
    blade = RotatingBlade(HERE / "01_uniform_blade" / "uniform_blade.bmi")
    modal = blade.run(n_modes=8)
    omega = 2.0 * np.pi * np.asarray(modal.frequencies)
    # L=31.623, EI=1e8, rho A=100 -> sqrt(EI/(rho A L^4)) = 1.0 rad/s exactly,
    # so omega_n = (beta_n L)^2.
    flaps = [omega[i] for i, s in enumerate(modal.shapes) if _flap_dominated(s)][:3]
    ok = True
    refs_betaL = [1.875104, 4.694091, 7.854757]
    for j, w in enumerate(flaps):
        ok &= _check(case, w, refs_betaL[j] ** 2,
                     f"flap-{j+1} omega (rad/s)")
    return ok


def case_02_tower_topmass() -> bool:
    """Cantilever tower with concentrated top mass — Blevins / Karnovsky."""
    case = "02_tower_topmass"
    tower = Tower(HERE / "02_tower_topmass" / "tower_topmass.bmi")
    modal = tower.run(n_modes=8)
    # mu = 1.0 root: beta_1 L = 1.24793 (Karnovsky table 3.6 row mu=1)
    # sqrt(EI/(rho A L^4)) = sqrt(5e10/(5000*80^4)) = 0.49410 rad/s
    sf = float(np.sqrt(5.0e10 / (5000.0 * 80.0 ** 4)))
    omega = 2.0 * np.pi * np.asarray(modal.frequencies)
    return _check(case, omega[0], (1.24793 ** 2) * sf,
                  "1st FA omega (rad/s)")


def case_03_rotating_uniform_blade() -> bool:
    """Rotating cantilever blade — Bir 2009 Table 3a, Ω = 6 rad/s row."""
    case = "03_rotating_uniform_blade"
    blade = RotatingBlade(HERE / "03_rotating_uniform_blade" / "rotating_blade.bmi")
    modal = blade.run(n_modes=12)
    omega = 2.0 * np.pi * np.asarray(modal.frequencies)
    flaps = [omega[i] for i, s in enumerate(modal.shapes) if _flap_dominated(s)][:3]
    ok = True
    refs = [7.360, 26.809, 66.684]   # Bir Table 3a, Omega = 6
    for j, w in enumerate(flaps):
        ok &= _check(case, w, refs[j], f"flap-{j+1} omega (rad/s)")
    return ok


def case_04_pinned_free_cable() -> bool:
    """Pinned-free cable — Bir 2009 Eq. 8: ω_k = Ω · √(k(2k − 1))."""
    case = "04_pinned_free_cable"
    cable = RotatingBlade(HERE / "04_pinned_free_cable" / "cable.bmi")
    modal = cable.run(n_modes=12)
    omega = 2.0 * np.pi * np.asarray(modal.frequencies)
    flaps = [omega[i] for i, s in enumerate(modal.shapes) if _flap_dominated(s)][:3]
    ok = True
    omega_rot = 10.0  # rot_rpm = 95.49297 -> Omega = 10 rad/s
    refs = [omega_rot * float(np.sqrt(k * (2.0 * k - 1.0))) for k in (1, 2, 3)]
    for j, w in enumerate(flaps):
        ok &= _check(case, w, refs[j], f"flap-{j+1} omega (rad/s)")
    return ok


def main() -> int:
    print("pyBmodes sample-input verification")
    print("=" * 50)
    cases = [
        ("01 uniform blade",         case_01_uniform_blade),
        ("02 tower with top mass",   case_02_tower_topmass),
        ("03 rotating uniform blade", case_03_rotating_uniform_blade),
        ("04 pinned-free cable",     case_04_pinned_free_cable),
    ]
    n_total = len(cases)
    n_passed = 0
    for label, fn in cases:
        print(f"\n{label}:")
        try:
            ok = fn()
        except Exception as exc:
            print(f"  ERROR {exc!r}")
            ok = False
        if ok:
            n_passed += 1

    print()
    print("=" * 50)
    print(f"Result: {n_passed}/{n_total} sample case(s) passed.")
    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
