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

"""Extensible-elastic-catenary residual + analytical Jacobian.

The Newton kernel inside :meth:`pybmodes.mooring.Line.solve_static`.
Two branches keyed on the seabed-contact toggle and the iterate's
``V`` vs ``W·L`` ratio:

- **Fully suspended** — Jonkman 2007 eq. B-1 / B-2 with EA-stretch
  correction (Irvine 1981 §2.4 eqn 2.49).
- **Anchor on seabed, frictionless** — Jonkman 2007 eq. B-7 / B-8
  with ``CB = 0``.

The analytical 2×2 Jacobian is from Irvine 1981 §2.4 eqn (2.51) for
the fully-suspended branch; the seabed-contact-branch Jacobian is
re-derived from B-7 / B-8 and matches Jonkman 2010 OC3 reference
computations.
"""
from __future__ import annotations

import math

import numpy as np


def _catenary_residual(
    H: float, V: float, dx_h: float, dz: float,
    L: float, W: float, EA: float,
    *, seabed_contact: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``[f1, f2]`` and analytical Jacobian for the catenary
    boundary-condition equations.

    Branches:

    - ``seabed_contact=True`` and ``V < W · L``: anchor portion rests on
      seabed at zero friction — Jonkman 2007 B-7 / B-8 with ``CB = 0``.
    - Otherwise: fully suspended — Jonkman 2007 B-1 / B-2. This is used
      for any line with both endpoints in free air (``seabed_contact=
      False``), plus for FOWT lines whose Newton iterate happens to land
      at ``V ≥ W · L``.

    The Jacobian is the analytical 2 × 2 derived in Irvine 1981 §2.4
    eqn (2.51) (fully-suspended branch); the seabed-contact-branch
    Jacobian is re-derived from B-7 / B-8 and matches Jonkman 2010 OC3
    reference computations.
    """
    WL = W * L
    if seabed_contact and V < WL:
        use_seabed = True
    else:
        use_seabed = False
    if not use_seabed:
        # Fully suspended — Jonkman 2007 eq. B-1, B-2.
        u = V / H
        v = (V - WL) / H
        su = math.sqrt(1.0 + u * u)
        sv = math.sqrt(1.0 + v * v)
        asu_diff = math.asinh(u) - math.asinh(v)
        ssu_diff = su - sv
        inv_diff = 1.0 / su - 1.0 / sv
        usu_diff = u / su - v / sv

        f1 = (H / W) * asu_diff + H * L / EA - dx_h
        f2 = (H / W) * ssu_diff + (L / EA) * (V - 0.5 * WL) - dz

        # ∂F1/∂H = (1/W)[asinh(u) − asinh(v) − (u/su − v/sv)] + L/EA
        # ∂F1/∂V = (1/W)[1/su − 1/sv]
        # ∂F2/∂H = ∂F1/∂V                            (Hessian symmetry)
        # ∂F2/∂V = (1/W)[u/su − v/sv] + L/EA
        J11 = (asu_diff - usu_diff) / W + L / EA
        J12 = inv_diff / W
        J21 = J12
        J22 = usu_diff / W + L / EA
    else:
        # Seabed contact, no friction — Jonkman 2007 eq. B-7, B-8 with
        # CB = 0. Suspended length L_S = V/W; resting length L_B = L − L_S.
        # The seabed portion contributes L_B to ΔX and zero to ΔZ; the
        # suspended portion's lower endpoint sees V = 0, so the standard
        # catenary holds with V_A → 0.
        u = V / H
        su = math.sqrt(1.0 + u * u)
        L_S = V / W
        L_B = L - L_S
        if L_B < 0.0:
            # Numerical edge; should be caught by branch condition.
            L_B = 0.0

        f1 = L_B + (H / W) * math.asinh(u) + H * L / EA - dx_h
        f2 = (H / W) * (su - 1.0) + V * V / (2.0 * W * EA) - dz

        J11 = (math.asinh(u) - u / su) / W + L / EA
        J12 = (1.0 / su - 1.0) / W
        J21 = J12
        J22 = (u / su) / W + V / (W * EA)

    return (
        np.array([f1, f2]),
        np.array([[J11, J12], [J21, J22]]),
    )
