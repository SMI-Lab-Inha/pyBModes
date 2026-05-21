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

"""MAC-based mode-shape tracking for the Campbell sweep.

Two layers:

- :func:`_shape_vector` / :func:`_mac_matrix` — thin back-compat
  wrappers around :mod:`pybmodes.mac` (the public shape-vector +
  pairwise-MAC API). Kept so existing internal callers and tests
  don't break; new code should call :mod:`pybmodes.mac` directly.
- :func:`_hungarian_assignment` — global MAC-maximising assignment
  via :func:`scipy.optimize.linear_sum_assignment`, the standard
  industry approach to mode tracking across an rpm sweep.
- :func:`_greedy_assignment` — deprecated alias.
"""
from __future__ import annotations

import numpy as np

from pybmodes.fem.normalize import NodeModeShape


def _shape_vector(shape: NodeModeShape) -> np.ndarray:
    """Thin wrapper kept for backwards-compatibility with the internal
    Campbell tracker. New code should call
    ``pybmodes.mac.shape_to_vector`` directly."""
    from pybmodes.mac import shape_to_vector
    return shape_to_vector(shape)


def _mac_matrix(
    curr: list[NodeModeShape],
    prev: list[NodeModeShape],
) -> np.ndarray:
    """Thin wrapper around :func:`pybmodes.mac.mac_matrix` kept for
    backwards-compatibility inside the Campbell tracker. New code
    should call ``pybmodes.mac.mac_matrix`` directly."""
    from pybmodes.mac import mac_matrix
    return mac_matrix(curr, prev)


def _hungarian_assignment(mac: np.ndarray) -> np.ndarray:
    """Global MAC-maximising assignment via the Hungarian (Munkres)
    algorithm.

    Returns ``order[i] = j`` mapping current-step mode ``i`` to the
    previous-step slot ``j`` that maximises the sum of MAC values
    across all matched pairs. This is the standard industry approach
    for mode tracking — it avoids the failure mode of the older
    greedy ``argmax(mac)`` scheme, which can lock in a slightly-
    better first match and force later modes into worse pairings.

    Non-square inputs are handled natively by
    ``scipy.optimize.linear_sum_assignment``: it returns
    ``min(n_curr, n_prev)`` matched pairs, and any current-step row
    that did not receive a previous-step pairing stays at the
    sentinel ``-1`` in the output. The caller (``_solve_blade_sweep``)
    fills those slots from any free previous-step indices, so a
    non-square call still produces a well-defined ordering for every
    current-step mode. In practice the Campbell sweep always supplies
    square ``(n_modes, n_modes)`` inputs; the non-square fallback is
    defensive.
    """
    from scipy.optimize import linear_sum_assignment

    n_curr, _ = mac.shape
    row_ind, col_ind = linear_sum_assignment(mac, maximize=True)
    order = -np.ones(n_curr, dtype=int)
    order[row_ind] = col_ind
    return order


def _greedy_assignment(mac: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`_hungarian_assignment` — kept for
    backwards compatibility; new code should call the Hungarian
    version directly."""
    return _hungarian_assignment(mac)
