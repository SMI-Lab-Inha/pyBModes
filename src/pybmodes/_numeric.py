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

"""Small numerical helpers shared across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["trapezoid"]


def trapezoid(y: ArrayLike, x: ArrayLike) -> float:
    """Trapezoidal integral of ``y`` over ``x``.

    Implemented directly rather than through ``np.trapezoid``, which only
    exists from NumPy 2.0, or ``np.trapz``, which NumPy 2 deprecates, so
    the package works across the whole advertised ``numpy>=1.26`` range.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
