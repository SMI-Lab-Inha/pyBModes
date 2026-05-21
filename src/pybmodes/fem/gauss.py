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

"""Gauss quadrature points and weights for the interval [0, 1]."""

from __future__ import annotations

import numpy as np


def gauss_6pt() -> tuple[np.ndarray, np.ndarray]:
    """6-point Gauss-Legendre quadrature on [0, 1].

    Returns (points, weights), both shape (6,).
    Used for spatial element integration.
    """
    pts, wts = np.polynomial.legendre.leggauss(6)
    return (pts + 1.0) / 2.0, wts / 2.0


def gauss_5pt() -> tuple[np.ndarray, np.ndarray]:
    """5-point Gauss-Legendre quadrature on [0, 1].

    Returns (points, weights), both shape (5,).
    Not required for free-vibration analysis.
    """
    pts, wts = np.polynomial.legendre.leggauss(5)
    return (pts + 1.0) / 2.0, wts / 2.0
