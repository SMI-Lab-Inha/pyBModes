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

"""3-2-1 intrinsic Euler rotation — primitive shared by
:meth:`pybmodes.mooring.Point.r_world` (and transitively by
:meth:`pybmodes.mooring.MooringSystem.restoring_force` through it).
Matches ElastoDyn's platform attitude convention.
"""
from __future__ import annotations

import math

import numpy as np


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 intrinsic Euler rotation: ``R = R_z(yaw) · R_y(pitch) · R_x(roll)``.

    Matches ElastoDyn's platform attitude convention so a ``r_body``
    expressed in body coords maps to world coords via ``R · r_body``.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
