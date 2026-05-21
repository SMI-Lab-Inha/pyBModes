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

"""Mode-name heuristics for the Campbell sweep.

Pure-numpy + string-formatting helpers — no FEM solves, no plotting.
The sweep driver labels each output slot using participation-argmax
(``1st flap`` / ``2nd edge`` for a blade; ``1st tower FA`` /
``1st tower SS`` for a tower), with an override path for floating
platforms where the FEM's :class:`~pybmodes.fem.platform_modes`
classification already named the rigid-body DOFs.
"""
from __future__ import annotations

import numpy as np

from pybmodes.fem.normalize import NodeModeShape


def _participation(shape: NodeModeShape) -> np.ndarray:
    """Energy fractions in axes 0 / 1 / 2 (sum to 1; zeros if shape is null).

    For a blade these read flap / edge / torsion; for a tower they read
    FA / SS / torsion (same FEM DOF layout, different physical naming).
    """
    flap = float(np.dot(shape.flap_disp, shape.flap_disp))
    edge = float(np.dot(shape.lag_disp, shape.lag_disp))
    twist = float(np.dot(shape.twist, shape.twist))
    total = flap + edge + twist
    if total <= 0.0:
        return np.zeros(3)
    return np.array([flap, edge, twist]) / total


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _label_blade_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st flap"`` / ``"2nd edge"`` / … from participation at one rotor speed."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("flap", "edge", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} {names[axis]}")
    return out


def _label_tower_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st tower FA"`` / ``"1st tower SS"`` / …."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("FA", "SS", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} tower {names[axis]}")
    return out


def _label_tower_modes_with_overrides(
    participation: np.ndarray,
    mode_labels: "list[str | None] | None",
) -> list[str]:
    """Tower-column labels, preferring the FEM's own classification.

    For a free-free floating tower the leading modes are the platform
    rigid-body modes (surge / sway / heave / roll / pitch / yaw), which
    :func:`pybmodes.fem.platform_modes.classify_platform_modes`
    already names on the :class:`~pybmodes.models.result.ModalResult`
    (``mode_labels``) and which BModes-cross-validates. Participation
    argmax (flap/edge/torsion energy) is meaningless for those rigid
    modes — it produced spurious ``"1st tower FA"`` … names for the
    platform DOFs (issue #47). So: where ``mode_labels[i]`` is a
    classified platform DOF, use it verbatim; everywhere else fall
    back to the participation-derived ``"Nth tower FA/SS/torsion"``
    label, with the ordinal counted over the *flexible* tower modes
    only so the first real bending mode is ``"1st tower FA"`` even
    when six rigid modes precede it. ``mode_labels=None`` (every
    cantilever / monopile tower) reproduces :func:`_label_tower_modes`
    exactly.
    """
    n = participation.shape[0]
    counts = [0, 0, 0]
    names = ("FA", "SS", "torsion")
    out: list[str] = []
    for i in range(n):
        plat = (
            mode_labels[i]
            if mode_labels is not None and i < len(mode_labels)
            else None
        )
        if plat is not None:
            out.append(str(plat))
            continue
        axis = int(np.argmax(participation[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} tower {names[axis]}")
    return out
