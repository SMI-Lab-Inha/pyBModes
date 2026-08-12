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

"""The raw inputs a geometry-derived model was built from (issue #102).

:func:`pybmodes.checks.check_model` runs on a parsed model, whose section
properties are *derived* — and derived properties cannot support the
domain checks a non-specialist most needs. The scoping work on issue #102
found the concrete trap: a longitudinal wave-speed guard ``sqrt(EA / rho A)``
that catches the classic "forgot the e9" unit error false-positives on
every land and monopile reference deck, because an ElastoDyn-convention
tower is modelled **axially rigid** and its ``axial_stff`` is a large
placeholder rather than a physical ``E A``.

So the material and shell checks have to see what the *user* typed, not
what the reduction produced. The geometry constructors record it here and
hang it on the model as ``_construction``; the checks read it when
present and skip cleanly when it is absent (a deck-derived model, where
those numbers genuinely do not exist).

Nothing in this module feeds the FE pipeline. It exists so the checks can
be honest about what they are looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstructionInputs:
    """What a geometry-derived model was built from, per tube segment.

    Attributes
    ----------
    segments : one :class:`TubeSegment` per independently-reduced tube.
        A plain tower has one; a spliced monopile + tower has two, each
        with its own wall schedule and steel grade.
    is_monopile : the model includes a monopile, i.e. a driven pile whose
        base is in the seabed. Enables the geotechnical gates.
    has_soil : a soil foundation is attached, so the base is not a rigid
        clamp. Suppresses the non-conservative-clamp note.
    embedded_length : embedded pile length in metres when known.
    """

    segments: list[TubeSegment] = field(default_factory=list)
    is_monopile: bool = False
    has_soil: bool = False
    embedded_length: float | None = None


@dataclass
class TubeSegment:
    """One circular-tube segment as the user supplied it.

    ``station_grid`` is normalised ``[0, 1]`` from the segment base to its
    top; ``outer_diameter`` and ``wall_thickness`` are in metres per
    station; ``E`` (Pa), ``rho`` (kg/m^3) and ``nu`` are the isotropic
    material.
    """

    name: str
    station_grid: np.ndarray
    outer_diameter: np.ndarray
    wall_thickness: np.ndarray
    E: float
    rho: float
    nu: float

    @property
    def diameter_thickness(self) -> np.ndarray:
        """``D / t`` per station, with non-positive walls masked out."""
        od = np.asarray(self.outer_diameter, dtype=float)
        wt = np.asarray(self.wall_thickness, dtype=float)
        good = np.isfinite(od) & np.isfinite(wt) & (wt > 0.0)
        return od[good] / wt[good]
