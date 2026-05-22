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

"""Platform-scalar parsing + inertia-matrix assembly helpers.

Used by :meth:`pybmodes.models.Tower.from_elastodyn_with_mooring` and
:meth:`pybmodes.models.Tower.from_windio_floating` (deck-fallback
tier). Extracted from :mod:`pybmodes.models.tower` in Phase 3 PR C3:
neither helper belongs *inside* the :class:`Tower` class, but both
are tower-specific (the geometry / cantilever paths don't touch
``Ptfm*`` scalars), so they live as private siblings of ``tower.py``
under :mod:`pybmodes.models`.

``tower.py`` re-exports both names for back-compat with the existing
test import paths
(``from pybmodes.models.tower import _scan_platform_fields,
_platform_inertia_matrix``).
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _scan_platform_fields(dat_path: pathlib.Path) -> dict[str, float]:
    """Scan an ElastoDyn ``.dat`` for the platform scalars used to
    assemble a floating ``PlatformSupport``.

    Two field groups are extracted:

    - **Geometry / inertia** (``PtfmMass``, ``PtfmRIner``,
      ``PtfmPIner``, ``PtfmYIner``, ``PtfmCMxt``, ``PtfmCMyt``,
      ``PtfmCMzt``, ``PtfmRefzt``) — feed the BMI's ``mass_pform``,
      ``cm_pform``, ``i_matrix``, ``ref_msl``.
    - **Additional linear platform stiffness**
      (``PtfmSurgeStiff``, ``PtfmSwayStiff``, ``PtfmHeaveStiff``,
      ``PtfmRollStiff``, ``PtfmPitchStiff``, ``PtfmYawStiff``) — these
      are ElastoDyn-side springs added on top of HydroDyn / MoorDyn
      contributions. ``PtfmYawStiff`` is how the OC3 spec carries the
      delta-line crowfoot's yaw spring (~ 9.83e7 N·m/rad), which is NOT
      in the MoorDyn ``.dat``. :meth:`Tower.from_elastodyn_with_mooring`
      folds them into the diagonal of ``mooring_K``.

    The full ElastoDyn parser in :mod:`pybmodes.io._elastodyn` doesn't
    surface these (they're irrelevant for the cantilever path); this
    helper is a tiny shim used by :meth:`Tower.from_elastodyn_with_mooring`
    to avoid extending the main parser for a single use case. Missing
    fields default to ``0.0``. Fortran-style D / d exponents
    (``7.466D+06``) are normalised to ``E`` before parsing.
    """
    # Every field this helper extracts is load-bearing: the inertia
    # scalars define the platform's rigid-body mass-matrix
    # contribution, and the ``Ptfm*Stiff`` springs carry restoring
    # contributions that are NOT in HydroDyn or MoorDyn (the OC3
    # delta-line crowfoot yaw spring at ~ 9.83e7 N·m/rad lives in
    # ``PtfmYawStiff`` only). A malformed value silently falling back
    # to 0.0 would produce a physically wrong floating model with no
    # warning, so we raise on any parse failure.
    fields: dict[str, float] = {
        "PtfmMass": 0.0, "PtfmRIner": 0.0, "PtfmPIner": 0.0,
        "PtfmYIner": 0.0, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
        "PtfmCMzt": 0.0, "PtfmRefzt": 0.0,
        "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
        "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
        "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
    }
    from pybmodes.io.wamit_reader import _parse_fortran_float

    with pathlib.Path(dat_path).open(
        "r", encoding="utf-8", errors="replace",
    ) as fh:
        for raw in fh:
            parts = raw.split()
            if len(parts) < 2:
                continue
            value, label = parts[0], parts[1]
            if label in fields:
                try:
                    fields[label] = _parse_fortran_float(value)
                except ValueError as err:
                    raise ValueError(
                        f"Malformed value for {label!r} in "
                        f"{dat_path}: {value!r} cannot be parsed as a "
                        f"float (even with Fortran-style D/d exponent "
                        f"normalisation). The platform model would be "
                        f"physically meaningless or silently lose a "
                        f"restoring contribution without this scalar."
                    ) from err
    return fields


def _platform_inertia_matrix(ptfm: dict[str, float]) -> np.ndarray:
    """Assemble the platform 6×6 inertia matrix AT THE CM in
    **OpenFAST DOF order** ``[surge, sway, heave, roll, pitch, yaw]``
    from the ``Ptfm*`` scalars produced by :func:`_scan_platform_fields`.

    Diagonal-only — translation slots 0–2 carry ``PtfmMass``,
    rotation slots 3 / 4 / 5 carry ``PtfmRIner`` / ``PtfmPIner`` /
    ``PtfmYIner`` respectively. Cross-coupling terms (``[0,4]`` for
    surge-pitch, ``[1,3]`` for sway-roll) are zero on the at-CM
    matrix; the downstream :func:`pybmodes.fem.nondim.nondim_platform`
    applies the rigid-arm CM → tower-base transfer using
    ``cm_pform - draft``, so adding a parallel-axis term here would
    double-count (caught by a pre-1.0 review).

    The DOF order is the canonical convention documented in
    :mod:`pybmodes.coords` and consumed by ``nondim_platform``. A
    pre-1.0 review caught a latent swap (``PtfmPIner`` at slot 3,
    ``PtfmRIner`` at slot 4) that was invisible on OC3 — where roll
    and pitch inertia are equal by symmetry — but would silently
    mis-couple roll and pitch on any asymmetric semi or
    submersible. :func:`tests.test_mooring.test_platform_inertia_matrix_dof_order`
    pins the convention.
    """
    import numpy as np

    i_mat = np.zeros((6, 6))
    i_mat[0, 0] = ptfm["PtfmMass"]    # surge mass
    i_mat[1, 1] = ptfm["PtfmMass"]    # sway  mass
    i_mat[2, 2] = ptfm["PtfmMass"]    # heave mass
    i_mat[3, 3] = ptfm["PtfmRIner"]   # roll  inertia about CM (DOF 3)
    i_mat[4, 4] = ptfm["PtfmPIner"]   # pitch inertia about CM (DOF 4)
    i_mat[5, 5] = ptfm["PtfmYIner"]   # yaw   inertia about CM
    return i_mat
