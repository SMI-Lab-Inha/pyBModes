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

"""Helpers shared between :mod:`pybmodes.models.tower` and
:mod:`pybmodes.models.blade`.

Phase 3 PR C3 of the v1.x architecture refactor pulled
:func:`_run_validation_and_warn` out of ``tower.py``: it was always a
cross-model helper (both ``Tower.from_elastodyn`` and
``RotatingBlade.from_elastodyn`` use it), so a sibling module under
:mod:`pybmodes.models` is the honest home. ``tower.py`` re-exports
the name for back-compat with callers / tests that still import via
``from pybmodes.models.tower import _run_validation_and_warn``.
"""
from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybmodes.elastodyn.validate import ValidationResult


def _run_validation_and_warn(
    main_dat_path: pathlib.Path,
) -> "ValidationResult":
    """Validate coefficient blocks in an ElastoDyn deck and warn on issues.

    Helper shared by ``Tower.from_elastodyn`` and
    ``RotatingBlade.from_elastodyn``. Returns the
    :class:`~pybmodes.elastodyn.ValidationResult`. Emits a
    :class:`UserWarning` if the overall verdict is WARN or FAIL, with
    per-block details for FAIL.
    """
    from pybmodes.elastodyn.validate import validate_dat_coefficients

    result = validate_dat_coefficients(main_dat_path)
    failing = result.failing_blocks()
    warning = result.warning_blocks()

    if failing:
        details = "\n  ".join(
            f"{b.name}: file_rms={b.file_rms:.4f}, "
            f"pyB_rms={b.pybmodes_rms:.4f}, ratio={b.ratio:.0f}"
            for b in failing
        )
        warnings.warn(
            f"{result.summary}\n  {details}\n  "
            f"Run `pybmodes patch {main_dat_path}` to regenerate.",
            UserWarning,
            stacklevel=3,
        )
    elif warning:
        warnings.warn(result.summary, UserWarning, stacklevel=3)

    return result
