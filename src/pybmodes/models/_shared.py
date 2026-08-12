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

import numpy as np

if TYPE_CHECKING:
    from pybmodes.elastodyn.validate import ValidationResult
    from pybmodes.io.bmi import BMIFile
    from pybmodes.io.sec_props import SectionProperties


# Two span stations closer than this (in normalised span) encode a
# deliberate property *step* rather than two distinct sections — the
# convention the ElastoDyn / WindIO adapters use to put a wall-thickness
# or material discontinuity into a table that is otherwise interpolated.
_STEP_TOL = 1.0e-9


def refine_deck_mesh(
    bmi: BMIFile,
    sp: SectionProperties | None,
    n_nodes: int,
) -> None:
    """Re-grid a deck-derived model onto ``n_nodes`` evenly-spaced FE nodes.

    The uniform mesh replaces the deck's own ``el_loc`` in place; the
    tabulated section properties are left untouched and the FE pipeline
    re-samples them at the new element midpoints, exactly as it already
    does for the deck's native mesh.

    This is the deck-reader half of issue #58, and it is deliberately a
    *different* operation from the ``n_nodes`` on the geometry-derived
    constructors (``from_geometry`` / ``from_windio*``). Those re-grid
    continuous geometry and recompute closed-form tube properties at each
    new station, so refinement is exact. A deck carries an already
    tabulated property table, so refinement re-samples it — and a uniform
    grid will generally not put a node on a deliberate property step,
    which then gets averaged across the straddling element. That is the
    documented cost, and it is warned about rather than left silent.

    Raises ``ValueError`` for a bad ``n_nodes`` or for a model whose
    support block attaches to *node numbers* (tension wires), since those
    indices refer to the old mesh and would silently move to a different
    elevation under the new one.
    """
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 2:
        raise ValueError(f"n_nodes must be an integer >= 2; got {n_nodes!r}")

    from pybmodes.io.bmi import PlatformSupport, TensionWireSupport

    wires = None
    if isinstance(bmi.support, TensionWireSupport):
        wires = bmi.support
    elif isinstance(bmi.support, PlatformSupport):
        wires = bmi.support.wires
    if wires is not None and getattr(wires, "n_attachments", 0) > 0:
        raise ValueError(
            "n_nodes cannot re-grid a model with tension-wire supports: the "
            "wires attach to FE *node numbers*, which refer to the deck's own "
            "mesh and would silently move to a different elevation on a new "
            "one. Solve this deck on its native mesh, or rebuild it from "
            "geometry (Tower.from_geometry) where the attachment can be "
            "expressed as a height."
        )

    new_el_loc = np.linspace(0.0, 1.0, n_nodes)

    if sp is not None:
        span = np.asarray(sp.span_loc, dtype=float)
        if span.size > 1:
            steps = span[:-1][np.diff(span) <= _STEP_TOL]
            missed = [
                float(s) for s in steps
                if np.min(np.abs(new_el_loc - s)) > _STEP_TOL
            ]
            if missed:
                warnings.warn(
                    f"n_nodes={n_nodes} re-grids this deck onto a uniform "
                    f"mesh that does not land on {len(missed)} deliberate "
                    f"property step(s) in the section-property table (first "
                    f"at normalised span {missed[0]:.4f}). The element "
                    f"straddling a step takes a single mid-element value, so "
                    f"the step is smoothed. Omit n_nodes to keep the deck's "
                    f"own mesh, which places nodes on the steps.",
                    UserWarning,
                    stacklevel=3,
                )

    bmi.el_loc = new_el_loc
    bmi.n_elements = n_nodes - 1


def _run_validation_and_warn(
    main_dat_path: pathlib.Path,
) -> ValidationResult:
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
