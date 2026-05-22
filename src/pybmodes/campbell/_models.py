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

"""Input-model dispatcher for :func:`campbell_sweep`.

Resolves a heterogeneous input — a path (``.bmi`` or ElastoDyn
``.dat``), an already-loaded :class:`~pybmodes.models.RotatingBlade`
or :class:`~pybmodes.models.Tower` (issue #51), plus the optional
``tower_input`` keyword — to a pair of ``(blade, tower)`` model
tuples, each ``(BMIFile, SectionProperties | None)``.

The deferred imports (``models.blade``, ``models.tower``,
``io.elastodyn_reader``) keep this module loadable without dragging
the heavy IO + model machinery into a notebook that only uses
:class:`~pybmodes.campbell.CampbellResult`.
"""
from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

from pybmodes.io.bmi import BMIFile, read_bmi
from pybmodes.io.sec_props import SectionProperties

if TYPE_CHECKING:
    pass  # all RotatingBlade/Tower imports are deferred (cycle break)


# A "model" is a (BMIFile, SectionProperties|None) tuple. ``None`` for the
# section-properties slot signals that ``run_fem`` should re-read them
# from disk via ``BMIFile.resolve_sec_props_path``; ElastoDyn-derived
# models supply them directly.
_Model = tuple[BMIFile, SectionProperties | None]


def _model_pair(obj: object) -> tuple[str, _Model] | None:
    """``(role, (_bmi, _sp))`` for an already-loaded ``RotatingBlade``
    / ``Tower`` (issue #51 — single point of load-in; also the only
    way to Campbell a ``from_windio`` / ``from_elastodyn`` model,
    which carry pre-built section properties no path can re-read).
    ``None`` if ``obj`` is not a loaded model (treat as a path).

    Deferred import keeps ``campbell`` free of a ``models`` import
    cycle. ``role`` is taken from ``beam_type`` so a blade or tower
    can be passed in either positional slot."""
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.tower import Tower

    if not isinstance(obj, (RotatingBlade, Tower)):
        return None
    bmi = obj._bmi
    sp = obj._sp
    if bmi.beam_type == 1:
        return "blade", (bmi, sp)
    if bmi.beam_type == 2:
        return "tower", (bmi, sp)
    raise ValueError(
        f"loaded {type(obj).__name__} has unsupported beam_type "
        f"{bmi.beam_type} (expected 1 = blade or 2 = tower)"
    )


def _load_models(
    input_path: str | pathlib.Path | object,
    tower_input: str | pathlib.Path | object | None,
) -> tuple[_Model | None, _Model | None]:
    """Resolve the input(s) to (blade, tower) model pairs.

    ``input_path`` may be a path **or an already-loaded
    ``RotatingBlade`` / ``Tower``** (issue #51): a loaded model is
    used directly — no disk re-read — which is the only way to sweep a
    ``from_windio`` / ``from_elastodyn`` model and gives a single
    point of load-in. For an ElastoDyn ``.dat`` file the deck carries
    both, so we load both unless the corresponding files can't be
    resolved. ``.bmi`` inputs are routed to blade or tower by their
    ``beam_type``. The ``tower_input`` keyword (a tower ``.bmi`` path
    *or* a loaded ``Tower``) pairs an explicit tower with a blade
    input; if the primary input was an ElastoDyn deck and
    ``tower_input`` is also given, ``tower_input`` overrides the
    deck-supplied tower (useful when the deck's tower file points
    somewhere unhelpful).
    """
    blade: _Model | None = None
    tower: _Model | None = None

    # Already-loaded RotatingBlade / Tower — use verbatim.
    mp = _model_pair(input_path)
    if mp is not None:
        role, pair = mp
        if role == "blade":
            blade = pair
        else:
            tower = pair
        if tower_input is not None:
            tmp = _model_pair(tower_input)
            if tmp is not None:
                if tmp[0] != "tower":
                    raise ValueError(
                        "tower_input must be a Tower (beam_type=2) or a "
                        "tower .bmi"
                    )
                tower = tmp[1]
            else:
                tp = pathlib.Path(tower_input)  # type: ignore[arg-type]
                if tp.suffix.lower() != ".bmi":
                    raise ValueError(
                        f"tower_input must be a .bmi file; got "
                        f"{tp.suffix!r}"
                    )
                tbmi = read_bmi(tp)
                if tbmi.beam_type != 2:
                    raise ValueError(
                        f"tower_input {tp} has beam_type "
                        f"{tbmi.beam_type}, expected 2 (tower)"
                    )
                tower = (tbmi, None)
        return blade, tower

    input_path = pathlib.Path(input_path)  # type: ignore[arg-type]
    suffix = input_path.suffix.lower()

    if suffix == ".dat":
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
            to_pybmodes_blade,
            to_pybmodes_tower,
        )
        main = read_elastodyn_main(input_path)
        bld_path = input_path.parent / main.bld_file[0]
        blade_data = read_elastodyn_blade(bld_path)
        blade = to_pybmodes_blade(main, blade_data)

        twr_path = input_path.parent / main.twr_file
        if twr_path.is_file():
            tower_data = read_elastodyn_tower(twr_path)
            tower = to_pybmodes_tower(main, tower_data, blade=blade_data)
        else:
            # ElastoDyn main references a TwrFile that we couldn't
            # locate on disk. Continuing blade-only is a useful
            # degraded-mode for blade-focused Campbell sweeps, but
            # silently dropping the tower modes from the result has
            # surprised users. Warn explicitly so the absence is
            # visible — caller can still opt in to blade-only by
            # ignoring the warning.
            warnings.warn(
                f"campbell_sweep: TwrFile referenced by {input_path} "
                f"as {main.twr_file!r} not found at {twr_path}. "
                f"Continuing blade-only — the resulting CampbellResult "
                f"will carry zero tower modes. To suppress this "
                f"warning explicitly, pass a .bmi blade file directly "
                f"instead of the ElastoDyn main.",
                UserWarning,
                stacklevel=2,
            )
    elif suffix == ".bmi":
        bmi = read_bmi(input_path)
        if bmi.beam_type == 1:
            blade = (bmi, None)
        elif bmi.beam_type == 2:
            tower = (bmi, None)
        else:
            raise ValueError(
                f"unsupported beam_type {bmi.beam_type} in {input_path}"
            )
    else:
        raise ValueError(
            f"campbell_sweep input must be .bmi or ElastoDyn .dat; "
            f"got {input_path.suffix!r}"
        )

    if tower_input is not None:
        tmp = _model_pair(tower_input)
        if tmp is not None:                      # a loaded Tower
            if tmp[0] != "tower":
                raise ValueError(
                    "tower_input must be a Tower (beam_type=2) or a "
                    "tower .bmi"
                )
            tower = tmp[1]
        else:
            tpath = pathlib.Path(tower_input)    # type: ignore[arg-type]
            if tpath.suffix.lower() != ".bmi":
                raise ValueError(
                    f"tower_input must be a .bmi file; got "
                    f"{tpath.suffix!r}"
                )
            tower_bmi = read_bmi(tpath)
            if tower_bmi.beam_type != 2:
                raise ValueError(
                    f"tower_input {tpath} has beam_type "
                    f"{tower_bmi.beam_type}, expected 2 (tower)"
                )
            tower = (tower_bmi, None)

    return blade, tower
