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

"""Read the structural subset of a WindIO ontology ``.yaml`` for a
tubular tower or monopile (issue #35).

WindIO describes a tower / monopile as a circular tube via:

* ``components.<component>.outer_shape.outer_diameter.{grid, values}``
* ``components.<component>.structure.layers[]`` — each
  ``{material, thickness.{grid, values}}`` (summed for the wall)
* ``components.<component>.structure.outfitting_factor`` — the
  non-structural mass multiplier (internals / flanges / paint)
* ``components.<component>.reference_axis.z.{grid, values}`` — physical
  station heights (m); the span = ``|z[-1] - z[0]|``
* top-level ``materials[]`` — the layer's material ``{E, rho, nu}``

That is exactly what :func:`pybmodes.io.geometry.tubular_section_props`
needs, so :meth:`pybmodes.models.Tower.from_windio` is a thin wrapper.

This module is the *tubular* (tower / monopile) reader only. A WindIO
blade is a composite layup whose beam properties need a PreComp-class
thin-wall cross-section reduction — that lives in
:mod:`pybmodes.io.windio_blade` (:func:`~pybmodes.io.windio_blade.
read_windio_blade` / :meth:`pybmodes.models.RotatingBlade.from_windio`),
not here.

Requires the optional ``[windio]`` extra (PyYAML); the runtime core
stays ``numpy + scipy`` only, mirroring the ``[plots]`` /
``[notebook]`` extras.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np


def _require_yaml():
    """Import PyYAML or raise the documented friendly error.

    Mirrors ``pybmodes.plots._require_matplotlib`` — the YAML
    dependency is opt-in so a core install is numpy+scipy only.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise ModuleNotFoundError(
            "Reading WindIO .yaml needs PyYAML, which ships in the "
            "optional 'windio' extra. Install it with:\n"
            "    pip install 'pybmodes[windio]'\n"
            "(the pyBmodes runtime core is intentionally numpy+scipy "
            "only; YAML is opt-in)."
        ) from exc
    return yaml


_LOADER_CACHE: dict = {}


def _dup_anchor_loader(yaml):
    """A SafeLoader that tolerates *duplicate* YAML anchors (last wins).

    WindIO ontology files emitted by the WISDEM toolchain (ruamel-based)
    routinely redefine an anchor — e.g. IEA-10's ``materials`` block
    reuses ``&id004``. Strict PyYAML rejects that with ``ComposerError``;
    ruamel and the YAML-1.2 alias-resolution model accept it (an alias
    binds to the most recent *prior* definition). We subclass
    ``SafeLoader`` and drop only the duplicate-anchor guard from
    ``compose_node``, keeping the genuine *undefined-alias* error and
    everything else byte-for-byte. Cached per ``yaml`` module object.
    """
    cached = _LOADER_CACHE.get("loader")
    if cached is not None:
        return cached

    from yaml.composer import ComposerError  # type: ignore[import-untyped]
    from yaml.events import (  # type: ignore[import-untyped]
        AliasEvent,
        MappingStartEvent,
        ScalarEvent,
        SequenceStartEvent,
    )

    class _DupAnchorSafeLoader(yaml.SafeLoader):
        def compose_node(self, parent, index):  # noqa: D102 - PyYAML override
            if self.check_event(AliasEvent):
                event = self.get_event()
                anchor = event.anchor
                if anchor not in self.anchors:
                    raise ComposerError(
                        None, None,
                        "found undefined alias %r" % anchor,
                        event.start_mark,
                    )
                return self.anchors[anchor]
            event = self.peek_event()
            anchor = event.anchor
            # Duplicate-anchor guard intentionally omitted: compose_*_node
            # overwrites self.anchors[anchor], so a later definition wins
            # and prior aliases keep the value current at their position
            # (ruamel / YAML-1.2 semantics — what WindIO files assume).
            self.descend_resolver(parent, index)
            if self.check_event(ScalarEvent):
                node = self.compose_scalar_node(anchor)
            elif self.check_event(SequenceStartEvent):
                node = self.compose_sequence_node(anchor)
            elif self.check_event(MappingStartEvent):
                node = self.compose_mapping_node(anchor)
            self.ascend_resolver()
            return node

    _LOADER_CACHE["loader"] = _DupAnchorSafeLoader
    return _DupAnchorSafeLoader


@dataclass
class WindIOTubular:
    """Geometry + material extracted from a WindIO tower / monopile."""

    station_grid: np.ndarray   # normalised [0, 1], base -> top
    outer_diameter: np.ndarray  # m, per station
    wall_thickness: np.ndarray  # m, per station (summed layers)
    flexible_length: float      # m, |z_top - z_base|
    E: float
    rho: float
    nu: float
    outfitting_factor: float


def _interp(grid: np.ndarray, values: np.ndarray, at: np.ndarray,
            how: str) -> np.ndarray:
    """Interpolate a WindIO ``(grid, values)`` curve onto ``at``.

    ``"linear"`` — WindIO-native piecewise-linear (``np.interp``).
    ``"piecewise_constant"`` — WISDEM-style: each station takes the
    value of the nearest grid point at or below it (the last segment
    governs). The two differ measurably for the 2nd tower-bending
    mode; the caller chooses.
    """
    if how == "linear":
        return np.interp(at, grid, values)
    if how == "piecewise_constant":
        idx = np.searchsorted(grid, at, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return np.asarray(values)[idx]
    raise ValueError(
        f"thickness_interp must be 'linear' or 'piecewise_constant'; "
        f"got {how!r}"
    )


def _shape_and_structure(comp: dict, component: str) -> tuple[dict, dict]:
    """Return ``(outer_shape_block, structure_block)`` across WindIO dialects.

    WindIO has shipped under two key spellings for the same content:

    * **modern** (IEA-15 WT_Ontology, every WISDEM example yaml):
      ``outer_shape`` + ``structure``
    * **older** (IEA-3.4 / IEA-10 / IEA-22 RWT ontology yamls):
      ``outer_shape_bem`` + ``internal_structure_2d_fem``

    The payload (``outer_diameter``, ``layers``, ``outfitting_factor``,
    ``reference_axis``) is identical; only the container key differs.
    """
    shape = comp.get("outer_shape", comp.get("outer_shape_bem"))
    structure = comp.get("structure", comp.get("internal_structure_2d_fem"))
    if shape is None or structure is None:
        raise KeyError(
            f"components.{component} has neither the modern "
            f"'outer_shape'/'structure' nor the older "
            f"'outer_shape_bem'/'internal_structure_2d_fem' blocks; "
            f"this does not look like a WindIO tower/monopile component."
        )
    return shape, structure


def _reference_axis_z(comp: dict, shape: dict, structure: dict,
                      component: str) -> dict:
    """Resolve the ``reference_axis.z`` curve across WindIO dialects.

    Modern files carry ``reference_axis`` at the component level; older
    ones nest it inside ``outer_shape_bem`` (and alias it into
    ``internal_structure_2d_fem`` via a YAML anchor, which PyYAML has
    already expanded by the time we get here). Accept any of the three.
    """
    for holder in (comp, shape, structure):
        ref = holder.get("reference_axis")
        if ref is not None and "z" in ref:
            return ref["z"]
    raise KeyError(
        f"components.{component} has no reference_axis.z (needed for the "
        f"physical span); looked at the component, the outer-shape block, "
        f"and the structure block."
    )


def read_windio_tubular(
    yaml_path: str | pathlib.Path,
    *,
    component: str = "tower",
    thickness_interp: str = "linear",
) -> WindIOTubular:
    """Parse the structural subset of ``component`` from a WindIO file.

    Handles both WindIO key dialects (modern ``outer_shape``/``structure``
    and older ``outer_shape_bem``/``internal_structure_2d_fem``); see
    :func:`_shape_and_structure`.
    """
    yaml = _require_yaml()
    yaml_path = pathlib.Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))

    try:
        comp = doc["components"][component]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"WindIO file {yaml_path} has no components.{component!r} "
            f"block (expected 'tower' or 'monopile')."
        ) from exc

    shape, structure = _shape_and_structure(comp, component)

    od = shape["outer_diameter"]
    grid = np.asarray(od["grid"], dtype=float)
    outer_d = np.asarray(od["values"], dtype=float)

    outfitting = float(structure.get("outfitting_factor", 1.0))
    layers = structure["layers"]
    if not layers:
        raise ValueError(f"components.{component}.structure.layers is empty")

    # Sum every layer's thickness onto the outer-diameter grid; require
    # one consistent material (tower / monopile are single-material
    # steel tubes — a multi-material wall would need a composite
    # reduction, which is out of scope).
    mat_names = {ly["material"] for ly in layers}
    if len(mat_names) != 1:
        raise ValueError(
            f"components.{component} has layers of multiple materials "
            f"{sorted(mat_names)}; only a single-material tubular wall "
            f"is supported (a layered composite needs a PreComp/BECAS "
            f"cross-section reduction, out of scope)."
        )
    wall_t = np.zeros_like(grid)
    for ly in layers:
        th = ly["thickness"]
        wall_t = wall_t + _interp(
            np.asarray(th["grid"], dtype=float),
            np.asarray(th["values"], dtype=float),
            grid, thickness_interp,
        )

    mat_name = next(iter(mat_names))
    mat = _find_material(doc, mat_name, yaml_path)
    E = float(mat["E"])
    rho_val = mat.get("rho", mat.get("density"))
    if rho_val is None:  # pragma: no cover - guarded in _find_material
        raise KeyError(
            f"WindIO material {mat_name!r} has neither 'rho' nor 'density'."
        )
    rho = float(rho_val)
    nu = float(mat.get("nu", 0.3))

    z = _reference_axis_z(comp, shape, structure, component)
    z_vals = np.asarray(z["values"], dtype=float)
    flexible_length = float(abs(z_vals[-1] - z_vals[0]))

    return WindIOTubular(
        station_grid=grid,
        outer_diameter=outer_d,
        wall_thickness=wall_t,
        flexible_length=flexible_length,
        E=E, rho=rho, nu=nu,
        outfitting_factor=outfitting,
    )


def _find_material(doc: dict, name: str, yaml_path: pathlib.Path) -> dict:
    for mat in doc.get("materials", []):
        if mat.get("name") == name:
            if "E" not in mat or ("rho" not in mat and "density" not in mat):
                raise KeyError(
                    f"WindIO material {name!r} in {yaml_path} is missing "
                    f"'E' and/or 'rho' — an isotropic E + rho (+ optional "
                    f"nu) is required for a tubular section."
                )
            # Older RWT ontology files list orthotropic composites
            # (triax/biax: E/G/nu are 3-vectors) alongside the
            # isotropic tower 'steel'. A tube needs a single isotropic
            # modulus; a layered composite would need a PreComp/BECAS
            # reduction (out of scope, same stance as the multi-material
            # guard above).
            if isinstance(mat["E"], (list, tuple)):
                raise ValueError(
                    f"WindIO material {name!r} in {yaml_path} is "
                    f"orthotropic (E is a {len(mat['E'])}-vector); only an "
                    f"isotropic (scalar E, rho, nu) tubular wall material "
                    f"is supported — a composite layup needs a PreComp/"
                    f"BECAS cross-section reduction, out of scope."
                )
            return mat
    raise KeyError(
        f"WindIO material {name!r} (referenced by a "
        f"{yaml_path.name} structural layer) not found in the top-level "
        f"'materials' list."
    )
