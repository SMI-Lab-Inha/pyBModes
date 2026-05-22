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

from pybmodes.io.sec_props import SectionProperties


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
        def compose_node(self, parent, index):
            if self.check_event(AliasEvent):
                event = self.get_event()
                anchor = event.anchor
                if anchor not in self.anchors:
                    raise ComposerError(
                        None, None,
                        f"found undefined alias {anchor!r}",
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
    z_base: float = 0.0         # m, absolute base elevation (reference_axis.z[0])
    z_top: float = 0.0          # m, absolute top elevation (reference_axis.z[-1])


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
        z_base=float(z_vals[0]),
        z_top=float(z_vals[-1]),
    )


@dataclass
class WindIOMonopileTower:
    """A monopile + tower spliced into a single fixed-bottom cantilever.

    The combined cantilever runs from the monopile base (mudline,
    ``z_base``) up through the transition piece (``z_transition``, where
    the monopile meets the tower) to the tower top (``z_top``). The
    section-property table carries both segments' own wall schedules and
    materials, joined at the transition with a near-coincident station
    pair so the FE interpolant captures the cross-section step.
    """

    section_props: SectionProperties
    combined_length: float            # m, z_top - z_base
    el_loc: np.ndarray                # normalised FE node boundaries [0, 1]
    transition_frac: float            # normalised transition-piece location
    z_base: float                     # mudline elevation (m)
    z_transition: float               # transition-piece elevation (m)
    z_top: float                      # tower-top elevation (m)


def read_windio_monopile_tower(
    yaml_path: str | pathlib.Path,
    *,
    component_tower: str = "tower",
    component_monopile: str = "monopile",
    thickness_interp: str = "linear",
    n_nodes: int | None = None,
) -> WindIOMonopileTower:
    """Reduce the ``monopile`` and ``tower`` components and splice them
    into one fixed-bottom cantilever (issue #92).

    Each component is reduced independently through the closed-form
    circular-tube relations (so they keep their own wall schedule and
    steel grade), then concatenated bottom-to-top at the transition
    piece — the elevation where the monopile top meets the tower base.
    The result is the WindIO analog of
    :func:`pybmodes.io.subdyn_reader.to_pybmodes_pile_tower` (the
    ElastoDyn + SubDyn splice).

    Parameters
    ----------
    yaml_path : path to a WindIO ontology file carrying both a
        ``monopile`` and a ``tower`` component.
    component_tower, component_monopile : component names to splice
        (defaults ``"tower"`` / ``"monopile"``).
    thickness_interp : ``"linear"`` or ``"piecewise_constant"`` — passed
        through to each component's reduction (see
        :func:`read_windio_tubular`).
    n_nodes : optional FE-mesh refinement, applied **per segment**: each
        of the monopile and tower is re-gridded onto ``n_nodes`` evenly-
        spaced stations (geometry interpolated, tube properties
        recomputed exactly), mirroring :meth:`Tower.from_windio`'s
        ``n_nodes``. ``None`` keeps each component's native WindIO grid.

    Raises
    ------
    ValueError : when the monopile top and tower base do not meet at a
        common transition-piece elevation (a gap or overlap of more than
        1 mm), since a non-contiguous pair cannot be spliced into one
        beam.
    """
    import numpy as _np

    from pybmodes.io._elastodyn.adapter import _tower_element_boundaries
    from pybmodes.io.geometry import tubular_section_props

    mp = read_windio_tubular(
        yaml_path, component=component_monopile, thickness_interp=thickness_interp,
    )
    tw = read_windio_tubular(
        yaml_path, component=component_tower, thickness_interp=thickness_interp,
    )

    # The monopile top and tower base must describe the same transition
    # piece. WindIO encodes both as absolute reference_axis.z, so they
    # should coincide (e.g. IEA-15: monopile z ends at +15 m, tower z
    # starts at +15 m). A gap/overlap means the segments aren't
    # contiguous and can't be spliced into a single beam.
    gap = abs(mp.z_top - tw.z_base)
    if gap > 1.0e-3:
        raise ValueError(
            f"WindIO monopile top (z={mp.z_top:g} m) and tower base "
            f"(z={tw.z_base:g} m) do not meet at a common transition piece "
            f"(gap {gap:g} m). read_windio_monopile_tower splices contiguous "
            f"segments; check the components' reference_axis.z."
        )

    z_base = mp.z_base
    z_transition = mp.z_top
    z_top = tw.z_top
    combined_length = z_top - z_base
    if combined_length <= 0.0:
        raise ValueError(
            f"WindIO monopile+tower combined length must be positive; got "
            f"{combined_length:g} m (monopile base z={z_base:g}, tower top "
            f"z={z_top:g}). Check the components' reference_axis.z ordering "
            f"(base -> top)."
        )
    transition_frac = (z_transition - z_base) / combined_length

    def _reduce(t: WindIOTubular) -> tuple[np.ndarray, SectionProperties]:
        grid = _np.asarray(t.station_grid, dtype=float)
        od = _np.asarray(t.outer_diameter, dtype=float)
        wt = _np.asarray(t.wall_thickness, dtype=float)
        if n_nodes is not None:
            if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) \
                    or n_nodes < 2:
                raise ValueError(
                    f"n_nodes must be an integer >= 2; got {n_nodes!r}"
                )
            fine = _np.linspace(float(grid[0]), float(grid[-1]), n_nodes)
            od = _np.interp(fine, grid, od)
            wt = _np.interp(fine, grid, wt)
            grid = fine
        sp = tubular_section_props(
            grid, od, wt, E=t.E, rho=t.rho, nu=t.nu,
            outfitting_factor=t.outfitting_factor,
        )
        return grid, sp

    mp_grid, mp_sp = _reduce(mp)
    tw_grid, tw_sp = _reduce(tw)

    # Map each component's own [0, 1] grid onto the combined [0, 1].
    mp_frac = mp_grid * transition_frac
    tw_frac = transition_frac + tw_grid * (1.0 - transition_frac)

    # Near-coincident station pair at the transition so the section-table
    # interpolant resolves the cross-section step (same device as
    # to_pybmodes_pile_tower): nudge the monopile's top station down by a
    # tiny eps; the tower's bottom station sits exactly at the transition.
    eps = 1.0e-9
    mp_frac_sp = mp_frac.copy()
    if mp_frac_sp.size >= 2:
        mp_frac_sp[-1] = max(mp_frac_sp[-1] - eps, mp_frac_sp[-2] + eps / 2)

    span_loc = _np.concatenate([mp_frac_sp, tw_frac])
    zeros = _np.zeros_like(span_loc)
    sp = SectionProperties(
        title="WindIO monopile + tower (combined cantilever)",
        n_secs=int(span_loc.size),
        span_loc=span_loc,
        str_tw=zeros.copy(),
        tw_iner=zeros.copy(),
        mass_den=_np.concatenate([mp_sp.mass_den, tw_sp.mass_den]),
        flp_iner=_np.concatenate([mp_sp.flp_iner, tw_sp.flp_iner]),
        edge_iner=_np.concatenate([mp_sp.edge_iner, tw_sp.edge_iner]),
        flp_stff=_np.concatenate([mp_sp.flp_stff, tw_sp.flp_stff]),
        edge_stff=_np.concatenate([mp_sp.edge_stff, tw_sp.edge_stff]),
        tor_stff=_np.concatenate([mp_sp.tor_stff, tw_sp.tor_stff]),
        axial_stff=_np.concatenate([mp_sp.axial_stff, tw_sp.axial_stff]),
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
    )

    # FE mesh: a clean node sits exactly at the transition (no degenerate
    # eps-length element). Use each segment's (un-gapped) station grid as
    # element boundaries, with the duplicate-station guard, then drop the
    # shared transition node from the tower segment.
    mp_el = _tower_element_boundaries(mp_frac)
    tw_el = _tower_element_boundaries(tw_frac)
    el_loc = _np.concatenate([mp_el, tw_el[1:]])

    return WindIOMonopileTower(
        section_props=sp,
        combined_length=float(combined_length),
        el_loc=el_loc,
        transition_frac=float(transition_frac),
        z_base=float(z_base),
        z_transition=float(z_transition),
        z_top=float(z_top),
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
