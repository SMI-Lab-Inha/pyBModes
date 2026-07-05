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
from typing import TYPE_CHECKING, Any

import numpy as np

from pybmodes.io.sec_props import SectionProperties

if TYPE_CHECKING:
    from pybmodes.io.bmi import TipMassProps

def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x``.

    Implemented directly rather than via ``np.trapezoid`` (NumPy >= 2.0) /
    ``np.trapz`` (the deprecated <2.0 spelling) so the reader works across
    the advertised ``numpy>=1.26`` floor without a version shim.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


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


def _read_water_depth(
    yaml_path: str | pathlib.Path, override: float | None,
) -> float | None:
    """Resolve the water depth (m, positive) used to place the mudline.

    Priority: an explicit ``override`` argument, then the ontology's
    ``environment.water_depth`` block, then ``None`` (unknown, so the
    monopile base is taken as the clamp). An explicit ``override`` must be
    positive and finite or a ``ValueError`` is raised (it is the caller's
    contract). A non-positive or non-finite value read from the ontology
    is treated as absent and returns ``None``.
    """
    if override is not None:
        if isinstance(override, bool):
            raise ValueError(
                f"water_depth must be a number in metres, not a bool; got "
                f"{override!r}"
            )
        wd = float(override)
        if not np.isfinite(wd) or wd <= 0.0:
            raise ValueError(
                f"water_depth must be a positive, finite depth in metres; got "
                f"{override!r}"
            )
        return wd
    yaml = _require_yaml()
    with pathlib.Path(yaml_path).open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))
    env = doc.get("environment") if isinstance(doc, dict) else None
    if isinstance(env, dict) and env.get("water_depth") is not None:
        raw = env["water_depth"]
        if isinstance(raw, bool):
            return None
        wd = float(raw)
        return wd if (np.isfinite(wd) and wd > 0.0) else None
    return None


def _truncate_tubular_base(
    t: WindIOTubular, new_z_base: float, thickness_interp: str,
) -> WindIOTubular:
    """Clamp a monopile at ``new_z_base`` (the mudline), dropping the
    embedded stations below it.

    Every station at or above ``new_z_base`` is kept; when none sits
    exactly there a station is inserted with geometry interpolated at the
    mudline. The surviving grid is renormalised to ``[0, 1]``. A
    wall-schedule step some ontologies place at the mudline (a
    near-coincident station pair) is preserved because both members of the
    pair survive the cut.
    """
    span = t.z_top - t.z_base
    if not (span > 0.0 and t.z_base < new_z_base < t.z_top):
        raise ValueError(
            f"_truncate_tubular_base requires z_base < new_z_base < z_top; "
            f"got z_base={t.z_base:g}, new_z_base={new_z_base:g}, "
            f"z_top={t.z_top:g}."
        )
    cut = (new_z_base - t.z_base) / span
    ftol = 1.0e-6 / span
    grid = np.asarray(t.station_grid, dtype=float)
    od = np.asarray(t.outer_diameter, dtype=float)
    wt = np.asarray(t.wall_thickness, dtype=float)

    keep = grid >= cut - ftol
    new_grid = grid[keep]
    new_od = od[keep]
    new_wt = wt[keep]
    if new_grid.size == 0 or new_grid[0] > cut + ftol:
        od_cut = float(np.interp(cut, grid, od))
        wt_cut = float(_interp(grid, wt, np.array([cut]), thickness_interp)[0])
        new_grid = np.concatenate([[cut], new_grid])
        new_od = np.concatenate([[od_cut], new_od])
        new_wt = np.concatenate([[wt_cut], new_wt])

    new_grid = (new_grid - cut) / (1.0 - cut)
    new_grid[0] = 0.0
    return WindIOTubular(
        station_grid=new_grid,
        outer_diameter=new_od,
        wall_thickness=new_wt,
        flexible_length=float(t.z_top - new_z_base),
        E=t.E, rho=t.rho, nu=t.nu,
        outfitting_factor=t.outfitting_factor,
        z_base=float(new_z_base),
        z_top=t.z_top,
    )


def read_windio_monopile_tower(
    yaml_path: str | pathlib.Path,
    *,
    component_tower: str = "tower",
    component_monopile: str = "monopile",
    thickness_interp: str = "linear",
    n_nodes: int | None = None,
    water_depth: float | None = None,
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
    water_depth : water depth in metres (positive). Places the mudline at
        ``z = -water_depth`` and clamps the combined cantilever there,
        dropping any embedded monopile length below the seabed (issue
        #121). Needed when the monopile ``reference_axis.z`` includes the
        embedded pile (e.g. IEA-15: axis -75 -> +15 with the mudline at
        -30). Defaults to the ontology's ``environment.water_depth`` when
        present; ``None`` with no ontology value keeps the monopile base
        as the clamp (correct only when the axis already starts at the
        mudline).

    Raises
    ------
    ValueError : when the monopile top and tower base do not meet at a
        common transition-piece elevation (a gap or overlap of more than
        1 mm), since a non-contiguous pair cannot be spliced into one
        beam; when an explicit ``water_depth`` is not positive and finite;
        or when the resolved mudline falls at or above the transition
        piece, or below the monopile base (the pile does not reach the
        seabed).
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

    # Rigid fixed-base monopiles are clamped at the mudline, not at the
    # embedded pile tip. When the monopile reference_axis extends below the
    # seabed (e.g. IEA-15: axis -75 -> +15 with the mudline at -30), the
    # embedded length must be dropped so it is not modelled as a free
    # cantilever (issue #121). The seabed is -water_depth; water_depth comes
    # from the caller or the ontology's environment block. With no water
    # depth available the monopile base is taken as the clamp (the previous
    # behaviour, correct when the axis already begins at the mudline).
    wd = _read_water_depth(yaml_path, water_depth)
    if wd is not None:
        mudline_z = -wd
        if mudline_z >= mp.z_top:
            raise ValueError(
                f"water_depth={wd:g} m places the mudline (z={mudline_z:g} m) "
                f"at or above the monopile transition piece (z={mp.z_top:g} m); "
                f"the mudline must sit below the transition. Check water_depth "
                f"and the monopile reference_axis.z."
            )
        if mudline_z < mp.z_base - 1.0e-6:
            raise ValueError(
                f"water_depth={wd:g} m places the mudline (z={mudline_z:g} m) "
                f"below the monopile base (z={mp.z_base:g} m); the monopile "
                f"does not reach the seabed. Check water_depth and the monopile "
                f"reference_axis.z."
            )
        if mudline_z > mp.z_base + 1.0e-6:
            mp = _truncate_tubular_base(mp, mudline_z, thickness_interp)
        # else: the mudline coincides with the monopile base (within tol),
        # so the base is already the clamp and no truncation is needed.

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


# ---------------------------------------------------------------------------
# RNA (rotor-nacelle assembly) tower-top lumped mass (issue #82)
# ---------------------------------------------------------------------------


def _require_mapping(node: object, path: str) -> dict:
    """Return ``node`` as a mapping or raise a KeyError naming ``path``."""
    if not isinstance(node, dict):
        raise KeyError(
            f"WindIO ontology has no '{path}' block, which is required to "
            f"lump the RNA (elastic_properties_mb schema). Ontologies without "
            f"it (e.g. IEA-15) cannot supply the RNA mass; pass tip_mass "
            f"explicitly instead."
        )
    return node


def _require_key(node: dict, key: str, path: str) -> Any:
    """Return ``node[key]`` or raise a KeyError naming ``path.key``."""
    if key not in node:
        raise KeyError(
            f"WindIO ontology is missing '{path}.{key}', required to lump "
            f"the RNA."
        )
    return node[key]


def _finite_float(value: Any, what: str) -> float:
    """Coerce ``value`` to a finite float (bool rejected), else raise."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, not a bool; got {value!r}.")
    f = float(value)
    if not np.isfinite(f):
        raise ValueError(f"{what} must be finite; got {value!r}.")
    return f


def _positive_mass(value: Any, what: str) -> float:
    """Coerce ``value`` to a positive, finite mass (kg), else raise."""
    m = _finite_float(value, what)
    if m <= 0.0:
        raise ValueError(
            f"{what} must be a positive, finite mass in kg; got {value!r}."
        )
    return m


def _finite_vector(value: Any, n: int, what: str) -> np.ndarray:
    """Coerce ``value`` to a finite length-``n`` 1-D array, else raise.

    Rejects bool entries (which ``np.asarray(..., dtype=float)`` would
    otherwise coerce to 0.0 / 1.0) so a stray YAML ``true`` / ``false`` in
    a vector field fails like the scalar fields rather than corrupting the
    CM / inertia.
    """
    if np.asarray(value).dtype == bool or (
        isinstance(value, (list, tuple))
        and any(isinstance(x, bool) for x in value)
    ):
        raise ValueError(f"{what} entries must be numbers, not bools; got {value!r}.")
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{what} must be a {n}-vector; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must be finite; got {np.asarray(value).tolist()}.")
    return arr


def _sym_tensor_from_6vec(vec: Any, what: str) -> np.ndarray:
    """Build a symmetric 3x3 inertia tensor from a WindIO inertia vector.

    Accepts the full ``[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`` 6-vector or the
    diagonal ``[Ixx, Iyy, Izz]`` triplet (products of inertia taken as
    zero) some ontologies use. The returned tensor is
    ``[[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]``; every entry
    must be finite.
    """
    size = np.asarray(vec).size
    if size not in (3, 6):
        raise ValueError(
            f"{what} inertia must be a 3-vector [Ixx, Iyy, Izz] (diagonal) or "
            f"a 6-vector [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]; got {size} value(s)."
        )
    arr = _finite_vector(vec, int(size), f"{what} inertia")
    if size == 3:
        ixx, iyy, izz = (float(v) for v in arr)
        ixy = ixz = iyz = 0.0
    else:
        ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in arr)
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float
    )


def _check_monotone_grid(g: np.ndarray, label: str) -> None:
    """Require a finite, strictly increasing 1-D grid of >= 2 stations.

    WindIO ``{grid, values}`` curves are piecewise-linear on an ascending
    grid; a non-finite or non-monotone grid breaks ``np.interp`` /
    ``np.union1d`` silently, so reject it up front.
    """
    if g.size < 2 or not np.all(np.isfinite(g)) or np.any(np.diff(g) <= 0.0):
        raise ValueError(
            f"{label} must be a finite, strictly increasing grid of at least "
            f"2 stations."
        )


def _blade_reference_axis(six: dict, comp: dict, blade_component: str) -> dict:
    """Resolve the blade reference axis across WindIO layouts.

    The newer ``six_x_six`` layout (IEA-22) nests ``reference_axis`` inside
    ``six_x_six``; older layouts put it on the blade component /
    ``outer_shape`` / ``structure`` — the same places the blade reader
    (:func:`pybmodes.io.windio_blade._reference_axis`) looks. Prefer the
    ``six_x_six`` one when present, then fall back to the component so
    ``lumped_rna_cal`` works for every ontology the blade reader supports.
    """
    outer = comp.get("outer_shape") or comp.get("outer_shape_bem") or {}
    struct = comp.get("structure") or comp.get("internal_structure_2d_fem") or {}
    for holder in (six, comp, outer, struct):
        if isinstance(holder, dict):
            ra = holder.get("reference_axis")
            if isinstance(ra, dict) and "z" in ra:
                return ra
    raise KeyError(
        f"components.{blade_component} has no reference_axis with a z curve "
        f"(needed to integrate the blade span mass); looked in six_x_six, "
        f"the component, outer_shape and structure."
    )


def _blade_span_mass_inertia(
    six: dict, ref: dict, blade_component: str, hub_r: float, cone_cos: float,
) -> tuple[float, float]:
    """Integrate a WindIO blade's span into ``(mass, polar_second_moment)``.

    ``six`` is
    ``components.<blade>.elastic_properties_mb.six_x_six`` and ``ref`` is
    the resolved blade ``reference_axis`` (see
    :func:`_blade_reference_axis`). The per-station mass/length is the
    first entry (``M[0, 0]``) of each upper-triangular 21-vector in
    ``inertia_matrix.values``, integrated over the arc length of the
    reference axis. Only the ``z`` curve is required; ``x`` / ``y``
    (prebend / sweep) are optional and default to zero (a straight span).

    Returns the single-blade mass and the single-blade polar second moment
    about the rotor axis, ``∫ (dm/ds) · r(s)² ds`` with the radial distance
    ``r(s) = hub_r + z(s)·cone_cos``, so the caller can build the rotor's
    diametral inertia from the spanwise mass distribution (issue #130).
    """
    im = _require_mapping(six.get("inertia_matrix"), "blade six_x_six.inertia_matrix")
    grid = np.asarray(_require_key(im, "grid", "blade inertia_matrix"), dtype=float)
    rows = _require_key(im, "values", "blade inertia_matrix")
    try:
        mass_per_len = np.array([float(r[0]) for r in rows], dtype=float)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"components.{blade_component} inertia_matrix.values must be a "
            f"list of numeric rows (upper-triangular 6x6)."
        ) from exc
    if grid.size != mass_per_len.size:
        raise ValueError(
            f"components.{blade_component} inertia_matrix grid ({grid.size}) "
            f"and values ({mass_per_len.size}) must be equal-length arrays."
        )
    _check_monotone_grid(grid, f"components.{blade_component} inertia_matrix.grid")
    if not np.all(np.isfinite(mass_per_len)) or np.any(mass_per_len < 0.0):
        raise ValueError(
            f"components.{blade_component} blade mass/length has non-finite "
            f"or negative entries."
        )
    # Collect the present reference-axis curves (z required; x / y prebend
    # / sweep optional -> a straight span in that axis).
    axis_curves = {}
    for axis in ("x", "y", "z"):
        a = ref.get(axis)
        if not isinstance(a, dict):
            if axis == "z":
                raise ValueError(
                    f"blade reference_axis.z is required to integrate the "
                    f"span mass of components.{blade_component}."
                )
            continue
        ag = np.asarray(_require_key(a, "grid", f"reference_axis.{axis}"), dtype=float)
        av = np.asarray(_require_key(a, "values", f"reference_axis.{axis}"), dtype=float)
        if ag.size != av.size:
            raise ValueError(
                f"blade reference_axis.{axis} grid ({ag.size}) and values "
                f"({av.size}) must be equal-length arrays."
            )
        _check_monotone_grid(ag, f"blade reference_axis.{axis}.grid")
        if not np.all(np.isfinite(av)):
            raise ValueError(
                f"blade reference_axis.{axis}.values has non-finite entries."
            )
        axis_curves[axis] = (ag, av)

    # Integrate on the union of the inertia grid and every reference-axis
    # knot, restricted to the inertia grid's span (where mass/length is
    # defined). WindIO curves are piecewise-linear on their own grids, so
    # sampling onto the union keeps every segment and makes the trapezoidal
    # integral exact — sampling onto the inertia grid alone would chord over
    # intermediate prebend/sweep knots and undercount the mass.
    merged = grid
    for ag, _ in axis_curves.values():
        merged = np.union1d(merged, ag)
    merged = merged[(merged >= grid[0]) & (merged <= grid[-1])]

    mpl = np.interp(merged, grid, mass_per_len)
    coords = []
    for axis in ("x", "y", "z"):
        if axis in axis_curves:
            ag, av = axis_curves[axis]
            coords.append(np.interp(merged, ag, av))
        else:
            coords.append(np.zeros_like(merged))
    xyz = np.vstack(coords).T
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    mass = _trapezoid(mpl, s)
    # Radial distance from the rotor axis: the hub radius plus the spanwise
    # position projected onto the rotor plane by the cone angle.
    radial = hub_r + coords[2] * cone_cos
    polar_second_moment = _trapezoid(mpl * radial * radial, s)
    return mass, polar_second_moment


def read_windio_rna(
    yaml_path: str | pathlib.Path,
    *,
    component_hub: str = "hub",
    component_nacelle: str = "nacelle",
    component_blade: str = "blade",
) -> TipMassProps:
    """Lump the WindIO rotor-nacelle assembly into a tower-top ``TipMassProps``.

    Reads the ``elastic_properties_mb`` blocks of the ``hub`` and
    ``nacelle.drivetrain`` components plus the integrated blade span mass,
    and assembles nacelle + hub + blades into a single rigid-body mass and
    inertia at the tower top (issue #82). This mirrors the ElastoDyn
    assembler
    :func:`pybmodes.io._elastodyn.adapter._tower_top_assembly_mass`, so the
    yaml path and the deck path share one tower-top convention.

    Requires an ontology that carries the hub and nacelle
    ``elastic_properties_mb`` lumped-mass blocks (IEA-22-class). Ontologies
    that omit them (IEA-10 carries only the blade block; IEA-15 neither)
    raise a ``KeyError`` naming the missing block — the RNA mass cannot be
    synthesised without them, so pass ``tip_mass`` explicitly in that case.

    Frame: tower-top ``x = downwind, y = lateral, z = up``. The nacelle
    inertia is the ontology's own tensor about the nacelle CM; the hub is
    its tensor about the hub centre; the rotor is a rigid body at the apex
    (total blade mass plus the diametral inertia from the spanwise blade
    mass). The result is expressed at the tower top with ``cm_offset = 0``
    and the vertical CM lever in ``cm_axial`` so the FEM nondimensionaliser
    does not re-apply parallel-axis.

    ``cm_offset`` is intentionally ``0``, not the horizontal CM / overhang.
    The auto-RNA is consumed only by the clamped-base (``hub_conn = 1``)
    path, where :func:`pybmodes.fem.nondim.nondim_tip_mass` ignores
    ``cm_offset`` entirely (it carries ``cm_axial`` as the lever and zeroes
    the axial term). The horizontal overhang is instead folded into the
    returned inertia tensor: the parallel-axis shift to the tower top puts
    the ``m·overhang²`` and ``m·height²`` terms into ``ixx`` / ``iyy`` /
    ``izz``. Setting ``cm_offset = cm[0]`` on top of the already-shifted
    tensor would double-count the overhang.

    Blade inertia (issue #130): the rotor carries the diametral inertia
    ``N_bl · ∫ (dm/ds) · r² ds`` from the blade mass spread along the span
    (``r = hub_radius + span·cos(cone)``), as ``diag([I_polar, I_polar/2,
    I_polar/2])`` about the hub (the same perpendicular-axis form as the
    hub tensor). Only each blade's own *sectional* spin inertia is left out
    (its parallel-axis / span contribution is captured). This intentionally
    goes beyond the ElastoDyn deck path's bare point-mass lumping, which the
    WindIO ontology's per-station blade mass makes possible.

    Requires the optional ``[windio]`` extra (PyYAML).
    """
    from pybmodes.io.bmi import TipMassProps

    yaml = _require_yaml()
    yaml_path = pathlib.Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))

    comps = _require_mapping(
        doc.get("components") if isinstance(doc, dict) else None, "components"
    )
    assembly = _require_mapping(
        doc.get("assembly") if isinstance(doc, dict) else None, "assembly"
    )

    # --- Nacelle: components.<nacelle>[.drivetrain] ---
    # The nacelle lumped-mass block and drivetrain geometry live under
    # ``nacelle.drivetrain`` (IEA-22) or directly on ``nacelle`` (the WISDEM
    # documented layout, where ``elastic_properties_mb`` is a sibling of
    # ``drivetrain``). Resolve from either, preferring ``drivetrain``.
    nacelle = _require_mapping(
        comps.get(component_nacelle), f"components.{component_nacelle}"
    )
    drivetrain = nacelle.get("drivetrain")
    _dt = drivetrain if isinstance(drivetrain, dict) else {}

    def _nac_get(key: str) -> Any:
        return _dt[key] if key in _dt else nacelle.get(key)

    nac_ep = _require_mapping(
        _nac_get("elastic_properties_mb"),
        f"components.{component_nacelle}[.drivetrain].elastic_properties_mb",
    )
    # ``system_mass`` is the complete nacelle mass; the sibling
    # ``yaw_mass`` is a sub-component breakdown, not an additive term
    # (adding it double-counts and overshoots the ElastoDyn NacMass).
    m_nac = _positive_mass(
        _require_key(nac_ep, "system_mass", "nacelle elastic_properties_mb"),
        "nacelle system_mass",
    )
    r_nac = _finite_vector(
        _require_key(nac_ep, "system_center_mass", "nacelle elastic_properties_mb"),
        3, "nacelle system_center_mass",
    )
    i_nac = _sym_tensor_from_6vec(
        _require_key(nac_ep, "system_inertia", "nacelle elastic_properties_mb"),
        "nacelle",
    )

    # --- Drivetrain geometry (rotor apex vs tower top); uptilt is radians ---
    def _nac_geom(key: str) -> float:
        val = _nac_get(key)
        if val is None:
            raise KeyError(
                f"components.{component_nacelle}[.drivetrain].{key} is "
                f"required to place the rotor apex for the RNA lump."
            )
        return _finite_float(val, f"nacelle.drivetrain.{key}")

    overhang = _nac_geom("overhang")
    uptilt = _nac_geom("uptilt")
    dist_tt_hub = _nac_geom("distance_tt_hub")

    # --- Hub: components.<hub>.elastic_properties_mb ---
    hub = _require_mapping(comps.get(component_hub), f"components.{component_hub}")
    hub_ep = _require_mapping(
        hub.get("elastic_properties_mb"),
        f"components.{component_hub}.elastic_properties_mb",
    )
    m_hub = _positive_mass(
        _require_key(hub_ep, "system_mass", "hub elastic_properties_mb"),
        "hub system_mass",
    )
    i_hub = _sym_tensor_from_6vec(
        _require_key(hub_ep, "system_inertia", "hub elastic_properties_mb"),
        "hub",
    )
    # Rotor geometry for the blade-span inertia (issue #130): the hub radius
    # offsets the blade root from the rotor axis, and the cone angle
    # projects the coned span onto the rotor plane. Both optional (default
    # no hub offset / no cone).
    hub_diam = hub.get("diameter")
    hub_r = (
        0.5 * _finite_float(hub_diam, "hub diameter") if hub_diam is not None else 0.0
    )
    if hub_r < 0.0:
        raise ValueError(f"hub diameter must be non-negative; got {hub_diam!r}.")
    cone = hub.get("cone_angle")
    cone_cos = (
        float(np.cos(_finite_float(cone, "hub cone_angle")))
        if cone is not None else 1.0
    )

    # --- Blades: assembly.number_of_blades x integrated span mass ---
    n_blades = _require_key(assembly, "number_of_blades", "assembly")
    if isinstance(n_blades, bool) or not isinstance(n_blades, int) or n_blades < 0:
        raise ValueError(
            f"assembly.number_of_blades must be a non-negative integer; got "
            f"{n_blades!r}."
        )
    orientation = str(assembly.get("rotor_orientation", "upwind")).strip().lower()
    if orientation not in {"upwind", "downwind"}:
        raise ValueError(
            f"assembly.rotor_orientation must be 'Upwind' or 'Downwind'; got "
            f"{assembly.get('rotor_orientation')!r}."
        )
    m_blade_each = 0.0
    i_polar_each = 0.0
    if n_blades > 0:
        blade = _require_mapping(
            comps.get(component_blade), f"components.{component_blade}"
        )
        blade_ep = _require_mapping(
            blade.get("elastic_properties_mb"),
            f"components.{component_blade}.elastic_properties_mb",
        )
        six = _require_mapping(
            blade_ep.get("six_x_six"),
            f"components.{component_blade}.elastic_properties_mb.six_x_six",
        )
        ref = _blade_reference_axis(six, blade, component_blade)
        m_blade_each, i_polar_each = _blade_span_mass_inertia(
            six, ref, component_blade, hub_r, cone_cos,
        )
    m_blades = n_blades * m_blade_each

    # Rotor inertia from the spanwise blade mass (issue #130). For N >= 3
    # symmetric blades the rotor tensor about the hub is
    # diag([I_polar, I_polar/2, I_polar/2]) in the shaft frame (polar about
    # the shaft, half on each transverse axis by the perpendicular-axis
    # theorem), the same form as the hub. Lumping the blades as a bare point
    # mass at the apex would drop this, which is the dominant part of the
    # rotor's contribution to the tower-top rotary inertia.
    i_polar_rotor = n_blades * i_polar_each
    i_blades = np.diag(
        [i_polar_rotor, 0.5 * i_polar_rotor, 0.5 * i_polar_rotor]
    )

    # Rotor apex relative to the tower top. An upwind rotor sits at negative
    # x (downwind-positive frame); the vertical hub position is
    # distance_tt_hub directly (= Twr2Shft + overhang*sin(uptilt)).
    sign_x = -1.0 if orientation == "upwind" else 1.0
    apex = np.array(
        [sign_x * abs(overhang) * float(np.cos(uptilt)), 0.0, dist_tt_hub],
        dtype=float,
    )

    # The hub and rotor inertia tensors are in the shaft frame; rotate them
    # into the tower-top frame by the shaft tilt before they are summed as
    # tower-top tensors (a tilted rotor otherwise loses the tensor izx
    # product and misallocates the large polar term between ixx and izz).
    # The shaft unit axis is ``[sign_x*cos(uptilt), 0, sin(uptilt)]``; R maps
    # the shaft x-axis onto it and fixes the lateral y-axis. The nacelle
    # tensor is already in the tower frame and is left alone.
    c_t, s_t = float(np.cos(uptilt)), float(np.sin(uptilt))
    r_tilt = np.array(
        [[sign_x * c_t, 0.0, -s_t], [0.0, 1.0, 0.0], [s_t, 0.0, sign_x * c_t]]
    )
    i_hub = r_tilt @ i_hub @ r_tilt.T
    i_blades = r_tilt @ i_blades @ r_tilt.T

    bodies = [
        (m_nac, r_nac, i_nac),
        (m_hub, apex, i_hub),
        (m_blades, apex.copy(), i_blades),
    ]

    m_total = float(sum(m for m, _, _ in bodies))
    if m_total <= 0.0:
        raise ValueError(
            "WindIO RNA assembled to zero total mass; check the hub / nacelle "
            "system_mass and assembly.number_of_blades."
        )
    cm: np.ndarray = (
        sum((m * r for m, r, _ in bodies), start=np.zeros(3)) / m_total
    )
    eye = np.eye(3)
    i_tt = np.zeros((3, 3))
    for m, r, i_body in bodies:
        rsq = float(r @ r)
        i_tt = i_tt + i_body + m * (rsq * eye - np.outer(r, r))

    return TipMassProps(
        mass=m_total,
        cm_offset=0.0,
        cm_axial=float(cm[2]),
        ixx=float(i_tt[0, 0]),
        iyy=float(i_tt[1, 1]),
        izz=float(i_tt[2, 2]),
        ixy=float(i_tt[0, 1]),
        izx=float(i_tt[2, 0]),
        iyz=float(i_tt[1, 2]),
    )
