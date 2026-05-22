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

"""Resolve WindIO blade web / layer ``nd_arc`` bands onto the span grid
(issue #35).

A WindIO blade locates every web and every shell/web layer by a
*normalised arc* band ``[start_nd_arc, end_nd_arc]`` along the airfoil
perimeter (the :mod:`pybmodes.io._precomp.profile` ``nd_arc`` spine),
varying along span. The band is given in one of two dialects:

* **older** (IEA-3.4 / 10 / 22 ``internal_structure_2d_fem``): each
  web / layer carries explicit ``start_nd_arc``/``end_nd_arc``
  ``{grid, values}`` curves directly.
* **modern** (IEA-15 ``WT_Ontology``, every WISDEM example incl. the
  floating ones): each web / layer's ``start_nd_arc``/``end_nd_arc``
  is ``{anchor: {name, handle}}`` — an *indirection* into the
  blade-level ``structure.anchors[]`` registry. In the published
  IEA / WISDEM RWT files every registry entry ships the **fully
  resolved** explicit ``{grid, values}`` for both handles (verified:
  IEA-15 WT_Ontology + VolturnUS-S FOWT carry 50-point resolved
  curves for layers *and* webs, beside the parametric
  ``plane_intersection`` recipe).

So resolution is a **dereference + dialect-normalisation**: follow the
``anchor`` to its registry entry, take the explicit ``{grid, values}``
for the requested ``handle``, and interpolate onto the blade span
grid. The purely-parametric geometric fallback (``plane_intersection``
offset/rotation, ``midpoint_nd_arc`` + ``width`` — WISDEM's
``build_layer`` 1–6 in ``gc_WT_DataStruc.py``) is only needed for
hand-authored yamls that omit the resolved values; that case raises a
clear, actionable error rather than silently guessing.

Interpolation mirrors WISDEM (``PchipInterpolator(extrapolate=False)``
then ``nan_to_num`` → 0 outside the defined grid) so the bands match
the WISDEM / BeamDyn validation oracle with minimal numerical noise.
Clean-room reimplementation; the WISDEM source is studied as the
reference, not vendored (independence stance; see ``VALIDATION.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class ResolvedWeb:
    """A shear web's perimeter attachment band along span."""

    name: str
    start_nd: np.ndarray   # (n_span,) nd_arc of the suction-side foot
    end_nd: np.ndarray     # (n_span,) nd_arc of the pressure-side foot


@dataclass
class ResolvedLayer:
    """A composite layer: material, thickness/orientation, and the
    perimeter band it covers (or the web it sits on)."""

    name: str
    material: str
    thickness: np.ndarray         # (n_span,) m
    fiber_orientation: np.ndarray  # (n_span,) rad
    start_nd: np.ndarray          # (n_span,) shell-arc band start
    end_nd: np.ndarray            # (n_span,) shell-arc band end
    web: str | None               # web name if this is an on-web layer


@dataclass
class ResolvedBladeStructure:
    """Every web + layer resolved onto a common blade span grid."""

    nd_span: np.ndarray
    webs: list[ResolvedWeb]
    layers: list[ResolvedLayer]


def _interp(grid, values, nd_span: np.ndarray) -> np.ndarray:
    """WISDEM-faithful curve interpolation onto the span grid.

    ``PchipInterpolator`` with ``extrapolate=False`` (NaN outside the
    defined grid) then ``nan_to_num`` → 0, exactly as
    ``gc_WT_InitModel.py`` does, so a region defined only over part of
    the span (e.g. a web starting at r/R = 0.1) is zero elsewhere and
    matches the WISDEM / BeamDyn oracle. A 2-point grid degenerates to
    linear, which is the intended behaviour for ``[0, 1]`` constants.
    """
    g = np.asarray(grid, dtype=float)
    v = np.asarray(values, dtype=float)
    if g.size < 2:
        # A single point: constant over the whole span.
        return np.full_like(np.asarray(nd_span, dtype=float), float(v.flat[0]))
    order = np.argsort(g)
    g, v = g[order], v[order]
    # PCHIP needs strictly increasing x; collapse exact dups (keep last).
    keep = np.concatenate([np.diff(g) > 0, [True]])
    g, v = g[keep], v[keep]
    out = PchipInterpolator(g, v, extrapolate=False)(np.asarray(nd_span,
                                                                dtype=float))
    return np.nan_to_num(out, nan=0.0)


def _resolve_handle(
    spec: dict,
    anchors_by_name: dict,
    nd_span: np.ndarray,
    *,
    what: str,
) -> np.ndarray:
    """Resolve one ``start_nd_arc`` / ``end_nd_arc`` spec to a span curve.

    Explicit ``{grid, values}`` → interpolate. ``{anchor: {name,
    handle}}`` → dereference the ``structure.anchors[]`` registry entry
    ``name`` and take its explicit ``{grid, values}`` for ``handle``.
    A registry entry that carries only a parametric recipe
    (``plane_intersection`` / ``midpoint_nd_arc`` + ``width``) with no
    resolved values raises a clear, actionable error.
    """
    if "grid" in spec and "values" in spec:
        return _interp(spec["grid"], spec["values"], nd_span)

    if "anchor" in spec:
        a_name = spec["anchor"]["name"]
        a_handle = spec["anchor"]["handle"]
        anchor = anchors_by_name.get(a_name)
        if anchor is None:
            raise KeyError(
                f"WindIO blade {what} references anchor {a_name!r}, which "
                f"is not in the blade structure.anchors[] registry "
                f"(have: {sorted(anchors_by_name)})."
            )
        sub = anchor.get(a_handle)
        if isinstance(sub, dict) and "grid" in sub and "values" in sub:
            return _interp(sub["grid"], sub["values"], nd_span)
        recipe = next((k for k in ("plane_intersection", "midpoint_nd_arc",
                                   "width") if k in anchor), None)
        raise NotImplementedError(
            f"WindIO blade anchor {a_name!r} has no resolved "
            f"{a_handle}.{{grid,values}} — only a parametric "
            f"{recipe!r} recipe. pyBmodes resolves blade structure by "
            f"dereferencing resolved arc curves; the published IEA / "
            f"WISDEM RWT yamls include them. Geometric parametric "
            f"resolution (offset/rotation/width → nd_arc) is not yet "
            f"implemented — supply a WISDEM-resolved blade yaml."
        )

    raise ValueError(
        f"WindIO blade {what} arc spec is neither explicit "
        f"{{grid, values}} nor an {{anchor}}: keys {sorted(spec)}"
    )


def _side_kind(spec):
    """Classify a ``start_nd_arc`` / ``end_nd_arc`` spec.

    Returns ``(kind, payload)`` with kind ∈ {``absent``, ``value``,
    ``anchor``, ``te``, ``le``, ``region``, ``bad``}. Explicit
    ``{grid, values}`` wins even when a ``fixed:`` tag co-exists (the
    IEA-3.4 layers carry both — the resolved values are authoritative).
    """
    if not isinstance(spec, dict):
        return ("absent", None)
    if "grid" in spec and "values" in spec:
        return ("value", spec)
    if "anchor" in spec:
        return ("anchor", spec)
    if "fixed" in spec:
        f = spec["fixed"]
        if f == "TE":
            return ("te", None)
        if f == "LE":
            return ("le", None)
        return ("region", f)
    return ("bad", sorted(spec))


def _perimeter_geometry(profiles, chords, nd_span):
    """Per-station physical perimeter length and LE ``nd_arc``.

    Needed to turn a ``width`` (metres along the perimeter) into an
    arc fraction and to place an ``LE``/``TE`` anchor. ``None`` when
    the caller did not supply geometry (then any parametric form
    raises the documented error)."""
    if profiles is None or chords is None:
        return None, None
    chords = np.asarray(chords, dtype=float)
    perim = np.empty(len(nd_span))
    s_le = np.empty(len(nd_span))
    for i, p in enumerate(profiles):
        seg = np.hypot(np.diff(p.xc), np.diff(p.yc))
        perim[i] = float(seg.sum()) * float(chords[i])
        s_le[i] = float(p.s_le)
    return perim, s_le


def resolve_blade_structure(
    structure: dict,
    nd_span: np.ndarray,
    *,
    profiles=None,
    chords=None,
) -> ResolvedBladeStructure:
    """Resolve a WindIO blade ``structure`` block onto ``nd_span``.

    ``structure`` is the blade structure mapping — modern
    ``components.blade.structure`` or older
    ``components.blade.internal_structure_2d_fem``. It carries
    ``webs`` / ``layers`` and (modern) an ``anchors`` registry.

    Resolution order per side: explicit ``{grid, values}`` →
    ``{anchor}`` dereference → ``{fixed: TE/LE}`` → a missing side
    derived from ``width`` (± a ``midpoint_nd_arc``) → ``{fixed:
    <region>}`` locked to a sibling region (WISDEM ``build_layer``
    3–6). The ``width``/``LE`` forms need the per-station airfoil
    perimeter, so ``profiles`` (a :class:`~pybmodes.io._precomp.
    profile.Profile` per ``nd_span`` station) and ``chords`` must be
    supplied for those; absent them a parametric form raises a clear,
    actionable error (explicit/anchor yamls are unaffected).
    """
    nd_span = np.asarray(nd_span, dtype=float)
    anchors_by_name = {
        a["name"]: a for a in structure.get("anchors", []) if "name" in a
    }
    web_names = {w.get("name") for w in structure.get("webs", [])}
    perim, s_le = _perimeter_geometry(profiles, chords, nd_span)
    zeros = np.zeros_like(nd_span)
    ones = np.ones_like(nd_span)

    def need_geo(what):
        if perim is None:
            raise NotImplementedError(
                f"WindIO blade {what} uses a parametric arc form "
                f"(fixed: LE/TE, width or midpoint_nd_arc); resolving it "
                f"needs the per-station airfoil — call via "
                f"read_windio_blade (which supplies profiles/chords). "
                f"Pure explicit/anchor yamls do not need this."
            )

    def width_arr(entry):
        w = entry.get("width")
        if not isinstance(w, dict):
            return None
        return _interp(w["grid"], w["values"], nd_span)

    # Pass 1: resolve every web + layer side that does not depend on a
    # sibling region. Region-locked sides are recorded for pass 2.
    region_targets: dict = {}      # name -> {"start": rname, "end": rname}
    resolved: dict = {}            # name -> [start_arr|None, end_arr|None]

    def _resolve_entry(entry, what):
        ks, _ = _side_kind(entry.get("start_nd_arc"))
        ke, _ = _side_kind(entry.get("end_nd_arc"))
        sd = entry.get("start_nd_arc")
        ed = entry.get("end_nd_arc")
        start = end = None
        rs = re_ = None

        if ks == "value" or ks == "anchor":
            start = _resolve_handle(sd, anchors_by_name, nd_span, what=what)
        elif ks == "te":
            start = zeros.copy()
        elif ks == "le":
            need_geo(what)
            start = s_le.copy()
        elif ks == "region":
            rs = _side_kind(sd)[1]
        elif ks == "bad":
            raise ValueError(
                f"WindIO blade {what} start_nd_arc is neither explicit, "
                f"anchor, fixed, nor width-derived: keys {_side_kind(sd)[1]}"
            )

        if ke == "value" or ke == "anchor":
            end = _resolve_handle(ed, anchors_by_name, nd_span, what=what)
        elif ke == "te":
            end = ones.copy()
        elif ke == "le":
            need_geo(what)
            end = s_le.copy()
        elif ke == "region":
            re_ = _side_kind(ed)[1]
        elif ke == "bad":
            raise ValueError(
                f"WindIO blade {what} end_nd_arc is neither explicit, "
                f"anchor, fixed, nor width-derived: keys {_side_kind(ed)[1]}"
            )

        # Width fills a missing (absent / region) side.
        w = width_arr(entry)
        if w is not None:
            need_geo(what)
            frac = w / perim
            mid = entry.get("midpoint_nd_arc")
            if start is None and end is None and isinstance(mid, dict):
                mk, _ = _side_kind(mid)
                if mk == "le":
                    centre = s_le
                elif mk == "value":
                    centre = _interp(mid["grid"], mid["values"], nd_span)
                elif mk == "te":
                    centre = zeros
                else:
                    raise NotImplementedError(
                        f"WindIO blade {what}: unsupported midpoint_nd_arc "
                        f"{sorted(mid)} (only fixed: LE/TE or grid/values)"
                    )
                start = np.clip(centre - 0.5 * frac, 0.0, 1.0)
                end = np.clip(centre + 0.5 * frac, 0.0, 1.0)
                rs = re_ = None
            elif start is not None and end is None and re_ is None:
                end = np.clip(start + frac, 0.0, 1.0)
            elif end is not None and start is None and rs is None:
                start = np.clip(end - frac, 0.0, 1.0)

        resolved[what_name(what)] = [start, end]
        if rs is not None or re_ is not None:
            region_targets[what_name(what)] = {"start": rs, "end": re_}

    def what_name(what):
        return what.split("'")[1] if "'" in what else what

    webs: list[ResolvedWeb] = []
    for w in structure.get("webs", []):
        nm = w.get("name", f"web{len(webs)}")
        _resolve_entry(w, f"web {nm!r}")
        webs.append(ResolvedWeb(name=nm, start_nd=zeros.copy(),
                                end_nd=ones.copy()))

    layer_meta: list = []
    for ly in structure.get("layers", []):
        nm = ly.get("name", f"layer{len(layer_meta)}")
        if "material" not in ly:
            raise KeyError(f"WindIO blade layer {nm!r} has no 'material'")
        if "thickness" not in ly:
            raise KeyError(f"WindIO blade layer {nm!r} has no 'thickness'")
        th = ly["thickness"]
        thickness = _interp(th["grid"], th["values"], nd_span)
        fo = ly.get("fiber_orientation")
        fiber = (_interp(fo["grid"], fo["values"], nd_span)
                 if isinstance(fo, dict) and "grid" in fo
                 else np.zeros_like(nd_span))
        web = ly.get("web")
        if web is None:
            for side in ("start_nd_arc", "end_nd_arc"):
                spec = ly.get(side, {})
                a = spec.get("anchor", {}) if isinstance(spec, dict) else {}
                if a.get("name") in web_names:
                    web = a["name"]
                    break
        if web is None and "start_nd_arc" not in ly and "end_nd_arc" not in ly \
                and "width" not in ly:
            # Truly unplaced layer: treat as full perimeter.
            resolved[nm] = [zeros.copy(), ones.copy()]
        elif web is None:
            _resolve_entry(ly, f"layer {nm!r}")
        layer_meta.append((nm, ly["material"], thickness, fiber, web))

    # Pass 2: lock region-referenced sides to a sibling's resolved edge
    # (WISDEM build_layer 6: a filler butts against its neighbours).
    for _ in range(4):
        progressed = False
        for nm, tgt in list(region_targets.items()):
            st, en = resolved.get(nm, [None, None])
            if tgt.get("start") and st is None:
                ref = resolved.get(tgt["start"])
                if ref and ref[1] is not None:
                    st = ref[1].copy()
                    progressed = True
            if tgt.get("end") and en is None:
                ref = resolved.get(tgt["end"])
                if ref and ref[0] is not None:
                    en = ref[0].copy()
                    progressed = True
            resolved[nm] = [st, en]
        if not progressed:
            break

    out_webs: list[ResolvedWeb] = []
    for rw in webs:
        st, en = resolved.get(rw.name, [None, None])
        if st is None or en is None:
            raise ValueError(
                f"WindIO blade web {rw.name!r} arc band unresolved "
                f"(start={st is not None}, end={en is not None})"
            )
        out_webs.append(ResolvedWeb(name=rw.name, start_nd=st, end_nd=en))

    out_layers: list[ResolvedLayer] = []
    for nm, mat, thickness, fiber, web in layer_meta:
        if web is not None:
            out_layers.append(ResolvedLayer(
                name=nm, material=mat, thickness=thickness,
                fiber_orientation=fiber, start_nd=zeros.copy(),
                end_nd=zeros.copy(), web=web,
            ))
            continue
        st, en = resolved.get(nm, [None, None])
        if st is None or en is None:
            raise ValueError(
                f"WindIO blade layer {nm!r} arc band unresolved — a "
                f"`fixed: <region>` reference could not be matched to a "
                f"sibling region (start={st is not None}, "
                f"end={en is not None})"
            )
        out_layers.append(ResolvedLayer(
            name=nm, material=mat, thickness=thickness,
            fiber_orientation=fiber, start_nd=st, end_nd=en, web=None,
        ))

    return ResolvedBladeStructure(nd_span=nd_span, webs=out_webs,
                                  layers=out_layers)
