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

"""Read a WindIO ontology ``.yaml`` blade and reduce it to the FEM
section-property table (issue #35).

This is the public glue that ties Phase-2 together, mirroring
:mod:`pybmodes.io.windio` (tower) / :func:`pybmodes.io.geometry.
tubular_section_props`:

* :func:`read_windio_blade` — parse the blade component
  (dialect-robust, reusing the duplicate-anchor-tolerant loader from
  :mod:`pybmodes.io.windio`): span axis, chord, twist, reference-axis
  chordwise location, the spanwise airfoil set, the resolved web /
  layer ``nd_arc`` bands (:mod:`pybmodes.io._precomp.arc_resolver`),
  and the material table.
* :func:`windio_blade_section_props` — walk the span, blend the
  airfoil, build each station's shell-layer / web stacks, run the
  thin-wall reduction (:mod:`pybmodes.io._precomp.reduction`), and
  assemble a :class:`pybmodes.io.sec_props.SectionProperties` ready
  for :class:`pybmodes.models.RotatingBlade`.

Both WindIO key dialects are handled — modern ``outer_shape`` /
``structure`` (IEA-15 WT_Ontology, every WISDEM example incl. the
floating ones) and older ``outer_shape_bem`` /
``internal_structure_2d_fem`` (IEA-3.4 / 10 / 22). Needs the optional
``[windio]`` extra (PyYAML); the runtime core stays numpy+scipy.
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass

import numpy as np

from pybmodes.io._precomp.arc_resolver import (
    ResolvedBladeStructure,
    resolve_blade_structure,
)
from pybmodes.io._precomp.laminate import material_plane_stress
from pybmodes.io._precomp.profile import Profile
from pybmodes.io._precomp.reduction import (
    LayerStation,
    WebStation,
    reduce_section,
)
from pybmodes.io.sec_props import SectionProperties
from pybmodes.io.windio import _dup_anchor_loader, _require_yaml


@dataclass
class WindIOBlade:
    """Geometry + layup of a WindIO blade, resolved onto a span grid."""

    span_grid: np.ndarray          # normalised [0, 1], root → tip
    flexible_length: float         # m, |z_tip − z_root|
    chord: np.ndarray              # m, per station
    twist_deg: np.ndarray          # deg, per station (structural twist)
    ref_axis_xc: np.ndarray        # reference-axis chord fraction
    profiles: list[Profile]        # blended airfoil per station
    resolved: ResolvedBladeStructure
    materials: dict
    #: Pre-computed distributed beam properties parsed straight from
    #: the WindIO ``elastic_properties`` / ``elastic_properties_mb``
    #: block (the published reference), interpolated onto
    #: ``span_grid``; ``None`` when the file carries only the layup.
    #: Keys: ``mass_den``/``flp_iner``/``edge_iner``/``flp_stff``/
    #: ``edge_stff``/``tor_stff``/``axial_stff``/``cg_offst``.
    elastic: dict | None = None
    #: Non-``None`` when a published block *was* present but could not
    #: be parsed (schema drift / malformed). ``elastic`` is then
    #: ``None`` too, but this distinguishes "absent" (silent PreComp
    #: fallback is correct) from "present-but-broken" (``"auto"``
    #: warns; ``"file"`` raises) — so a typo can't hide behind a
    #: plausible lower-fidelity result.
    elastic_parse_error: str | None = None


def _blade_shape_and_structure(comp: dict, component: str):
    """``(outer_shape, structure)`` across both WindIO dialects
    (mirrors ``pybmodes.io.windio._shape_and_structure``)."""
    shape = comp.get("outer_shape", comp.get("outer_shape_bem"))
    structure = comp.get("structure",
                          comp.get("internal_structure_2d_fem"))
    if shape is None or structure is None:
        raise KeyError(
            f"components.{component} has neither modern "
            f"'outer_shape'/'structure' nor older "
            f"'outer_shape_bem'/'internal_structure_2d_fem'."
        )
    return shape, structure


def _reference_axis(comp: dict, shape: dict, structure: dict,
                    component: str) -> dict:
    for holder in (comp, shape, structure):
        ra = holder.get("reference_axis")
        if ra is not None and "z" in ra:
            return ra
    raise KeyError(f"components.{component} has no reference_axis.z")


def _curve(spec: dict, at: np.ndarray) -> np.ndarray:
    """Linear-interpolate a WindIO ``{grid, values}`` onto ``at``
    (WindIO-native interpolation; mirrors the tower reader)."""
    g = np.asarray(spec["grid"], dtype=float)
    v = np.asarray(spec["values"], dtype=float)
    return np.interp(at, g, v)


# Blade aerodynamic/structural twist is at most ~25–30° anywhere on a
# modern blade. The windIO ontology *nominally* stores twist in
# radians (IEA-3.4-130-RWT: root ≈ 0.349 rad), but real WISDEM
# reference files ship it in degrees (IEA-15-240-RWT: root ≈ 15.6).
# A radian-convention twist therefore never exceeds ~0.6; anything
# past ~2 rad (≈ 115°) is unphysical as radians and must already be
# in degrees. Decide by magnitude rather than trusting the spec.
_TWIST_RADIAN_CEILING = 2.0


def _twist_to_degrees(twist_raw: np.ndarray) -> np.ndarray:
    """Return blade twist in **degrees**, auto-detecting the source
    unit (issue #47 follow-up — static review).

    ``np.degrees`` was previously applied unconditionally, which is
    correct for radian-convention windIO files (IEA-3.4) but turned a
    degree-convention file's 15.6° root twist (IEA-15) into ≈ 894°.
    """
    arr = np.asarray(twist_raw, dtype=float)
    if arr.size and float(np.nanmax(np.abs(arr))) > _TWIST_RADIAN_CEILING:
        return arr                 # already degrees (WISDEM IEA-15 style)
    return np.degrees(arr)         # radians (windIO spec / IEA-3.4 style)


def _read_blade_elastic(
    holders: tuple[dict, ...], span: np.ndarray
) -> tuple[dict | None, str | None]:
    """Parse the WindIO blade *published* distributed beam properties
    onto ``span``.

    Returns ``(props, error)``:

    * ``(dict, None)`` — a published block was present and parsed;
    * ``(None, None)`` — the file genuinely carries *no* published
      block (only a layup) → silent PreComp fallback is correct;
    * ``(None, str)`` — a published block *is* present but could not
      be parsed (schema drift / malformed data). The caller must not
      silently degrade to the approximate PreComp path on this case
      (it would hide a typo behind a plausible but lower-fidelity
      result, issue #47 follow-up — static review): in
      ``elastic="auto"`` it warns, in ``"file"`` it raises.

    ``holders`` are the candidate dicts the block may live under —
    the modern ``elastic_properties`` is nested in the ``structure``
    block (IEA-15), while ``elastic_properties_mb`` is a direct
    ``components.blade`` child (IEA-22); search both.

    Supports both dialects:

    * modern ``components.blade.<...>.elastic_properties`` —
      ``inertia_matrix`` (named ``mass`` / ``i_flap`` / ``i_edge`` /
      ``cm_x`` arrays) + ``stiffness_matrix`` (named ``K11``..``K66``);
    * ``components.blade.elastic_properties_mb.six_x_six`` —
      ``stiff_matrix`` / ``inertia_matrix`` as a ``{grid, values}``
      with each ``values`` row the 21-element upper-triangular
      flatten of the symmetric 6×6.

    The 6×6 is **decoupled** to pyBmodes' Euler–Bernoulli beam at the
    elastic / shear centres and principal elastic axes (issue #50):
    the raw reference-axis diagonal
    ``K44``/``K55`` is *not* ``EI_flap``/``EI_edge`` for an offset /
    pre-twisted blade — see :mod:`pybmodes.io._precomp.decouple`. Both
    dialects carry the full 6×6 (modern ``stiffness_matrix`` ships the
    upper triangle ``K11``..``K66``; ``elastic_properties_mb`` ships
    the 21-element flatten), so the coupling is honoured, not dropped.
    """
    from pybmodes.io._precomp.decouple import (
        _assign_flap_edge,
        _principal_2x2,
        decouple_inertia,
        decouple_stiffness,
    )

    def _principal_pair(a: float, b: float, c: float) -> tuple[float, float]:
        """(flap, edge) principal moments of the symmetric 2×2
        ``[[a, c], [c, b]]``, assigned to the *named* axes (axis-1 =
        flap) by principal-axis alignment — the same rule the full-6×6
        path uses — **not** magnitude-sorted. A schema-labelled
        ``i_flap > i_edge`` (or ``i_cp = 0``, already diagonal) is
        therefore preserved, never silently swapped (issue #50
        follow-up — static review)."""
        la, lb, _ang, V = _principal_2x2(np.array([[a, c], [c, b]]))
        return _assign_flap_edge(la, lb, V)

    def _find(key: str):
        for h in holders:
            if isinstance(h, dict) and isinstance(h.get(key), dict):
                return h[key]
        return None

    def _sym6_from_named(km: dict, gi: int) -> np.ndarray:
        """Symmetric 6×6 at source-grid index ``gi`` from named
        ``Kij`` arrays (upper triangle; missing entry ⇒ 0)."""
        K = np.zeros((6, 6))
        for i in range(6):
            for j in range(i, 6):
                vals = km.get(f"K{i + 1}{j + 1}")
                if vals is None:
                    continue
                v = float(np.asarray(vals, float)[gi])
                K[i, j] = K[j, i] = v
        return K

    def _sym6_from_upper21(row: np.ndarray) -> np.ndarray:
        """Symmetric 6×6 from a 21-element row-major upper-triangular
        flatten (``six_x_six`` ``values`` convention)."""
        K = np.zeros((6, 6))
        p = 0
        for i in range(6):
            for j in range(i, 6):
                K[i, j] = K[j, i] = float(row[p])
                p += 1
        return K

    def _decoupled_over_grid(
        mats: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Decouple each source-grid 6×6 then return per-grid scalar
        arrays (decoupling is non-linear, so decouple *before*
        interpolating onto the output span)."""
        ds = [decouple_stiffness(K) for K in mats]
        return {
            "axial_stff": np.array([d.EA for d in ds]),
            "flp_stff": np.array([d.EI_flap for d in ds]),
            "edge_stff": np.array([d.EI_edge for d in ds]),
            "tor_stff": np.array([d.GJ for d in ds]),
            "x_tc": np.array([d.x_tc for d in ds]),
        }

    def _interp_grid(g: np.ndarray, arr: np.ndarray) -> np.ndarray:
        return np.interp(span, g, arr)

    ep = _find("elastic_properties")
    ep_present = isinstance(ep, dict) and "stiffness_matrix" in ep
    mb = _find("elastic_properties_mb")
    s6 = mb.get("six_x_six") if isinstance(mb, dict) else None
    mb_present = isinstance(s6, dict) and "stiff_matrix" in s6

    if not ep_present and not mb_present:
        return None, None          # genuinely absent — layup only

    try:
        if ep_present:
            km = ep["stiffness_matrix"]
            im = ep.get("inertia_matrix", {})
            kg = np.asarray(km["grid"], float)
            kmat = [_sym6_from_named(km, gi) for gi in range(kg.size)]
            dec = _decoupled_over_grid(kmat)

            ig = np.asarray(im["grid"], float)
            mass = np.asarray(im["mass"], float)
            i_fl = np.asarray(im["i_flap"], float)
            i_ed = np.asarray(im["i_edge"], float)
            i_cp = (np.asarray(im["i_cp"], float)
                    if "i_cp" in im else np.zeros_like(mass))
            cm_x = (np.asarray(im["cm_x"], float)
                    if "cm_x" in im else np.zeros_like(mass))
            # Principal mass moments from the 2×2 [[i_flap,i_cp],
            # [i_cp,i_edge]] (about the c.g.; i_cp ≠ 0 ⇒ not
            # principal), assigned to flap/edge by axis alignment so
            # the schema labels survive (no magnitude sort).
            fe = np.array([_principal_pair(fl, ed, cp)
                           for fl, ed, cp in zip(i_fl, i_ed, i_cp)])
            i_flap_p, i_edge_p = fe[:, 0], fe[:, 1]
            # Tension centre evaluated on the inertia grid.
            tc_on_ig = np.interp(ig, kg, dec["x_tc"])
            return {
                "axial_stff": _interp_grid(kg, dec["axial_stff"]),
                "flp_stff": _interp_grid(kg, dec["flp_stff"]),
                "edge_stff": _interp_grid(kg, dec["edge_stff"]),
                "tor_stff": _interp_grid(kg, dec["tor_stff"]),
                "mass_den": np.interp(span, ig, mass),
                "flp_iner": np.interp(span, ig, i_flap_p),
                "edge_iner": np.interp(span, ig, i_edge_p),
                # c.g. offset relative to the elastic (tension) centre.
                "cg_offst": np.interp(span, ig, cm_x - tc_on_ig),
            }, None

        assert isinstance(s6, dict)   # narrowed by mb_present above
        sk = s6["stiff_matrix"]
        si = s6["inertia_matrix"]
        gk = np.asarray(sk["grid"], float)
        gi = np.asarray(si["grid"], float)
        kmat = [_sym6_from_upper21(r)
                for r in np.asarray(sk["values"], float)]
        mmat = [_sym6_from_upper21(r)
                for r in np.asarray(si["values"], float)]
        dec = _decoupled_over_grid(kmat)
        di = [decouple_inertia(M) for M in mmat]
        mass = np.array([d.mass for d in di])
        i_fl = np.array([d.i_flap for d in di])
        i_ed = np.array([d.i_edge for d in di])
        x_cg = np.array([d.x_cg for d in di])
        tc_on_gi = np.interp(gi, gk, dec["x_tc"])
        return {
            "axial_stff": _interp_grid(gk, dec["axial_stff"]),
            "flp_stff": _interp_grid(gk, dec["flp_stff"]),
            "edge_stff": _interp_grid(gk, dec["edge_stff"]),
            "tor_stff": _interp_grid(gk, dec["tor_stff"]),
            "mass_den": np.interp(span, gi, mass),
            "flp_iner": np.interp(span, gi, i_fl),
            "edge_iner": np.interp(span, gi, i_ed),
            "cg_offst": np.interp(span, gi, x_cg - tc_on_gi),
        }, None
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        # Block is *present* but unparseable — never silently degrade.
        return None, (
            "WindIO blade carries a published elastic-properties block "
            f"that could not be parsed ({type(exc).__name__}: {exc}); "
            "this usually means schema drift or a malformed grid/values "
            "table."
        )


def read_windio_blade(
    yaml_path: str | pathlib.Path,
    *,
    component: str = "blade",
    n_span: int = 30,
) -> WindIOBlade:
    """Parse the structural subset of a WindIO blade component."""
    yaml = _require_yaml()
    yaml_path = pathlib.Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_dup_anchor_loader(yaml))

    try:
        comp = doc["components"][component]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"WindIO file {yaml_path} has no components.{component!r}."
        ) from exc

    shape, structure = _blade_shape_and_structure(comp, component)
    ra = _reference_axis(comp, shape, structure, component)
    z = ra["z"]
    z_grid = np.asarray(z["grid"], dtype=float)
    z_vals = np.asarray(z["values"], dtype=float)
    flexible_length = float(abs(z_vals[-1] - z_vals[0]))

    # Output span stations: a uniform grid over the defined span
    # (offsets/twist/chord interpolated linearly onto it).
    span = np.linspace(float(z_grid[0]), float(z_grid[-1]), n_span)

    chord = _curve(shape["chord"], span)
    twist_deg = _twist_to_degrees(_curve(shape["twist"], span))
    if "pitch_axis" in shape:                    # older: chord fraction
        ref_xc = _curve(shape["pitch_axis"], span)
    elif "section_offset_y" in shape:            # modern: metres / chord
        ref_xc = _curve(shape["section_offset_y"], span) / chord
    else:
        ref_xc = np.full(n_span, 0.5)            # fallback: mid-chord

    # Airfoil set: name → Profile, and the spanwise schedule.
    af_coords = {a["name"]: a["coordinates"] for a in doc.get("airfoils", [])}

    def _profile(name: str) -> Profile:
        c = af_coords[name]
        return Profile.from_windio_coords(c["x"], c["y"])

    if "airfoil_position" in shape:              # older dialect
        af_grid = np.asarray(shape["airfoil_position"]["grid"], float)
        af_labels = list(shape["airfoil_position"]["labels"])
    else:                                        # modern dialect
        afs = sorted(shape["airfoils"],
                     key=lambda a: a["spanwise_position"])
        af_grid = np.asarray([a["spanwise_position"] for a in afs], float)
        af_labels = [a["name"] for a in afs]

    if af_grid.size == 0 or not af_labels:
        raise ValueError(
            f"WindIO blade {component!r} has no airfoil schedule "
            "(empty airfoil_position / airfoils)."
        )

    cache: dict[str, Profile] = {}

    def _blended(s: float) -> Profile:
        # A single-airfoil blade (constant profile) is a valid input
        # shape; without this guard ``len(af_grid) - 2 = -1`` makes
        # ``np.clip`` and the ``j + 1`` index misbehave (static
        # review). Reuse the one profile everywhere.
        if len(af_grid) < 2:
            only = af_labels[0]
            return cache.setdefault(only, _profile(only))
        j = int(np.clip(np.searchsorted(af_grid, s) - 1, 0,
                        len(af_grid) - 2))
        nlo, nhi = af_labels[j], af_labels[j + 1]
        plo = cache.setdefault(nlo, _profile(nlo))
        if nhi == nlo:
            return plo
        phi = cache.setdefault(nhi, _profile(nhi))
        span_lo, span_hi = af_grid[j], af_grid[j + 1]
        w = 0.0 if span_hi <= span_lo else (s - span_lo) / (span_hi -
                                                            span_lo)
        return plo.blend(phi, float(np.clip(w, 0.0, 1.0)))

    profiles = [_blended(float(s)) for s in span]
    resolved = resolve_blade_structure(
        structure, span, profiles=profiles, chords=chord
    )
    materials = {m["name"]: m for m in doc.get("materials", [])
                 if "name" in m}

    elastic, elastic_err = _read_blade_elastic(
        (comp, structure, shape), span
    )
    return WindIOBlade(
        span_grid=span, flexible_length=flexible_length, chord=chord,
        twist_deg=twist_deg, ref_axis_xc=ref_xc, profiles=profiles,
        resolved=resolved, materials=materials,
        elastic=elastic, elastic_parse_error=elastic_err,
    )


def windio_blade_section_props(
    blade: WindIOBlade,
    *,
    n_perim: int = 300,
    title: str = "WindIO composite-blade section properties",
    elastic: str = "auto",
) -> SectionProperties:
    """Reduce every span station to the FEM section-property table.

    ``elastic`` selects the property source (issue #48 — keep deltas
    to the reference model small):

    * ``"auto"`` (default) — use the WindIO *published* distributed
      beam properties (``elastic_properties`` /
      ``elastic_properties_mb``) when the file carries them, so
      pyBmodes matches the reference model's stiffness/inertia
      exactly; fall back to the PreComp thin-wall reduction of the
      layup only when they are absent.
    * ``"precomp"`` — always run the PreComp reduction (the pre-1.5
      behaviour), even when published properties exist.
    * ``"file"`` — require the published properties; raise
      ``ValueError`` if the file has only the layup *or* carries a
      published block that could not be parsed.

    If a published block is **present but unparseable** (schema drift
    / malformed), ``"auto"`` does not silently fall back to the
    lower-fidelity PreComp result — it emits a ``UserWarning`` naming
    the parse problem before reducing the layup, and ``"file"``
    raises (issue #47 follow-up — static review).
    """
    if elastic not in ("auto", "precomp", "file"):
        raise ValueError(
            f"elastic must be 'auto', 'precomp' or 'file'; got "
            f"{elastic!r}"
        )
    n = len(blade.span_grid)

    use_published = (
        elastic != "precomp" and blade.elastic is not None
    )
    if elastic == "file" and blade.elastic is None:
        if blade.elastic_parse_error is not None:
            raise ValueError(
                "elastic='file' but the WindIO blade's published "
                f"block is unusable: {blade.elastic_parse_error}"
            )
        raise ValueError(
            "elastic='file' but the WindIO blade carries no "
            "elastic_properties / elastic_properties_mb block "
            "(only a layup) — use elastic='auto' or 'precomp' to "
            "reduce the layup via PreComp."
        )
    if (elastic == "auto" and blade.elastic is None
            and blade.elastic_parse_error is not None):
        warnings.warn(
            f"{blade.elastic_parse_error} Falling back to the "
            "approximate PreComp layup reduction; pass "
            "elastic='precomp' to silence this, or fix the block / "
            "use elastic='file' to require it.",
            UserWarning,
            stacklevel=2,
        )
    if use_published:
        e = blade.elastic
        assert e is not None        # narrowed by use_published
        z = np.asarray(blade.span_grid, dtype=float)
        zeros = np.zeros(n)
        return SectionProperties(
            title=title + " (WindIO published elastic properties)",
            n_secs=n,
            span_loc=z,
            str_tw=np.asarray(blade.twist_deg, dtype=float),
            tw_iner=zeros.copy(),
            mass_den=np.asarray(e["mass_den"], float),
            flp_iner=np.asarray(e["flp_iner"], float),
            edge_iner=np.asarray(e["edge_iner"], float),
            flp_stff=np.asarray(e["flp_stff"], float),
            edge_stff=np.asarray(e["edge_stff"], float),
            tor_stff=np.asarray(e["tor_stff"], float),
            axial_stff=np.asarray(e["axial_stff"], float),
            cg_offst=np.asarray(e["cg_offst"], float),
            sc_offst=zeros.copy(),     # decoupled beam: coupling
            tc_offst=zeros.copy(),     # terms intentionally not modelled
        )

    cols = {k: np.zeros(n) for k in (
        "mass_den", "flp_iner", "edge_iner", "flp_stff", "edge_stff",
        "tor_stff", "axial_stff", "cg_offst", "sc_offst", "tc_offst",
    )}

    for i in range(n):
        web_plies: dict[str, list] = {}
        shell: list[LayerStation] = []
        for ly in blade.resolved.layers:
            if ly.material not in blade.materials:
                raise KeyError(
                    f"WindIO blade layer {ly.name!r} references material "
                    f"{ly.material!r} not in the top-level materials list"
                )
            pe = material_plane_stress(blade.materials[ly.material])
            t = float(ly.thickness[i])
            if t <= 0.0:
                continue
            th = float(ly.fiber_orientation[i])
            if ly.web is not None:
                web_plies.setdefault(ly.web, []).append((pe, t, th))
            else:
                shell.append(LayerStation(pe, t, th,
                                          float(ly.start_nd[i]),
                                          float(ly.end_nd[i])))
        webs = [
            WebStation(float(w.start_nd[i]), float(w.end_nd[i]),
                       web_plies.get(w.name, []))
            for w in blade.resolved.webs
        ]
        res = reduce_section(
            blade.profiles[i], float(blade.chord[i]),
            float(blade.ref_axis_xc[i]), shell, webs, n_perim=n_perim,
        )
        cols["mass_den"][i] = res.mass
        cols["flp_iner"][i] = res.flap_iner
        cols["edge_iner"][i] = res.edge_iner
        cols["flp_stff"][i] = res.EI_flap
        cols["edge_stff"][i] = res.EI_edge
        cols["tor_stff"][i] = res.GJ
        cols["axial_stff"][i] = res.EA
        cols["cg_offst"][i] = res.x_cg
        cols["sc_offst"][i] = res.x_sc
        cols["tc_offst"][i] = res.x_tc

    z = np.asarray(blade.span_grid, dtype=float)
    zeros = np.zeros(n)
    return SectionProperties(
        title=title,
        n_secs=n,
        span_loc=z,
        str_tw=np.asarray(blade.twist_deg, dtype=float),
        tw_iner=zeros.copy(),
        mass_den=cols["mass_den"],
        flp_iner=cols["flp_iner"],
        edge_iner=cols["edge_iner"],
        flp_stff=cols["flp_stff"],
        edge_stff=cols["edge_stff"],
        tor_stff=cols["tor_stff"],
        axial_stff=cols["axial_stff"],
        cg_offst=cols["cg_offst"],
        sc_offst=cols["sc_offst"],
        tc_offst=cols["tc_offst"],
    )
