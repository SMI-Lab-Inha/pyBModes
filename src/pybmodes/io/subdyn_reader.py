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

"""Minimal parser and adapter for OpenFAST SubDyn ``.dat`` substructure files.

Scope is intentionally narrow — only what's needed to drive a clamped-base
monopile model from the existing reference decks shipped with the OpenFAST
regression-test corpus. Specifically:

* JOINTS — XYZ positions and joint type.
* BASE REACTION JOINTS — read the single clamped-base joint id and its DOF
  flags; SSI files are not handled (the 5MW OC3 deck has none).
* INTERFACE JOINTS — read the transition-piece joint id.
* MEMBERS — beam connectivity (joint pairs and property-set ids).
* CIRCULAR BEAM CROSS-SECTION PROPERTIES — ``E``, ``G``, ``MatDens``,
  outer diameter, wall thickness.

Sections we *don't* parse: rectangular / arbitrary cross-sections, cables,
rigid links, spring elements, cosine matrices, concentrated masses, output
settings. These are zero in the bundled OC3 monopile and SubDyn raises an
explicit error if asked to read tables we don't know about, so silently
skipping them is fine; we just consume their rows until the next section
divider. Extending the parser to cover one of those sections is a localised
change — drop another ``_consume_table`` call into :func:`_parse` and add a
typed list field on :class:`SubDynFile`.

The companion :func:`to_pybmodes_pile_tower` adapter takes the parsed
SubDyn together with the ElastoDyn main + tower files and synthesises a
single combined-cantilever ``BMIFile`` + ``SectionProperties`` for
pyBmodes' tower modal solver — a "rigid base + flexible pile + flexible
tower" model with no soil flexibility (matching the OC3 reference design,
which clamps the pile rigidly at the seabed).
"""

from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubDynJoint:
    """One node in the substructure graph (SS-coordinate system, metres)."""

    joint_id: int
    x: float
    y: float
    z: float
    joint_type: int = 1   # 1 = cantilever, 2/3/4 = universal/revolute/spherical


@dataclass
class SubDynMember:
    """One beam member connecting two joints."""

    member_id: int
    joint_a: int
    joint_b: int
    prop_set_a: int
    prop_set_b: int
    member_type: str = "1c"   # 1c = circular beam, 1r = rectangular, 2 = cable, …


@dataclass
class SubDynCircProp:
    """Circular beam cross-section property set."""

    prop_set_id: int
    E: float          # Young's modulus, Pa
    G: float          # Shear modulus, Pa
    rho: float        # Density, kg/m³
    D: float          # Outer diameter, m
    t: float          # Wall thickness, m

    @property
    def area(self) -> float:
        """Cross-sectional area of the thin-walled tube, m²."""
        d_inner = self.D - 2.0 * self.t
        return math.pi / 4.0 * (self.D ** 2 - d_inner ** 2)

    @property
    def mass_per_length(self) -> float:
        """Distributed mass density, kg/m."""
        return self.rho * self.area

    @property
    def I(self) -> float:  # noqa: E743 — standard structural symbol for area moment of inertia
        """Second moment of area, m⁴ (same about either bending axis)."""
        d_inner = self.D - 2.0 * self.t
        return math.pi / 64.0 * (self.D ** 4 - d_inner ** 4)

    @property
    def EI(self) -> float:
        """Bending stiffness about either transverse axis, N·m²."""
        return self.E * self.I

    @property
    def J(self) -> float:
        """Polar moment of area, m⁴."""
        return 2.0 * self.I

    @property
    def GJ(self) -> float:
        """Torsional stiffness, N·m²."""
        return self.G * self.J

    @property
    def EA(self) -> float:
        """Axial stiffness, N."""
        return self.E * self.area


@dataclass
class SubDynFile:
    """All parameters parsed from a SubDyn ``.dat`` file."""

    header: str = ""
    title: str = ""
    source_file: pathlib.Path | None = None
    joints: list[SubDynJoint] = field(default_factory=list)
    members: list[SubDynMember] = field(default_factory=list)
    circ_props: list[SubDynCircProp] = field(default_factory=list)
    reaction_joint_id: int = 0
    interface_joint_id: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_RE_SECTION = re.compile(r"^\s*-{3,}\s*(.+?)\s*-{3,}\s*$")


def read_subdyn(path: str | pathlib.Path) -> SubDynFile:
    """Parse a SubDyn ``.dat`` file. See module docstring for the supported
    subset of sections."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="latin-1")
    return _parse(text.splitlines(), source_file=path)


def _strip_comment(line: str) -> str:
    """Drop trailing ``!``-style or ``-`` comments after a value/label line."""
    # SubDyn comments start with ``!`` or ``-`` after the label; keep
    # everything up to the first such marker that comes after at least one
    # non-whitespace token.
    return line


def _parse_float(tok: str) -> float:
    return float(tok.strip().replace("d", "e").replace("D", "E"))


def _parse_int(tok: str) -> int:
    return int(tok.strip())


def _next_data_line(lines: list[str], i: int) -> tuple[int, str]:
    """Skip blank lines starting at ``i``, return ``(idx, stripped_line)``."""
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        raise EOFError("unexpected end of SubDyn file")
    return i, lines[i].rstrip()


def _consume_table(
    lines: list[str], i: int, n_rows: int, header_rows: int = 2,
) -> tuple[int, list[list[str]]]:
    """Skip ``header_rows`` non-blank lines, then read ``n_rows`` data rows."""
    consumed = 0
    while consumed < header_rows:
        i, _ = _next_data_line(lines, i)
        i += 1
        consumed += 1
    rows: list[list[str]] = []
    for _ in range(n_rows):
        i, line = _next_data_line(lines, i)
        rows.append(line.split())
        i += 1
    return i, rows


def _read_count(lines: list[str], i: int) -> tuple[int, int]:
    """Read a ``<count> Label - comment`` line and return ``(idx_after, count)``."""
    i, line = _next_data_line(lines, i)
    return i + 1, _parse_int(line.split()[0])


def _parse(lines: list[str], source_file: pathlib.Path | None = None) -> SubDynFile:
    obj = SubDynFile(source_file=source_file)
    if lines:
        obj.header = lines[0].rstrip()
    title_idx = 1
    while title_idx < len(lines) and not lines[title_idx].strip():
        title_idx += 1
    if title_idx < len(lines):
        obj.title = lines[title_idx].rstrip()

    # Pre-scan for the section dividers we care about; map section-name
    # keywords to the line index of the divider.
    section_index: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = _RE_SECTION.match(line)
        if not m:
            continue
        name = m.group(1).strip().upper()
        # Only record the first occurrence — subsequent dividers with the same
        # word in their title won't shadow the canonical one.
        if name not in section_index:
            section_index[name] = i

    def find_section(prefix: str) -> int:
        """Return the line index of the first divider whose name **starts with**
        ``prefix`` (case-insensitive). Prefix matching is necessary because the
        SubDyn divider lines often append free-form descriptive text after the
        canonical section name, and substring matching fires false positives
        (e.g. the *Structure Joints* divider's prose contains the word
        "members")."""
        target = prefix.upper()
        for name, idx in section_index.items():
            if name.startswith(target):
                return idx
        raise ValueError(
            f"SubDyn parser: no section header starting with {prefix!r} "
            f"found in {source_file}; got headers: {sorted(section_index)}"
        )

    # ----- JOINTS --------------------------------------------------------
    i = find_section("STRUCTURE JOINTS") + 1
    i, n_joints = _read_count(lines, i)
    i, rows = _consume_table(lines, i, n_joints, header_rows=2)
    for r in rows:
        obj.joints.append(SubDynJoint(
            joint_id=_parse_int(r[0]),
            x=_parse_float(r[1]),
            y=_parse_float(r[2]),
            z=_parse_float(r[3]),
            joint_type=_parse_int(r[4]) if len(r) >= 5 else 1,
        ))

    # ----- BASE REACTION JOINTS -----------------------------------------
    i = find_section("BASE REACTION") + 1
    i, _n_react = _read_count(lines, i)
    # Header rows then 1 data row (we only support the single-reaction case).
    i, rows = _consume_table(lines, i, 1, header_rows=2)
    obj.reaction_joint_id = _parse_int(rows[0][0])

    # ----- INTERFACE JOINTS ---------------------------------------------
    i = find_section("INTERFACE JOINTS") + 1
    i, _n_interf = _read_count(lines, i)
    i, rows = _consume_table(lines, i, 1, header_rows=2)
    obj.interface_joint_id = _parse_int(rows[0][0])

    # ----- MEMBERS -------------------------------------------------------
    i = find_section("MEMBERS") + 1
    i, n_mem = _read_count(lines, i)
    i, rows = _consume_table(lines, i, n_mem, header_rows=2)
    for row_idx, r in enumerate(rows):
        # A member row needs at least 5 columns
        # (MemberID MJointID1 MJointID2 MPropSetID1 MPropSetID2);
        # the 6th column (MType) is optional and falls back to "1c"
        # (circular beam). A short member row must not raise a
        # bare IndexError.
        if len(r) < 5:
            raise ValueError(
                f"SubDyn: malformed MEMBERS row {row_idx + 1} in "
                f"{source_file}: expected >= 5 columns "
                f"(MemberID MJointID1 MJointID2 MPropSetID1 "
                f"MPropSetID2), got {len(r)}: {' '.join(r)!r}"
            )
        obj.members.append(SubDynMember(
            member_id=_parse_int(r[0]),
            joint_a=_parse_int(r[1]),
            joint_b=_parse_int(r[2]),
            prop_set_a=_parse_int(r[3]),
            prop_set_b=_parse_int(r[4]),
            member_type=r[5] if len(r) >= 6 else "1c",
        ))

    # ----- CIRCULAR BEAM CROSS-SECTION PROPERTIES -----------------------
    i = find_section("CIRCULAR BEAM") + 1
    i, n_cyl = _read_count(lines, i)
    i, rows = _consume_table(lines, i, n_cyl, header_rows=2)
    for r in rows:
        obj.circ_props.append(SubDynCircProp(
            prop_set_id=_parse_int(r[0]),
            E=_parse_float(r[1]),
            G=_parse_float(r[2]),
            rho=_parse_float(r[3]),
            D=_parse_float(r[4]),
            t=_parse_float(r[5]),
        ))

    return obj


# ---------------------------------------------------------------------------
# Adapter — combine SubDyn pile + ElastoDyn tower into one cantilever
# ---------------------------------------------------------------------------

def _lookup_joint(subdyn: SubDynFile, joint_id: int, role: str) -> SubDynJoint:
    """Find the joint with the given ID, or raise a clear ``ValueError``.

    Used by :func:`to_pybmodes_pile_tower` to resolve the reaction
    and interface joint IDs against the joint list. A bare
    ``next(j for j in ... if j.joint_id == ...)`` raises an
    uninformative ``StopIteration`` on missing IDs; this helper names
    the missing ID, the role (``reaction`` / ``interface``), the
    source file, and the known joint IDs so a misreferenced SubDyn
    deck produces an actionable error. Pass-2 review.
    """
    for j in subdyn.joints:
        if j.joint_id == joint_id:
            return j
    known = sorted(j.joint_id for j in subdyn.joints)
    raise ValueError(
        f"SubDyn: no joint with id={joint_id} (declared as the "
        f"{role} joint) in {subdyn.source_file}; known joint IDs = "
        f"{known}. Check the SubDyn ``BASE REACTION`` / "
        f"``INTERFACE JOINTS`` block against the ``STRUCTURE JOINTS`` "
        f"table."
    )


def _circ_prop_for(subdyn: SubDynFile, prop_set_id: int) -> SubDynCircProp:
    for p in subdyn.circ_props:
        if p.prop_set_id == prop_set_id:
            return p
    raise ValueError(
        f"SubDyn: no circular cross-section property set with id={prop_set_id}; "
        f"the OC3 monopile path here only supports circular sections referenced "
        f"by member endpoints (rectangular / arbitrary / cable / rigid sets are "
        f"not implemented)."
    )


def _pile_axial_stations(subdyn: SubDynFile) -> tuple[np.ndarray, list[SubDynCircProp]]:
    """Sort joints by ``z`` (ascending), return ``(z_coords, prop_per_segment)``.

    Returns ``n`` z-coordinates and ``n - 1`` cross-section properties (one
    per inter-joint segment / member). Assumes joints are connected
    sequentially by member ordering — the OC3 monopile satisfies this; a
    multi-leg jacket would require a graph-based traversal that this simple
    adapter does not perform.
    """
    if len(subdyn.joints) < 2 or len(subdyn.members) < 1:
        raise ValueError("SubDyn: need at least 2 joints and 1 member for the pile model")

    # Joints in ascending-z order.
    joints_sorted = sorted(subdyn.joints, key=lambda j: j.z)
    z_coords = np.array([j.z for j in joints_sorted], dtype=float)

    # For each adjacent pair (k, k+1), find the member spanning that pair
    # and use its prop_set (averaged across its endpoints if they differ).
    seg_props: list[SubDynCircProp] = []
    for k in range(len(joints_sorted) - 1):
        ja_id = joints_sorted[k].joint_id
        jb_id = joints_sorted[k + 1].joint_id
        member = next(
            (m for m in subdyn.members
             if {m.joint_a, m.joint_b} == {ja_id, jb_id}),
            None,
        )
        if member is None:
            raise ValueError(
                f"SubDyn: no member connects adjacent joints "
                f"{ja_id} (z={joints_sorted[k].z}) and "
                f"{jb_id} (z={joints_sorted[k + 1].z}); the OC3-style "
                f"sequential-pile assumption is violated."
            )
        # OC3 has uniform members (prop_set_a == prop_set_b); take the
        # average if a future deck has tapered members.
        if member.prop_set_a == member.prop_set_b:
            seg_props.append(_circ_prop_for(subdyn, member.prop_set_a))
        else:
            pa = _circ_prop_for(subdyn, member.prop_set_a)
            pb = _circ_prop_for(subdyn, member.prop_set_b)
            seg_props.append(SubDynCircProp(
                prop_set_id=-1,
                E=0.5 * (pa.E + pb.E),
                G=0.5 * (pa.G + pb.G),
                rho=0.5 * (pa.rho + pb.rho),
                D=0.5 * (pa.D + pb.D),
                t=0.5 * (pa.t + pb.t),
            ))

    return z_coords, seg_props


def to_pybmodes_pile_tower(
    main,                       # ElastoDynMain (forward-ref)
    tower,                      # ElastoDynTower
    subdyn: SubDynFile,
    blade=None,                 # ElastoDynBlade — optional, used for RNA mass
):
    """Build a combined pile + tower BMI + SectionProperties for OC3-style
    rigid-base monopiles.

    Layout (axial coordinate ``z`` increases upward)::

        z = z_seabed   --- rigid clamped base (SubDyn reaction joint)
        ...                pile section, properties from SubDyn members
        z = z_TP       --- transition piece (SubDyn interface joint
                            / ElastoDyn ``TowerBsHt``)
        ...                tower section, properties from ElastoDyn tower
        z = z_top      --- tower top, lumped RNA tip mass

    The returned ``BMIFile`` describes the full beam from ``z_seabed`` to
    ``z_top`` as a single cantilever; the ``SectionProperties`` array
    splices the pile and tower property tables at the transition piece
    with two stations very close together so the FE interpolant captures
    the cross-section discontinuity correctly.

    No soil flexibility, no hydrodynamic added mass — the OC3 design
    fixes the pile rigidly at the seabed and the user-selected scope here
    excludes hydro coupling. See the case-study script in
    ``cases/nrel5mw_monopile/run.py`` for context.
    """
    from pybmodes.io.bmi import BMIFile, ScalingFactors
    from pybmodes.io.elastodyn_reader import (
        _stack_tower_section_props,
        _tower_top_assembly_mass,
    )
    from pybmodes.io.sec_props import SectionProperties

    # --- Pile geometry from SubDyn ------------------------------------
    z_pile, seg_props = _pile_axial_stations(subdyn)

    # SubDyn reaction joint sits at z_seabed; interface at z_TP.
    # Bare ``next(...)`` raises StopIteration on missing IDs, which
    # propagates as a cryptic traceback — convert to a clear
    # ValueError that names the missing ID and the source file
    # (matches the adjacent ``_pile_axial_stations`` /
    # ``_circ_prop_for`` error style). Pass-2 review.
    reaction_joint = _lookup_joint(
        subdyn, subdyn.reaction_joint_id, "reaction"
    )
    interface_joint = _lookup_joint(
        subdyn, subdyn.interface_joint_id, "interface"
    )
    z_seabed = float(reaction_joint.z)
    z_tp = float(interface_joint.z)
    if z_tp <= z_seabed:
        raise ValueError(
            f"SubDyn: interface joint z={z_tp} must be above reaction joint "
            f"z={z_seabed}"
        )

    # --- Tower geometry from ElastoDyn --------------------------------
    # ElastoDyn TowerHt is the tower top relative to mean sea level (offshore).
    # TowerBsHt is the tower base; for OC3 it sits at the transition piece
    # which is 10 m above MSL. The combined cantilever length is therefore
    # (TowerHt - z_seabed), measured upward from the seabed.
    z_tower_top = float(main.tower_ht)
    tower_base_z = float(main.tower_bs_ht)

    # Sanity: the SubDyn TP and the ElastoDyn TowerBsHt should agree.
    if abs(tower_base_z - z_tp) > 1.0:
        # Soft check: warn-but-proceed equivalent done by raising. The OC3
        # deck has TowerBsHt = 10 m and SubDyn interface at z = +10 m, so
        # this never trips for the supported case.
        raise ValueError(
            f"ElastoDyn TowerBsHt ({tower_base_z} m) and SubDyn interface "
            f"joint z ({z_tp} m) differ by more than 1 m; this adapter "
            f"assumes they describe the same physical transition piece."
        )

    combined_length = z_tower_top - z_seabed
    pile_length = z_tp - z_seabed
    pile_frac = pile_length / combined_length

    # --- Section properties: pile segments + tower stations -----------
    # Build span_loc in [0, 1] from z_seabed to z_tower_top.
    sp_tower = _stack_tower_section_props(tower)
    tower_z_norm = sp_tower.span_loc          # 0 at TP, 1 at tower top in tower frame
    tower_combined_frac = pile_frac + tower_z_norm * (1.0 - pile_frac)

    # Pile stations: one at each SubDyn joint. Properties are piecewise
    # uniform per segment, so duplicate the segment property at the upper
    # node of each segment to give the FE interpolant a step at the
    # segment join. For OC3 the pile is fully uniform so this collapses
    # to a single (D, t, ρ) value across all pile stations.
    n_pile = len(z_pile)
    pile_combined_frac = (z_pile - z_seabed) / combined_length

    # Per-pile-station properties: take the property of the segment
    # *above* each station (so the bottom station inherits the bottom
    # member's section, and each station thereafter the segment leaving
    # it upward); the topmost pile station inherits from the segment
    # below (it's collocated with the TP).
    pile_props_per_station: list[SubDynCircProp] = []
    for k in range(n_pile - 1):
        pile_props_per_station.append(seg_props[k])
    pile_props_per_station.append(seg_props[-1])

    pile_mass_den = np.array([p.mass_per_length for p in pile_props_per_station])
    pile_EI       = np.array([p.EI               for p in pile_props_per_station])
    pile_GJ       = np.array([p.GJ               for p in pile_props_per_station])
    pile_EA       = np.array([p.EA               for p in pile_props_per_station])

    # SectionProperties uses ``np.interp`` over span_loc to evaluate
    # properties at any spanwise position. To get a sharp discontinuity at
    # the transition piece (where the pile cross-section meets the tower
    # cross-section), the section-property table includes two stations
    # almost-but-not-exactly at the TP: one carrying the pile properties
    # at ``pile_frac - eps`` and one carrying the tower properties at
    # exactly ``pile_frac``. The tiny normalised gap ``eps`` is small
    # enough that interpolation at any sane element midpoint resolves to
    # one side or the other unambiguously.
    eps = 1.0e-9
    pile_combined_frac[-1] = max(pile_combined_frac[-1] - eps, pile_combined_frac[-2] + eps / 2)

    span_loc = np.concatenate([pile_combined_frac, tower_combined_frac])
    mass_den = np.concatenate([pile_mass_den, sp_tower.mass_den])
    flp_stff = np.concatenate([pile_EI, sp_tower.flp_stff])
    edge_stff = np.concatenate([pile_EI, sp_tower.edge_stff])
    tor_stff = np.concatenate([pile_GJ, sp_tower.tor_stff])
    axial_stff = np.concatenate([pile_EA, sp_tower.axial_stff])

    # Rotary-inertia floor — same handling as the land-based path.
    from pybmodes.io.elastodyn_reader import _rotary_inertia_floor
    flp_iner = _rotary_inertia_floor(mass_den, 3.0)
    edge_iner = _rotary_inertia_floor(mass_den, 3.0)

    zeros = np.zeros_like(span_loc)
    sp = SectionProperties(
        title="OC3 monopile pile + ElastoDyn tower (combined cantilever)",
        n_secs=int(span_loc.size),
        span_loc=span_loc,
        str_tw=zeros.copy(),
        tw_iner=zeros.copy(),
        mass_den=mass_den,
        flp_iner=flp_iner,
        edge_iner=edge_iner,
        flp_stff=flp_stff,
        edge_stff=edge_stff,
        tor_stff=tor_stff,
        axial_stff=axial_stff,
        cg_offst=zeros.copy(),
        sc_offst=zeros.copy(),
        tc_offst=zeros.copy(),
        source_file=subdyn.source_file,
    )

    # --- BMI with combined-length cantilever --------------------------
    # FE node distribution is decoupled from the section-property table:
    # the section table has an ε-gap at the TP for the interpolation
    # discontinuity, but the FE mesh places one clean node *at* the TP
    # (no degenerate ε-length element). All FE elements straddle either
    # purely-pile or purely-tower cross-sections; element-midpoint
    # interpolation of the section table gives the right side every time.
    pile_nodes = np.linspace(0.0, pile_frac, n_pile)
    n_tower_elt = max(tower_combined_frac.size - 1, 1)
    tower_nodes = np.linspace(pile_frac, 1.0, n_tower_elt + 1)
    el_loc = np.concatenate([pile_nodes, tower_nodes[1:]])
    n_elements = el_loc.size - 1

    # RNA tip mass at the tower top, same lumping as the land-based case.
    tip = _tower_top_assembly_mass(main, blade)

    bmi = BMIFile(
        title=f"OC3 monopile (mudline z={z_seabed} m, TP z={z_tp} m, top z={z_tower_top} m)",
        echo=False,
        beam_type=2,
        rot_rpm=0.0, rpm_mult=1.0,
        radius=combined_length, hub_rad=0.0,
        precone=0.0, bl_thp=0.0,
        hub_conn=1, n_modes_print=20,
        tab_delim=True, mid_node_tw=False,
        tip_mass=tip,
        id_mat=1, sec_props_file="",
        scaling=ScalingFactors(),
        n_elements=n_elements,
        el_loc=el_loc,
        tow_support=0,
        support=None,
        source_file=None,
    )

    return bmi, sp


__all__ = [
    "SubDynCircProp",
    "SubDynFile",
    "SubDynJoint",
    "SubDynMember",
    "read_subdyn",
    "to_pybmodes_pile_tower",
]
