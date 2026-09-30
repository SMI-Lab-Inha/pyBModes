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

"""Filled-member (flooded ballast) mass properties from a HydroDyn ``.dat``.

Some floating decks keep their water ballast out of ElastoDyn's
``PtfmMass`` and declare it instead as HydroDyn *filled member groups*.
The OC4 DeepCwind semi is the standard example, where ``PtfmMass`` is the
3.852e6 kg of steel and the 9.6e6 kg of ballast water lives in two fill
groups (upper and base columns). OpenFAST adds that ballast through
HydroDyn's Morison loads at run time, so a model assembled from the
ElastoDyn scalars alone is missing most of the platform mass, puts the
centre of gravity 5 m too high and loses its roll and pitch restoring.

HydroDyn treats a closed fill group as body-fixed ballast. The fill mass,
centroid and inertia of each element are precomputed once
(``FloodedBallastPartSegmentCyl`` in ``Morison.f90``) and the internal
pressure datum follows the instantaneous high point of the ballast, so
the net fill load is the weight of a rigid mass at the fill centroid,
with no free-surface correction. This module reproduces those mass
properties so they can be lumped into the platform.

Only the geometry needed for that is parsed, namely the member joints,
the cross-section property sets, the member table and the fill groups.
Both member-table layouts in circulation are accepted (with and without
the ``MSecGeom`` / ``MSpinOrient`` columns), as is the older single
``NPropSets`` property table.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from pybmodes.io.wamit_reader import (
    _is_fortran_default,
    _parse_fortran_float,
)

__all__ = ["FilledBallast", "read_filled_ballast"]

#: Gauss-Legendre order for the along-axis integrals. The integrands are
#: polynomials of degree at most 4 in the axial coordinate (the inner
#: radius tapers linearly), so five points integrate them exactly.
_GAUSS_ORDER = 5


@dataclass
class FilledBallast:
    """Rigid-body mass properties of all filled members in a HydroDyn deck.

    Attributes
    ----------
    mass : float
        Total fill mass, kg.
    cg : np.ndarray
        Centre of mass ``(x, y, z)`` in the HydroDyn global frame (z up
        from MSL), metres.
    inertia_cg : np.ndarray
        3×3 inertia tensor about ``cg`` in kg·m², axes (x, y, z), i.e.
        roll, pitch, yaw. Off-diagonal entries are the negated products
        of inertia, ``I_ij = ∫ (|r|² δ_ij − r_i r_j) dm``.
    member_mass : dict[int, float]
        Fill mass of each filled member, keyed by ``MemberID``, for
        diagnostics.
    """

    mass: float
    cg: np.ndarray
    inertia_cg: np.ndarray
    member_mass: dict[int, float] = field(default_factory=dict)


@dataclass
class _Member:
    member_id: int
    joint1: int
    joint2: int
    prop1: int
    prop2: int
    sec_geom: int


def _tokens(line: str) -> list[str]:
    return line.split("!", 1)[0].split()


def _count_after(lines: list[str], label: str) -> int | None:
    """Index of the line whose second token is ``label``, or ``None``."""
    for i, raw in enumerate(lines):
        parts = raw.split()
        if len(parts) >= 2 and parts[1] == label:
            return i
    return None


def _table(
    lines: list[str], label: str, path: pathlib.Path,
) -> tuple[list[str], list[list[str]]]:
    """Return ``(header_tokens, rows)`` for the table counted by ``label``.

    HydroDyn tables are a count line, a column-name line, a units line
    and then ``count`` data rows. A missing label means an empty table.
    """
    i = _count_after(lines, label)
    if i is None:
        return [], []
    count_tok = lines[i].split()[0]
    try:
        n = int(count_tok)
    except ValueError as err:
        raise ValueError(
            f"{label} in {path} is not an integer: {count_tok!r}"
        ) from err
    if n < 0:
        raise ValueError(f"{label} in {path} is negative: {n}")
    header = _tokens(lines[i + 1]) if i + 1 < len(lines) else []
    rows = []
    for j in range(i + 3, i + 3 + n):
        if j >= len(lines):
            raise ValueError(
                f"{path} ends inside the {label} table "
                f"(expected {n} rows)"
            )
        rows.append(_tokens(lines[j]))
    return header, rows


def _float(tok: str, what: str, path: pathlib.Path) -> float:
    """Parse one deck scalar, rejecting anything non-finite.

    A NaN would slip past every later range check (comparisons with NaN
    are false) and reach the platform mass matrix, so finiteness is
    checked here rather than left to the shared parser.
    """
    try:
        value = _parse_fortran_float(tok)
    except ValueError as err:
        raise ValueError(f"Malformed {what} in {path}: {tok!r}") from err
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {what} in {path}: {tok!r}")
    return value


def _int(tok: str, what: str, path: pathlib.Path) -> int:
    try:
        return int(tok)
    except ValueError as err:
        raise ValueError(f"Malformed {what} in {path}: {tok!r}") from err


def _water_density(lines: list[str], path: pathlib.Path) -> float:
    """``WtrDens`` from the deck, or the paired SeaState file default."""
    i = _count_after(lines, "WtrDens")
    if i is not None:
        tok = lines[i].split()[0]
        if not _is_fortran_default(tok):
            return _float(tok, "WtrDens", path)
    return 1025.0


def read_filled_ballast(
    dat_path: str | pathlib.Path,
    *,
    water_density: float | None = None,
) -> FilledBallast | None:
    """Mass properties of the filled members declared in a HydroDyn deck.

    Parameters
    ----------
    dat_path
        HydroDyn ``.dat`` file.
    water_density
        Density used for fill groups whose ``FillDens`` is ``DEFAULT``.
        When omitted, the deck's own ``WtrDens`` is used, falling back to
        1025 kg/m³ (the SeaState default) for decks that delegate it.

    Returns
    -------
    FilledBallast or None
        ``None`` when the deck declares no fill groups or none of them
        holds any fluid.

    Each filled member is a straight frustum whose outer diameter and
    wall thickness vary linearly between its two property sets, as in
    HydroDyn. The fill runs from the lower end up to ``FillFSLoc``
    (metres, relative to MSL), measured along the member axis exactly as
    HydroDyn does. Rectangular members are rejected with a clear error
    rather than silently skipped.
    """
    path = pathlib.Path(dat_path)
    if not path.is_file():
        raise FileNotFoundError(f"HydroDyn .dat not found at {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    _, fill_rows = _table(lines, "NFillGroups", path)
    if not fill_rows:
        return None

    if water_density is None:
        water_density = _water_density(lines, path)
    if not np.isfinite(water_density) or water_density <= 0.0:
        raise ValueError(
            f"water_density must be finite and positive, got {water_density!r}"
        )

    _, joint_rows = _table(lines, "NJoints", path)
    joints: dict[int, np.ndarray] = {}
    for row in joint_rows:
        if len(row) < 4:
            raise ValueError(f"Short joint row in {path}: {row!r}")
        jid = _int(row[0], "JointID", path)
        joints[jid] = np.array(
            [_float(t, f"joint {jid} coordinate", path) for t in row[1:4]]
        )

    props: dict[int, tuple[float, float]] = {}
    label = "NPropSetsCyl" if _count_after(lines, "NPropSetsCyl") is not None \
        else "NPropSets"
    for row in _table(lines, label, path)[1]:
        if len(row) < 3:
            raise ValueError(f"Short property-set row in {path}: {row!r}")
        pid = _int(row[0], "PropSetID", path)
        diam = _float(row[1], f"PropD of set {pid}", path)
        thick = _float(row[2], f"PropThck of set {pid}", path)
        if diam <= 0.0 or thick < 0.0 or 2.0 * thick > diam:
            raise ValueError(
                f"Property set {pid} in {path} has a non-physical section "
                f"(PropD = {diam}, PropThck = {thick})"
            )
        props[pid] = (diam, thick)

    header, member_rows = _table(lines, "NMembers", path)
    cols = {name: k for k, name in enumerate(header)}
    for need in ("MemberID", "MJointID1", "MJointID2",
                 "MPropSetID1", "MPropSetID2"):
        if need not in cols:
            raise ValueError(
                f"Member table in {path} has no {need} column; "
                f"header was {header!r}"
            )
    members: dict[int, _Member] = {}
    for row in member_rows:
        if len(row) <= max(cols[c] for c in ("MemberID", "MJointID1",
                                             "MJointID2", "MPropSetID1",
                                             "MPropSetID2")):
            raise ValueError(f"Short member row in {path}: {row!r}")
        mid = _int(row[cols["MemberID"]], "MemberID", path)
        sec_geom = 1
        if "MSecGeom" in cols:
            sec_geom = _int(row[cols["MSecGeom"]], "MSecGeom", path)
        members[mid] = _Member(
            member_id=mid,
            joint1=_int(row[cols["MJointID1"]], "MJointID1", path),
            joint2=_int(row[cols["MJointID2"]], "MJointID2", path),
            prop1=_int(row[cols["MPropSetID1"]], "MPropSetID1", path),
            prop2=_int(row[cols["MPropSetID2"]], "MPropSetID2", path),
            sec_geom=sec_geom,
        )

    xg, wg = np.polynomial.legendre.leggauss(_GAUSS_ORDER)
    mass = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))     # ∫ (|r|² I − r rᵀ) dm about the origin
    member_mass: dict[int, float] = {}

    for row in fill_rows:
        n_mem = _int(row[0], "FillNumM", path)
        if n_mem < 1 or len(row) < n_mem + 3:
            raise ValueError(f"Malformed fill-group row in {path}: {row!r}")
        ids = [_int(t, "FillMList entry", path) for t in row[1:1 + n_mem]]
        fs_z = _float(row[1 + n_mem], "FillFSLoc", path)
        dens_tok = row[2 + n_mem]
        dens = water_density if _is_fortran_default(dens_tok) \
            else _float(dens_tok, "FillDens", path)
        if dens < 0.0:
            raise ValueError(f"Negative FillDens in {path}: {dens}")

        for mid in ids:
            if mid not in members:
                raise ValueError(
                    f"Fill group in {path} names member {mid}, which is not "
                    f"in the member table"
                )
            mem = members[mid]
            if mem.sec_geom != 1:
                raise NotImplementedError(
                    f"Filled member {mid} in {path} is rectangular "
                    f"(MSecGeom = {mem.sec_geom}); only cylindrical filled "
                    f"members are supported"
                )
            for jid in (mem.joint1, mem.joint2):
                if jid not in joints:
                    raise ValueError(
                        f"Member {mid} in {path} uses undefined joint {jid}"
                    )
            for pid in (mem.prop1, mem.prop2):
                if pid not in props:
                    raise ValueError(
                        f"Member {mid} in {path} uses undefined property "
                        f"set {pid}"
                    )

            # Orient from the lower end, as HydroDyn does (Za <= Zb).
            p_a, p_b = joints[mem.joint1], joints[mem.joint2]
            (d_a, t_a), (d_b, t_b) = props[mem.prop1], props[mem.prop2]
            if p_b[2] < p_a[2]:
                p_a, p_b = p_b, p_a
                d_a, t_a, d_b, t_b = d_b, t_b, d_a, t_a
            axis = p_b - p_a
            length = float(np.linalg.norm(axis))
            if length <= 0.0:
                raise ValueError(f"Member {mid} in {path} has zero length")
            e = axis / length
            rin_a = 0.5 * d_a - t_a
            rin_b = 0.5 * d_b - t_b

            if fs_z >= p_b[2]:
                l_fill = length
            elif p_a[2] >= fs_z:
                l_fill = 0.0
            else:
                # Only reached when p_b[2] > fs_z > p_a[2], so e[2] > 0.
                l_fill = (fs_z - p_a[2]) / e[2]
            if l_fill <= 0.0 or dens == 0.0:
                member_mass[mid] = member_mass.get(mid, 0.0)
                continue

            s = 0.5 * l_fill * (xg + 1.0)
            w = 0.5 * l_fill * wg
            rin = rin_a + (rin_b - rin_a) * s / length
            dm = dens * np.pi * rin ** 2 * w          # mass per Gauss slice
            pos = p_a[None, :] + s[:, None] * e[None, :]

            m_mem = float(dm.sum())
            mass += m_mem
            member_mass[mid] = member_mass.get(mid, 0.0) + m_mem
            first += dm @ pos
            # Point-mass part plus each slice's own disc inertia,
            # (r²/4)(I + e eᵀ) per unit mass for a thin disc of radius r
            # normal to e. The slices' axial extent is recovered exactly
            # by the quadrature through the point-mass part.
            r2 = np.einsum("ki,ki->k", pos, pos)
            second += (dm * r2).sum() * np.eye(3) - np.einsum(
                "k,ki,kj->ij", dm, pos, pos,
            )
            second += float((dm * rin ** 2 / 4.0).sum()) * (
                np.eye(3) + np.outer(e, e)
            )

    if mass <= 0.0:
        return None
    cg = first / mass
    inertia_cg = second - mass * (float(cg @ cg) * np.eye(3) - np.outer(cg, cg))
    return FilledBallast(
        mass=mass, cg=cg, inertia_cg=inertia_cg, member_mass=member_mass,
    )
