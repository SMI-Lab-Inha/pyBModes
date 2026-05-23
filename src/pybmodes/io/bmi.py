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

"""Parser for .bmi main input files.

The .bmi format is line-ordered: values precede their labels. The reader
follows the format conventions::

    ReadCom  -> consume one line verbatim (section headers / blank lines)
    ReadStr  -> consume one line, return it as a string
    ReadVar  -> skip blanks, return first whitespace token of next non-blank line
    ReadAry  -> skip blanks, return first N tokens of next non-blank line
"""

from __future__ import annotations

import math
import pathlib
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class TipMassProps:
    """Mass and inertia of the blade tip or tower-top concentrated mass.

    Lumps the rotor-nacelle assembly (RNA) at the tower top, or a tip
    mass at the blade tip. Offsets are in the span-end **section frame**:
    ``z`` along the span (positive toward / beyond the top), the
    transverse axis aligned with the lateral/sway direction. See
    :doc:`/conventions`.

    Attributes
    ----------
    mass : float
        Concentrated mass, kg. Omitting the RNA (mass = 0) on a tower
        makes the 1st fore-aft frequency ~10–30 % too high.
    cm_offset : float
        CM offset **transverse** to the span (along the tip-section axis
        aligned with lateral/sway), metres.
    cm_axial : float
        CM offset **along the span (z)**, metres — how far the lumped CM
        sits beyond the span end (e.g. the RNA CM above the tower top).
    ixx, iyy, izz, ixy, izx, iyz : float
        Mass moments / products of inertia about the CM, kg·m².
    """

    mass: float        # kg
    cm_offset: float   # m  (transverse, along tip-section y-axis)
    cm_axial: float    # m  (axial offset along z)
    ixx: float         # kg*m^2
    iyy: float
    izz: float
    ixy: float
    izx: float
    iyz: float


@dataclass
class ScalingFactors:
    """Multiplicative scaling factors applied to all section properties."""

    sec_mass: float = 1.0
    flp_iner: float = 1.0
    lag_iner: float = 1.0
    flp_stff: float = 1.0
    edge_stff: float = 1.0
    tor_stff: float = 1.0
    axial_stff: float = 1.0
    cg_offst: float = 1.0
    sc_offst: float = 1.0
    tc_offst: float = 1.0


@dataclass
class TensionWireSupport:
    """Tension wire (guy wire) support for land-based towers."""

    n_attachments: int
    n_wires: list[int]
    node_attach: list[int]
    wire_stiffness: list[float]  # N/m, one per attachment set
    th_wire: list[float]         # deg, angle with respect to tower axis


@dataclass
class PlatformSupport:
    """Offshore floating-platform or monopile support.

    All quantities follow the conventions in
    :doc:`/conventions` — read that page if anything below is unclear.
    The **single origin is the tower base**; the **vertical datum is
    mean sea level (MSL, z = 0)**; the 6×6 matrices use **OpenFAST DOF
    order** (surge, sway, heave, roll, pitch, yaw — see
    :mod:`pybmodes.coords`).

    Attributes
    ----------
    draft : float
        Signed elevation of the **tower base** relative to MSL, in
        metres, **negative = above MSL** (a tower base 15 m above the
        waterline is ``draft = -15``).

        .. note::

           Despite the name, this is **not** the naval-architecture
           draft (the keel depth below the waterline). The field name is
           inherited verbatim from the BModes ``.bmi`` platform block;
           here it means "tower-base z relative to MSL, negative up".
           pyBmodes forms the CM→tower-base vertical lever internally as
           ``cm_pform - draft``, so the base elevation enters the maths
           exactly once — do **not** also add it to ``cm_pform`` /
           ``ref_msl``.
    cm_pform : float
        Platform centre-of-mass depth **below MSL**, metres, positive
        downward. The vertical reference of ``i_matrix``.
    mass_pform : float
        Platform (substructure) mass, kg.
    i_matrix : np.ndarray
        Platform 6×6 rigid-body inertia (kg, kg·m²) about the platform
        CM, in OpenFAST DOF order. Transferred to the tower base by the
        rigid-arm transform using ``cm_pform`` (vertical) and
        ``cm_pform_x`` / ``cm_pform_y`` (horizontal).
    ref_msl : float
        Depth **below MSL** of the reference point for the hydro and
        mooring matrices, metres, positive downward (the WAMIT/HydroDyn
        ``PtfmRefzt``; usually 0, i.e. at the waterline). This reference
        is assumed to lie **on the tower axis** (``PtfmRefxt =
        PtfmRefyt = 0``); ``hydro_*`` / ``mooring_K`` receive **no**
        horizontal arm (see ``cm_pform_x``).
    hydro_M : np.ndarray
        Infinite-frequency hydrodynamic added mass ``A_inf``, 6×6, about
        the ``ref_msl`` reference, OpenFAST DOF order.
    hydro_K : np.ndarray
        Hydrostatic restoring ``C_hst``, 6×6, about the ``ref_msl``
        reference. May be (legitimately) negative-definite in
        roll/pitch for an unstable spar stabilised by mooring.
    mooring_K : np.ndarray
        Linearised mooring stiffness, 6×6, about the ``ref_msl``
        reference, OpenFAST DOF order.
    distr_m_z, distr_m : np.ndarray
        Optional distributed added mass vs depth (Morison), if used.
    distr_k_z, distr_k : np.ndarray
        Optional distributed Winkler soil/foundation stiffness vs depth
        (the ``hub_conn = 3`` soft-monopile path).
    wires : TensionWireSupport | None
        Optional guy-wire support.
    cm_pform_x, cm_pform_y : float
        Horizontal offset of the platform CM **from the tower axis**
        (see the inline note below).
    """

    draft: float
    cm_pform: float
    mass_pform: float
    i_matrix: np.ndarray
    ref_msl: float
    hydro_M: np.ndarray
    hydro_K: np.ndarray
    mooring_K: np.ndarray
    distr_m_z: np.ndarray
    distr_m: np.ndarray
    distr_k_z: np.ndarray
    distr_k: np.ndarray
    wires: TensionWireSupport | None = None
    # Horizontal offset of the platform centre of mass from the TOWER
    # AXIS, in the tower-base frame (x = downwind / surge-aligned,
    # y = lateral / sway-aligned), metres. This is a *local* offset of
    # the CM relative to the tower centreline — NOT a position in any
    # global / OpenFAST / WAMIT coordinate frame. For a standard floater
    # the tower sits on the platform centroid, so these are ≈ 0 (every
    # axisymmetric spar / symmetric semi: exactly 0). A value comparable
    # to the platform's own size (its yaw radius of gyration) is almost
    # always a coordinate-origin offset leaking in; it injects spurious
    # surge/sway↔yaw coupling and mislabels the rigid-body modes — the
    # ``check_model`` platform-CM-offset gate warns on it (issue #95).
    # Both default to 0.0 so every existing deck / sample is byte-
    # identical. Consumed by ``pybmodes.fem.nondim.nondim_platform`` as
    # the horizontal components of the CM → tower-base rigid arm (the
    # vertical component stays ``cm_pform``). Added 1.2.0.
    cm_pform_x: float = 0.0
    cm_pform_y: float = 0.0


@dataclass
class BMIFile:
    """All parameters parsed from a .bmi main input file.

    See :doc:`/conventions` for the coordinate system, origin (tower
    base) and boundary-condition definitions.

    Key fields
    ----------
    beam_type : int
        ``1`` = blade (rotating), ``2`` = tower.
    radius : float
        Span length of the finite-element beam, metres — the blade
        ``radius`` or the flexible tower length (``TowerHt -
        TowerBsHt``). The beam runs from base (``z = 0``) to this length;
        the *absolute* base elevation is conveyed separately (via the
        ``PlatformSupport.draft`` for a floater).
    hub_rad : float
        Hub radius (blade root radial offset from the rotation axis), m.
    hub_conn : int
        Base boundary condition: ``1`` cantilever (clamped), ``2``
        free-free floating (reactions from ``support``), ``3`` soft
        monopile (lateral + rocking free), ``4`` pinned-free (bending
        slopes free). See the ``hub_conn`` table in :doc:`/conventions`.
    rot_rpm, precone : float
        Rotor speed (rpm) and coning angle (deg) for a rotating blade.
    tip_mass : TipMassProps
        Tower-top RNA / blade-tip lumped mass.
    tow_support : int
        ``0`` none / wires-only, ``1`` inline platform block, normalised
        to a ``PlatformSupport`` by the parser.
    support : TensionWireSupport | PlatformSupport | None
        The offshore platform or guy-wire support, when present.
    """

    title: str
    echo: bool
    beam_type: int
    rot_rpm: float
    rpm_mult: float
    radius: float
    hub_rad: float
    precone: float
    bl_thp: float
    hub_conn: int
    n_modes_print: int
    tab_delim: bool
    mid_node_tw: bool
    tip_mass: TipMassProps
    id_mat: int
    sec_props_file: str
    scaling: ScalingFactors
    n_elements: int
    el_loc: np.ndarray
    tow_support: int = 0
    support: TensionWireSupport | PlatformSupport | None = None
    source_file: pathlib.Path | None = None

    def resolve_sec_props_path(self) -> pathlib.Path:
        """Return absolute path to the section properties file."""
        p = pathlib.Path(self.sec_props_file)
        if p.is_absolute():
            return p
        if self.source_file is not None:
            return (self.source_file.parent / p).resolve()
        return p.resolve()


@dataclass
class _GeneralParams:
    """Common top-level BMI fields parsed from the general-parameters section."""

    echo: bool
    beam_type: int
    rot_rpm: float
    rpm_mult: float
    radius: float
    hub_rad: float
    precone: float
    bl_thp: float
    hub_conn: int
    n_modes_print: int
    tab_delim: bool
    mid_node_tw: bool


@dataclass
class _SectionPropsRef:
    """Identifiers pointing to the distributed section-properties file."""

    id_mat: int
    sec_props_file: str


@dataclass
class _Discretization:
    """Finite-element discretization settings."""

    n_elements: int
    el_loc: np.ndarray


class _LineReader:
    """Stateful line-by-line reader following .bmi file format conventions."""

    def __init__(self, lines: list[str]):
        processed = []
        for raw in lines:
            line = raw.rstrip("\r\n")
            bang = _find_comment_start(line)
            if bang >= 0:
                line = line[:bang]
            processed.append(line)
        self._lines = processed
        self._pos = 0

    def read_com(self) -> None:
        """Advance one line verbatim, including blanks."""
        self._pos += 1

    def read_str(self) -> str:
        """Return the stripped contents of the next line."""
        line = self._lines[self._pos]
        self._pos += 1
        return line.strip()

    def read_var(self) -> str:
        """Return the first token from the next non-blank line."""
        self._skip_blanks()
        line = self._lines[self._pos]
        self._pos += 1
        tokens = line.split()
        if not tokens:
            raise ValueError(f"Empty data line at position {self._pos}")
        return tokens[0]

    def read_ary(self, n: int) -> list[str]:
        """Return the first ``n`` tokens from the next non-blank line.

        Raises ``ValueError`` if the line has fewer than ``n`` tokens.
        The previous behaviour was to silently truncate to whatever
        was present, which deferred the failure to downstream
        ``[_parse_float(t) for t in ...]``-style broadcasts that
        produced shape errors with no contextual path / row info.
        """
        self._skip_blanks()
        line = self._lines[self._pos]
        self._pos += 1
        tokens = line.split()
        if len(tokens) < n:
            raise ValueError(
                f"Truncated array at BMI line {self._pos}: expected "
                f"{n} tokens, got {len(tokens)}: {line.strip()!r}. "
                f"The most common cause is a wrapped line or a "
                f"missing element-count scalar (n_elements / n_att / "
                f"matrix dimension) earlier in the deck."
            )
        return tokens[:n]

    def peek_token(self) -> str:
        """Return the first token of the next non-blank line without advancing."""
        pos = self._pos
        while pos < len(self._lines) and not self._lines[pos].strip():
            pos += 1
        if pos >= len(self._lines):
            return ""
        tokens = self._lines[pos].split()
        return tokens[0] if tokens else ""

    def read_line_tokens(self) -> list[str]:
        """Skip blanks and return ALL whitespace-split tokens of the
        next line, advancing one line.

        ``read_var`` returns only the first token and ``read_ary``
        requires an exact count; this is for the few fields whose line
        carries a *variable* leading numeric run followed by a label
        (the optional same-line horizontal platform-CM extension —
        ``<cm_pform> [<x> <y>] cm_pform : <comment>``)."""
        self._skip_blanks()
        line = self._lines[self._pos]
        self._pos += 1
        return line.split()

    def _skip_blanks(self) -> None:
        while self._pos < len(self._lines) and not self._lines[self._pos].strip():
            self._pos += 1
        if self._pos >= len(self._lines):
            raise EOFError("Unexpected end of input file")


def _find_comment_start(line: str) -> int:
    """Return index of the first ! not inside a quoted string, or -1."""
    in_sq = False
    in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "!" and not in_sq and not in_dq:
            return i
    return -1


def _leading_floats(tokens: list[str]) -> list[float]:
    """Return the leading run of finite-float tokens, stopping at the
    first token that is not one (e.g. the inline ``cm_pform`` label
    word).

    Backs the backward-compatible same-line horizontal platform-CM
    extension: a legacy line ``<cm_pform>  cm_pform : <comment>`` has a
    single leading number (the label word stops the run → x = y = 0);
    an asymmetric deck writes ``<cm_pform> <x> <y>  cm_pform : …`` and
    yields three. Zero new lines, so every pre-1.2.1 deck parses
    identically. Added 1.2.1.
    """
    out: list[float] = []
    for tok in tokens:
        try:
            out.append(_parse_float(tok))
        except ValueError:
            break
    return out


def _parse_bool(token: str) -> bool:
    t = token.strip().lower().strip("'\"")
    if t in ("t", "true"):
        return True
    if t in ("f", "false"):
        return False
    raise ValueError(f"Cannot parse boolean from: {token!r}")


def _parse_float(token: str) -> float:
    value = float(token.strip().strip("'\"").replace("d", "e").replace("D", "E"))
    # Reject NaN / ±Inf — a stray ``nan`` or ``inf`` literal in a BMI
    # numeric field silently produces a non-physical model whose
    # eigensolve returns NaN frequencies.
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite float in BMI deck: {token!r} parses to "
            f"{value!r}. Physical quantities must be finite; a stray "
            f"``nan`` / ``inf`` is almost certainly a transcription "
            f"error in the upstream deck."
        )
    return value


def _parse_int(token: str) -> int:
    return int(token.strip().strip("'\""))


def _parse_str(token: str) -> str:
    return token.strip().strip("'\"")


def _is_float(token: str) -> bool:
    """Return True if the token can be parsed as a float."""
    try:
        _parse_float(token)
        return True
    except (ValueError, AttributeError):
        return False


def read_bmi(path: str | pathlib.Path) -> BMIFile:
    """Parse a .bmi main input file and return a :class:`BMIFile`."""
    path = pathlib.Path(path)
    lines = path.read_text(encoding="latin-1").splitlines()
    reader = _LineReader(lines)
    return _parse(reader, source_file=path)


def _parse_header(r: _LineReader) -> str:
    """Parse the file header and return the user title."""
    r.read_com()
    return r.read_str()


def _parse_general_params(r: _LineReader) -> _GeneralParams:
    """Parse the common general-parameters section."""
    r.read_com()
    r.read_com()
    return _GeneralParams(
        echo=_parse_bool(r.read_var()),
        beam_type=_parse_int(r.read_var()),
        rot_rpm=_parse_float(r.read_var()),
        rpm_mult=_parse_float(r.read_var()),
        radius=_parse_float(r.read_var()),
        hub_rad=_parse_float(r.read_var()),
        precone=_parse_float(r.read_var()),
        bl_thp=_parse_float(r.read_var()),
        hub_conn=_parse_int(r.read_var()),
        n_modes_print=_parse_int(r.read_var()),
        tab_delim=_parse_bool(r.read_var()),
        mid_node_tw=_parse_bool(r.read_var()),
    )


def _parse_tip_mass(r: _LineReader) -> TipMassProps:
    """Parse the blade-tip / tower-top concentrated-mass section."""
    r.read_com()
    r.read_com()
    return TipMassProps(
        mass=_parse_float(r.read_var()),
        cm_offset=_parse_float(r.read_var()),
        cm_axial=_parse_float(r.read_var()),
        ixx=_parse_float(r.read_var()),
        iyy=_parse_float(r.read_var()),
        izz=_parse_float(r.read_var()),
        ixy=_parse_float(r.read_var()),
        izx=_parse_float(r.read_var()),
        iyz=_parse_float(r.read_var()),
    )


def _parse_section_props_ref(r: _LineReader) -> _SectionPropsRef:
    """Parse the distributed-property identifier section."""
    r.read_com()
    r.read_com()
    # Rewrite Windows-style backslashes in ``sec_props_file`` to forward
    # slashes so a BMI authored on Windows with ``subdir\props.dat``
    # resolves correctly when consumed on Linux / macOS. Matches the
    # equivalent normalisation in
    # :func:`pybmodes.io._elastodyn.parser._normalise_subfile_path` for
    # ElastoDyn ``TwrFile`` / ``BldFile`` paths. ``pathlib.Path`` treats
    # backslash as a literal character on POSIX, so the unaltered string
    # resolves to a non-existent ``subdir\props.dat`` file rather than
    # ``subdir/props.dat``.
    return _SectionPropsRef(
        id_mat=_parse_int(r.read_var()),
        sec_props_file=_parse_str(r.read_var()).replace("\\", "/"),
    )


def _parse_scaling(r: _LineReader) -> ScalingFactors:
    """Parse property scaling factors."""
    r.read_com()
    r.read_com()
    return ScalingFactors(
        sec_mass=_parse_float(r.read_var()),
        flp_iner=_parse_float(r.read_var()),
        lag_iner=_parse_float(r.read_var()),
        flp_stff=_parse_float(r.read_var()),
        edge_stff=_parse_float(r.read_var()),
        tor_stff=_parse_float(r.read_var()),
        axial_stff=_parse_float(r.read_var()),
        cg_offst=_parse_float(r.read_var()),
        sc_offst=_parse_float(r.read_var()),
        tc_offst=_parse_float(r.read_var()),
    )


def _parse_discretization(r: _LineReader) -> _Discretization:
    """Parse the finite-element discretization section."""
    r.read_com()
    r.read_com()
    n_elements = _parse_int(r.read_var())
    r.read_com()
    return _Discretization(
        n_elements=n_elements,
        el_loc=np.array([_parse_float(t) for t in r.read_ary(n_elements + 1)]),
    )


def _parse(r: _LineReader, source_file: pathlib.Path | None = None) -> BMIFile:
    title = _parse_header(r)
    general = _parse_general_params(r)
    tip_mass = _parse_tip_mass(r)
    section_props = _parse_section_props_ref(r)
    scaling = _parse_scaling(r)
    discretization = _parse_discretization(r)

    tow_support = 0
    support: TensionWireSupport | PlatformSupport | None = None
    if general.beam_type == 2:
        r.read_com()
        r.read_com()
        tow_support, support = _parse_tower_support(r)

    return BMIFile(
        title=title,
        echo=general.echo,
        beam_type=general.beam_type,
        rot_rpm=general.rot_rpm,
        rpm_mult=general.rpm_mult,
        radius=general.radius,
        hub_rad=general.hub_rad,
        precone=general.precone,
        bl_thp=general.bl_thp,
        hub_conn=general.hub_conn,
        n_modes_print=general.n_modes_print,
        tab_delim=general.tab_delim,
        mid_node_tw=general.mid_node_tw,
        tip_mass=tip_mass,
        id_mat=section_props.id_mat,
        sec_props_file=section_props.sec_props_file,
        scaling=scaling,
        n_elements=discretization.n_elements,
        el_loc=discretization.el_loc,
        tow_support=tow_support,
        support=support,
        source_file=source_file,
    )


def _parse_tower_support(
    r: _LineReader,
) -> tuple[int, TensionWireSupport | PlatformSupport | None]:
    """Parse and normalize the tower-support section.

    Internal support codes are:
      0 -> none
      1 -> tension wires
      2 -> offshore platform/monopile
    """
    file_support_code = _parse_int(r.read_var())

    if file_support_code == 0:
        return 0, None
    if file_support_code == 1:
        support_format = _detect_tower_support_format(r)
        if support_format == "wires":
            return 1, _parse_tension_wires(r)
        return 2, _parse_platform_extended(r)
    if file_support_code == 2:
        return 2, _parse_platform_legacy(r)

    raise ValueError(f"Unsupported tower support code: {file_support_code}")


def _detect_tower_support_format(r: _LineReader) -> str:
    """Detect which reader to use for a ``tow_support == 1`` block.

    Two file dialects share the ``1`` code for tower support: one stores
    land-based tension wires, the other stores an offshore platform inline
    (with the platform code remapped from 2 to 1).  The next token tells us
    which dialect applies:

    - numeric token: the platform block starts with ``draft``
    - text token: the tension-wire block starts with its label line
    """
    return "platform_extended" if _is_float(r.peek_token()) else "wires"


def _parse_tension_wires(r: _LineReader) -> TensionWireSupport:
    """Parse the standalone tension-wire support block."""
    r.read_com()
    n_att = _parse_int(r.read_var())
    return TensionWireSupport(
        n_attachments=n_att,
        n_wires=[_parse_int(t) for t in r.read_ary(n_att)],
        node_attach=[_parse_int(t) for t in r.read_ary(n_att)],
        wire_stiffness=[_parse_float(t) for t in r.read_ary(n_att)],
        th_wire=[_parse_float(t) for t in r.read_ary(n_att)],
    )


def _read_square_matrix(r: _LineReader, size: int) -> np.ndarray:
    """Read a dense square matrix stored one row per line."""
    matrix = np.zeros((size, size))
    for row in range(size):
        matrix[row, :] = [_parse_float(t) for t in r.read_ary(size)]
    return matrix


def _read_optional_row_array_pair(r: _LineReader) -> tuple[np.ndarray, np.ndarray]:
    """Read a counted pair of row arrays used for distributed support data."""
    n_vals = _parse_int(r.read_var())
    if n_vals > 0:
        z_vals = np.array([_parse_float(t) for t in r.read_ary(n_vals)])
        data_vals = np.array([_parse_float(t) for t in r.read_ary(n_vals)])
        return z_vals, data_vals

    # When the count is zero, optional placeholder rows are skipped and no
    # distributed data are activated.
    if _is_float(r.peek_token()):
        r.read_com()
        if _is_float(r.peek_token()):
            r.read_com()
    return np.array([]), np.array([])


def _read_platform_common_tail(
    r: _LineReader,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Read the shared hydrodynamic/mooring/distributed-data tail of a platform block."""
    ref_msl = _parse_float(r.read_var())

    r.read_com()
    hydro_M = _read_square_matrix(r, 6)

    r.read_com()
    hydro_K = _read_square_matrix(r, 6)

    r.read_com()
    mooring_K = _read_square_matrix(r, 6)

    r.read_com()
    r.read_com()
    z_distr_m, distr_m = _read_optional_row_array_pair(r)

    if distr_m.size > 0:
        warnings.warn(
            "Distributed hydrodynamic added mass (distr_m) is parsed but not "
            "yet wired into the FEM mass matrix; this input is currently "
            "ignored by the modal solver. Track the issue at "
            "https://github.com/SMI-Lab-Inha/pyBModes/issues",
            UserWarning,
            stacklevel=4,
        )

    r.read_com()
    r.read_com()
    z_distr_k, distr_k = _read_optional_row_array_pair(r)

    return ref_msl, hydro_M, hydro_K, mooring_K, z_distr_m, distr_m, z_distr_k, distr_k


def _read_platform_inertia(r: _LineReader, mass_pform: float) -> np.ndarray:
    """Read an offshore platform inertia block into a 6x6 structural mass matrix.

    The legacy and extended-platform ``tow_support`` dialects encode this
    block identically — translational ``mass_pform`` on the diagonal plus a
    3x3 rotational-inertia sub-block — so both share this reader.
    """
    i_mat = np.zeros((6, 6))
    for i in range(3):
        i_mat[i, i] = mass_pform

    r.read_com()
    i_mat[3:6, 3:6] = _read_square_matrix(r, 3)
    return i_mat


def _read_cm_pform_line(r: _LineReader) -> tuple[float, float, float]:
    """Read the ``cm_pform`` line, returning ``(cm_pform,
    cm_pform_x, cm_pform_y)``.

    Legacy form ``<cm_pform>  cm_pform : <comment>`` → the label word
    stops the leading-float run, so ``x = y = 0`` (every pre-1.2.1
    deck). Asymmetric form ``<cm_pform> <x> <y>  cm_pform : <comment>``
    → all three. See :func:`_leading_floats`.

    The horizontal offsets are an (x, y) **pair**: the leading numeric
    run must be exactly 1 (symmetric) or 3 (asymmetric). A run of 2
    (one offset omitted) or ≥ 4 is a malformed hand-authored line —
    silently defaulting the missing coordinate to 0.0 would produce a
    plausible but wrong platform geometry instead of an input error,
    so we raise.
    """
    nums = _leading_floats(r.read_line_tokens())
    if len(nums) not in (1, 3):
        raise ValueError(
            "Malformed BMI platform block: the cm_pform line must have "
            "exactly 1 leading number (symmetric: <cm_pform>) or 3 "
            "(asymmetric: <cm_pform> <cm_pform_x> <cm_pform_y>); got "
            f"{len(nums)}: {nums!r}. The horizontal CM offsets are an "
            "(x, y) pair — supply both or neither."
        )
    cm_pform = nums[0]
    cm_pform_x = nums[1] if len(nums) == 3 else 0.0
    cm_pform_y = nums[2] if len(nums) == 3 else 0.0
    return cm_pform, cm_pform_x, cm_pform_y


def _parse_platform_legacy(r: _LineReader) -> PlatformSupport:
    """Parse the legacy offshore platform block (`tow_support == 2`)."""
    draft = _parse_float(r.read_var())
    cm_pform, cm_pform_x, cm_pform_y = _read_cm_pform_line(r)
    mass_pform = _parse_float(r.read_var())

    i_mat = _read_platform_inertia(r, mass_pform)
    (
        ref_msl,
        hydro_M,
        hydro_K,
        mooring_K,
        z_distr_m,
        distr_m,
        z_distr_k,
        distr_k,
    ) = _read_platform_common_tail(r)

    return PlatformSupport(
        draft=draft,
        cm_pform=cm_pform,
        mass_pform=mass_pform,
        i_matrix=i_mat,
        ref_msl=ref_msl,
        hydro_M=hydro_M,
        hydro_K=hydro_K,
        mooring_K=mooring_K,
        distr_m_z=z_distr_m,
        distr_m=distr_m,
        distr_k_z=z_distr_k,
        distr_k=distr_k,
        cm_pform_x=cm_pform_x,
        cm_pform_y=cm_pform_y,
    )


def _parse_platform_extended(r: _LineReader) -> PlatformSupport:
    """Parse the extended-platform offshore block stored under `tow_support == 1`."""
    draft = _parse_float(r.read_var())
    cm_pform, cm_pform_x, cm_pform_y = _read_cm_pform_line(r)
    mass_pform = _parse_float(r.read_var())

    i_mat = _read_platform_inertia(r, mass_pform)
    (
        ref_msl,
        hydro_M,
        hydro_K,
        mooring_K,
        z_distr_m,
        distr_m,
        z_distr_k,
        distr_k,
    ) = _read_platform_common_tail(r)

    r.read_com()
    wires = _parse_tension_wires(r)

    return PlatformSupport(
        draft=draft,
        cm_pform=cm_pform,
        mass_pform=mass_pform,
        i_matrix=i_mat,
        ref_msl=ref_msl,
        hydro_M=hydro_M,
        hydro_K=hydro_K,
        mooring_K=mooring_K,
        distr_m_z=z_distr_m,
        distr_m=distr_m,
        distr_k_z=z_distr_k,
        distr_k=distr_k,
        wires=wires,
        cm_pform_x=cm_pform_x,
        cm_pform_y=cm_pform_y,
    )
