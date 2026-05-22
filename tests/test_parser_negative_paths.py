"""Comprehensive parser negative-path audit.

For every parser entry point, this module pins the documented
behaviour on:

1. **Well-formed minimal input** — sanity that the parser works at all.
2. **Truncated count / table** — declared count exceeds available rows.
3. **Bad numeric token** — non-numeric in a column expected to parse.
4. **Non-finite numeric token** — ``nan`` / ``inf`` in a numeric field.
5. **Cross-reference mismatch** — ID referenced but not defined.
6. **Path normalization** — quoted / backslash-separated file paths.

Earlier passes already covered most of these for individual parsers
piecemeal; this audit gathers them in one file so the rubric is
visible at a glance and adding a new parser triggers an obvious
"where are the negative-path tests?" question at review time.

Parsers covered (one section each):
    A. BMI (``pybmodes.io.bmi.read_bmi``)
    B. section-properties (``pybmodes.io.sec_props.read_sec_props``)
    C. SubDyn (``pybmodes.io.subdyn_reader.read_subdyn``)
    D. WAMIT (``pybmodes.io.wamit_reader.WamitReader``)
    E. HydroDyn (``pybmodes.io.wamit_reader.HydroDynReader``)
    F. ElastoDyn (``pybmodes.io.elastodyn_reader.read_elastodyn_*``)
    G. MoorDyn (``pybmodes.mooring.MooringSystem.from_moordyn``)
    H. .out output (``pybmodes.io.out_parser.read_out``)

Most negative behaviours either raise ``ValueError`` with file +
row context (the goal) or silently ``continue`` past a header-like
row (deliberate for WAMIT / .out where the file format mixes free-
form header text with numeric data). This file pins which is which
so the policy is visible.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# ===========================================================================
# A. BMI parser
# ===========================================================================
#
# ``read_bmi`` parses a structured deck with explicit count-then-array
# sections (n_elements / el_loc, n_attachments / wire arrays, matrix
# rows). Static review caught that ``read_ary(n)`` silently truncated
# to whatever tokens were present; pass-5 raised on truncation.

class TestBMIParserNegativePaths:

    def _write_bmi_with_short_el_loc(self, path: pathlib.Path) -> pathlib.Path:
        """Synthetic BMI with ``n_elements = 4`` (so 5 el_loc tokens
        expected) but only 3 tokens on the el_loc line."""
        lines = [
            "================= synthetic .bmi =================",
            "'short_el_loc'",
            "--------------- general parameters ---------------",
            "--------------- (echo through mid_node_tw) -------",
            "f    ! echo",
            "1    ! beam_type",
            "0.0    ! rot_rpm",
            "1.0    ! rpm_mult",
            "50.0    ! radius",
            "0.0    ! hub_rad",
            "0.0    ! precone",
            "0.0    ! bl_thp",
            "1    ! hub_conn",
            "10    ! n_modes_print",
            "f    ! tab_delim",
            "f    ! mid_node_tw",
            "--------------- tip-mass props -------------------",
            "--------------- (9 values) -----------------------",
            "0.0    ! tip_mass",
            "0.0    ! cm_offset",
            "0.0    ! cm_axial",
            "0.0    ! ixx",
            "0.0    ! iyy",
            "0.0    ! izz",
            "0.0    ! ixy",
            "0.0    ! izx",
            "0.0    ! iyz",
            "--------------- distributed-prop ref -------------",
            "--------------- (id_mat + filename) --------------",
            "1    ! id_mat",
            "'secs.dat'    ! sec_props_file",
            "--------------- scaling factors ------------------",
            "--------------- (10 unity multipliers) -----------",
            "1.0    ! sec_mass",
            "1.0    ! flp_iner",
            "1.0    ! lag_iner",
            "1.0    ! flp_stff",
            "1.0    ! edge_stff",
            "1.0    ! tor_stff",
            "1.0    ! axial_stff",
            "1.0    ! cg_offst",
            "1.0    ! sc_offst",
            "1.0    ! tc_offst",
            "--------------- fe discretisation ---------------",
            "--------------- (nselt + el_loc) ----------------",
            "4    ! n_elements",
            "--- el_loc ---",
            # n_elements + 1 = 5 expected, only 3 present
            "0.0  0.25  0.5",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_truncated_el_loc_raises_with_context(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``read_ary(n_elements + 1)`` raises when the el_loc line has
        fewer tokens. Used to silently truncate, deferring the failure
        to a downstream NumPy broadcasting error."""
        path = self._write_bmi_with_short_el_loc(tmp_path / "short.bmi")
        from pybmodes.io.bmi import read_bmi
        with pytest.raises(
            ValueError, match="Truncated array.*expected 5 tokens, got 3",
        ):
            read_bmi(path)

    def test_bmi_sec_props_path_normalises_backslashes(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A BMI authored on Windows with ``subdir\\props.dat`` parses
        and resolves correctly on POSIX. Already covered by
        ``test_post_1_0_review_fixes.test_bmi_sec_props_file_normalises_windows_backslashes``;
        this test is the audit-table breadcrumb."""
        from pybmodes.io.bmi import read_bmi
        from tests._synthetic_bmi import write_bmi, write_uniform_sec_props

        subdir = tmp_path / "props_sub"
        subdir.mkdir()
        write_uniform_sec_props(subdir / "props.dat")
        bmi_path = tmp_path / "tower.bmi"
        write_bmi(
            bmi_path, beam_type=2, radius=90.0, hub_conn=1,
            sec_props_file=r"props_sub\props.dat",
            n_elements=10, tip_mass=200_000.0,
        )
        parsed = read_bmi(bmi_path)
        assert parsed.sec_props_file == "props_sub/props.dat"
        assert parsed.resolve_sec_props_path().is_file()


# ===========================================================================
# B. section-properties parser
# ===========================================================================
#
# Already heavily covered by pass-4 (``test_static_review_pass_4_fixes.
# TestParsersRejectNonFinite``). One additional anchor: declared
# n_secs not matching parsed-row count.

class TestSecPropsNegativePaths:

    def test_declared_n_secs_must_match_parsed_rows(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``n_secs`` mismatch raises with the declared vs actual count.
        The reader breaks the loop on trailing notes (silently OK) but
        if the resulting row count != ``n_secs``, raises."""
        from pybmodes.io.sec_props import read_sec_props

        path = tmp_path / "sec.dat"
        path.write_text(
            "synthetic\n"
            "5  n_secs\n"
            "\n"
            "col_header\n"
            "units_row\n"
            "0.00  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n"
            "0.50  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n"
            "1.00  0  0  100  10  10  1e9  1e9  1e8  1e10  0  0  0\n",
            encoding="latin-1",
        )
        with pytest.raises(
            ValueError, match="expected 5 data rows, found 3",
        ):
            read_sec_props(path)


# ===========================================================================
# C. SubDyn parser
# ===========================================================================
#
# Earlier passes covered missing-section and adapter friendly errors;
# this section pins per-row failure modes the rubric calls out.

_SUBDYN_TEMPLATE = """\
----------- SubDyn MultiMember Support Structure Input File ---------------------------
Audit synthetic deck.
-------------------------- SIMULATION CONTROL -----------------------------------------
False            Echo        - flag
---- STRUCTURE JOINTS: joints connect structure members ------------------
             3   NJoints     - count
JointID  JointXss  JointYss  JointZss  JointType  JointDirX  JointDirY  JointDirZ  JointStiff
  (-)      (m)       (m)       (m)       (-)        (-)        (-)        (-)      (Nm/rad)
{joint_rows}
------------------- BASE REACTION JOINTS ----------------------------------------------
             1   NReact      - count
RJointID  RctTDXss  RctTDYss  RctTDZss  RctRDXss  RctRDYss  RctRDZss  SSIfile
  (-)      (flag)    (flag)    (flag)    (flag)    (flag)    (flag)   (string)
   1        1         1         1         1         1         1       ""
------- INTERFACE JOINTS --------------------------------------------------------------
             1   NInterf     - count
IJointID  TPID  ItfTDXss  ItfTDYss  ItfTDZss  ItfRDXss  ItfRDYss  ItfRDZss
  (-)     (-)    (flag)    (flag)    (flag)    (flag)    (flag)    (flag)
   3       1       1         1         1         1         1         1
----------------------------------- MEMBERS -------------------------------------------
             2   NMembers    - count
MemberID  MJointID1  MJointID2  MPropSetID1  MPropSetID2  MType  MSpin/COSMID
  (-)        (-)        (-)         (-)          (-)       (-)    (deg/-)
{member_rows}
------------------ CIRCULAR BEAM CROSS-SECTION PROPERTIES -----------------------------
             1   NPropSetsCyl - count
PropSetID  YoungE       ShearG       MatDens   XsecD    XsecT
  (-)      (N/m2)       (N/m2)       (kg/m3)    (m)      (m)
   1     2.10000e+11  8.08000e+10   8500.00    6.000    0.060
"""

_GOOD_JOINT_ROWS = (
    "   1     0.000     0.000   -20.000        1        0.0        0.0        0.0        0.0\n"
    "   2     0.000     0.000   -10.000        1        0.0        0.0        0.0        0.0\n"
    "   3     0.000     0.000     0.000        1        0.0        0.0        0.0        0.0\n"
)
_GOOD_MEMBER_ROWS = (
    "   1          1          2           1            1         1c       0\n"
    "   2          2          3           1            1         1c       0\n"
)


class TestSubDynNegativePaths:

    def test_well_formed_minimal_input(self, tmp_path: pathlib.Path) -> None:
        """Audit checkpoint #1 — the synthetic template above is a
        well-formed minimal SubDyn input."""
        from pybmodes.io.subdyn_reader import read_subdyn

        path = tmp_path / "good.dat"
        path.write_text(
            _SUBDYN_TEMPLATE.format(
                joint_rows=_GOOD_JOINT_ROWS, member_rows=_GOOD_MEMBER_ROWS,
            ),
            encoding="latin-1",
        )
        parsed = read_subdyn(path)
        assert len(parsed.joints) == 3
        assert len(parsed.members) == 2
        assert parsed.reaction_joint_id == 1
        assert parsed.interface_joint_id == 3

    def test_bad_numeric_joint_row_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Non-numeric ``JointXss`` value raises from ``_parse_float``."""
        from pybmodes.io.subdyn_reader import read_subdyn

        bad_rows = (
            "   1     not_a_number     0.000   -20.000        1\n"
            "   2     0.000     0.000   -10.000        1\n"
            "   3     0.000     0.000     0.000        1\n"
        )
        path = tmp_path / "bad_joint.dat"
        path.write_text(
            _SUBDYN_TEMPLATE.format(
                joint_rows=bad_rows, member_rows=_GOOD_MEMBER_ROWS,
            ),
            encoding="latin-1",
        )
        with pytest.raises(ValueError):
            read_subdyn(path)

    def test_short_member_row_raises_with_row_index(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A member row with < 5 columns raises ``ValueError`` naming
        the row and source file. Used to be a bare ``IndexError``."""
        from pybmodes.io.subdyn_reader import read_subdyn

        bad_rows = (
            "   1          1          2           1            1         1c       0\n"
            "   2          2          3\n"  # only 3 columns
        )
        path = tmp_path / "short_member.dat"
        path.write_text(
            _SUBDYN_TEMPLATE.format(
                joint_rows=_GOOD_JOINT_ROWS, member_rows=bad_rows,
            ),
            encoding="latin-1",
        )
        with pytest.raises(
            ValueError, match="malformed MEMBERS row 2.*expected >= 5",
        ):
            read_subdyn(path)

    def test_declared_count_exceeds_available_rows_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``NMembers = 5`` but only 2 rows present → ``_consume_table``
        runs off the end and raises (``EOFError`` from
        ``_next_data_line``). The error type is currently EOFError;
        either ValueError or EOFError is acceptable here, but the
        behaviour MUST be raise-not-return."""
        from pybmodes.io.subdyn_reader import read_subdyn

        # Splice a bumped count without adding more rows.
        text = _SUBDYN_TEMPLATE.format(
            joint_rows=_GOOD_JOINT_ROWS, member_rows=_GOOD_MEMBER_ROWS,
        ).replace("             2   NMembers", "             5   NMembers")
        path = tmp_path / "overcount.dat"
        path.write_text(text, encoding="latin-1")
        with pytest.raises((EOFError, ValueError)):
            read_subdyn(path)

    def test_undefined_prop_set_id_raises_at_adapter_time(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Cross-reference: a member referencing ``MPropSetID = 99``
        (no such circular property set exists) raises when the
        adapter tries to look it up. The PARSER tolerates this
        because the cross-reference can be resolved against the
        property table that comes later in the file."""
        from pybmodes.io.subdyn_reader import read_subdyn, to_pybmodes_pile_tower

        bad_rows = (
            "   1          1          2          99            99         1c       0\n"
            "   2          2          3          99            99         1c       0\n"
        )
        path = tmp_path / "bad_propset.dat"
        path.write_text(
            _SUBDYN_TEMPLATE.format(
                joint_rows=_GOOD_JOINT_ROWS, member_rows=bad_rows,
            ),
            encoding="latin-1",
        )
        subdyn = read_subdyn(path)
        # Parse alone succeeds (cross-ref is unresolved but not
        # checked at parse time); adapter raises.
        assert len(subdyn.members) == 2
        from dataclasses import dataclass

        @dataclass
        class _StubMain:
            tower_ht: float = 90.0
            tower_bs_ht: float = 0.0

        with pytest.raises(
            ValueError, match="no circular cross-section property set with id=99",
        ):
            to_pybmodes_pile_tower(
                main=_StubMain(), tower=None, subdyn=subdyn,
            )


# ===========================================================================
# D. WAMIT parser
# ===========================================================================

def _write_wamit_pair(
    dir: pathlib.Path, root: str, dot1_body: str, hst_body: str,
) -> None:
    """Write a ``<root>.1`` and ``<root>.hst`` pair into ``dir``."""
    (dir / f"{root}.1").write_text(dot1_body, encoding="utf-8")
    (dir / f"{root}.hst").write_text(hst_body, encoding="utf-8")


_GOOD_DOT1 = (
    "-1.0  1 1   1.000\n"   # A_inf surge-surge
    "-1.0  3 3   2.000\n"   # A_inf heave-heave
    " 0.0  1 1   0.500\n"   # A_0 surge-surge
)
_GOOD_HST = (
    "1 1   0.000\n"
    "3 3   1.500\n"  # heave hydrostatic
    "4 4   2.000\n"  # roll
)


class TestWamitNegativePaths:

    def test_well_formed_minimal_input(self, tmp_path: pathlib.Path) -> None:
        """Audit checkpoint #1."""
        from pybmodes.io.wamit_reader import WamitReader

        _write_wamit_pair(tmp_path, "good", _GOOD_DOT1, _GOOD_HST)
        r = WamitReader("good", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        data = r.read()
        # Heave-heave A_inf = 2.0 · ρ · L³ = 2.0 · 1025 · 1 = 2050
        assert data.A_inf[2, 2] == pytest.approx(2050.0, rel=0.001)
        # Heave-heave C_hst = 1.5 · ρ · g · L² = 1.5 · 1025 · 9.81 = 15083.4
        assert data.C_hst[2, 2] == pytest.approx(1.5 * 1025 * 9.81, rel=0.001)

    def test_out_of_range_dof_in_dot1_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``i = 7`` (only 6 rigid-body DOFs) raises ``ValueError``."""
        from pybmodes.io.wamit_reader import WamitReader

        bad_dot1 = "-1.0  7 1   1.0\n"
        _write_wamit_pair(tmp_path, "bad", bad_dot1, _GOOD_HST)
        r = WamitReader("bad", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        with pytest.raises(ValueError, match=r"outside.*6 rigid-body DOFs"):
            r.read()

    def test_out_of_range_dof_in_hst_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        from pybmodes.io.wamit_reader import WamitReader

        bad_hst = "0 1   1.0\n"  # i = 0 (1-indexed convention requires 1-6)
        _write_wamit_pair(tmp_path, "bad", _GOOD_DOT1, bad_hst)
        r = WamitReader("bad", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        with pytest.raises(ValueError, match=r"outside.*6 rigid-body DOFs"):
            r.read()

    def test_non_finite_dot1_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``A(3,3) = nan`` raises rather than landing in the matrix."""
        from pybmodes.io.wamit_reader import WamitReader

        bad_dot1 = "-1.0  3 3   nan\n"
        _write_wamit_pair(tmp_path, "bad", bad_dot1, _GOOD_HST)
        r = WamitReader("bad", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        with pytest.raises(ValueError, match="Non-finite WAMIT entry"):
            r.read()

    def test_non_finite_hst_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        from pybmodes.io.wamit_reader import WamitReader

        bad_hst = "3 3   inf\n"
        _write_wamit_pair(tmp_path, "bad", _GOOD_DOT1, bad_hst)
        r = WamitReader("bad", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        with pytest.raises(ValueError, match="Non-finite WAMIT entry"):
            r.read()

    def test_non_finite_dot1_period_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A ``nan`` in the period column used to fall through the
        ``period == -1.0`` / ``period == 0.0`` / ``else: continue``
        dispatch as if it were a finite-period (frequency-dependent)
        row — silently dropping an otherwise schema-matching
        ``A_inf`` / ``A_0`` row. The fix adds an explicit
        ``_require_finite`` on ``period`` outside the schema-probe
        try block."""
        from pybmodes.io.wamit_reader import WamitReader

        bad_dot1 = "nan  3 3   1.0\n"
        _write_wamit_pair(tmp_path, "bad_period", bad_dot1, _GOOD_HST)
        r = WamitReader("bad_period", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        with pytest.raises(ValueError, match="Non-finite WAMIT entry.*period"):
            r.read()

    def test_short_dot1_row_silently_ignored(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A row with fewer than 4 tokens is treated as header /
        comment text and skipped. Documented behaviour — WAMIT files
        sometimes preface the data with a one-line title that
        otherwise parses as 1-3 tokens."""
        from pybmodes.io.wamit_reader import WamitReader

        # Insert a 2-token "TITLE LINE" before the good data.
        body = "TITLE LINE\n" + _GOOD_DOT1
        _write_wamit_pair(tmp_path, "short", body, _GOOD_HST)
        r = WamitReader("short", tmp_path, rho=1025.0, g=9.81, ulen=1.0)
        data = r.read()
        # The title line was silently skipped; the data parsed normally.
        assert data.A_inf[2, 2] != 0.0


# ===========================================================================
# E. HydroDyn parser
# ===========================================================================

def _write_hydrodyn(path: pathlib.Path, **kwargs) -> pathlib.Path:
    """Write a HydroDyn ``.dat`` with the keys passed in. Defaults
    produce a minimal valid file pointing at ``./bodyA``."""
    defaults = {
        "WtrDpth": "200.0",
        "WAMITULEN": "1.0",
        "PotMod": "1",
        "PotFile": '"bodyA"',
        "PtfmRefzt": "0.0",
    }
    defaults.update(kwargs)
    lines = [
        "------- HydroDyn Input File ----------------------",
        "Audit synthetic.",
    ]
    for key, val in defaults.items():
        lines.append(f"{val}    {key}    - description")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestHydroDynNegativePaths:

    def test_potmod_zero_raises_on_read_platform_matrices(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A HydroDyn deck with ``PotMod = 0`` (no WAMIT) raises a
        clear ``ValueError`` rather than attempting to read absent
        WAMIT files."""
        from pybmodes.io.wamit_reader import HydroDynReader

        path = _write_hydrodyn(tmp_path / "no_wamit.dat", PotMod="0")
        reader = HydroDynReader(path)
        with pytest.raises(ValueError, match="PotMod=0"):
            reader.read_platform_matrices()

    def test_missing_pot_file_raises_file_not_found(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``PotFile`` points at a root with no ``.1`` / ``.hst`` files
        — ``WamitReader.read`` raises ``FileNotFoundError`` naming the
        expected path."""
        from pybmodes.io.wamit_reader import HydroDynReader

        path = _write_hydrodyn(
            tmp_path / "deck.dat", PotFile='"missing_body"',
        )
        reader = HydroDynReader(path)
        with pytest.raises(FileNotFoundError, match=r"\.1 file not found"):
            reader.read_platform_matrices()

    def test_malformed_wamitulen_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``WAMITULEN = not_a_number`` raises when the property is
        accessed."""
        from pybmodes.io.wamit_reader import HydroDynReader

        path = _write_hydrodyn(
            tmp_path / "bad_ulen.dat", WAMITULEN="not_a_number",
        )
        reader = HydroDynReader(path)
        with pytest.raises(ValueError):
            _ = reader.ulen

    def test_non_finite_wamitulen_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A finite WAMITULEN is load-bearing for the v7
        redimensionalisation (``ρ · L^k`` factors). ``inf`` would
        produce nonsense matrices; reject at parse time."""
        from pybmodes.io.wamit_reader import HydroDynReader

        path = _write_hydrodyn(tmp_path / "inf_ulen.dat", WAMITULEN="inf")
        reader = HydroDynReader(path)
        with pytest.raises(ValueError, match="Non-finite"):
            _ = reader.ulen

    def test_missing_pot_file_key_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A deck with no ``PotFile`` line raises ``KeyError`` on
        access."""
        from pybmodes.io.wamit_reader import HydroDynReader

        path = tmp_path / "no_potfile.dat"
        path.write_text(
            "------- HydroDyn Input File ----------\n"
            "Audit synthetic.\n"
            "200.0    WtrDpth    - depth\n"
            "1.0      WAMITULEN  - reference length\n"
            "1        PotMod     - WAMIT enabled\n",
            encoding="utf-8",
        )
        reader = HydroDynReader(path)
        with pytest.raises(KeyError, match="PotFile"):
            _ = reader.pot_file

    def test_pot_file_path_normalisation(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``PotFile`` accepts quoted Windows-style paths
        (``"sub\\body"``) and resolves them across platforms."""
        from pybmodes.io.wamit_reader import WamitReader

        sub = tmp_path / "wamit_sub"
        sub.mkdir()
        _write_wamit_pair(sub, "body", _GOOD_DOT1, _GOOD_HST)
        # Build a WamitReader directly with a Windows-style path —
        # exercises the path-normalisation branch in
        # ``_resolve_pot_path``.
        r = WamitReader(
            r'"wamit_sub\body"', tmp_path,
            rho=1025.0, g=9.81, ulen=1.0,
        )
        assert r.pot_file_root == (sub / "body").resolve()


# ===========================================================================
# F. ElastoDyn parser
# ===========================================================================
#
# Bundled-deck roundtrip coverage already exists
# (``tests/test_elastodyn_writer.py``); this section pins a few
# malformed-known-scalar paths the rubric calls out.

_ELASTODYN_MAIN_TEMPLATE = """\
------- ELASTODYN V1.00.* INPUT FILE -----------------------------------------
Audit synthetic.
---------------------- SIMULATION CONTROL ------------------------------------
{num_bl}      NumBl       - Number of blades
{tip_rad}    TipRad      - Tip radius (m)
1.5     HubRad      - Hub radius (m)
"{twr_file}"   TwrFile     - Name of file with tower properties
0       PreCone(1)  - blade 1 precone
0       PreCone(2)
0       PreCone(3)
"""


class TestElastoDynMainNegativePaths:

    def test_malformed_numbl_raises(self, tmp_path: pathlib.Path) -> None:
        """``NumBl = not_a_number`` raises when the typed field is
        assigned — the parser raises loudly on KNOWN labels per the
        ``_KNOWN_MAIN_CANON`` set added in earlier pre-1.0 review."""
        from pybmodes.io.elastodyn_reader import read_elastodyn_main

        path = tmp_path / "bad_numbl.dat"
        path.write_text(
            _ELASTODYN_MAIN_TEMPLATE.format(
                num_bl="not_a_number", tip_rad="63.0",
                twr_file="tower.dat",
            ),
            encoding="latin-1",
        )
        with pytest.raises(ValueError, match="NumBl"):
            read_elastodyn_main(path)

    def test_non_finite_tip_rad_raises(self, tmp_path: pathlib.Path) -> None:
        """``TipRad = inf`` raises via the strict-finite
        ``_parse_float`` added in pass-4."""
        from pybmodes.io.elastodyn_reader import read_elastodyn_main

        path = tmp_path / "inf_tip.dat"
        path.write_text(
            _ELASTODYN_MAIN_TEMPLATE.format(
                num_bl="3", tip_rad="inf", twr_file="tower.dat",
            ),
            encoding="latin-1",
        )
        with pytest.raises(ValueError, match="TipRad"):
            read_elastodyn_main(path)

    def test_twr_file_backslash_path_normalised(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """``TwrFile`` rewrites Windows-style backslashes to forward
        slashes — already covered by the ElastoDyn parser pass-2 fix.
        Audit-table breadcrumb."""
        from pybmodes.io.elastodyn_reader import read_elastodyn_main

        path = tmp_path / "bs_twrfile.dat"
        path.write_text(
            _ELASTODYN_MAIN_TEMPLATE.format(
                num_bl="3", tip_rad="63.0",
                twr_file=r"sub\Mytower.dat",
            ),
            encoding="latin-1",
        )
        main = read_elastodyn_main(path)
        assert main.twr_file == "sub/Mytower.dat"


# ===========================================================================
# G. MoorDyn parser
# ===========================================================================
#
# Heavily covered by pass-2 (``test_static_review_pass_2_fixes.py``)
# and pass-4 (``test_static_review_pass_4_fixes.py``). Audit-table
# breadcrumbs only.

class TestMoorDynAuditBreadcrumbs:

    def test_strict_parsing_documented_elsewhere(self) -> None:
        """The MoorDyn strict-parsing rubric is verified in
        ``tests/test_static_review_pass_2_fixes.py`` (LINE TYPES /
        POINTS / LINES strict parse on malformed rows; header-variant
        tolerance) and
        ``tests/test_static_review_pass_4_fixes.py`` (OPTIONS strict
        parse on the three load-bearing keys; non-finite rejection).
        Keeping a sentinel here so the audit-file enumeration stays
        complete."""
        assert True


# ===========================================================================
# H. .out output parser
# ===========================================================================
#
# The ``.out`` parser is deliberately permissive — it consumes
# whatever the BModes Fortran writer happens to produce, including
# wrapped lines, trailing blank rows, and the occasional non-numeric
# token in column comments. Pin the documented tolerance.

_OUT_HEADER = (
    "============================================================\n"
    "Rotating-Blade Frequencies\n"
    "============================================================\n"
    "synthetic title line\n"
)


def _out_mode_block(mode: int, freq: float, n_rows: int = 5) -> str:
    """One mode block: header + column line + n_rows data."""
    rows = "\n".join(
        f"  {x:.3f}  0.0  0.0  0.0  0.0  {x:.3f}"
        for x in np.linspace(0.0, 1.0, n_rows)
    )
    return (
        f"-------- Mode No. {mode}  (freq = {freq:.4e} Hz)\n"
        f"  span   u_x   u_y   u_z   th_x   twist\n"
        + rows
        + "\n"
    )


class TestOutParserTolerance:

    def test_well_formed_input(self, tmp_path: pathlib.Path) -> None:
        from pybmodes.io.out_parser import read_out

        path = tmp_path / "good.out"
        path.write_text(
            _OUT_HEADER + _out_mode_block(1, 0.5) + _out_mode_block(2, 1.2),
            encoding="latin-1",
        )
        out = read_out(path)
        assert len(out.modes) == 2
        assert out.modes[0].frequency == pytest.approx(0.5)
        assert out.modes[1].frequency == pytest.approx(1.2)

    def test_short_data_row_silently_skipped(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A data row with fewer than 6 tokens is silently skipped.
        Documented tolerance — BModes output sometimes wraps a
        long column comment onto a new line."""
        from pybmodes.io.out_parser import read_out

        body = (
            "-------- Mode No. 1  (freq = 0.5 Hz)\n"
            "  span   u_x   u_y   u_z   th_x   twist\n"
            "  0.0  0.0  0.0  0.0  0.0  0.0\n"
            "  short_row_only_3_tokens\n"
            "  1.0  1.0  1.0  1.0  1.0  1.0\n"
        )
        path = tmp_path / "short.out"
        path.write_text(_OUT_HEADER + body, encoding="latin-1")
        out = read_out(path)
        # Two data rows survived; the short row was silently dropped.
        assert out.modes[0].span_loc.size == 2

    def test_duplicate_mode_numbers_both_stored(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """The parser does NOT deduplicate by mode number — both
        appearances land in the result. Documented for visibility;
        downstream tooling that needs uniqueness has to enforce it
        itself."""
        from pybmodes.io.out_parser import read_out

        path = tmp_path / "dup.out"
        path.write_text(
            _OUT_HEADER + _out_mode_block(1, 0.5) + _out_mode_block(1, 0.8),
            encoding="latin-1",
        )
        out = read_out(path)
        assert len(out.modes) == 2
        # Both stored, in file order.
        assert [m.mode_number for m in out.modes] == [1, 1]
        assert out.modes[1].frequency == pytest.approx(0.8)

    def test_nan_in_data_row_tolerated(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """A data row with ``nan`` in any column is currently kept —
        ``float('nan')`` succeeds and the value lands in the array.
        Documented for visibility; if downstream consumers care
        about finiteness they should run
        :func:`pybmodes.checks._check_section_properties_finite` or
        an equivalent before using the result."""
        from pybmodes.io.out_parser import read_out

        body = (
            "-------- Mode No. 1  (freq = 0.5 Hz)\n"
            "  span   u_x   u_y   u_z   th_x   twist\n"
            "  0.0  0.0  0.0  0.0  0.0  0.0\n"
            "  0.5  nan  0.0  0.0  0.0  0.0\n"
            "  1.0  1.0  1.0  1.0  1.0  1.0\n"
        )
        path = tmp_path / "nan_data.out"
        path.write_text(_OUT_HEADER + body, encoding="latin-1")
        out = read_out(path)
        # NaN landed in column 1 (u_x) of the middle row.
        assert np.isnan(out.modes[0].col1[1])
