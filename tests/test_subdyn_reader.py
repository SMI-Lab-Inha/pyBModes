"""Default-suite tests for :mod:`pybmodes.io.subdyn_reader`.

The integration-marked ``tests/test_certtest.py::test_certtest_cs_monopile``
exercises this module end-to-end against the OC3 monopile reference
deck, but that test only runs when upstream OpenFAST data is present
under ``docs/``. On a fresh clone the parser gets zero default-suite
coverage. This module covers the parser surface with a minimal
synthetic SubDyn snippet emitted to ``tmp_path`` and the
``SubDynCircProp`` derived properties (area / mass_per_length / I /
EI / J / GJ / EA), plus the ``_pile_axial_stations`` adapter helper
on the same synthetic input.

The snippet follows the SubDyn ``.dat`` format used by every OpenFAST
release from v2.x onward — a thin pile with three joints, two members,
and one circular cross-section property set. That's the smallest
shape the parser's section-prescan needs to find ``STRUCTURE JOINTS``,
``BASE REACTION``, ``INTERFACE JOINTS``, ``MEMBERS``, and
``CIRCULAR BEAM`` headers and produce a self-consistent
:class:`~pybmodes.io.subdyn_reader.SubDynFile`.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from pybmodes.io.subdyn_reader import (
    SubDynCircProp,
    SubDynFile,
    SubDynJoint,
    SubDynMember,
    _pile_axial_stations,
    read_subdyn,
    to_pybmodes_pile_tower,
)

# ---------------------------------------------------------------------------
# Minimal synthetic SubDyn snippet
# ---------------------------------------------------------------------------

# Three joints stacked vertically (z = -20, -10, 0), two members between
# them, one cross-section property set referenced by both members.
# Reaction joint 1 (bottom, clamped); interface joint 3 (top, attached
# to the transition piece). Sections we don't parse are included with
# zero-count headers so the file looks like a real SubDyn deck.
_SYNTHETIC_SUBDYN = """\
----------- SubDyn MultiMember Support Structure Input File ---------------------------
Synthetic 3-joint monopile for the pybmodes default-suite parser test.
-------------------------- SIMULATION CONTROL -----------------------------------------
False            Echo        - Echo input data
"DEFAULT"        SDdeltaT    - Local Integration Step
             3   IntMethod   - Integration Method
True             SttcSolve   - Solve dynamics about static equilibrium
-------------------- FEA AND CRAIG-BAMPTON PARAMETERS ---------------------------------
             3   FEMMod      - FEM switch
             3   NDiv        - Number of sub-elements per member
             0   Nmodes      - Number of internal modes
             1   JDampings   - Damping ratios
             0   GuyanDampMod - Guyan damping mode
---- STRUCTURE JOINTS: joints connect structure members ------------------
             3   NJoints     - Number of joints
JointID  JointXss  JointYss  JointZss  JointType  JointDirX  JointDirY  JointDirZ  JointStiff
  (-)      (m)       (m)       (m)       (-)        (-)        (-)        (-)      (Nm/rad)
   1     0.000     0.000   -20.000        1        0.0        0.0        0.0        0.0
   2     0.000     0.000   -10.000        1        0.0        0.0        0.0        0.0
   3     0.000     0.000     0.000        1        0.0        0.0        0.0        0.0
------------------- BASE REACTION JOINTS ----------------------------------------------
             1   NReact      - Number of reaction joints
RJointID  RctTDXss  RctTDYss  RctTDZss  RctRDXss  RctRDYss  RctRDZss  SSIfile
  (-)      (flag)    (flag)    (flag)    (flag)    (flag)    (flag)   (string)
   1         1         1         1         1         1         1       ""
------- INTERFACE JOINTS --------------------------------------------------------------
             1   NInterf     - Number of interface joints
IJointID  TPID  ItfTDXss  ItfTDYss  ItfTDZss  ItfRDXss  ItfRDYss  ItfRDZss
  (-)     (-)    (flag)    (flag)    (flag)    (flag)    (flag)    (flag)
   3       1       1         1         1         1         1         1
----------------------------------- MEMBERS -------------------------------------------
             2   NMembers    - Number of members
MemberID  MJointID1  MJointID2  MPropSetID1  MPropSetID2  MType  MSpin/COSMID
  (-)        (-)        (-)         (-)          (-)       (-)    (deg/-)
   1          1          2           1            1         1c       0
   2          2          3           1            1         1c       0
------------------ CIRCULAR BEAM CROSS-SECTION PROPERTIES -----------------------------
             1   NPropSetsCyl - Number of circular cross-sections
PropSetID  YoungE       ShearG       MatDens   XsecD    XsecT
  (-)      (N/m2)       (N/m2)       (kg/m3)    (m)      (m)
   1     2.10000e+11  8.08000e+10   8500.00    6.000    0.060
"""


@pytest.fixture(scope="module")
def synthetic_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """One-off SubDyn deck shared across the parser tests."""
    p = tmp_path_factory.mktemp("subdyn") / "synth.dat"
    p.write_text(_SYNTHETIC_SUBDYN, encoding="latin-1")
    return p


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParseSynthetic:

    @pytest.fixture(scope="class")
    def parsed(self, tmp_path_factory: pytest.TempPathFactory) -> SubDynFile:
        p = tmp_path_factory.mktemp("subdyn_cls") / "synth.dat"
        p.write_text(_SYNTHETIC_SUBDYN, encoding="latin-1")
        return read_subdyn(p)

    def test_header_and_title(self, parsed: SubDynFile) -> None:
        assert parsed.header.startswith("----------- SubDyn")
        assert "Synthetic 3-joint monopile" in parsed.title

    def test_source_file_captured(self, parsed: SubDynFile) -> None:
        assert parsed.source_file is not None
        assert parsed.source_file.name == "synth.dat"

    def test_three_joints_in_z_order(self, parsed: SubDynFile) -> None:
        assert len(parsed.joints) == 3
        # Joints are stored in file order — z values cover the full
        # synthesised range.
        zs = sorted(j.z for j in parsed.joints)
        assert zs == [-20.0, -10.0, 0.0]
        for j in parsed.joints:
            assert isinstance(j, SubDynJoint)
            assert j.x == 0.0 and j.y == 0.0
            assert j.joint_type == 1

    def test_reaction_and_interface_joint_ids(self, parsed: SubDynFile) -> None:
        assert parsed.reaction_joint_id == 1
        assert parsed.interface_joint_id == 3

    def test_two_circular_members(self, parsed: SubDynFile) -> None:
        assert len(parsed.members) == 2
        m1, m2 = parsed.members
        assert isinstance(m1, SubDynMember)
        assert m1.member_id == 1 and m1.joint_a == 1 and m1.joint_b == 2
        assert m2.member_id == 2 and m2.joint_a == 2 and m2.joint_b == 3
        assert m1.prop_set_a == m1.prop_set_b == 1
        assert m1.member_type == "1c"

    def test_single_circular_property_set(self, parsed: SubDynFile) -> None:
        assert len(parsed.circ_props) == 1
        p = parsed.circ_props[0]
        assert isinstance(p, SubDynCircProp)
        assert p.prop_set_id == 1
        assert pytest.approx(2.10e11) == p.E
        assert pytest.approx(8.08e10) == p.G
        assert p.rho == pytest.approx(8500.0)
        assert pytest.approx(6.0) == p.D
        assert p.t == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Derived cross-section properties
# ---------------------------------------------------------------------------

class TestCircPropDerived:
    """Cross-check the closed-form thin-walled tube formulas in
    :class:`SubDynCircProp` against direct computation.
    """

    @pytest.fixture(scope="class")
    def prop(self) -> SubDynCircProp:
        return SubDynCircProp(
            prop_set_id=1, E=2.1e11, G=8.08e10, rho=8500.0, D=6.0, t=0.06,
        )

    def test_area_matches_annular_formula(self, prop: SubDynCircProp) -> None:
        d_inner = prop.D - 2.0 * prop.t
        expected = math.pi / 4.0 * (prop.D ** 2 - d_inner ** 2)
        assert prop.area == pytest.approx(expected)

    def test_mass_per_length_matches_rho_times_area(
        self, prop: SubDynCircProp,
    ) -> None:
        assert prop.mass_per_length == pytest.approx(prop.rho * prop.area)

    def test_second_moment_of_area_matches_formula(
        self, prop: SubDynCircProp,
    ) -> None:
        d_inner = prop.D - 2.0 * prop.t
        expected = math.pi / 64.0 * (prop.D ** 4 - d_inner ** 4)
        assert pytest.approx(expected) == prop.I

    def test_bending_stiffness_is_E_times_I(self, prop: SubDynCircProp) -> None:
        assert pytest.approx(prop.E * prop.I) == prop.EI

    def test_polar_moment_is_twice_bending(self, prop: SubDynCircProp) -> None:
        assert pytest.approx(2.0 * prop.I) == prop.J

    def test_torsional_stiffness_is_G_times_J(self, prop: SubDynCircProp) -> None:
        assert pytest.approx(prop.G * prop.J) == prop.GJ

    def test_axial_stiffness_is_E_times_area(self, prop: SubDynCircProp) -> None:
        assert pytest.approx(prop.E * prop.area) == prop.EA


# ---------------------------------------------------------------------------
# Adapter helper: _pile_axial_stations
# ---------------------------------------------------------------------------

def test_pile_axial_stations_returns_sorted_z_and_segment_props(
    synthetic_path: pathlib.Path,
) -> None:
    """``_pile_axial_stations`` sorts the SubDyn joints by z (ascending)
    and returns one cross-section property per inter-joint segment. For
    the synthetic deck — three joints, uniform property set — that's
    three z values and two identical segment properties."""
    parsed = read_subdyn(synthetic_path)
    z_coords, seg_props = _pile_axial_stations(parsed)
    np.testing.assert_array_equal(z_coords, np.array([-20.0, -10.0, 0.0]))
    assert len(seg_props) == 2
    for p in seg_props:
        assert p.prop_set_id == 1
        assert pytest.approx(6.0) == p.D
        assert p.t == pytest.approx(0.06)


def test_pile_axial_stations_rejects_under_specified_subdyn() -> None:
    """The adapter rejects a SubDyn with too few joints or members —
    the OC3-style sequential-pile assumption needs ≥ 2 joints + 1
    member."""
    empty = SubDynFile()
    with pytest.raises(ValueError, match="need at least 2 joints"):
        _pile_axial_stations(empty)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_read_subdyn_missing_section_raises(tmp_path: pathlib.Path) -> None:
    """If a required section header is absent, ``read_subdyn`` raises a
    ``ValueError`` naming the missing prefix."""
    truncated = (
        "----------- SubDyn MultiMember Support Structure Input File ----\n"
        "title line\n"
        "---- STRUCTURE JOINTS: joints connect structure members ----\n"
        "             1   NJoints     - count\n"
        "JointID  JointXss  JointYss  JointZss  JointType\n"
        "  (-)      (m)       (m)       (m)       (-)\n"
        "   1     0.000     0.000     0.000        1\n"
        # Deliberately omit BASE REACTION, INTERFACE, MEMBERS, CIRCULAR
    )
    p = tmp_path / "truncated.dat"
    p.write_text(truncated, encoding="latin-1")
    with pytest.raises(ValueError, match="no section header starting with"):
        read_subdyn(p)


# ---------------------------------------------------------------------------
# Adapter friendly errors on missing reaction / interface joint IDs
# ---------------------------------------------------------------------------

_SUBDYN_VALID = """\
----------- SubDyn MultiMember Support Structure Input File ---------------------------
Synthetic 3-joint monopile.
-------------------------- SIMULATION CONTROL -----------------------------------------
False            Echo        - Echo input data
---- STRUCTURE JOINTS: joints connect structure members ------------------
             3   NJoints     - Number of joints
JointID  JointXss  JointYss  JointZss  JointType  JointDirX  JointDirY  JointDirZ  JointStiff
  (-)      (m)       (m)       (m)       (-)        (-)        (-)        (-)      (Nm/rad)
   1     0.000     0.000   -20.000        1        0.0        0.0        0.0        0.0
   2     0.000     0.000   -10.000        1        0.0        0.0        0.0        0.0
   3     0.000     0.000     0.000        1        0.0        0.0        0.0        0.0
------------------- BASE REACTION JOINTS ----------------------------------------------
             1   NReact      - Number of reaction joints
RJointID  RctTDXss  RctTDYss  RctTDZss  RctRDXss  RctRDYss  RctRDZss  SSIfile
  (-)      (flag)    (flag)    (flag)    (flag)    (flag)    (flag)   (string)
   {react_id}     1     1     1     1     1     1     ""
------- INTERFACE JOINTS --------------------------------------------------------------
             1   NInterf     - Number of interface joints
IJointID  TPID  ItfTDXss  ItfTDYss  ItfTDZss  ItfRDXss  ItfRDYss  ItfRDZss
  (-)     (-)    (flag)    (flag)    (flag)    (flag)    (flag)    (flag)
   {iface_id}   1   1   1   1   1   1   1
----------------------------------- MEMBERS -------------------------------------------
             2   NMembers    - Number of members
MemberID  MJointID1  MJointID2  MPropSetID1  MPropSetID2  MType  MSpin/COSMID
  (-)        (-)        (-)         (-)          (-)       (-)    (deg/-)
   1          1          2           1            1         1c       0
   2          2          3           1            1         1c       0
------------------ CIRCULAR BEAM CROSS-SECTION PROPERTIES -----------------------------
             1   NPropSetsCyl - Number of circular cross-sections
PropSetID  YoungE       ShearG       MatDens   XsecD    XsecT
  (-)      (N/m2)       (N/m2)       (kg/m3)    (m)      (m)
   1     2.10000e+11  8.08000e+10   8500.00    6.000    0.060
"""


def test_subdyn_adapter_missing_reaction_joint_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """A SubDyn deck whose ``BASE REACTION`` references a joint ID
    absent from ``STRUCTURE JOINTS`` raises a clear ``ValueError``
    naming the missing ID — not a bare ``StopIteration``."""
    deck = tmp_path / "bad_react.dat"
    deck.write_text(
        _SUBDYN_VALID.format(react_id=99, iface_id=3),  # 99 not in {1,2,3}
        encoding="latin-1",
    )

    # Minimal ElastoDyn main stub so the adapter can reach the joint
    # lookup. The adapter inspects ``main.tower_ht`` / ``main.tower_bs_ht``
    # AFTER the joint lookup, so attribute presence is enough.
    from dataclasses import dataclass

    @dataclass
    class _StubMain:
        tower_ht: float = 90.0
        tower_bs_ht: float = 0.0

    subdyn = read_subdyn(deck)
    with pytest.raises(ValueError, match="no joint with id=99.*reaction"):
        to_pybmodes_pile_tower(main=_StubMain(), tower=None, subdyn=subdyn)


def test_subdyn_adapter_missing_interface_joint_raises_valueerror(
    tmp_path: pathlib.Path,
) -> None:
    """Same diagnostic for a missing interface-joint ID."""
    deck = tmp_path / "bad_iface.dat"
    deck.write_text(
        _SUBDYN_VALID.format(react_id=1, iface_id=77),  # 77 not in {1,2,3}
        encoding="latin-1",
    )

    from dataclasses import dataclass

    @dataclass
    class _StubMain:
        tower_ht: float = 90.0
        tower_bs_ht: float = 0.0

    subdyn = read_subdyn(deck)
    with pytest.raises(ValueError, match="no joint with id=77.*interface"):
        to_pybmodes_pile_tower(main=_StubMain(), tower=None, subdyn=subdyn)
