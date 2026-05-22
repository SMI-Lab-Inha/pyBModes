"""WindIO floating-platform reader + hydro/mooring assembly
(1.4.0, issue #35).

Validation ladder (default rungs self-contained; integration rungs
anchor on the IEA-15 UMaine VolturnUS-S yaml vs its companion
WAMIT / MoorDyn / ElastoDyn decks):

* floating geometry: joint parsing (cartesian + cylindrical),
  member geometry, axial-joint resolution, transition joint.
  **(this file, below)**
* hydrostatic restoring C_hst vs closed-form cylinder + WAMIT.
* Morison added mass + buoyancy + rigid-body inertia.
* mooring catenary stiffness vs companion MoorDyn.
* PlatformSupport assembly → floating rigid-body + tower modes.
"""

from __future__ import annotations

import pathlib
import re
import textwrap

import numpy as np
import pytest

_DOCS = (pathlib.Path(__file__).resolve().parents[1]
         / "external" / "OpenFAST_files")
_IEA15_FLOAT_Y = (_DOCS / "IEA-15-240-RWT/WT_Ontology/"
                  "IEA-15-240-RWT_VolturnUS-S.yaml")
_IEA15_HD = (_DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/"
             "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat")
_IEA15_ED_F = (_DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/"
               "IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat")
_IEA15_MD = (_DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi/"
             "IEA-15-240-RWT-UMaineSemi_MoorDyn.dat")

# ---------------------------------------------------------------------------
# Floating-platform geometry reader (default; no external data)
# ---------------------------------------------------------------------------

_MIN_FLOAT = textwrap.dedent("""\
    components:
      floating_platform:
        transition_piece_mass: 100000.0
        joints:
          - {name: keel, location: [0.0, 0.0, -20.0]}
          - {name: top, location: [0.0, 0.0, 10.0], transition: true}
          - {name: c1k, location: [50.0, 120.0, -20.0], cylindrical: true}
        members:
          - name: main
            joint1: keel
            joint2: top
            Ca: [1.0, 1.0]
            Cd: [0.8, 0.8]
            outer_shape:
              shape: circular
              outer_diameter: {grid: [0.0, 1.0], values: [10.0, 10.0]}
            axial_joints:
              - {name: fairlead, grid: 0.25}
            structure:
              layers:
                - name: wall
                  material: steel
                  thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}
              bulkhead:
                material: steel
                thickness: {grid: [0.0, 1.0], values: [0.04, 0.04]}
      mooring:
        nodes: []
        lines: []
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def _read(tmp_path, text=_MIN_FLOAT):
    from pybmodes.io.windio_floating import read_windio_floating
    p = tmp_path / "float.yaml"
    p.write_text(text, encoding="utf-8")
    return read_windio_floating(p)


def test_floating_joints_cartesian_and_cylindrical(tmp_path) -> None:
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    np.testing.assert_allclose(f.joints["keel"], [0.0, 0.0, -20.0])
    np.testing.assert_allclose(f.joints["top"], [0.0, 0.0, 10.0])
    # cylindrical [r=50, θ=120°, z=-20] → (50cos120, 50sin120, -20)
    np.testing.assert_allclose(
        f.joints["c1k"],
        [50.0 * np.cos(np.deg2rad(120.0)),
         50.0 * np.sin(np.deg2rad(120.0)), -20.0],
        atol=1e-9,
    )
    assert f.transition_joint == "top"
    assert f.transition_piece_mass == 100000.0
    assert f.materials["steel"]["rho"] == 7800.0


def test_floating_member_geometry(tmp_path) -> None:
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    assert [m.name for m in f.members] == ["main"]
    m = f.members[0]
    assert m.length == pytest.approx(30.0)
    np.testing.assert_allclose(m.axis, [0.0, 0.0, 1.0])
    assert m.diameter_at(0.5) == pytest.approx(10.0)
    assert m.wall_t_at(0.7) == pytest.approx(0.05)
    assert m.ca == 1.0
    assert m.bulkhead_material == "steel"
    assert m.bulkhead_t == pytest.approx(0.04)
    np.testing.assert_allclose(m.point_at(0.0), [0.0, 0.0, -20.0])
    np.testing.assert_allclose(m.point_at(1.0), [0.0, 0.0, 10.0])


def test_floating_axial_joint_resolved_into_table(tmp_path) -> None:
    """A named axial joint (e.g. a fairlead) becomes a referenceable
    joint at its member fraction — mooring needs this."""
    pytest.importorskip("yaml")
    f = _read(tmp_path)
    assert "fairlead" in f.joints
    # keel + 0.25·(top−keel) = (0,0,−20 + 0.25·30) = (0,0,−12.5)
    np.testing.assert_allclose(f.joints["fairlead"], [0.0, 0.0, -12.5])


def test_floating_missing_joint_reference_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    bad = _MIN_FLOAT.replace("joint1: keel", "joint1: ghost")
    with pytest.raises(KeyError, match="ghost|joints list"):
        _read(tmp_path, bad)


def test_floating_member_without_wall_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    bad = textwrap.dedent("""\
        components:
          floating_platform:
            joints:
              - {name: a, location: [0.0, 0.0, -10.0]}
              - {name: b, location: [0.0, 0.0, 5.0]}
            members:
              - name: m
                joint1: a
                joint2: b
                outer_shape:
                  shape: circular
                  outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
                structure:
                  layers: []
        materials:
          - {name: steel, E: 2.0e11, rho: 7800.0}
        """)
    with pytest.raises(ValueError, match="structure.layers"):
        _read(tmp_path, bad)


def test_floating_missing_component_raises(tmp_path) -> None:
    pytest.importorskip("yaml")
    with pytest.raises(KeyError, match="floating_platform"):
        _read(tmp_path, "components:\n  tower: {}\n")


# ---------------------------------------------------------------------------
# Hydrostatic restoring vs closed form (default; no external data)
# ---------------------------------------------------------------------------


def _cyl_yaml(r: float, keel_z: float, top_z: float) -> str:
    return textwrap.dedent(f"""\
        components:
          floating_platform:
            joints:
              - {{name: k, location: [0.0, 0.0, {keel_z}]}}
              - {{name: t, location: [0.0, 0.0, {top_z}], transition: true}}
            members:
              - name: col
                joint1: k
                joint2: t
                Ca: [1.0, 1.0]
                outer_shape:
                  shape: circular
                  outer_diameter:
                    grid: [0.0, 1.0]
                    values: [{2 * r}, {2 * r}]
                structure:
                  layers:
                    - name: w
                      material: steel
                      thickness: {{grid: [0.0, 1.0], values: [0.05, 0.05]}}
          mooring: {{nodes: [], lines: []}}
        materials:
          - {{name: steel, E: 2.0e11, rho: 7800.0}}
        """)


def test_hydrostatic_single_cylinder_closed_form(tmp_path) -> None:
    """A vertical surface-piercing cylinder reduces to the exact
    waterplane + buoyancy restoring (WAMIT/.hst convention, no
    gravity term): C₃₃ = ρg·πr², C₄₄ = C₅₅ = ρg(πr⁴/4 − πr²·d²/2);
    a single deep cylinder is hydrostatically unstable (C₄₄ < 0)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        RHO_SW,
        G,
        hydrostatic_restoring,
        read_windio_floating,
    )

    r, d, h = 5.0, 20.0, 10.0          # radius, draft, freeboard
    p = tmp_path / "cyl.yaml"
    p.write_text(_cyl_yaml(r, -d, h), encoding="utf-8")
    C = hydrostatic_restoring(read_windio_floating(p), n_seg=400)

    rg = RHO_SW * G
    S = np.pi * r**2
    Iwp = np.pi * r**4 / 4.0
    vol = np.pi * r**2 * d
    zb = -d / 2.0
    assert C[2, 2] == pytest.approx(rg * S, rel=1e-4)
    assert C[3, 3] == pytest.approx(rg * Iwp + rg * vol * zb, rel=5e-3)
    assert C[4, 4] == pytest.approx(rg * Iwp + rg * vol * zb, rel=5e-3)
    assert C[3, 3] < 0.0                       # unstable bare cylinder
    # Axisymmetric & on-axis: no surge/sway/yaw, no cross terms.
    for i, j in [(2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]:
        assert abs(C[i, j]) < 1e-6 * abs(C[2, 2])
    assert np.allclose(C, C.T)
    assert C[0, 0] == 0.0 and C[1, 1] == 0.0 and C[5, 5] == 0.0


def test_hydrostatic_offset_column_parallel_axis(tmp_path) -> None:
    """An off-axis column adds the parallel-axis ρg·A·x² to the
    pitch restoring (the semi-submersible stabilising mechanism)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        RHO_SW,
        G,
        hydrostatic_restoring,
        read_windio_floating,
    )

    r, d = 4.0, 15.0
    y_off = 30.0
    txt = textwrap.dedent(f"""\
        components:
          floating_platform:
            joints:
              - {{name: k, location: [0.0, {y_off}, {-d}]}}
              - {{name: t, location: [0.0, {y_off}, 8.0]}}
            members:
              - name: c
                joint1: k
                joint2: t
                outer_shape:
                  shape: circular
                  outer_diameter: {{grid: [0.0, 1.0], values: [{2*r}, {2*r}]}}
                structure:
                  layers:
                    - {{name: w, material: steel,
                        thickness: {{grid: [0.0, 1.0], values: [0.05, 0.05]}}}}
          mooring: {{nodes: [], lines: []}}
        materials:
          - {{name: steel, rho: 7800.0, E: 2.0e11}}
        """)
    p = tmp_path / "off.yaml"
    p.write_text(txt, encoding="utf-8")
    C = hydrostatic_restoring(read_windio_floating(p), n_seg=400)

    rg = RHO_SW * G
    S = np.pi * r**2
    # Roll restoring picks up ρg·A·y² (parallel axis) about global x.
    expect_C44 = rg * (np.pi * r**4 / 4.0 + S * y_off**2) \
        + rg * (S * d) * (-d / 2.0)
    assert C[3, 3] == pytest.approx(expect_C44, rel=1e-2)
    assert C[3, 3] > 0.0           # offset column → roll-stable


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_FLOAT_Y.is_file() and _IEA15_HD.is_file()),
    reason="IEA-15 VolturnUS-S yaml / UMaineSemi HydroDyn deck absent",
)
def test_hydrostatic_iea15_volturnus_vs_wamit_hst() -> None:
    """Geometry-exact anchor: the IEA-15 UMaine VolturnUS-S WindIO
    floating geometry, through the member waterplane reduction,
    reproduces the companion **potential-flow WAMIT `.hst`**
    (HydroDynReader) heave / roll / pitch restoring. Hydrostatics are
    geometry-only, so panel-method and member-integration agree
    closely — measured heave 0.8 %, roll/pitch 1.6 %."""
    pytest.importorskip("yaml")
    from pybmodes.io.wamit_reader import HydroDynReader
    from pybmodes.io.windio_floating import (
        hydrostatic_restoring,
        read_windio_floating,
    )

    C = hydrostatic_restoring(read_windio_floating(_IEA15_FLOAT_Y))
    Cw = np.asarray(
        HydroDynReader(_IEA15_HD).read_platform_matrices().C_hst, float
    )

    def rel(i, j):
        return abs(C[i, j] - Cw[i, j]) / abs(Cw[i, j])

    assert rel(2, 2) < 0.03        # heave  (measured 0.8 %)
    assert rel(3, 3) < 0.04        # roll   (measured 1.6 %)
    assert rel(4, 4) < 0.04        # pitch  (measured 1.6 %)
    # Same physical sign structure as the panel method.
    assert C[2, 2] > 0.0 and C[3, 3] > 0.0 and C[4, 4] > 0.0


# ---------------------------------------------------------------------------
# Morison added mass + rigid-body inertia vs closed form
# ---------------------------------------------------------------------------


def test_added_mass_single_cylinder_closed_form(tmp_path) -> None:
    """A vertical cylinder: transverse surge/sway = Ca·ρ·πr²·d
    (submerged depth d); the single submerged end (the keel) adds the
    RAFT end-cap axial heave term ρ·Ca_End·(2/3)πr³ exactly (the
    above-MSL top end is excluded); the about-keel-ref rotational
    term = a'·d³/3 (an on-axis purely-axial cap adds no moment)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        RHO_SW,
        added_mass,
        read_windio_floating,
    )

    r, d, h = 5.0, 20.0, 10.0
    p = tmp_path / "c.yaml"
    p.write_text(_cyl_yaml(r, -d, h), encoding="utf-8")
    A = added_mass(read_windio_floating(p), n_seg=600)   # ref = origin

    ap = 1.0 * RHO_SW * np.pi * r**2          # Ca·ρ·πr² per length
    a_end = RHO_SW * 0.6 * (2.0 / 3.0) * np.pi * r**3   # one keel cap
    assert A[0, 0] == pytest.approx(ap * d, rel=1e-3)
    assert A[1, 1] == pytest.approx(ap * d, rel=1e-3)
    assert A[2, 2] == pytest.approx(a_end, rel=1e-9)     # exact end cap
    assert A[3, 3] == pytest.approx(ap * d**3 / 3.0, rel=2e-2)
    assert A[4, 4] == pytest.approx(ap * d**3 / 3.0, rel=2e-2)
    assert np.allclose(A, A.T)


def test_rigid_body_inertia_single_cylinder_closed_form(tmp_path) -> None:
    """Thin-wall steel cylinder: mass = ρ·πD·t·L, c.g. at mid-length,
    translational 6×6 block = m·I₃."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        read_windio_floating,
        rigid_body_inertia,
    )

    r, d, h, t, rho_s = 5.0, 20.0, 10.0, 0.05, 7800.0
    p = tmp_path / "c.yaml"
    p.write_text(_cyl_yaml(r, -d, h), encoding="utf-8")
    m, M, cg = rigid_body_inertia(read_windio_floating(p), n_seg=400)

    L = d + h
    expect = rho_s * np.pi * (2.0 * r) * t * L
    assert m == pytest.approx(expect, rel=1e-3)
    np.testing.assert_allclose(cg, [0.0, 0.0, 0.5 * (-d + h)], atol=1e-6)
    assert M[0, 0] == pytest.approx(m, rel=1e-3)
    assert M[1, 1] == pytest.approx(m, rel=1e-3)
    assert M[2, 2] == pytest.approx(m, rel=1e-3)
    assert np.allclose(M, M.T)


def test_rigid_body_inertia_counts_fixed_ballast_skips_variable(
    tmp_path,
) -> None:
    """Fixed ballast (explicit volume × material ρ) adds mass; a
    variable-flag ballast entry is excluded (it is an assembled-
    turbine trim quantity, see the docstring)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import (
        read_windio_floating,
        rigid_body_inertia,
    )

    no_ballast = textwrap.dedent("""\
        components:
          floating_platform:
            joints:
              - {name: k, location: [0.0, 0.0, -20.0]}
              - {name: t, location: [0.0, 0.0, 5.0]}
            members:
              - name: c
                joint1: k
                joint2: t
                outer_shape:
                  shape: circular
                  outer_diameter: {grid: [0.0, 1.0], values: [10.0, 10.0]}
                structure:
                  layers:
                    - {name: w, material: steel,
                       thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}}
          mooring: {nodes: [], lines: []}
        materials:
          - {name: steel, rho: 7800.0, E: 2.0e11}
          - {name: slurry, rho: 2500.0}
        """)
    with_ballast = textwrap.dedent("""\
        components:
          floating_platform:
            joints:
              - {name: k, location: [0.0, 0.0, -20.0]}
              - {name: t, location: [0.0, 0.0, 5.0]}
            members:
              - name: c
                joint1: k
                joint2: t
                outer_shape:
                  shape: circular
                  outer_diameter: {grid: [0.0, 1.0], values: [10.0, 10.0]}
                structure:
                  layers:
                    - {name: w, material: steel,
                       thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}}
                  ballast:
                    - {variable_flag: false, material: slurry,
                       volume: 100.0, grid: [0.0, 0.1]}
                    - {variable_flag: true, grid: [0.1, 0.3]}
          mooring: {nodes: [], lines: []}
        materials:
          - {name: steel, rho: 7800.0, E: 2.0e11}
          - {name: slurry, rho: 2500.0}
        """)
    pa = tmp_path / "no.yaml"
    pb = tmp_path / "yes.yaml"
    pa.write_text(no_ballast, encoding="utf-8")
    pb.write_text(with_ballast, encoding="utf-8")
    m_no = rigid_body_inertia(read_windio_floating(pa))[0]
    m_yes = rigid_body_inertia(read_windio_floating(pb))[0]
    # Fixed ballast adds exactly 2500·100 kg; the variable entry adds 0.
    assert m_yes - m_no == pytest.approx(2500.0 * 100.0, rel=1e-6)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_FLOAT_Y.is_file() and _IEA15_HD.is_file()
         and _IEA15_ED_F.is_file()),
    reason="IEA-15 VolturnUS-S yaml / UMaineSemi HydroDyn+ElastoDyn absent",
)
def test_added_mass_and_mass_iea15_documented_bounds() -> None:
    """IEA-15 UMaine VolturnUS-S, vs the companion decks. These are
    the *documented-approximate* parts of the WindIO floating path
    (the deck-fallback supplies the exact matrices):

    * Morison + RAFT end-cap A_inf vs potential-flow WAMIT: surge /
      sway / yaw within ~30 %, roll / pitch ~25 % (improved by the
      RAFT ``Ca_End`` end-cap term, 36→25 %), heave still within a
      factor ~2 (~53 %) — a complex heave-plate semi needs BEM for
      accurate heave, which is why RAFT itself uses potential flow
      there and the WAMIT deck-fallback is the accurate path.
    * Structural + *fixed*-ballast mass is a deliberate lower bound
      on the ElastoDyn ``PtfmMass`` — the difference is the variable
      (trim) ballast, an assembled-turbine equilibrium quantity not
      in the floating component (measured 6.5e6 vs 17.8e6 kg)."""
    pytest.importorskip("yaml")
    import re

    from pybmodes.io.wamit_reader import HydroDynReader
    from pybmodes.io.windio_floating import (
        added_mass,
        read_windio_floating,
        rigid_body_inertia,
    )

    fl = read_windio_floating(_IEA15_FLOAT_Y)
    A = added_mass(fl)
    Aw = np.asarray(
        HydroDynReader(_IEA15_HD).read_platform_matrices().A_inf, float
    )

    def rel(i):
        return abs(A[i, i] - Aw[i, i]) / abs(Aw[i, i])

    for i in (0, 1, 5):                       # surge, sway, yaw
        assert rel(i) < 0.45
    for i in (2, 3, 4):                       # heave, roll, pitch
        assert 0.4 < A[i, i] / Aw[i, i] < 2.5   # factor-~2 envelope
    # Symmetric to the matrix scale (float accumulation over the
    # member segmentation leaves ~1e-6 dust against ~1e10 entries).
    assert np.max(np.abs(A - A.T)) < 1e-9 * np.max(np.abs(A))

    m, _M, cg = rigid_body_inertia(fl)
    ptfm_mass = float(re.search(
        r"([-\d.eE+]+)\s+PtfmMass",
        _IEA15_ED_F.read_text(errors="ignore")).group(1))
    assert 0.0 < m < ptfm_mass               # lower bound (no trim ballast)
    assert m > 0.20 * ptfm_mass              # but a substantial fraction
    assert cg[2] < 0.0                        # c.g. below MSL


# ---------------------------------------------------------------------------
# Mooring from WindIO (catenary engine reuse) (default; no data)
# ---------------------------------------------------------------------------

_MOOR_FLOAT = textwrap.dedent("""\
    components:
      floating_platform:
        joints:
          - {name: a1, location: [800.0, 0.0, -200.0]}
          - {name: a2, location: [-400.0, 692.82, -200.0]}
          - {name: a3, location: [-400.0, -692.82, -200.0]}
          - {name: keel, location: [0.0, 0.0, -20.0]}
          - {name: top, location: [0.0, 0.0, 15.0], transition: true}
        members:
          - name: col
            joint1: keel
            joint2: top
            outer_shape:
              shape: circular
              outer_diameter: {grid: [0.0, 1.0], values: [10.0, 10.0]}
            axial_joints:
              - {name: f1, grid: 0.2}
              - {name: f2, grid: 0.2}
              - {name: f3, grid: 0.2}
            structure:
              layers:
                - {name: w, material: steel,
                   thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}}
      mooring:
        nodes:
          - {name: na1, node_type: fixed,  joint: a1}
          - {name: na2, node_type: fixed,  joint: a2}
          - {name: na3, node_type: fixed,  joint: a3}
          - {name: nf1, node_type: vessel, joint: f1}
          - {name: nf2, node_type: vessel, joint: f2}
          - {name: nf3, node_type: vessel, joint: f3}
        lines:
          - {name: l1, node1: na1, node2: nf1, line_type: chain,
             unstretched_length: 850.0}
          - {name: l2, node1: na2, node2: nf2, line_type: chain,
             unstretched_length: 850.0}
          - {name: l3, node1: na3, node2: nf3, line_type: chain,
             unstretched_length: 850.0}
        line_types:
          - {name: chain, diameter: 0.185, type: chain,
             mass_density: 686.0, EA: 3.27e9}
    materials:
      - {name: steel, rho: 7800.0, E: 2.0e11}
    """)


def test_from_windio_mooring_topology_and_props(tmp_path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.mooring import MooringSystem

    f = _read(tmp_path, _MOOR_FLOAT)
    ms = MooringSystem.from_windio_mooring(f, depth=200.0)
    assert len(ms.lines) == 3
    nf = sum(p.attachment == "Fixed" for p in ms.points.values())
    nv = sum(p.attachment == "Vessel" for p in ms.points.values())
    assert (nf, nv) == (3, 3)
    lt = ms.line_types["chain"]
    assert lt.diam == 0.185
    assert lt.mass_per_length_air == pytest.approx(686.0)
    assert pytest.approx(3.27e9) == lt.EA
    # wet weight from explicit mass: (m − ρ·π/4·d²)·g
    w_exp = (686.0 - 1025.0 * 0.25 * np.pi * 0.185**2) * 9.80665
    assert lt.w == pytest.approx(w_exp, rel=1e-6)


def test_from_windio_mooring_stiffness_symmetry(tmp_path) -> None:
    """A 120°-symmetric 3-line system → symmetric 6×6 with equal,
    positive surge/sway stiffness (the catenary engine is the
    validated `pybmodes.mooring` one — reused unchanged)."""
    pytest.importorskip("yaml")
    from pybmodes.mooring import MooringSystem

    ms = MooringSystem.from_windio_mooring(
        _read(tmp_path, _MOOR_FLOAT), depth=200.0
    )
    K = ms.stiffness_matrix()
    assert K.shape == (6, 6)
    assert np.max(np.abs(K - K.T)) < 1e-6 * np.max(np.abs(K))
    assert K[0, 0] > 0.0 and K[1, 1] > 0.0
    assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-2)   # 120° symmetry


# Strip the explicit mass/EA from the single chain line_type
# (whitespace-agnostic; the flow map spans two lines).
_MOOR_BARE_LT = re.sub(
    r"\{name: chain,.*?\}",
    "{name: chain, diameter: 0.185, type: chain}",
    _MOOR_FLOAT,
    flags=re.S,
)
assert "mass_density" not in _MOOR_BARE_LT and "diameter: 0.185" in \
    _MOOR_BARE_LT  # guard the fixture edit actually applied


def test_from_windio_mooring_regression_warns(tmp_path) -> None:
    """No explicit mass/EA and no MoorDyn fallback → the rough
    studless-chain diameter regression with a clear UserWarning."""
    pytest.importorskip("yaml")
    from pybmodes.mooring import MooringSystem

    with pytest.warns(UserWarning, match="studless-chain|moordyn_fallback"):
        ms = MooringSystem.from_windio_mooring(
            _read(tmp_path, _MOOR_BARE_LT), depth=200.0
        )
    assert ms.line_types["chain"].mass_per_length_air > 0.0
    assert ms.line_types["chain"].EA > 0.0


def test_from_windio_mooring_bad_refs_raise(tmp_path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.mooring import MooringSystem

    bad_joint = _MOOR_FLOAT.replace("joint: a1", "joint: ghost")
    with pytest.raises(KeyError, match="ghost|joints"):
        MooringSystem.from_windio_mooring(
            _read(tmp_path, bad_joint), depth=200.0
        )
    bad_lt = _MOOR_FLOAT.replace("node2: nf1, line_type: chain",
                                 "node2: nf1, line_type: nope")
    with pytest.raises(KeyError, match="nope|line_type"):
        MooringSystem.from_windio_mooring(
            _read(tmp_path, bad_lt), depth=200.0
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_FLOAT_Y.is_file() and _IEA15_MD.is_file()),
    reason="IEA-15 VolturnUS-S yaml / UMaineSemi MoorDyn deck absent",
)
def test_from_windio_mooring_vs_from_moordyn_iea15() -> None:
    """Cross-path consistency anchor (IEA-15 UMaine VolturnUS-S):
    `from_windio_mooring` (WindIO topology + deck-fallback line
    props) vs `from_moordyn` (the same deck). With identical line
    properties (deck-fallback is exact) and the *same* reused
    catenary engine, the only difference is geometry — WindIO column-
    centreline axial joints vs the MoorDyn explicit fairlead
    attachment radius. Roll/pitch/heave/yaw agree to ≤ ~15 %;
    surge/sway is the most fairlead-radius-sensitive (~32 %, the
    column-radius offset) and is bounded, not a model error."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_floating import read_windio_floating
    from pybmodes.mooring import MooringSystem

    ref = MooringSystem.from_moordyn(_IEA15_MD)
    fl = read_windio_floating(_IEA15_FLOAT_Y)
    ws = MooringSystem.from_windio_mooring(
        fl, depth=ref.depth, moordyn_fallback=_IEA15_MD
    )

    # Line properties resolved via deck-fallback must be exact.
    for nm, lt in ws.line_types.items():
        rt = ref.line_types.get(nm) or next(iter(ref.line_types.values()))
        assert lt.mass_per_length_air == pytest.approx(
            rt.mass_per_length_air, rel=1e-9)
        assert pytest.approx(rt.EA, rel=1e-9) == lt.EA
        assert lt.w == pytest.approx(rt.w, rel=1e-9)

    Kw = ws.stiffness_matrix()
    Km = ref.stiffness_matrix()
    assert np.max(np.abs(Kw - Kw.T)) < 1e-6 * np.max(np.abs(Kw))
    assert np.all(np.diag(Kw)[:5] > 0.0)

    def rel(i):
        return abs(Kw[i, i] - Km[i, i]) / abs(Km[i, i])

    assert rel(3) < 0.15 and rel(4) < 0.15      # roll / pitch (~3 %)
    assert rel(2) < 0.15                        # heave (~9 %)
    assert rel(5) < 0.20                        # yaw (~11 %)
    assert rel(0) < 0.40 and rel(1) < 0.40      # surge/sway (~32 %)


# ---------------------------------------------------------------------------
# Tower.from_windio_floating — coupled assembly + modes
# ---------------------------------------------------------------------------

_FLOAT_TURBINE = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [10.0, 6.5]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: w, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.05, 0.02]}}
        reference_axis:
          z: {grid: [0.0, 1.0], values: [12.0, 140.0]}
      floating_platform:
        transition_piece_mass: 100000.0
        joints:
          - {name: keel, location: [0.0, 0.0, -8.0]}
          - {name: tp,   location: [0.0, 0.0, 12.0], transition: true}
          - {name: a1, location: [800.0, 0.0, -200.0]}
          - {name: a2, location: [-400.0, 692.82, -200.0]}
          - {name: a3, location: [-400.0, -692.82, -200.0]}
        members:
          - name: col
            joint1: keel
            joint2: tp
            Ca: [1.0, 1.0]
            outer_shape:
              shape: circular
              outer_diameter: {grid: [0.0, 1.0], values: [40.0, 40.0]}
            axial_joints:
              - {name: f1, grid: 0.3}
              - {name: f2, grid: 0.3}
              - {name: f3, grid: 0.3}
            structure:
              layers:
                - {name: w, material: steel,
                   thickness: {grid: [0.0, 1.0], values: [0.05, 0.05]}}
      mooring:
        nodes:
          - {name: na1, node_type: fixed,  joint: a1}
          - {name: na2, node_type: fixed,  joint: a2}
          - {name: na3, node_type: fixed,  joint: a3}
          - {name: nf1, node_type: vessel, joint: f1}
          - {name: nf2, node_type: vessel, joint: f2}
          - {name: nf3, node_type: vessel, joint: f3}
        lines:
          - {name: l1, node1: na1, node2: nf1, line_type: chain,
             unstretched_length: 850.0}
          - {name: l2, node1: na2, node2: nf2, line_type: chain,
             unstretched_length: 850.0}
          - {name: l3, node1: na3, node2: nf3, line_type: chain,
             unstretched_length: 850.0}
        line_types:
          - {name: chain, diameter: 0.185, type: chain,
             mass_density: 686.0, EA: 3.27e9}
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def test_from_windio_floating_yaml_only_modal_smoke(tmp_path) -> None:
    """End-to-end (no external data): a hydrostatically-stable
    single-column FOWT assembled purely from WindIO yaml solves to a
    physical coupled spectrum — soft platform rigid-body modes well
    below the first tower-bending pair, finite and ascending. (A
    *symmetric* single column with three centreline-coincident
    fairleads genuinely has ~zero yaw restoring, so the softest mode
    sits at ≈ 0 — correct physics for this minimal geometry, not a
    defect; real spread-fairlead FOWTs restrain all six DOF, which the
    IEA-15 integration anchor exercises.)"""
    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    # No companion decks → every platform leg is the WindIO screening
    # model, which must announce itself as NOT industry-grade.
    with pytest.warns(UserWarning, match="SCREENING-fidelity"):
        f = Tower.from_windio_floating(p, water_depth=200.0).run(
            n_modes=10, check_model=False
        ).frequencies
    f = np.asarray(f, float)
    assert np.all(np.isfinite(f))
    assert np.all(f >= -1e-9)               # ≥0 (zero-yaw mode allowed)
    assert np.all(np.diff(f) >= -1e-9)      # ascending
    # Soft platform rigid-body modes far below the 1st tower-bending
    # pair.
    assert f[5] < 0.5
    assert f[6] > 5.0 * f[5]


def test_from_windio_floating_screening_honors_rna_tip(tmp_path) -> None:
    """Regression (issue #83): in the screening tier (no companion
    ElastoDyn deck, no injected ``platform_support``) the caller-
    supplied ``rna_tip`` must be carried onto the tower top, not
    silently replaced with a zero lump. Pre-fix the screening branch
    re-assigned ``rna_tip`` to an all-zero ``TipMassProps``, dropping
    the argument; the injected-platform branch already honoured it."""
    pytest.importorskip("yaml")
    from pybmodes.io.bmi import TipMassProps
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")

    rna = TipMassProps(
        mass=1.2e6, cm_offset=0.0, cm_axial=4.0,
        ixx=5.0e7, iyy=3.0e7, izz=3.0e7, ixy=0.0, izx=0.0, iyz=0.0,
    )
    with pytest.warns(UserWarning, match="SCREENING-fidelity"):
        m = Tower.from_windio_floating(p, water_depth=200.0, rna_tip=rna)
    # The passed RNA lump survives onto the assembled tower top.
    assert m._bmi.tip_mass.mass == pytest.approx(1.2e6)
    assert m._bmi.tip_mass.cm_axial == pytest.approx(4.0)
    assert m._bmi.tip_mass.iyy == pytest.approx(3.0e7)

    # Default (no rna_tip) still yields a bare tower top — the zero
    # default is preserved, only the silent override is removed.
    with pytest.warns(UserWarning, match="SCREENING-fidelity"):
        m0 = Tower.from_windio_floating(p, water_depth=200.0)
    assert m0._bmi.tip_mass.mass == 0.0


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_FLOAT_Y.is_file() and _IEA15_HD.is_file()
         and _IEA15_MD.is_file() and _IEA15_ED_F.is_file()),
    reason="IEA-15 VolturnUS-S yaml / UMaineSemi companion decks absent",
)
def test_from_windio_floating_iea15_vs_from_elastodyn_with_mooring() -> None:
    """Industry-grade coupled anchor (IEA-15 UMaine VolturnUS-S):
    with all companion decks supplied, `from_windio_floating` uses the
    *complete* deck model (full MoorDyn system, WAMIT A_inf+C_hst,
    ElastoDyn PtfmMass/RIner+RNA+draft) — byte-identical to the
    BModes-JJ-validated `from_elastodyn_with_mooring` except the tower
    is the (Phase-1 machine-exact) WindIO one. So:

    * every platform rigid-body mode (surge/sway/heave/roll/pitch/yaw)
      AND the 1st tower fore-aft/side-side bending match to ≤ 1 %
      (measured 0.0–0.3 %) — reference grade;
    * higher tower harmonics (2nd+ bending) carry only the Phase-1
      WindIO-vs-ElastoDyn *tower-discretisation* residual (≤ 8 %
      measured ≤ 6 %), orthogonal to the platform;
    * NO screening UserWarning — all legs are deck-backed."""
    pytest.importorskip("yaml")
    import warnings

    from pybmodes.models import Tower

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no preview warn
        fw = np.asarray(Tower.from_windio_floating(
            _IEA15_FLOAT_Y, hydrodyn_dat=_IEA15_HD,
            moordyn_dat=_IEA15_MD, elastodyn_dat=_IEA15_ED_F,
        ).run(n_modes=12, check_model=False).frequencies, float)
    fe = np.asarray(Tower.from_elastodyn_with_mooring(
        _IEA15_ED_F, _IEA15_MD, _IEA15_HD,
    ).run(n_modes=12, check_model=False).frequencies, float)

    assert np.all(np.isfinite(fw)) and np.all(fw > 0.0)
    assert np.all(np.diff(fw) >= -1e-9)

    def rel(i):
        return abs(fw[i] - fe[i]) / abs(fe[i])

    # modes 0-5 platform rigid body, 6,7 1st tower FA/SS bending:
    # reference grade (measured ≤ 0.3 %).
    for i in range(8):
        assert rel(i) < 0.01, f"mode {i} off by {rel(i):.2%}"
    # 2nd+ tower harmonics: the documented Phase-1 WindIO-vs-ED tower
    # discretisation residual only (NOT a platform-fidelity issue).
    for i in range(8, min(len(fw), len(fe))):
        assert rel(i) < 0.08


def test_from_windio_floating_missing_deck_fails_fast(tmp_path) -> None:
    """An explicitly-supplied companion deck that doesn't exist is a
    single clear FileNotFoundError naming the offending argument —
    NOT a deep stack trace from inside from_moordyn, and NOT a silent
    degrade to screening (which would hide a typo and hand back
    wrong-fidelity results)."""
    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    with pytest.raises(FileNotFoundError,
                       match=r"companion deck\(s\) not found.*moordyn_dat"):
        Tower.from_windio_floating(
            p, water_depth=200.0,
            moordyn_dat=str(tmp_path / "nope_MoorDyn.dat"),
        )


# ---------------------------------------------------------------------------
# Injected-platform tier (issue #35): WindIO tower + caller-supplied
# PlatformSupport (floater designed separately)
# ---------------------------------------------------------------------------

def _stable_platform_support():
    """A hydrostatically-stable diagonal PlatformSupport — soft
    rigid-body modes well below the first tower-bending pair."""
    from pybmodes.io import PlatformSupport

    m = 1.5e7
    i6 = np.diag([m, m, m, 8.0e9, 8.0e9, 1.3e10])
    hydro_K = np.diag([0.0, 0.0, 3.5e6, 1.0e9, 1.0e9, 0.0])
    mooring_K = np.diag([4.0e4, 4.0e4, 1.2e4, 8.0e7, 8.0e7, 1.0e7])
    a_inf = np.diag([9.0e6, 9.0e6, 9.0e6, 7.0e9, 7.0e9, 9.0e9])
    return PlatformSupport(
        draft=-20.0, cm_pform=-14.0, mass_pform=m, i_matrix=i6,
        ref_msl=0.0, hydro_M=a_inf, hydro_K=hydro_K, mooring_K=mooring_K,
        distr_m_z=np.zeros(0), distr_m=np.zeros(0),
        distr_k_z=np.zeros(0), distr_k=np.zeros(0),
    )


def test_from_windio_floating_injected_platform_support(tmp_path) -> None:
    """WindIO tower geometry + a caller-supplied PlatformSupport
    (floater designed separately) → the validated free-free FEM, no
    screening warning, soft rigid-body modes below tower bending."""
    import warnings

    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    ps = _stable_platform_support()

    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        m = Tower.from_windio_floating(p, platform_support=ps)
    assert not [w for w in wlog
                if "SCREENING" in str(w.message)]      # caller owns fidelity
    assert m._bmi.support is ps                        # injected verbatim
    assert m._bmi.hub_conn == 2 and m._bmi.tow_support == 1

    res = m.run(n_modes=12, check_model=False)
    f = np.asarray(res.frequencies, dtype=float)
    assert np.all(np.isfinite(f)) and np.all(f >= -1e-9)
    assert np.all(np.diff(f) >= -1e-6)                  # ascending
    # Lowest modes are soft platform rigid body; first tower bending
    # sits far above them.
    assert f[0] < 0.2 < f[8]


def test_injected_platform_tower_length_is_draft_invariant(tmp_path) -> None:
    """Static-review P1 regression (shipped in v1.4.2, fixed 1.4.3):
    the FEM beam length is ``radius + draft - hub_rad``
    (``make_params``), so the injected branch must pass ``radius =
    flexible_length - draft`` for it to cancel to ``flexible_length``.

    Structural-invariant regression: for *any* supplied platform
    ``draft`` the constructed ``bmi.radius + bmi.support.draft`` must
    equal the WindIO ``flexible_length`` (with ``hub_rad == 0``), so
    the FEM beam length is the true tower length regardless of the
    floater's draft. Pre-fix the injected branch passed ``radius =
    flexible_length`` so a non-zero draft shortened the beam by
    ``|draft|`` and shifted every modal frequency; the deck path
    cancels the same way via ``radius = tower_top``.

    (A draft-invariant *spectrum* is **not** the right contract: the
    rigid-body modes legitimately move with draft through the
    ``cm_pform - draft`` rigid-arm lever — that is correct physics,
    not the bug. The beam-length invariant isolates the defect.)"""
    pytest.importorskip("yaml")

    from pybmodes.io import PlatformSupport
    from pybmodes.io.windio import read_windio_tubular
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    flex_len = float(
        read_windio_tubular(p, component="tower").flexible_length
    )

    for draft in (0.0, -10.0, -30.0):
        ps = PlatformSupport(
            draft=draft, cm_pform=0.0, mass_pform=1.0e7,
            i_matrix=np.diag([1.0e7, 1.0e7, 1.0e7, 8e9, 8e9, 1e10]),
            ref_msl=0.0,
            hydro_M=np.diag([9e6, 9e6, 9e6, 7e9, 7e9, 9e9]),
            hydro_K=np.diag([0.0, 0.0, 3.5e6, 1e9, 1e9, 0.0]),
            mooring_K=np.diag([4e4, 4e4, 1.2e4, 8e7, 8e7, 1e7]),
            distr_m_z=np.zeros(0), distr_m=np.zeros(0),
            distr_k_z=np.zeros(0), distr_k=np.zeros(0),
        )
        m = Tower.from_windio_floating(p, platform_support=ps)
        assert m._bmi.hub_rad == 0.0
        assert (m._bmi.radius + m._bmi.support.draft) == pytest.approx(
            flex_len, rel=1e-12
        ), f"draft={draft}: FEM beam length != flexible_length"
        # Sanity: the model still solves to a finite spectrum.
        res = m.run(n_modes=12, check_model=False)
        assert np.all(np.isfinite(np.asarray(res.frequencies)))


def test_from_windio_floating_n_nodes_refines_mesh(tmp_path) -> None:
    """issue #58: ``from_windio_floating`` gains the same ``n_nodes``
    mesh-refinement kwarg as ``from_windio`` / ``from_geometry``. It
    re-grids only the tower beam (the platform assembly is untouched), so
    the FE station count tracks ``n_nodes`` exactly and the coupled model
    still solves. Uses the injected-platform tier (self-contained, no
    screening warning)."""
    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    ps = _stable_platform_support()

    native = Tower.from_windio_floating(p, platform_support=ps)
    refined = Tower.from_windio_floating(p, platform_support=ps, n_nodes=40)

    assert refined._bmi.el_loc.size == 40                 # exact mesh size
    assert refined._bmi.el_loc.size > native._bmi.el_loc.size
    f = np.asarray(
        refined.run(n_modes=12, check_model=False).frequencies, float
    )
    assert np.all(np.isfinite(f)) and np.all(np.diff(f) >= -1e-6)


def test_from_windio_floating_n_nodes_guard(tmp_path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    for bad in (1, 0, -3, 2.5, True):
        with pytest.raises(ValueError, match="n_nodes must be"):
            Tower.from_windio_floating(
                p, platform_support=_stable_platform_support(), n_nodes=bad,
            )


def test_injected_platform_mutually_exclusive_with_decks(tmp_path) -> None:
    """platform_support + a companion deck is ambiguous (two platform
    sources) → a clear ValueError, not a silent precedence rule."""
    pytest.importorskip("yaml")
    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    deck = tmp_path / "MoorDyn.dat"          # must exist so the
    deck.write_text("dummy", encoding="utf-8")  # fail-fast guard passes

    with pytest.raises(ValueError, match="mutually exclusive"):
        Tower.from_windio_floating(
            p, platform_support=_stable_platform_support(),
            moordyn_dat=str(deck),
        )


@pytest.mark.parametrize(
    "z, k, msg",
    [
        (np.array([0.0, 1.0, 2.0]), np.array([1e6, 1e6]),
         "same shape"),                                  # length mismatch
        (np.array([0.0, 1.0]), np.array([1e6, np.inf]),
         "non-finite"),                                  # inf stiffness
        (np.array([0.0, 1.0]), np.array([1e6, -1e6]),
         "non-negative"),                                # negative stiffness
        (np.array([2.0, 0.0, 1.0]), np.array([1e6, 1e6, 1e6]),
         "sorted ascending"),                            # unsorted coords
    ],
)
def test_injected_platform_distr_k_is_validated(tmp_path, z, k, msg) -> None:
    """A hand-injected PlatformSupport with an inconsistent
    distributed-soil array must fail loudly in the FEM pipeline, not
    poison the matrices (review Medium D)."""
    pytest.importorskip("yaml")
    import dataclasses

    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    ps = dataclasses.replace(
        _stable_platform_support(), distr_k_z=z, distr_k=k,
    )
    m = Tower.from_windio_floating(p, platform_support=ps)
    with pytest.raises(ValueError, match=msg):
        m.run(n_modes=8, check_model=False)


def test_injected_platform_distr_k_accepts_python_list(tmp_path) -> None:
    """A *valid* distr_k supplied as a Python list (not an ndarray)
    must work: validation coerces it and the FEM uses the coerced
    array, not the raw list (review Low/Medium #5 — `list / rmom2`
    would otherwise raise mid-pipeline)."""
    pytest.importorskip("yaml")
    import dataclasses

    from pybmodes.models import Tower

    p = tmp_path / "fowt.yaml"
    p.write_text(_FLOAT_TURBINE, encoding="utf-8")
    ps = dataclasses.replace(
        _stable_platform_support(),
        distr_k_z=[0.0, 20.0, 40.0],          # plain lists, ascending
        distr_k=[3.0e7, 2.0e7, 1.0e7],
    )
    m = Tower.from_windio_floating(p, platform_support=ps)
    res = m.run(n_modes=8, check_model=False)
    assert np.all(np.isfinite(np.asarray(res.frequencies)))
