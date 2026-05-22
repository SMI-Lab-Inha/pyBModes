"""Geometry / WindIO tower input (1.4.0, issue #35).

`Tower.from_geometry` derives structural properties from a tube's
outer diameter + wall thickness + material; `Tower.from_windio` reads
the structural subset of a WindIO ontology .yaml and feeds it through.

Default-suite coverage (self-contained):
1. Closed-form: a uniform steel tube cantilever reproduces the
   Euler-Bernoulli analytic 1st bending frequency.
2. `tubular_section_props` unit values vs hand-computed tube formulae,
   and the outfitting_factor mass-only / stiffness-untouched split.
3. WindIO parser on a hand-written minimal yaml fixture: layer-sum,
   material lookup, span from reference axis, and the
   linear-vs-piecewise-constant thickness interpretation actually
   differing on a tapered profile (the open question from the #35
   thread with Hisham Tariq — quantified here).

Integration (upstream WindIO + ElastoDyn for the SAME turbine):
4. The IEA-15 base WindIO tower, run through the geometry pipeline,
   reproduces the IEA-15 *Monopile* ElastoDyn deck's tabulated
   distributed mass / EI to ~0 % (the ElastoDyn deck was itself
   generated from this WindIO geometry) — an RNA-independent, exact
   like-for-like anchor — plus a `from_windio` modal smoke.
"""

from __future__ import annotations

import math
import pathlib
import textwrap

import numpy as np
import pytest

from pybmodes.io.geometry import tubular_section_props
from pybmodes.models import Tower

# ---------------------------------------------------------------------------
# 1. Closed-form: uniform steel tube cantilever vs Euler-Bernoulli
# ---------------------------------------------------------------------------

def test_uniform_tube_cantilever_matches_euler_bernoulli() -> None:
    D, t, L = 6.0, 0.03, 80.0
    E, rho = 2.0e11, 7850.0
    ro, ri = D / 2, D / 2 - t
    i_area = 0.25 * math.pi * (ro**4 - ri**4)
    area = math.pi * (ro**2 - ri**2)
    f1 = (1.875104**2) / (2 * math.pi * L**2) * math.sqrt(
        E * i_area / (rho * area)
    )
    n = 41
    grid = np.linspace(0.0, 1.0, n)
    res = Tower.from_geometry(
        grid, np.full(n, D), np.full(n, t),
        flexible_length=L, E=E, rho=rho,
    ).run(n_modes=4, check_model=False)
    assert res.frequencies[0] == pytest.approx(f1, rel=1e-3), (
        res.frequencies[0], f1,
    )


# ---------------------------------------------------------------------------
# 1b. issue #35 — n_nodes mesh refinement + tip_mass kwarg
# ---------------------------------------------------------------------------

def test_n_nodes_no_bias_on_uniform_tube() -> None:
    """Resampling must not bias frequencies: on a *uniform* tube the
    n_nodes path is identical to the native grid (interp of constants
    is exact) and both match Euler-Bernoulli (#35 validation bar)."""
    D, t, L, E, rho = 6.0, 0.03, 80.0, 2.0e11, 7850.0
    ro, ri = D / 2, D / 2 - t
    f1 = (1.875104**2) / (2 * math.pi * L**2) * math.sqrt(
        E * (0.25 * math.pi * (ro**4 - ri**4))
        / (rho * math.pi * (ro**2 - ri**2))
    )
    native = Tower.from_geometry(
        np.linspace(0.0, 1.0, 9), np.full(9, D), np.full(9, t),
        flexible_length=L, E=E, rho=rho,
    ).run(n_modes=4, check_model=False)
    refined = Tower.from_geometry(
        np.linspace(0.0, 1.0, 9), np.full(9, D), np.full(9, t),
        flexible_length=L, E=E, rho=rho, n_nodes=80,
    ).run(n_modes=4, check_model=False)
    assert refined.frequencies[0] == pytest.approx(f1, rel=1e-3)
    # Refinement only adds resolution — no systematic shift.
    assert refined.frequencies[0] == pytest.approx(
        native.frequencies[0], rel=5e-3)


def test_n_nodes_converges_and_resolves_higher_modes_on_taper() -> None:
    """On a tapered tower a finer mesh (a) converges — n_nodes=200 vs
    400 agree tightly on the 1st *and* 4th mode — and (b) resolves the
    higher mode shapes (more samples), the #35 ask."""
    L = 100.0
    g0 = np.linspace(0.0, 1.0, 6)
    od = np.array([8.0, 7.2, 6.4, 5.6, 4.8, 4.0])
    wt = np.array([0.05, 0.045, 0.04, 0.035, 0.03, 0.025])

    def _solve(nn):
        return Tower.from_geometry(
            g0, od, wt, flexible_length=L, n_nodes=nn,
        ).run(n_modes=4, check_model=False)

    coarse, fine, finer = _solve(12), _solve(200), _solve(400)
    # Self-convergence (no external oracle): the fine pair agree
    # tightly on the 1st and the higher (4th) mode.
    assert fine.frequencies[0] == pytest.approx(
        finer.frequencies[0], rel=2e-3)
    assert fine.frequencies[3] == pytest.approx(
        finer.frequencies[3], rel=2e-2)
    # Coarse is in the same basin (convergent, not divergent).
    assert coarse.frequencies[0] == pytest.approx(
        finer.frequencies[0], rel=5e-2)
    # Higher mode shapes get materially more samples.
    assert (len(finer.shapes[3].span_loc)
            > 3 * len(coarse.shapes[3].span_loc))


def test_n_nodes_guards() -> None:
    g = np.linspace(0.0, 1.0, 5)
    for bad in (1, 0, -3, 2.5, True):
        with pytest.raises(ValueError, match="n_nodes must be"):
            Tower.from_geometry(g, np.full(5, 6.0), np.full(5, 0.03),
                                flexible_length=50.0, n_nodes=bad)


def test_tip_mass_float_equals_tipmassprops_and_lowers_freq() -> None:
    """issue #35: a bare-float ``tip_mass`` (RNA mass, kg) is the
    common case and must equal the explicit TipMassProps form, equal
    the old ``_bmi.tip_mass`` workaround, and lower the 1st
    frequency."""
    from pybmodes.io.bmi import TipMassProps

    D, t, L = 6.0, 0.03, 80.0
    g = np.linspace(0.0, 1.0, 21)
    common = dict(flexible_length=L)

    no_tip = Tower.from_geometry(
        g, np.full(21, D), np.full(21, t), **common
    ).run(n_modes=3, check_model=False)
    by_float = Tower.from_geometry(
        g, np.full(21, D), np.full(21, t), tip_mass=2.0e5, **common
    ).run(n_modes=3, check_model=False)
    by_props = Tower.from_geometry(
        g, np.full(21, D), np.full(21, t),
        tip_mass=TipMassProps(mass=2.0e5, cm_offset=0.0, cm_axial=0.0,
                              ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0,
                              izx=0.0, iyz=0.0),
        **common,
    ).run(n_modes=3, check_model=False)

    np.testing.assert_allclose(by_float.frequencies,
                               by_props.frequencies, rtol=1e-12)
    assert by_float.frequencies[0] < no_tip.frequencies[0]   # mass softens

    with pytest.raises(ValueError, match="tip_mass"):
        Tower.from_geometry(g, np.full(21, D), np.full(21, t),
                            tip_mass=-5.0, **common)
    with pytest.raises(ValueError, match="tip_mass"):
        Tower.from_geometry(g, np.full(21, D), np.full(21, t),
                            tip_mass="heavy", **common)


def test_from_windio_tip_mass_and_n_nodes_end_to_end(
    tmp_path: pathlib.Path,
) -> None:
    """``Tower.from_windio`` forwards both new kwargs (issue #35):
    n_nodes refines the mesh, a float tip_mass lumps the RNA."""
    pytest.importorskip("yaml")
    p = tmp_path / "min.yaml"
    p.write_text(_MIN_WINDIO, encoding="utf-8")

    base = Tower.from_windio(p).run(n_modes=3, check_model=False)
    refined = Tower.from_windio(p, n_nodes=50).run(
        n_modes=3, check_model=False)
    converged = Tower.from_windio(p, n_nodes=200).run(
        n_modes=3, check_model=False)
    with_rna = Tower.from_windio(p, n_nodes=50, tip_mass=4.0e5).run(
        n_modes=3, check_model=False)

    # The fixture's native grid is only 3 stations — deliberately
    # under-resolved. n_nodes adds resolution and *converges* (50 vs
    # 200 agree tightly); the base 3-node value is the coarse-mesh
    # one #35 asks to refine away (so it legitimately differs — this
    # is the feature working, not a bias).
    assert len(refined.shapes[0].span_loc) > len(base.shapes[0].span_loc)
    assert refined.frequencies[0] == pytest.approx(
        converged.frequencies[0], rel=5e-3)
    # RNA lump softens the tower.
    assert with_rna.frequencies[0] < refined.frequencies[0]


# ---------------------------------------------------------------------------
# 2. tubular_section_props unit values + outfitting split
# ---------------------------------------------------------------------------

def test_tubular_section_props_closed_form_values() -> None:
    D, t, E, rho, nu = 8.0, 0.05, 2.0e11, 7800.0, 0.3
    sp = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, D), np.full(2, t),
        E=E, rho=rho, nu=nu,
    )
    ro, ri = D / 2, D / 2 - t
    area = math.pi * (ro**2 - ri**2)
    i_area = 0.25 * math.pi * (ro**4 - ri**4)
    G = E / (2 * (1 + nu))
    assert sp.mass_den[0] == pytest.approx(rho * area)
    assert sp.flp_stff[0] == pytest.approx(E * i_area)
    assert sp.edge_stff[0] == pytest.approx(E * i_area)        # FA == SS
    assert sp.tor_stff[0] == pytest.approx(G * 2 * i_area)     # J = 2I
    assert sp.axial_stff[0] == pytest.approx(E * area)
    assert sp.flp_iner[0] == pytest.approx(rho * i_area)


def test_outfitting_factor_scales_mass_not_stiffness() -> None:
    args = dict(E=2.0e11, rho=7800.0, nu=0.3)
    base = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, 8.0), np.full(2, 0.05), **args
    )
    fat = tubular_section_props(
        np.array([0.0, 1.0]), np.full(2, 8.0), np.full(2, 0.05),
        outfitting_factor=1.07, **args
    )
    assert fat.mass_den[0] == pytest.approx(1.07 * base.mass_den[0])
    assert fat.flp_iner[0] == base.flp_iner[0]                 # structural
    assert fat.flp_stff[0] == base.flp_stff[0]                 # untouched
    assert fat.axial_stff[0] == base.axial_stff[0]
    assert fat.tor_stff[0] == base.tor_stff[0]


def test_tubular_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        tubular_section_props(np.array([0.0, 1.0]), np.array([6.0, 6.0]),
                              np.array([0.0, 0.03]), E=2e11, rho=7800.0)
    with pytest.raises(ValueError, match="2.t < outer_diameter|outer radius"):
        tubular_section_props(np.array([0.0, 1.0]), np.array([6.0, 6.0]),
                              np.array([3.5, 3.5]), E=2e11, rho=7800.0)


# ---------------------------------------------------------------------------
# 3. WindIO parser on a minimal hand-written fixture
# ---------------------------------------------------------------------------

_MIN_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter:
            grid: [0.0, 0.5, 1.0]
            values: [8.0, 7.0, 6.0]
        structure:
          outfitting_factor: 1.1
          layers:
            - name: tower_wall
              material: steel
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
        reference_axis:
          z:
            grid: [0.0, 1.0]
            values: [20.0, 120.0]
    materials:
      - name: steel
        E: 2.0e11
        rho: 7800.0
        nu: 0.3
        G: 7.7e10
    """)


def test_windio_parser_minimal(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "min.yaml"
    p.write_text(_MIN_WINDIO, encoding="utf-8")
    g = read_windio_tubular(p, component="tower")

    assert g.E == 2.0e11 and g.rho == 7800.0 and g.nu == 0.3
    assert g.outfitting_factor == 1.1
    assert g.flexible_length == pytest.approx(100.0)        # |120 - 20|
    np.testing.assert_allclose(g.station_grid, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(g.outer_diameter, [8.0, 7.0, 6.0])
    # linear thickness interp onto the [0,0.5,1] grid: 0.05,0.035,0.02
    np.testing.assert_allclose(g.wall_thickness, [0.05, 0.035, 0.02])


def test_windio_thickness_interp_differs_on_taper(tmp_path: pathlib.Path) -> None:
    """The linear-vs-piecewise-constant choice (the open question
    raised with Hisham Tariq in #35) must measurably move the wall
    thickness — hence the 2nd-mode coefficients — on a *smoothly
    tapered* profile (it is ~0 on IEA-15's step profile, which is why
    this uses a taper)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "taper.yaml"
    p.write_text(_MIN_WINDIO, encoding="utf-8")
    lin = read_windio_tubular(p, component="tower", thickness_interp="linear")
    pc = read_windio_tubular(
        p, component="tower", thickness_interp="piecewise_constant"
    )
    # mid station: linear -> 0.035, piecewise-constant -> 0.05 (lower
    # grid point governs). Materially different.
    assert lin.wall_thickness[1] == pytest.approx(0.035)
    assert pc.wall_thickness[1] == pytest.approx(0.05)
    assert abs(lin.wall_thickness[1] - pc.wall_thickness[1]) > 0.01


# ---------------------------------------------------------------------------
# 3b. Monopile + tower splice (issue #92): Tower.from_windio_with_monopile
# ---------------------------------------------------------------------------

# Two distinct-material segments meeting at the transition piece z = 10 m:
# a uniform 9 m monopile (z -30 -> 10) below a tapered tower (z 10 -> 110).
# Different E per segment proves the splice keeps each component's own
# material rather than reducing both with one.
_MIN_WINDIO_MONOPILE = textwrap.dedent("""\
    components:
      monopile:
        outer_shape:
          outer_diameter:
            grid: [0.0, 1.0]
            values: [9.0, 9.0]
        structure:
          outfitting_factor: 1.0
          layers:
            - name: monopile_wall
              material: steel_mp
              thickness:
                grid: [0.0, 1.0]
                values: [0.08, 0.08]
        reference_axis:
          z:
            grid: [0.0, 1.0]
            values: [-30.0, 10.0]
      tower:
        outer_shape:
          outer_diameter:
            grid: [0.0, 0.5, 1.0]
            values: [9.0, 7.5, 6.0]
        structure:
          outfitting_factor: 1.1
          layers:
            - name: tower_wall
              material: steel_tw
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
        reference_axis:
          z:
            grid: [0.0, 1.0]
            values: [10.0, 110.0]
    materials:
      - name: steel_mp
        E: 2.0e11
        rho: 7850.0
        nu: 0.3
      - name: steel_tw
        E: 2.1e11
        rho: 7800.0
        nu: 0.3
    """)


def _write_monopile_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "monopile_tower.yaml"
    p.write_text(_MIN_WINDIO_MONOPILE, encoding="utf-8")
    return p


def test_monopile_tower_splice_geometry(tmp_path: pathlib.Path) -> None:
    """The splice spans mudline -> tower top, with the transition placed
    at the shared elevation and each segment keeping its own material."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_monopile_tower, read_windio_tubular

    p = _write_monopile_yaml(tmp_path)
    mt = read_windio_monopile_tower(p)

    assert mt.z_base == pytest.approx(-30.0)
    assert mt.z_transition == pytest.approx(10.0)
    assert mt.z_top == pytest.approx(110.0)
    assert mt.combined_length == pytest.approx(140.0)            # 110 - (-30)
    assert mt.transition_frac == pytest.approx(40.0 / 140.0)     # 40 m of pile

    # span_loc strictly increasing; a clean FE node sits at the transition.
    sl = mt.section_props.span_loc
    assert np.all(np.diff(sl) > 0.0)
    assert np.all(np.diff(mt.el_loc) > 0.0)
    assert np.min(np.abs(mt.el_loc - mt.transition_frac)) < 1e-9

    # Endpoints reduce to the standalone monopile / tower sections — i.e.
    # the bottom carries the monopile's (E, wall) and the top the tower's.
    g_mp = read_windio_tubular(p, component="monopile")
    g_tw = read_windio_tubular(p, component="tower")
    sp_mp = tubular_section_props(
        g_mp.station_grid, g_mp.outer_diameter, g_mp.wall_thickness,
        E=g_mp.E, rho=g_mp.rho, nu=g_mp.nu,
        outfitting_factor=g_mp.outfitting_factor,
    )
    sp_tw = tubular_section_props(
        g_tw.station_grid, g_tw.outer_diameter, g_tw.wall_thickness,
        E=g_tw.E, rho=g_tw.rho, nu=g_tw.nu,
        outfitting_factor=g_tw.outfitting_factor,
    )
    assert mt.section_props.mass_den[0] == pytest.approx(sp_mp.mass_den[0])
    assert mt.section_props.flp_stff[0] == pytest.approx(sp_mp.flp_stff[0])
    assert mt.section_props.mass_den[-1] == pytest.approx(sp_tw.mass_den[-1])
    assert mt.section_props.flp_stff[-1] == pytest.approx(sp_tw.flp_stff[-1])
    # Monopile base is far heavier / stiffer than the tower top.
    assert mt.section_props.mass_den[0] > 5.0 * mt.section_props.mass_den[-1]


def test_monopile_tower_non_contiguous_raises(tmp_path: pathlib.Path) -> None:
    """A gap between the monopile top and tower base can't be spliced."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_monopile_tower

    # Tower base bumped to z = 12 (2 m gap above the monopile top at 10).
    bad = _MIN_WINDIO_MONOPILE.replace(
        "values: [10.0, 110.0]", "values: [12.0, 110.0]"
    )
    p = tmp_path / "gap.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="common transition piece"):
        read_windio_monopile_tower(p)


def test_from_windio_with_monopile_solves_and_is_softer(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end: the combined cantilever solves to a physical spectrum,
    is clamped at the mudline, and is softer than the tower-only model
    clamped at the transition piece (the pile adds a long flexible base)."""
    pytest.importorskip("yaml")
    p = _write_monopile_yaml(tmp_path)

    t = Tower.from_windio_with_monopile(p, tip_mass=5.0e5)
    assert t._bmi.hub_conn == 1
    assert t._bmi.radius == pytest.approx(140.0)
    f = t.run(n_modes=6, check_model=False).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)
    # Axisymmetric tube -> degenerate FA/SS first pair.
    assert f[0] == pytest.approx(f[1], rel=1e-6)

    f_tower = Tower.from_windio(
        p, component="tower", tip_mass=5.0e5
    ).run(n_modes=4, check_model=False).frequencies
    assert f[0] < f_tower[0]


def test_from_windio_with_monopile_tip_mass_and_n_nodes(
    tmp_path: pathlib.Path,
) -> None:
    """A bare-float tip mass matches the explicit TipMassProps, and the
    per-segment n_nodes refinement re-grids each segment."""
    pytest.importorskip("yaml")
    from pybmodes.io.bmi import TipMassProps
    from pybmodes.io.windio import read_windio_monopile_tower

    p = _write_monopile_yaml(tmp_path)

    fa = Tower.from_windio_with_monopile(p, tip_mass=4.2e5).run(
        n_modes=6, check_model=False
    ).frequencies
    fb = Tower.from_windio_with_monopile(
        p,
        tip_mass=TipMassProps(
            mass=4.2e5, cm_offset=0.0, cm_axial=0.0,
            ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
        ),
    ).run(n_modes=6, check_model=False).frequencies
    np.testing.assert_allclose(fa, fb, rtol=1e-12)

    # n_nodes refines each segment to that many stations (2 segments).
    mt = read_windio_monopile_tower(p, n_nodes=25)
    assert mt.section_props.n_secs == 50
    f = Tower.from_windio_with_monopile(p, n_nodes=25, tip_mass=4.2e5).run(
        n_modes=6, check_model=False
    ).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)


def test_from_windio_friendly_error_without_pyyaml(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent PyYAML, the error names the [windio] extra (mirrors the
    matplotlib-gated plots)."""
    import builtins

    real_import = builtins.__import__

    def _no_yaml(name, *a, **k):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_yaml)
    from pybmodes.io.windio import _require_yaml

    with pytest.raises(ModuleNotFoundError, match=r"pybmodes\[windio\]"):
        _require_yaml()


# ---------------------------------------------------------------------------
# 3b. WindIO schema-dialect robustness (synthetic; no external data)
#
# Upstream RWT ontology files come in two key dialects and one strict-
# PyYAML-hostile anchor habit; gate all three in the DEFAULT suite via
# synthetic fixtures so a fresh clone proves the parser without needing
# the gitignored docs/ corpus (independence stance). The integration
# block below then exercises the *real* IEA-3.4/10/15/22 files.
# ---------------------------------------------------------------------------

# Older dialect: outer_shape_bem / internal_structure_2d_fem, with
# reference_axis nested in the shape block and aliased into the
# structure block (exactly IEA-3.4/10/22's layout). Numerically
# identical to _MIN_WINDIO so the two must parse to the same struct.
_OLD_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape_bem:
          reference_axis: &ref_axis_tower
            x: {grid: [0.0, 1.0], values: [0.0, 0.0]}
            y: {grid: [0.0, 1.0], values: [0.0, 0.0]}
            z:
              grid: [0.0, 1.0]
              values: [20.0, 120.0]
          outer_diameter:
            grid: [0.0, 0.5, 1.0]
            values: [8.0, 7.0, 6.0]
        internal_structure_2d_fem:
          outfitting_factor: 1.1
          reference_axis: *ref_axis_tower
          layers:
            - name: tower_wall
              material: steel
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
    materials:
      - name: steel
        E: 2.0e11
        rho: 7800.0
        nu: 0.3
        G: 7.7e10
    """)


def test_windio_older_dialect_matches_modern(tmp_path: pathlib.Path) -> None:
    """The older `outer_shape_bem` / `internal_structure_2d_fem` dialect
    (IEA-3.4/10/22) parses to the *same* WindIOTubular as the modern
    `outer_shape` / `structure` form for numerically identical input."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    pm = tmp_path / "modern.yaml"
    po = tmp_path / "older.yaml"
    pm.write_text(_MIN_WINDIO, encoding="utf-8")
    po.write_text(_OLD_WINDIO, encoding="utf-8")
    m = read_windio_tubular(pm, component="tower")
    o = read_windio_tubular(po, component="tower")

    assert (o.E, o.rho, o.nu, o.outfitting_factor) == (m.E, m.rho, m.nu,
                                                       m.outfitting_factor)
    assert o.flexible_length == pytest.approx(m.flexible_length)
    np.testing.assert_allclose(o.station_grid, m.station_grid)
    np.testing.assert_allclose(o.outer_diameter, m.outer_diameter)
    np.testing.assert_allclose(o.wall_thickness, m.wall_thickness)


# Duplicate anchor `&g` (no alias) — strict PyYAML raises ComposerError;
# ruamel / YAML-1.2 (and WISDEM's IEA-10 file) accept it. The shape-block
# reference_axis is resolved before the structure-block one, so the span
# is deterministically the shape value (|50 - 0| = 50).
_DUP_ANCHOR_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape_bem:
          reference_axis:
            z: &g
              grid: [0.0, 1.0]
              values: [0.0, 50.0]
          outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
        internal_structure_2d_fem:
          outfitting_factor: 1.0
          reference_axis:
            z: &g
              grid: [0.0, 1.0]
              values: [10.0, 110.0]
          layers:
            - name: w
              material: steel
              thickness: {grid: [0.0, 1.0], values: [0.03, 0.03]}
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def test_windio_tolerates_duplicate_anchors(tmp_path: pathlib.Path) -> None:
    """Strict PyYAML rejects a redefined anchor with ComposerError;
    WindIO files from the WISDEM toolchain (IEA-10 reuses `&id004`)
    routinely do this. The duplicate-anchor-tolerant loader must accept
    it (last-wins) and resolve the shape-block reference axis."""
    yaml = pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "dup.yaml"
    p.write_text(_DUP_ANCHOR_WINDIO, encoding="utf-8")

    # Sanity: the stock SafeLoader genuinely chokes on this fixture.
    with pytest.raises(yaml.composer.ComposerError):
        yaml.safe_load(_DUP_ANCHOR_WINDIO)

    g = read_windio_tubular(p, component="tower")
    assert g.flexible_length == pytest.approx(50.0)   # shape block wins
    assert g.E == 2.0e11 and g.rho == 7800.0


_ORTHO_WINDIO = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [6.0, 6.0]}
        structure:
          layers:
            - name: w
              material: triax
              thickness: {grid: [0.0, 1.0], values: [0.03, 0.03]}
        reference_axis:
          z: {grid: [0.0, 1.0], values: [0.0, 100.0]}
    materials:
      - name: triax
        E: [2.0e10, 1.4e10, 1.4e10]
        G: [9.4e9, 4.5e9, 4.5e9]
        rho: 1845.0
        nu: [0.48, 0.48, 0.48]
    """)


def test_windio_rejects_orthotropic_wall_material(
    tmp_path: pathlib.Path,
) -> None:
    """A list-valued (orthotropic composite) wall material — the
    triax/biax entries that sit beside `steel` in every RWT
    materials[] list — must raise a clear out-of-scope error, not a
    bare `float(list)` TypeError."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "ortho.yaml"
    p.write_text(_ORTHO_WINDIO, encoding="utf-8")
    with pytest.raises(ValueError, match="orthotropic"):
        read_windio_tubular(p, component="tower")


def test_windio_missing_shape_and_structure_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A component with neither dialect's blocks names both spellings."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    p = tmp_path / "bad.yaml"
    p.write_text("components:\n  tower:\n    foo: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="outer_shape_bem"):
        read_windio_tubular(p, component="tower")


# ---------------------------------------------------------------------------
# 4. Integration — real upstream WindIO corpus (IEA-3.4 / 10 / 15 / 22)
# ---------------------------------------------------------------------------

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "external" / "OpenFAST_files"
_IEA15_YAML = _DOCS / "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml"
_IEA15_MONO_ED = (
    _DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/"
    "IEA-15-240-RWT-Monopile_ElastoDyn.dat"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA15_YAML.is_file() and _IEA15_MONO_ED.is_file()),
    reason="IEA-15 WindIO yaml / Monopile ElastoDyn deck not present",
)
def test_windio_iea15_matches_elastodyn_section_props() -> None:
    """RNA-independent like-for-like anchor: the IEA-15 base WindIO
    tower, through the geometry pipeline, must reproduce the IEA-15
    Monopile ElastoDyn deck's tabulated distributed mass / EI for the
    same physical tower (that deck was generated from this geometry)."""
    pytest.importorskip("yaml")
    from pybmodes.io._elastodyn.parser import (
        read_elastodyn_main,
        read_elastodyn_tower,
    )
    from pybmodes.io.windio import read_windio_tubular

    g = read_windio_tubular(_IEA15_YAML, component="tower")
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    m = read_elastodyn_main(_IEA15_MONO_ED)
    ted = read_elastodyn_tower(_IEA15_MONO_ED.parent / m.twr_file)
    hf = np.asarray(ted.ht_fract, float)
    ed_mass = np.asarray(ted.t_mass_den, float)
    ed_ei = np.asarray(ted.tw_fa_stif, float)
    w_mass = np.interp(hf, sp.span_loc, sp.mass_den)
    w_ei = np.interp(hf, sp.span_loc, sp.flp_stff)

    assert np.max(np.abs(w_mass - ed_mass) / np.maximum(ed_mass, 1.0)) < 5e-3
    assert np.max(np.abs(w_ei - ed_ei) / np.maximum(ed_ei, 1.0)) < 5e-3


@pytest.mark.integration
@pytest.mark.skipif(
    not _IEA15_YAML.is_file(),
    reason="IEA-15 WindIO yaml not present",
)
def test_from_windio_modal_smoke() -> None:
    """`Tower.from_windio` solves to a physical bare-tower spectrum
    (positive, finite, ascending) for tower and monopile."""
    pytest.importorskip("yaml")
    for comp in ("tower", "monopile"):
        f = Tower.from_windio(_IEA15_YAML, component=comp).run(
            n_modes=4, check_model=False
        ).frequencies
        assert np.all(np.isfinite(f)) and np.all(f > 0.0)
        assert np.all(np.diff(f) >= -1e-9)


@pytest.mark.integration
@pytest.mark.skipif(
    not _IEA15_YAML.is_file(),
    reason="IEA-15 WindIO yaml not present",
)
def test_from_windio_with_monopile_iea15() -> None:
    """The IEA-15 monopile + tower splice (z -75 -> +15 -> +144.386)
    builds one fixed-bottom cantilever and solves to a physical spectrum
    that is softer than the tower alone clamped at the transition piece."""
    pytest.importorskip("yaml")
    from pybmodes.io.bmi import TipMassProps
    from pybmodes.io.windio import read_windio_monopile_tower

    mt = read_windio_monopile_tower(_IEA15_YAML)
    assert mt.z_base == pytest.approx(-75.0)
    assert mt.z_transition == pytest.approx(15.0)
    assert mt.z_top == pytest.approx(144.386)
    assert mt.combined_length == pytest.approx(219.386)

    rna = TipMassProps(
        mass=1.017e6, cm_offset=0.0, cm_axial=0.0,
        ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
    )
    t = Tower.from_windio_with_monopile(_IEA15_YAML, tip_mass=rna, n_nodes=40)
    assert t._bmi.hub_conn == 1
    f = t.run(n_modes=6, check_model=False).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)

    f_tower = Tower.from_windio(
        _IEA15_YAML, component="tower", tip_mass=rna, n_nodes=40
    ).run(n_modes=4, check_model=False).frequencies
    assert f[0] < f_tower[0]


# Full upstream corpus: every RWT ontology .yaml we ship-test against,
# spanning both key dialects and IEA-10's duplicate-anchor habit.
# (id, yaml relative to _DOCS, component). The id encodes the dialect so
# a failure pinpoints which parser path broke.
_WINDIO_CORPUS = [
    ("iea3.4-older-tower",
     "IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml", "tower"),
    ("iea10-older-dupanchor-tower",
     "IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml", "tower"),
    ("iea15-modern-tower",
     "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml", "tower"),
    ("iea15-modern-monopile",
     "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml", "monopile"),
    ("iea22-older-tower",
     "IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "tower"),
    ("iea22-older-monopile",
     "IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "monopile"),
    ("wisdem-nrel5mw-tower",
     "WISDEM/examples/05_tower_monopile/nrel5mw_tower.yaml", "tower"),
    ("wisdem-nrel5mw-monopile",
     "WISDEM/examples/05_tower_monopile/nrel5mw_monopile.yaml", "monopile"),
    ("wisdem-iea3p4-tower",
     "WISDEM/examples/02_reference_turbines/IEA-3p4-130-RWT.yaml", "tower"),
    ("wisdem-iea10-tower",
     "WISDEM/examples/02_reference_turbines/IEA-10-198-RWT.yaml", "tower"),
    ("wisdem-iea15-tower",
     "WISDEM/examples/02_reference_turbines/IEA-15-240-RWT.yaml", "tower"),
    ("wisdem-iea22-tower",
     "WISDEM/examples/02_reference_turbines/IEA-22-280-RWT.yaml", "tower"),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "rel,component",
    [(rel, comp) for _id, rel, comp in _WINDIO_CORPUS],
    ids=[i for i, _r, _c in _WINDIO_CORPUS],
)
def test_windio_corpus_parses_to_sane_tubular(rel: str, component: str) -> None:
    """Every shipped RWT ontology .yaml — both key dialects plus IEA-10's
    redefined-anchor file — parses to a physically sane steel tube:
    monotone-ish grid in [0,1], positive D / t with 2t < D at every
    station, a plausible steel modulus / density, and a positive span."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_tubular

    yaml_path = _DOCS / rel
    if not yaml_path.is_file():
        pytest.skip(f"{rel} not present")
    g = read_windio_tubular(yaml_path, component=component)

    assert g.station_grid[0] == pytest.approx(0.0)
    assert g.station_grid[-1] == pytest.approx(1.0)
    assert np.all(np.diff(g.station_grid) >= -1e-12)
    assert np.all(g.outer_diameter > 0.0)
    assert np.all(g.wall_thickness > 0.0)
    assert np.all(2.0 * g.wall_thickness < g.outer_diameter)
    assert 1.5e11 <= g.E <= 2.2e11          # structural steel, Pa
    assert 7000.0 <= g.rho <= 9000.0        # steel (+ outfit baked in some)
    assert 0.0 < g.nu < 0.5
    assert 0.9 <= g.outfitting_factor <= 1.2
    assert g.flexible_length > 1.0
    # The whole point: geometry -> exact section props, no exception.
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    assert np.all(sp.mass_den > 0.0)
    assert np.all(sp.flp_stff > 0.0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "rel,component",
    [
        ("IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml", "tower"),       # older
        ("IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml", "tower"),       # dup-anc
        ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "tower"),       # older
        ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml", "monopile"),
    ],
    ids=["iea3.4", "iea10", "iea22-tower", "iea22-monopile"],
)
def test_from_windio_corpus_modal_smoke(rel: str, component: str) -> None:
    """`Tower.from_windio` drives the full FEM pipeline to a physical
    bare-member spectrum on the *older* dialect (and IEA-10's
    duplicate-anchor file) — not just IEA-15's modern form."""
    pytest.importorskip("yaml")
    yaml_path = _DOCS / rel
    if not yaml_path.is_file():
        pytest.skip(f"{rel} not present")
    f = Tower.from_windio(yaml_path, component=component).run(
        n_modes=4, check_model=False
    ).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)


# IEA-3.4 / IEA-10 / IEA-22 ship an ElastoDyn `_tower.dat` for the *same*
# turbine, but — unlike IEA-15's Monopile deck — those tables were NOT
# regenerated by a 1:1 geometry round-trip from the ontology yaml, so
# the match is same-turbine *ballpark* (a few % to ~20 %), not the
# machine-exact anchor IEA-15 gives. Asserting a generous envelope still
# catches gross unit / layer-sum / dialect regressions against a wholly
# independent reference. The exact anchor stays IEA-15-only (above).
_BALLPARK = [
    ("IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml",
     "IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT_ElastoDyn_tower.dat"),
    ("IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml",
     "IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT_ElastoDyn_tower.dat"),
    ("IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml",
     "IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT-Monopile/"
     "IEA-22-280-RWT_ElastoDyn_tower.dat"),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "yrel,erel", _BALLPARK, ids=["iea3.4", "iea10", "iea22"]
)
def test_windio_older_dialect_same_turbine_ballpark(
    yrel: str, erel: str
) -> None:
    """Older-dialect yaml-derived distributed mass / EI lands within a
    same-turbine envelope of that turbine's own ElastoDyn tower table —
    an independent cross-check that the older parser path produces
    physically right-sized properties (NOT the exact IEA-15 anchor)."""
    pytest.importorskip("yaml")
    from pybmodes.io._elastodyn.parser import read_elastodyn_tower
    from pybmodes.io.windio import read_windio_tubular

    yaml_path, ed_path = _DOCS / yrel, _DOCS / erel
    if not (yaml_path.is_file() and ed_path.is_file()):
        pytest.skip("yaml / ElastoDyn tower deck not present")

    g = read_windio_tubular(yaml_path, component="tower")
    sp = tubular_section_props(
        g.station_grid, g.outer_diameter, g.wall_thickness,
        E=g.E, rho=g.rho, nu=g.nu, outfitting_factor=g.outfitting_factor,
    )
    ted = read_elastodyn_tower(ed_path)
    hf = np.asarray(ted.ht_fract, float)
    w_mass = np.interp(hf, sp.span_loc, sp.mass_den)
    w_ei = np.interp(hf, sp.span_loc, sp.flp_stff)
    e_m = np.max(np.abs(w_mass - ted.t_mass_den) / ted.t_mass_den)
    e_e = np.max(np.abs(w_ei - ted.tw_fa_stif) / ted.tw_fa_stif)
    assert e_m < 0.25, f"mass off by {e_m:.1%}"
    assert e_e < 0.30, f"EI off by {e_e:.1%}"
