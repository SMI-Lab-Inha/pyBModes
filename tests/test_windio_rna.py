"""WindIO auto-RNA tower-top lumped mass (issue #82).

`read_windio_rna` assembles the hub + nacelle + blades of an IEA-22-class
WindIO ontology (the `elastic_properties_mb` schema) into a single
tower-top `TipMassProps`, mirroring the ElastoDyn assembler
`_tower_top_assembly_mass`. `Tower.from_windio(..., lumped_rna_cal=True)`
and `Tower.from_windio_with_monopile(..., lumped_rna_cal=True)` use it to
fill the RNA lump automatically.

Default-suite coverage is self-contained: a synthetic ontology whose RNA
assembles to hand-computed mass / CM / inertia, plus fail-clean and
input-hardening checks. Integration (IEA-22 upstream ontology + ElastoDyn
deck) checks the mass and CM match the deck to <0.5 %.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "external" / "OpenFAST_files"


def _base_ontology() -> dict:
    """A minimal IEA-22-shaped ontology with the RNA blocks.

    Chosen so the assembled tower-top lump has a clean hand-computation
    (no shaft tilt: ``uptilt = 0``). Nacelle at CM (-4, 0, 3); rotor apex
    (upwind) at (-10, 0, 4). Blade uniform 100 kg/m over a 50 m straight
    span -> 5000 kg each, 3 blades.
    """
    return {
        "assembly": {"number_of_blades": 3, "rotor_orientation": "Upwind"},
        "components": {
            "hub": {
                "cone_angle": 0.0,
                "elastic_properties_mb": {
                    "system_mass": 100000.0,
                    "system_inertia": [2.0e6, 1.0e6, 1.0e6, 0.0, 0.0, 0.0],
                },
            },
            "nacelle": {
                "drivetrain": {
                    "uptilt": 0.0,
                    "distance_tt_hub": 4.0,
                    "overhang": 10.0,
                    "elastic_properties_mb": {
                        "system_mass": 500000.0,
                        "system_center_mass": [-4.0, 0.0, 3.0],
                        "system_inertia": [1.0e7, 1.2e7, 1.3e7, 0.0, 0.0, 0.0],
                    },
                },
            },
            "blade": {
                "elastic_properties_mb": {
                    "six_x_six": {
                        "reference_axis": {
                            "x": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
                            "y": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
                            "z": {"grid": [0.0, 1.0], "values": [0.0, 50.0]},
                        },
                        "inertia_matrix": {
                            "grid": [0.0, 1.0],
                            "values": [
                                [100.0] + [0.0] * 20,
                                [100.0] + [0.0] * 20,
                            ],
                        },
                    },
                },
            },
        },
    }


def _write(onto: dict, tmp_path: pathlib.Path, name: str = "rna.yaml") -> pathlib.Path:
    yaml = pytest.importorskip("yaml")
    p = tmp_path / name
    p.write_text(yaml.safe_dump(onto), encoding="utf-8")
    return p


def test_rna_assembly_matches_hand_computation(tmp_path: pathlib.Path) -> None:
    """The three-body parallel-axis assembly reproduces a hand-computed
    tower-top mass, CM and inertia tensor (issue #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    p = _write(_base_ontology(), tmp_path)
    rna = read_windio_rna(p)

    # mass = 500000 + 100000 + 3*5000
    assert rna.mass == pytest.approx(615000.0)
    # cm_z = (500000*3 + 100000*4 + 15000*4) / 615000
    assert rna.cm_axial == pytest.approx(1.96e6 / 615000.0)
    assert rna.cm_offset == 0.0
    # Parallel-axis tensor at the tower top (hand-computed), plus the rotor
    # diametral inertia from the spanwise blade mass (issue #130):
    # I_polar_rotor = 3 · ∫100·z² dz over [0, 50] = 3 · (100·50³/3)
    #              = 3 · 4.1667e6 = 1.25e7, added about the apex as
    # diag([1.25e7, 6.25e6, 6.25e6]). The exact span integral (Simpson,
    # exact for the cubic mass·z² integrand) replaces the earlier trapezoid.
    assert rna.ixx == pytest.approx(1.834e7 + 1.25e7)
    assert rna.iyy == pytest.approx(3.884e7 + 0.625e7)
    assert rna.izz == pytest.approx(3.35e7 + 0.625e7)
    assert rna.izx == pytest.approx(1.06e7)
    assert rna.ixy == pytest.approx(0.0, abs=1e-6)
    assert rna.iyz == pytest.approx(0.0, abs=1e-6)


def test_rna_rotor_inertia_from_span(tmp_path: pathlib.Path) -> None:
    """The rotor diametral inertia from the spanwise blade mass is included,
    and the hub radius increases it (issue #130). Concentrating the same
    blade mass near the rotor axis (short span) carries far less rotary
    inertia than spreading it along a long span."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    base = read_windio_rna(_write(_base_ontology(), tmp_path, "base_r.yaml"))
    # The pre-#130 value for izz was 3.35e7 (no blade inertia); it must now
    # be strictly larger because the rotor diametral term is included.
    assert base.izz > 3.35e7
    # The diametral term is the exact span integral 3·(100·50³/3)/2 = 6.25e6,
    # not the trapezoid over-estimate (9.375e6) a coarse two-station grid
    # would give for the cubic mass·z² integrand (Codex review on #130).
    assert base.izz == pytest.approx(3.35e7 + 6.25e6)

    # A hub radius offsets every blade section further from the rotor axis,
    # so the polar / diametral inertia grows.
    with_hub = _base_ontology()
    with_hub["components"]["hub"]["diameter"] = 8.0        # hub_R = 4 m
    wh = read_windio_rna(_write(with_hub, tmp_path, "hub_r.yaml"))
    assert wh.izz > base.izz
    assert wh.iyy > base.iyy
    assert wh.mass == pytest.approx(base.mass)             # mass unchanged

    # Spreading the same blade mass over a longer span carries much more
    # rotary inertia than keeping it near the root.
    longer = _base_ontology()
    longer["components"]["blade"]["elastic_properties_mb"]["six_x_six"][
        "inertia_matrix"
    ]["values"] = [[50.0] + [0.0] * 20, [50.0] + [0.0] * 20]   # half kg/m ...
    longer["components"]["blade"]["elastic_properties_mb"]["six_x_six"][
        "reference_axis"
    ]["z"] = {"grid": [0.0, 1.0], "values": [0.0, 100.0]}      # ... over 100 m
    lg = read_windio_rna(_write(longer, tmp_path, "long_r.yaml"))
    assert lg.mass == pytest.approx(base.mass)             # same 5000 kg/blade
    assert lg.izz > base.izz                              # longer span -> more inertia


def test_rna_rotor_inertia_tilt_rotation(tmp_path: pathlib.Path) -> None:
    """The shaft-frame rotor / hub inertia is rotated by the uptilt into the
    tower frame, so a tilted rotor gains an izx product while preserving the
    inertia trace (Codex review on #130)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    def _mk(uptilt: float):
        o = _base_ontology()
        dt = o["components"]["nacelle"]["drivetrain"]
        dt["overhang"] = 0.0                              # apex purely vertical
        dt["uptilt"] = uptilt
        dt["elastic_properties_mb"]["system_center_mass"] = [0.0, 0.0, 3.0]
        return read_windio_rna(_write(o, tmp_path, f"tilt_{uptilt}.yaml"))

    flat = _mk(0.0)
    tilt = _mk(0.3)                                       # ~17 deg
    # overhang 0 + CM x 0 -> all izx comes from the shaft-frame tensor
    # rotation; a flat rotor/hub are diagonal (izx 0), a tilted one gains izx.
    assert flat.izx == pytest.approx(0.0, abs=1.0)
    assert abs(tilt.izx) > 1.0e6
    # the rotation preserves the inertia trace (apex identical, overhang 0).
    assert (tilt.ixx + tilt.iyy + tilt.izz) == pytest.approx(
        flat.ixx + flat.iyy + flat.izz, rel=1e-9)


def test_rna_rotor_inertia_cone_axial(tmp_path: pathlib.Path) -> None:
    """A coned rotor adds the axial (span·sin(cone))² term to the transverse
    (diametral) moment, so the transverse inertia grows with cone even as
    the radial reach shrinks (Codex review on #130)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    def _mk(cone: float):
        o = _base_ontology()
        o["components"]["hub"]["cone_angle"] = cone
        dt = o["components"]["nacelle"]["drivetrain"]
        dt["uptilt"] = 0.0                               # no shaft-frame rotation
        dt["overhang"] = 0.0                             # apex purely vertical
        dt["elastic_properties_mb"]["system_center_mass"] = [0.0, 0.0, 3.0]
        return read_windio_rna(_write(o, tmp_path, f"cone_{cone}.yaml"))

    flat = _mk(0.0)
    coned = _mk(0.4)                                     # ~23 deg (exaggerated)
    # I_diam = I_polar/2 + N_bl·axial: a flat-disc assumption (I_polar/2 only)
    # would drop the axial term and give a smaller transverse moment.
    assert coned.iyy > flat.iyy
    assert coned.mass == pytest.approx(flat.mass)


@pytest.mark.parametrize("n_bl", [1, 2])
def test_rna_rejects_one_or_two_bladed_rotor(
    tmp_path: pathlib.Path, n_bl: int
) -> None:
    """A one- or two-bladed rotor is not azimuthally symmetric, so the
    ``I_polar/2`` transverse split is invalid and the auto-RNA rejects it
    with a message pointing at an explicit tip_mass (Codex review on #130)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    o = _base_ontology()
    o["assembly"]["number_of_blades"] = n_bl
    with pytest.raises(ValueError, match="azimuthally symmetric"):
        read_windio_rna(_write(o, tmp_path, f"nb_{n_bl}.yaml"))


def test_rna_rotor_inertia_includes_sweep(tmp_path: pathlib.Path) -> None:
    """Sweep (reference_axis.y) sets an in-plane tangential distance from the
    shaft axis, so it adds ``y²`` to the rotor polar lever. A constant
    tangential offset leaves the span arc length (and thus the blade mass)
    unchanged but raises the rotor inertia; a lever built from the z span
    alone would drop it (Codex review on #130)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    def _mk(y_off: float):
        o = _base_ontology()
        ra = o["components"]["blade"]["elastic_properties_mb"]["six_x_six"][
            "reference_axis"
        ]
        ra["y"] = {"grid": [0.0, 1.0], "values": [y_off, y_off]}
        return read_windio_rna(_write(o, tmp_path, f"sw_{y_off}.yaml"))

    straight = _mk(0.0)
    swept = _mk(10.0)
    assert swept.mass == pytest.approx(straight.mass)  # constant offset, same arc
    assert swept.ixx > straight.ixx
    assert swept.izz > straight.izz


def test_rna_coned_cm_shift_with_uptilt(tmp_path: pathlib.Path) -> None:
    """A coned rotor's CM sits off the hub apex along the (tilted) shaft, so
    with uptilt the tower-top vertical CM shifts by m_blades·offset·sin(uptilt)
    over the total mass. Pins that the rotor body is placed at its coned CM,
    not the apex, before the parallel-axis shift (Codex review on #130)."""
    pytest.importorskip("yaml")
    import numpy as np

    from pybmodes.io.windio import read_windio_rna

    uptilt = 0.10

    def _mk(cone: float):
        o = _base_ontology()
        o["components"]["hub"]["cone_angle"] = cone
        o["components"]["nacelle"]["drivetrain"]["uptilt"] = uptilt
        return read_windio_rna(_write(o, tmp_path, f"tc_{cone}.yaml"))

    flat = _mk(0.0)
    coned = _mk(0.15)
    # base blade: uniform 100 kg/m over 0..50 m -> mass-weighted mean span 25 m,
    # so the coned shaft-axis CM offset is sin(cone)·25.
    offset = float(np.sin(0.15)) * 25.0
    m_blades = 3 * 5000.0
    expected = m_blades * offset * float(np.sin(uptilt)) / flat.mass
    assert coned.mass == pytest.approx(flat.mass)
    assert coned.cm_axial - flat.cm_axial == pytest.approx(expected, rel=1e-6)


def test_rna_diagonal_inertia_accepted(tmp_path: pathlib.Path) -> None:
    """A diagonal 3-vector system_inertia [Ixx, Iyy, Izz] is accepted as a
    diagonal tensor, matching the equivalent zero-product 6-vector (Codex
    review #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    diag = _base_ontology()
    diag["components"]["hub"]["elastic_properties_mb"]["system_inertia"] = [
        2.0e6, 1.0e6, 1.0e6,
    ]
    diag["components"]["nacelle"]["drivetrain"]["elastic_properties_mb"][
        "system_inertia"
    ] = [1.0e7, 1.2e7, 1.3e7]
    d = read_windio_rna(_write(diag, tmp_path, "diag.yaml"))
    base = read_windio_rna(_write(_base_ontology(), tmp_path, "full.yaml"))
    for attr in ("mass", "cm_axial", "ixx", "iyy", "izz", "izx", "ixy", "iyz"):
        assert getattr(d, attr) == pytest.approx(getattr(base, attr), abs=1e-6)


def test_rna_orientation_sign_flips_izx(tmp_path: pathlib.Path) -> None:
    """Downwind rotor flips the rotor-apex x-sign, flipping its izx cross
    term while mass, CM and the sign-independent inertias are unchanged.

    The nacelle CM x is zeroed so the izx cross term comes only from the
    rotor apex (the nacelle CM is a fixed input, not flipped by the rotor
    orientation), which makes the sign flip exact.
    """
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    up_onto = _base_ontology()
    up_onto["components"]["nacelle"]["drivetrain"]["elastic_properties_mb"][
        "system_center_mass"
    ] = [0.0, 0.0, 3.0]
    up = read_windio_rna(_write(up_onto, tmp_path, "up.yaml"))

    dn_onto = _base_ontology()
    dn_onto["components"]["nacelle"]["drivetrain"]["elastic_properties_mb"][
        "system_center_mass"
    ] = [0.0, 0.0, 3.0]
    dn_onto["assembly"]["rotor_orientation"] = "Downwind"
    dn = read_windio_rna(_write(dn_onto, tmp_path, "dn.yaml"))

    assert dn.mass == pytest.approx(up.mass)
    assert dn.cm_axial == pytest.approx(up.cm_axial)
    assert dn.iyy == pytest.approx(up.iyy)          # depends on x^2, sign-free
    assert dn.izz == pytest.approx(up.izz)
    assert dn.izx == pytest.approx(-up.izx)         # depends on sign(x)
    assert up.izx != pytest.approx(0.0)             # and is non-trivial


def test_rna_blade_mass_uses_arc_length(tmp_path: pathlib.Path) -> None:
    """Blade mass integrates mass/length over the reference-axis arc
    length, so prebend/sweep lengthen the span."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    straight = read_windio_rna(_write(_base_ontology(), tmp_path, "straight.yaml"))

    # Add prebend: x sweeps to 30 m over the 50 m z-span -> arc length
    # sqrt(50^2 + 30^2) ~ 58.3 m, so each blade is heavier.
    swept = _base_ontology()
    swept["components"]["blade"]["elastic_properties_mb"]["six_x_six"][
        "reference_axis"
    ]["x"] = {"grid": [0.0, 1.0], "values": [0.0, 30.0]}
    curved = read_windio_rna(_write(swept, tmp_path, "swept.yaml"))

    # 3 blades * 100 kg/m * (58.31 - 50.0) extra metres.
    extra = 3 * 100.0 * (np.hypot(50.0, 30.0) - 50.0)
    assert curved.mass - straight.mass == pytest.approx(extra, rel=1e-6)


def test_rna_blade_mass_preserves_reference_axis_knots(
    tmp_path: pathlib.Path,
) -> None:
    """A prebend knot that is not on the inertia grid is kept, so the arc
    length (and blade mass) is not chorded over it (Codex review #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    onto = _base_ontology()
    six = onto["components"]["blade"]["elastic_properties_mb"]["six_x_six"]
    # inertia grid is [0, 1]; put a prebend peak at mid-span (0.5), a knot
    # absent from the inertia grid.
    six["reference_axis"] = {
        "x": {"grid": [0.0, 0.5, 1.0], "values": [0.0, 30.0, 0.0]},
        "y": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
        "z": {"grid": [0.0, 1.0], "values": [0.0, 40.0]},
    }
    rna = read_windio_rna(_write(onto, tmp_path, "prebend.yaml"))

    # True arc length over the union grid [0, 0.5, 1]: points (0,0,0),
    # (30,0,20), (0,0,40).
    pts = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 20.0], [0.0, 0.0, 40.0]])
    arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    expected = 500000.0 + 100000.0 + 3 * 100.0 * arc
    assert rna.mass == pytest.approx(expected)
    # ... and it exceeds the chord (endpoint-only) mass the old sampling
    # onto the inertia grid would have produced (3 * 100 * 40).
    assert rna.mass > 500000.0 + 100000.0 + 3 * 100.0 * 40.0


def test_rna_blade_reference_axis_on_component(tmp_path: pathlib.Path) -> None:
    """The blade reference axis resolves from the component / outer_shape
    when it is not nested in six_x_six (older layout; Codex review #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    moved = _base_ontology()
    six = moved["components"]["blade"]["elastic_properties_mb"]["six_x_six"]
    ref = six.pop("reference_axis")                 # off six_x_six ...
    moved["components"]["blade"]["outer_shape"] = {"reference_axis": ref}  # ... onto component
    on_component = read_windio_rna(_write(moved, tmp_path, "compref.yaml"))
    base = read_windio_rna(_write(_base_ontology(), tmp_path, "base_ref.yaml"))
    assert on_component.mass == pytest.approx(base.mass)


def test_rna_blade_reference_axis_z_only(tmp_path: pathlib.Path) -> None:
    """A reference axis with only z (no prebend/sweep) integrates the span
    mass fine — x / y are optional."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    onto = _base_ontology()
    onto["components"]["blade"]["elastic_properties_mb"]["six_x_six"][
        "reference_axis"
    ] = {"z": {"grid": [0.0, 1.0], "values": [0.0, 50.0]}}
    rna = read_windio_rna(_write(onto, tmp_path, "zonly.yaml"))
    assert rna.mass == pytest.approx(615000.0)      # straight 50 m span


def test_rna_nacelle_block_at_nacelle_level(tmp_path: pathlib.Path) -> None:
    """The nacelle mass block + drivetrain geometry resolve when they sit
    directly on the nacelle component (WISDEM layout), not under
    drivetrain (Codex review #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    hoisted = _base_ontology()
    dt = hoisted["components"]["nacelle"].pop("drivetrain")
    hoisted["components"]["nacelle"].update(dt)   # drivetrain keys -> nacelle level
    moved = read_windio_rna(_write(hoisted, tmp_path, "naclevel.yaml"))
    base = read_windio_rna(_write(_base_ontology(), tmp_path, "base_nac.yaml"))
    assert moved.mass == pytest.approx(base.mass)
    assert moved.cm_axial == pytest.approx(base.cm_axial)


def test_rna_missing_blocks_raise(tmp_path: pathlib.Path) -> None:
    """An ontology without the hub / nacelle elastic_properties_mb blocks
    (IEA-15-style) fails clean, naming the missing block."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    no_hub = _base_ontology()
    del no_hub["components"]["hub"]["elastic_properties_mb"]
    with pytest.raises(KeyError, match="hub.*elastic_properties_mb"):
        read_windio_rna(_write(no_hub, tmp_path, "no_hub.yaml"))

    no_nac = _base_ontology()
    del no_nac["components"]["nacelle"]["drivetrain"]["elastic_properties_mb"]
    with pytest.raises(KeyError, match="nacelle.*elastic_properties_mb"):
        read_windio_rna(_write(no_nac, tmp_path, "no_nac.yaml"))


def test_rna_input_hardening(tmp_path: pathlib.Path) -> None:
    """Non-finite / non-positive / bool masses, malformed vectors, and bad
    blade counts / orientation are all rejected (no silent fallback)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    def _run(mut):
        onto = _base_ontology()
        mut(onto)
        return read_windio_rna(_write(onto, tmp_path, "bad.yaml"))

    # Non-finite / non-positive / bool hub mass.
    for bad in (float("nan"), float("inf"), 0.0, -5.0, True):
        with pytest.raises(ValueError):
            _run(lambda o, b=bad: o["components"]["hub"][
                "elastic_properties_mb"].__setitem__("system_mass", b))

    # Malformed nacelle CM (not a finite 3-vector).
    with pytest.raises(ValueError, match="system_center_mass"):
        _run(lambda o: o["components"]["nacelle"]["drivetrain"][
            "elastic_properties_mb"].__setitem__(
                "system_center_mass", [1.0, 2.0]))

    # Malformed inertia (neither a 3- nor 6-vector).
    with pytest.raises(ValueError, match="inertia"):
        _run(lambda o: o["components"]["hub"]["elastic_properties_mb"]
             .__setitem__("system_inertia", [1.0, 2.0, 3.0, 4.0]))

    # Bool entries in vector fields must be rejected, not coerced to 0/1.
    with pytest.raises(ValueError, match="not bools"):
        _run(lambda o: o["components"]["nacelle"]["drivetrain"][
            "elastic_properties_mb"].__setitem__(
                "system_center_mass", [True, 0.0, 3.0]))
    with pytest.raises(ValueError, match="not bools"):
        _run(lambda o: o["components"]["hub"]["elastic_properties_mb"]
             .__setitem__("system_inertia", [True, 1.0, 1.0]))

    # Bad blade count / orientation / non-finite geometry.
    with pytest.raises(ValueError, match="number_of_blades"):
        _run(lambda o: o["assembly"].__setitem__("number_of_blades", -1))
    with pytest.raises(ValueError, match="rotor_orientation"):
        _run(lambda o: o["assembly"].__setitem__("rotor_orientation", "sideways"))
    with pytest.raises(ValueError, match="overhang"):
        _run(lambda o: o["components"]["nacelle"]["drivetrain"].__setitem__(
            "overhang", float("nan")))


def test_rna_grid_hardening(tmp_path: pathlib.Path) -> None:
    """Non-finite / non-monotone blade grids and non-finite reference-axis
    values are rejected rather than silently zeroing the blade mass (Codex
    review #82)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna

    def _run(mut, name):
        onto = _base_ontology()
        mut(onto)
        return read_windio_rna(_write(onto, tmp_path, name))

    def six(o):
        return o["components"]["blade"]["elastic_properties_mb"]["six_x_six"]

    # NaN in the inertia grid (size-only guard would have passed).
    with pytest.raises(ValueError, match="increasing"):
        _run(lambda o: six(o)["inertia_matrix"].__setitem__(
            "grid", [0.0, float("nan")]), "g1.yaml")
    # Non-monotone inertia grid.
    with pytest.raises(ValueError, match="increasing"):
        _run(lambda o: six(o)["inertia_matrix"].__setitem__(
            "grid", [1.0, 0.0]), "g2.yaml")
    # Non-finite reference-axis values.
    with pytest.raises(ValueError, match="non-finite"):
        _run(lambda o: six(o)["reference_axis"]["z"].__setitem__(
            "values", [0.0, float("nan")]), "g3.yaml")
    # Non-monotone reference-axis grid.
    with pytest.raises(ValueError, match="increasing"):
        _run(lambda o: six(o)["reference_axis"]["z"].__setitem__(
            "grid", [1.0, 0.0]), "g4.yaml")
    # Bool geometry.
    with pytest.raises(ValueError, match="not a bool"):
        _run(lambda o: o["components"]["nacelle"]["drivetrain"].__setitem__(
            "overhang", True), "g5.yaml")


def test_from_windio_lumped_rna_cal(tmp_path: pathlib.Path) -> None:
    """from_windio(lumped_rna_cal=True) attaches the auto-RNA and softens
    the tower; passing both tip_mass and the flag is rejected."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna
    from pybmodes.models import Tower

    # The base ontology has no tower component; add a simple tube tower so
    # from_windio can build a beam.
    onto = _base_ontology()
    onto["components"]["tower"] = {
        "outer_shape": {
            "outer_diameter": {"grid": [0.0, 1.0], "values": [8.0, 5.0]},
        },
        "structure": {
            "outfitting_factor": 1.0,
            "layers": [
                {
                    "name": "wall",
                    "material": "steel",
                    "thickness": {"grid": [0.0, 1.0], "values": [0.05, 0.02]},
                }
            ],
        },
        "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [0.0, 120.0]}},
    }
    onto["materials"] = [{"name": "steel", "E": 2.1e11, "rho": 7850.0, "nu": 0.3}]
    p = _write(onto, tmp_path)

    rna = read_windio_rna(p)
    t = Tower.from_windio(p, lumped_rna_cal=True)
    assert t._bmi.tip_mass.mass == pytest.approx(rna.mass)
    assert t._bmi.tip_mass.cm_axial == pytest.approx(rna.cm_axial)

    f_rna = t.run(n_modes=3, check_model=False).frequencies
    f_bare = Tower.from_windio(p).run(n_modes=3, check_model=False).frequencies
    assert np.all(np.isfinite(f_rna)) and np.all(f_rna > 0.0)
    assert f_rna[0] < f_bare[0]                      # RNA mass softens the tower

    with pytest.raises(ValueError, match="not both"):
        Tower.from_windio(p, tip_mass=1.0e5, lumped_rna_cal=True)

    # The pre-shifted tower-top inertia is a clamped-base (hub_conn=1)
    # record; a free-base / soil-flexible base would misplace it, so it is
    # rejected (Codex review on #82).
    with pytest.raises(ValueError, match="hub_conn=1"):
        Tower.from_windio(p, hub_conn=3, lumped_rna_cal=True)

    # component='monopile' would place the RNA at the transition piece, not
    # the tower top, so the flag is rejected there (Codex review on #82).
    with pytest.raises(ValueError, match="component='tower'"):
        Tower.from_windio(p, component="monopile", lumped_rna_cal=True)


def test_from_windio_with_monopile_lumped_rna_cal(tmp_path: pathlib.Path) -> None:
    """The same flag works on the monopile constructor and rejects the
    tip_mass conflict."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio import read_windio_rna
    from pybmodes.models import Tower

    onto = _base_ontology()
    onto["materials"] = [{"name": "steel", "E": 2.1e11, "rho": 7850.0, "nu": 0.3}]
    onto["components"]["monopile"] = {
        "outer_shape": {"outer_diameter": {"grid": [0.0, 1.0], "values": [9.0, 9.0]}},
        "structure": {
            "outfitting_factor": 1.0,
            "layers": [{"name": "w", "material": "steel",
                        "thickness": {"grid": [0.0, 1.0], "values": [0.08, 0.08]}}],
        },
        "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [-30.0, 10.0]}},
    }
    onto["components"]["tower"] = {
        "outer_shape": {"outer_diameter": {"grid": [0.0, 1.0], "values": [9.0, 6.0]}},
        "structure": {
            "outfitting_factor": 1.0,
            "layers": [{"name": "w", "material": "steel",
                        "thickness": {"grid": [0.0, 1.0], "values": [0.05, 0.02]}}],
        },
        "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [10.0, 110.0]}},
    }
    p = _write(onto, tmp_path)

    rna = read_windio_rna(p)
    t = Tower.from_windio_with_monopile(p, lumped_rna_cal=True)
    assert t._bmi.tip_mass.mass == pytest.approx(rna.mass)
    f = t.run(n_modes=4, check_model=False).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)

    with pytest.raises(ValueError, match="not both"):
        Tower.from_windio_with_monopile(p, tip_mass=2.0e5, lumped_rna_cal=True)


# ---------------------------------------------------------------------------
# Integration: IEA-22 ontology vs its ElastoDyn deck
# ---------------------------------------------------------------------------

_IEA22 = _DOCS / "IEA-22-280-RWT"
_IEA22_YAML = _IEA22 / "windIO/IEA-22-280-RWT.yaml"
_IEA22_ED = _IEA22 / "OpenFAST/IEA-22-280-RWT-Monopile/IEA-22-280-RWT_ElastoDyn.dat"
_IEA22_BLADE = _IEA22 / "OpenFAST/IEA-22-280-RWT/IEA-22-280-RWT_ElastoDyn_blade.dat"


@pytest.mark.integration
@pytest.mark.skipif(
    not (_IEA22_YAML.is_file() and _IEA22_ED.is_file() and _IEA22_BLADE.is_file()),
    reason="IEA-22 WindIO ontology + ElastoDyn deck not present",
)
def test_rna_iea22_matches_elastodyn_deck() -> None:
    """The WindIO auto-RNA mass matches the IEA-22 ElastoDyn deck's tower-top
    assembly to <0.5 % (issue #82), and its CM agrees to ~1 %. The small CM
    divergence is a documented, intentional one: the deck adapter lumps the
    blades as a point mass at the rotor apex, whereas the WindIO rigid-rotor
    path (issue #130) places the coned rotor at its true centre of mass,
    which sits ~2.6 m outboard of the apex for the 4 deg precone and so
    raises the tower-top vertical CM by ~0.06 m through the 6 deg shaft tilt
    (real ElastoDyn carries the same precone term). Inertia is WindIO-native
    and is not expected to byte-match the deck's NacYIner (ecosystem drift)."""
    pytest.importorskip("yaml")
    from pybmodes.io._elastodyn.adapter import _tower_top_assembly_mass
    from pybmodes.io.elastodyn_reader import (
        read_elastodyn_blade,
        read_elastodyn_main,
    )
    from pybmodes.io.windio import read_windio_rna
    from pybmodes.models import Tower

    rna = read_windio_rna(_IEA22_YAML)
    deck = _tower_top_assembly_mass(
        read_elastodyn_main(_IEA22_ED), read_elastodyn_blade(_IEA22_BLADE)
    )

    assert rna.mass == pytest.approx(deck.mass, rel=5e-3)       # <0.5 %
    # The precone CM offset the deck adapter drops raises the WindIO CM
    # slightly; it stays within ~1.5 % and lies above the apex-lumped value.
    assert rna.cm_axial == pytest.approx(deck.cm_axial, rel=1.5e-2)
    assert rna.cm_axial >= deck.cm_axial
    # WindIO-native inertia carries the fore-aft rotary term the same order
    # of magnitude as the deck, but not byte-identical (documented drift).
    assert rna.iyy > 0.0

    # End-to-end: the auto-RNA tower solves and is softer than bare.
    f_rna = Tower.from_windio(
        _IEA22_YAML, lumped_rna_cal=True, n_nodes=40
    ).run(n_modes=4, check_model=False).frequencies
    f_bare = Tower.from_windio(
        _IEA22_YAML, n_nodes=40
    ).run(n_modes=4, check_model=False).frequencies
    assert np.all(np.isfinite(f_rna)) and np.all(f_rna > 0.0)
    assert f_rna[0] < f_bare[0]
