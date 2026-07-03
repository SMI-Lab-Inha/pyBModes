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
    # Parallel-axis tensor at the tower top (hand-computed).
    assert rna.ixx == pytest.approx(1.834e7)
    assert rna.iyy == pytest.approx(3.884e7)
    assert rna.izz == pytest.approx(3.35e7)
    assert rna.izx == pytest.approx(1.06e7)
    assert rna.ixy == pytest.approx(0.0, abs=1e-6)
    assert rna.iyz == pytest.approx(0.0, abs=1e-6)


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

    # Malformed inertia (not a 6-vector).
    with pytest.raises(ValueError, match="inertia"):
        _run(lambda o: o["components"]["hub"]["elastic_properties_mb"]
             .__setitem__("system_inertia", [1.0, 2.0, 3.0]))

    # Bad blade count / orientation / non-finite geometry.
    with pytest.raises(ValueError, match="number_of_blades"):
        _run(lambda o: o["assembly"].__setitem__("number_of_blades", -1))
    with pytest.raises(ValueError, match="rotor_orientation"):
        _run(lambda o: o["assembly"].__setitem__("rotor_orientation", "sideways"))
    with pytest.raises(ValueError, match="overhang"):
        _run(lambda o: o["components"]["nacelle"]["drivetrain"].__setitem__(
            "overhang", float("nan")))


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
    """The WindIO auto-RNA mass and CM match the IEA-22 ElastoDyn deck's
    tower-top assembly to <0.5 % (issue #82). Inertia is WindIO-native and
    is not expected to byte-match the deck's NacYIner (ecosystem drift)."""
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
    assert rna.cm_axial == pytest.approx(deck.cm_axial, rel=5e-3)
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
