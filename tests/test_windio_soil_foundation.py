"""Integrated soil-pile interaction on the WindIO monopile path (issue #118).

`Tower.from_windio_with_monopile(..., soil=... | soil_E=...)` replaces the
rigid mudline clamp with a `MudlineFoundation` coupled-spring soil model
(`hub_conn=3`), and `MudlineFoundation.from_windio` auto-extracts the pile
geometry from the ontology. Self-contained: a tiny hand-written ontology in
`tmp_path`, no external data.
"""
from __future__ import annotations

import pathlib
import textwrap
import warnings

import pytest

from pybmodes.foundation import MudlineFoundation
from pybmodes.models import Tower

# monopile z -30..10 (base to transition), tower 10..110; a mudline placed by
# water_depth between -30 and 10 leaves an embedded pile below it.
_ONTO = textwrap.dedent("""\
    environment:
      water_depth: 20.0
    components:
      monopile:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [9.0, 9.0]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: monopile_wall, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.09, 0.09]}}
        reference_axis: {z: {grid: [0.0, 1.0], values: [-30.0, 10.0]}}
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 0.5, 1.0], values: [9.0, 7.5, 6.0]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: tower_wall, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.05, 0.02]}}
        reference_axis: {z: {grid: [0.0, 1.0], values: [10.0, 110.0]}}
    materials:
      - {name: steel, E: 2.0e11, rho: 7850.0, nu: 0.3}
    """)


def _yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    pytest.importorskip("yaml")
    p = tmp_path / "mp.yaml"
    p.write_text(_ONTO, encoding="utf-8")
    return p


def _f1(model: Tower, **run_kw) -> float:
    return float(model.run(n_modes=4, check_model=False, **run_kw).frequencies[0])


def test_soil_lowers_frequency_vs_rigid_clamp(tmp_path: pathlib.Path) -> None:
    """Soil flexibility softens the base, so the coupled 1st frequency is
    lower than the rigid mudline clamp (issue #118)."""
    p = _yaml(tmp_path)
    rigid = Tower.from_windio_with_monopile(p, water_depth=20.0)
    soft = Tower.from_windio_with_monopile(p, water_depth=20.0, soil_E=60e6)
    assert rigid._bmi.hub_conn == 1
    assert soft._bmi.hub_conn == 3
    assert _f1(soft) < _f1(rigid)


def test_soil_auto_matches_explicit_foundation(tmp_path: pathlib.Path) -> None:
    """`soil_E` auto-build and an explicit `soil=MudlineFoundation.from_windio`
    give the same model (issue #118)."""
    p = _yaml(tmp_path)
    auto = Tower.from_windio_with_monopile(p, water_depth=20.0, soil_E=60e6)
    found = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0)
    explicit = Tower.from_windio_with_monopile(p, water_depth=20.0, soil=found)
    assert _f1(auto) == pytest.approx(_f1(explicit))


def test_soil_no_spurious_check_warnings(tmp_path: pathlib.Path) -> None:
    """A soft-monopile (hub_conn=3, only mooring_K) triggers no floating-
    readiness check warnings (those gate on hub_conn=2) (issue #118)."""
    p = _yaml(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Tower.from_windio_with_monopile(
            p, water_depth=20.0, soil_E=60e6
        ).run(n_modes=4)  # check_model on (default)


_SLENDER = textwrap.dedent("""\
    environment:
      water_depth: 20.0
    components:
      monopile:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [2.0, 2.0]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: monopile_wall, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.06, 0.02]}}
        reference_axis: {z: {grid: [0.0, 1.0], values: [-50.0, 0.0]}}
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [2.0, 1.5]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: tower_wall, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.03, 0.02]}}
        reference_axis: {z: {grid: [0.0, 1.0], values: [0.0, 80.0]}}
    materials:
      - {name: steel, E: 2.0e11, rho: 7850.0, nu: 0.3}
    """)


def test_soil_auto_build_honors_E_override(tmp_path: pathlib.Path) -> None:
    """The E override reaches the mudline springs through pile_EI, not just the
    beam, so a flexible pile's foundation changes with E (Codex review #118)."""
    pytest.importorskip("yaml")
    p = tmp_path / "slender.yaml"
    p.write_text(_SLENDER, encoding="utf-8")
    base = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0)
    stiff = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0, E=4.0e11)
    assert base.pile_behaviour == "flexible"  # a flexible pile: EI (hence E) matters
    assert stiff.K_rr != pytest.approx(base.K_rr)


def test_soil_from_windio_honors_thickness_interp(tmp_path: pathlib.Path) -> None:
    """A piecewise-constant wall schedule takes the step value at the mudline,
    not a linear blend, so pile_EI (and the springs) match the beam reduction
    on a tapered pile (Codex review on #118)."""
    pytest.importorskip("yaml")
    p = tmp_path / "slender.yaml"
    p.write_text(_SLENDER, encoding="utf-8")
    lin = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0)
    pwc = MudlineFoundation.from_windio(
        p, soil_E=60e6, water_depth=20.0, thickness_interp="piecewise_constant"
    )
    assert lin.pile_behaviour == "flexible"
    assert pwc.K_rr != pytest.approx(lin.K_rr)


def test_soil_rejects_lumped_rna_cal(tmp_path: pathlib.Path) -> None:
    """The auto-RNA tip mass is clamped-base (hub_conn=1); it cannot be
    combined with a soil-flexible base (hub_conn=3) (Codex review on #118)."""
    p = _yaml(tmp_path)
    with pytest.raises(ValueError, match="lumped_rna_cal is not supported"):
        Tower.from_windio_with_monopile(
            p, water_depth=20.0, soil_E=60e6, lumped_rna_cal=True
        )


def test_soil_and_soil_E_mutually_exclusive(tmp_path: pathlib.Path) -> None:
    p = _yaml(tmp_path)
    found = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0)
    with pytest.raises(ValueError, match="not both"):
        Tower.from_windio_with_monopile(
            p, water_depth=20.0, soil=found, soil_E=60e6
        )


def test_explicit_soil_requires_water_depth(tmp_path: pathlib.Path) -> None:
    """The explicit-soil path also needs a resolved mudline depth, else the
    beam clamps at the pile toe and the springs act there, not at the mudline
    (Codex review on #118)."""
    pytest.importorskip("yaml")
    # ontology without environment.water_depth
    p = tmp_path / "no_env.yaml"
    p.write_text(_ONTO.split("environment:")[0] + _ONTO.split("\n", 2)[2],
                 encoding="utf-8")
    found = MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=20.0)
    with pytest.raises(ValueError, match="resolved water depth"):
        Tower.from_windio_with_monopile(p, soil=found)  # no water_depth


def test_from_windio_requires_water_depth(tmp_path: pathlib.Path) -> None:
    """No mudline (no water_depth and no environment block) -> clear error."""
    pytest.importorskip("yaml")
    # strip the environment block so there is no ontology water depth
    p = tmp_path / "no_env.yaml"
    p.write_text(_ONTO.split("environment:")[0] + _ONTO.split("\n", 2)[2],
                 encoding="utf-8")
    with pytest.raises(ValueError, match="water_depth is required"):
        MudlineFoundation.from_windio(p, soil_E=60e6)


def test_from_windio_requires_embedded_pile(tmp_path: pathlib.Path) -> None:
    """A mudline below the monopile base leaves no embedded length to model
    soil over (issue #118)."""
    p = _yaml(tmp_path)
    # monopile base is z=-30; water_depth=35 places the mudline at -35, below
    # the pile base, so there is no embedded pile between them.
    with pytest.raises(ValueError, match="does not extend below the mudline"):
        MudlineFoundation.from_windio(p, soil_E=60e6, water_depth=35.0)
