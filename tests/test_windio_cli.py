"""`pybmodes windio` — the one-click orchestrator (1.4.0, issue #35).

Default-suite rungs are self-contained synthetic yamls; the
integration rung drives the real IEA-15 UMaine VolturnUS-S RWT tree
(yaml + auto-discovered companion HydroDyn/MoorDyn/ElastoDyn → the
industry-grade deck-backed coupled model).
"""

from __future__ import annotations

import pathlib
import textwrap
import warnings

import pytest

from pybmodes.cli import main
from pybmodes.workflows.windio import discover_windio_inputs

_DOCS = (pathlib.Path(__file__).resolve().parents[1]
         / "external" / "OpenFAST_files")
_IEA15_RWT = _DOCS / "IEA-15-240-RWT"
_IEA15_FLOAT_Y = _IEA15_RWT / "WT_Ontology/IEA-15-240-RWT_VolturnUS-S.yaml"

_TOWER_ONLY = textwrap.dedent("""\
    components:
      tower:
        outer_shape:
          outer_diameter: {grid: [0.0, 1.0], values: [8.0, 5.0]}
        structure:
          outfitting_factor: 1.0
          layers:
            - {name: w, material: steel,
               thickness: {grid: [0.0, 1.0], values: [0.04, 0.02]}}
        reference_axis:
          z: {grid: [0.0, 1.0], values: [0.0, 110.0]}
    materials:
      - {name: steel, E: 2.0e11, rho: 7800.0, nu: 0.3}
    """)


def test_windio_cli_discovery_scoping(tmp_path) -> None:
    """A bare yaml in a scratch dir yields NO companion decks (the
    labelled-preview case) — discovery must never recursively scan an
    arbitrary parent (the WinError/wrong-turbine bug). A proper
    ``<root>/OpenFAST/*ElastoDyn.dat`` layout IS discovered, scoped to
    that turbine root."""
    pytest.importorskip("yaml")
    bare = tmp_path / "scratch" / "x.yaml"
    bare.parent.mkdir()
    bare.write_text(_TOWER_ONLY, encoding="utf-8")
    d = discover_windio_inputs(bare)
    assert d.yaml == bare
    assert d.hydrodyn is None and d.moordyn is None \
        and d.elastodyn is None

    rwt = tmp_path / "MY-RWT"
    (rwt / "yaml").mkdir(parents=True)
    (rwt / "OpenFAST").mkdir()
    y = rwt / "yaml" / "turb.yaml"
    y.write_text(_TOWER_ONLY, encoding="utf-8")
    (rwt / "OpenFAST" / "turb_ElastoDyn.dat").write_text("x", "utf-8")
    (rwt / "OpenFAST" / "turb_ElastoDyn_tower.dat").write_text("x",
                                                               "utf-8")
    d2 = discover_windio_inputs(y)
    # main deck found; the _tower aux file excluded.
    assert d2.elastodyn is not None
    assert d2.elastodyn.name == "turb_ElastoDyn.dat"


def test_windio_cli_fixed_tower(tmp_path) -> None:
    """`pybmodes windio <tower yaml>` → a cantilever-tower report."""
    pytest.importorskip("yaml")
    yp = tmp_path / "tower.yaml"
    yp.write_text(_TOWER_ONLY, encoding="utf-8")
    out = tmp_path / "rep.md"
    rc = main(["windio", str(yp), "--out", str(out), "--n-modes", "6"])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0


def test_windio_cli_floating_screening_preview(tmp_path) -> None:
    """A floating yaml with no companion decks → a report, with a
    clear SCREENING-fidelity warning (not silently 'accurate')."""
    pytest.importorskip("yaml")
    from tests.test_windio_floating import _FLOAT_TURBINE

    yp = tmp_path / "fowt.yaml"
    yp.write_text(_FLOAT_TURBINE, encoding="utf-8")
    out = tmp_path / "fowt.md"
    with pytest.warns(UserWarning, match="SCREENING-fidelity"):
        rc = main(["windio", str(yp), "--out", str(out),
                   "--water-depth", "200", "--n-modes", "8"])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0


def test_windio_cli_bad_input() -> None:
    rc = main(["windio", "does_not_exist_xyz.yaml", "--out",
               "x.md"])
    assert rc == 2


@pytest.mark.integration
@pytest.mark.skipif(
    not _IEA15_FLOAT_Y.is_file(),
    reason="IEA-15 VolturnUS-S RWT tree absent",
)
def test_windio_cli_iea15_industry_grade(tmp_path) -> None:
    """End-to-end one-click on the real IEA-15 UMaine VolturnUS-S:
    discovery picks the *UMaineSemi* (floating) companion decks → the
    industry-grade deck-backed coupled model (no screening warning),
    blade + coupled platform solved, report written."""
    pytest.importorskip("yaml")
    disc = discover_windio_inputs(_IEA15_FLOAT_Y)
    for k in ("hydrodyn", "moordyn", "elastodyn"):
        val = getattr(disc, k)
        assert val is not None, f"{k} not discovered"
        assert "umaine" in str(val).lower() \
            or "semi" in str(val).lower(), \
            f"{k} picked the wrong (non-floating) config: {val}"

    out = tmp_path / "iea15.md"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no preview warn
        rc = main(["windio", str(_IEA15_FLOAT_Y), "--out", str(out),
                   "--n-modes", "10"])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
