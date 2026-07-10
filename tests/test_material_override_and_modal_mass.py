"""Material / outfitting overrides on the WindIO paths (issue #133) and
per-mode generalised mass / stiffness on the result (issue #134).

Self-contained: builds tiny hand-written WindIO ontologies in ``tmp_path``
and analytical tower geometry inline, no external data.
"""
from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

from pybmodes.io.bmi import TipMassProps
from pybmodes.models import Tower

_MIN_TOWER = textwrap.dedent("""\
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

_MIN_MONOPILE_TOWER = textwrap.dedent("""\
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
              material: steel
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
          outfitting_factor: 1.0
          layers:
            - name: tower_wall
              material: steel
              thickness:
                grid: [0.0, 1.0]
                values: [0.05, 0.02]
        reference_axis:
          z:
            grid: [0.0, 1.0]
            values: [10.0, 110.0]
    materials:
      - name: steel
        E: 2.0e11
        rho: 7850.0
        nu: 0.3
    """)


def _tower_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    pytest.importorskip("yaml")
    p = tmp_path / "tower.yaml"
    p.write_text(_MIN_TOWER, encoding="utf-8")
    return p


def _f1(model: Tower) -> float:
    return float(model.run(n_modes=2, check_model=False).frequencies[0])


# --- issue #133: material / outfitting overrides on from_windio ------------

def test_from_windio_none_matches_ontology(tmp_path: pathlib.Path) -> None:
    """Passing ``None`` (the default) reproduces the ontology material exactly."""
    p = _tower_yaml(tmp_path)
    base = _f1(Tower.from_windio(p))
    same = _f1(Tower.from_windio(p, E=None, rho=None, nu=None, outfitting_factor=None))
    assert same == pytest.approx(base)


def test_from_windio_E_override_scales_frequency(tmp_path: pathlib.Path) -> None:
    """f scales as sqrt(E) at fixed mass, so overriding E from 2.0e11 to 1.5e11
    lowers the frequency by sqrt(1.5/2.0) (issue #133)."""
    p = _tower_yaml(tmp_path)
    base = _f1(Tower.from_windio(p))  # ontology E = 2.0e11
    soft = _f1(Tower.from_windio(p, E=1.5e11))
    assert soft / base == pytest.approx(np.sqrt(1.5e11 / 2.0e11), rel=1e-6)


def test_from_windio_outfitting_override_scales_frequency(
    tmp_path: pathlib.Path,
) -> None:
    """Outfitting scales mass, so f ~ 1/sqrt(outfitting). Overriding the
    ontology's 1.1 to 1.0 raises the frequency by sqrt(1.1) (issue #133)."""
    p = _tower_yaml(tmp_path)
    base = _f1(Tower.from_windio(p))  # ontology outfitting = 1.1
    bare = _f1(Tower.from_windio(p, outfitting_factor=1.0))
    assert bare / base == pytest.approx(np.sqrt(1.1), rel=1e-6)


def test_from_windio_rho_override_scales_frequency(tmp_path: pathlib.Path) -> None:
    """f ~ 1/sqrt(rho) at fixed stiffness (no tip mass)."""
    p = _tower_yaml(tmp_path)
    base = _f1(Tower.from_windio(p))  # ontology rho = 7800
    heavy = _f1(Tower.from_windio(p, rho=2 * 7800.0))
    assert heavy / base == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6)


def test_from_windio_with_monopile_override(tmp_path: pathlib.Path) -> None:
    """The override reaches both segments of the combined monopile+tower path
    and softens the whole structure (issue #133)."""
    pytest.importorskip("yaml")
    p = tmp_path / "mp.yaml"
    p.write_text(_MIN_MONOPILE_TOWER, encoding="utf-8")
    base = _f1(Tower.from_windio_with_monopile(p, water_depth=30.0))
    soft = _f1(Tower.from_windio_with_monopile(p, water_depth=30.0, E=1.0e11))
    assert soft < base  # halving E on both segments lowers the frequency


# --- issue #134: per-mode generalised mass / stiffness ---------------------

def _clamped_tower_with_rna() -> Tower:
    z = np.linspace(0.0, 1.0, 11)
    od = np.full_like(z, 6.0)
    wall = np.full_like(z, 0.03)
    rna = TipMassProps(
        mass=4.0e5, cm_offset=0.0, cm_axial=3.0,
        ixx=2.0e6, iyy=2.0e6, izz=2.0e6, ixy=0.0, izx=0.0, iyz=0.0,
    )
    return Tower.from_geometry(
        z, od, wall, flexible_length=100.0, E=2.0e11, rho=7850.0,
        hub_conn=1, tip_mass=rna, n_nodes=40,
    )


def test_generalized_mass_stiffness_populated() -> None:
    """The result carries a physical generalised mass (kg) and stiffness
    (N/m) per mode (issue #134)."""
    res = _clamped_tower_with_rna().run(n_modes=4, check_model=False)
    assert res.generalized_mass is not None
    assert res.generalized_stiffness is not None
    assert res.generalized_mass.shape == res.frequencies.shape
    assert res.generalized_stiffness.shape == res.frequencies.shape
    # first bending mode: finite, positive, and larger than the tip mass
    # (tip translation plus a share of the tower and the CM lever).
    assert np.isfinite(res.generalized_mass[0])
    assert res.generalized_mass[0] > 4.0e5


def test_generalized_mass_stiffness_recovers_frequency() -> None:
    """sqrt(K/M)/(2*pi) reproduces each mode's frequency (issue #134)."""
    res = _clamped_tower_with_rna().run(n_modes=4, check_model=False)
    m = res.generalized_mass
    k = res.generalized_stiffness
    ok = np.isfinite(m) & np.isfinite(k) & (m > 0)
    assert ok.any()
    f_from_km = np.sqrt(k[ok] / m[ok]) / (2.0 * np.pi)
    assert np.allclose(f_from_km, res.frequencies[ok], rtol=1e-6)
