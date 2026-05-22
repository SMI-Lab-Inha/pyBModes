"""Tests for elastodyn/params.py and elastodyn/writer.py.

Smoke tests that validate the public API surface against synthetic mode
shapes constructed directly in-test, without any bundled reference data.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from pybmodes.elastodyn import (
    BladeElastoDynParams,
    TowerElastoDynParams,
    TowerSelectionReport,
    compute_blade_params,
    compute_tower_params,
    compute_tower_params_report,
    patch_dat,
)
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.fitting.poly_fit import PolyFitResult
from pybmodes.models.result import ModalResult

# ===========================================================================
# Synthetic mode-shape helpers
# ===========================================================================

def _shape(*, mode_number, freq_hz, flap, lag, n=11):
    span = np.linspace(0.0, 1.0, n)
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=np.asarray(flap, dtype=float),
        flap_slope=np.zeros_like(span),
        lag_disp=np.asarray(lag, dtype=float),
        lag_slope=np.zeros_like(span),
        twist=np.zeros_like(span),
    )


def _polynomial_shape(coeffs):
    """Build a span-loc-keyed displacement that lives exactly in the model space."""
    span = np.linspace(0.0, 1.0, 11)
    c2, c3, c4, c5, c6 = coeffs
    return c2 * span**2 + c3 * span**3 + c4 * span**4 + c5 * span**5 + c6 * span**6


# ===========================================================================
# compute_blade_params
# ===========================================================================

class TestBladeParams:

    @pytest.fixture
    def modal(self):
        # Two flap-dominated and one edge-dominated mode, smooth polynomials.
        flap1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))     # pure x²
        flap2 = _polynomial_shape((-1.0, 0.0, 2.0, 0.0, 0.0))    # x² → x⁴ blend
        edge1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))     # x² + x⁶
        zeros = np.zeros(11)
        shapes = [
            _shape(mode_number=1, freq_hz=0.5, flap=flap1, lag=zeros),
            _shape(mode_number=2, freq_hz=2.0, flap=flap2, lag=zeros),
            _shape(mode_number=3, freq_hz=3.5, flap=zeros, lag=edge1),
        ]
        return ModalResult(frequencies=np.array([0.5, 2.0, 3.5]), shapes=shapes)

    def test_returns_blade_params(self, modal):
        params = compute_blade_params(modal)
        assert isinstance(params, BladeElastoDynParams)

    def test_constraints(self, modal):
        params = compute_blade_params(modal)
        for fit in (params.BldFl1Sh, params.BldFl2Sh, params.BldEdgSh):
            total = fit.c2 + fit.c3 + fit.c4 + fit.c5 + fit.c6
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self, modal):
        d = compute_blade_params(modal).as_dict()
        assert set(d.keys()) == {
            f"{n}({k})"
            for n in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh")
            for k in range(2, 7)
        }

    def test_insufficient_modes_raises(self):
        # Only one flap mode → cannot form BldFl2Sh
        flap1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))
        edge1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))
        zeros = np.zeros(11)
        modal = ModalResult(
            frequencies=np.array([0.5, 3.5]),
            shapes=[
                _shape(mode_number=1, freq_hz=0.5, flap=flap1, lag=zeros),
                _shape(mode_number=2, freq_hz=3.5, flap=zeros, lag=edge1),
            ],
        )
        with pytest.raises(ValueError, match="flap modes"):
            compute_blade_params(modal)


# ===========================================================================
# compute_tower_params / compute_tower_params_report
# ===========================================================================

class TestTowerParams:

    @pytest.fixture
    def modal(self):
        # Two FA-dominated + two SS-dominated modes, smooth polynomials.
        fa1 = _polynomial_shape((1.0, 0.0, 0.0, 0.0, 0.0))
        fa2 = _polynomial_shape((-1.0, 0.0, 2.0, 0.0, 0.0))
        ss1 = _polynomial_shape((0.5, 0.0, 0.0, 0.0, 0.5))
        ss2 = _polynomial_shape((0.0, 1.0, 0.0, 0.0, 0.0))
        zeros = np.zeros(11)
        shapes = [
            _shape(mode_number=1, freq_hz=0.4, flap=fa1, lag=zeros),
            _shape(mode_number=2, freq_hz=0.5, flap=zeros, lag=ss1),
            _shape(mode_number=3, freq_hz=2.0, flap=fa2, lag=zeros),
            _shape(mode_number=4, freq_hz=2.2, flap=zeros, lag=ss2),
        ]
        return ModalResult(frequencies=np.array([0.4, 0.5, 2.0, 2.2]), shapes=shapes)

    def test_returns_tower_params(self, modal):
        params = compute_tower_params(modal)
        assert isinstance(params, TowerElastoDynParams)

    def test_report_structure(self, modal):
        params, report = compute_tower_params_report(modal)
        assert isinstance(params, TowerElastoDynParams)
        assert isinstance(report, TowerSelectionReport)
        assert sum(m.selected for m in report.fa_family) == 2
        assert sum(m.selected for m in report.ss_family) == 2

    def test_constraints(self, modal):
        params = compute_tower_params(modal)
        for fit in (params.TwFAM1Sh, params.TwFAM2Sh,
                    params.TwSSM1Sh, params.TwSSM2Sh):
            total = fit.c2 + fit.c3 + fit.c4 + fit.c5 + fit.c6
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_as_dict_keys(self, modal):
        d = compute_tower_params(modal).as_dict()
        assert set(d.keys()) == {
            f"{n}({k})"
            for n in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh")
            for k in range(2, 7)
        }


# ===========================================================================
# patch_dat round-trip on synthetic ElastoDyn templates
# ===========================================================================

_BLADE_TEMPLATE = """\
------- BLADE MODE SHAPES ------------------------------------------
          1   BldFl1Sh(2)   - Blade 1 flap mode 1, coeff of x^2
          0   BldFl1Sh(3)   - Blade 1 flap mode 1, coeff of x^3
          0   BldFl1Sh(4)   - Blade 1 flap mode 1, coeff of x^4
          0   BldFl1Sh(5)   - Blade 1 flap mode 1, coeff of x^5
          0   BldFl1Sh(6)   - Blade 1 flap mode 1, coeff of x^6
          1   BldFl2Sh(2)   - Blade 1 flap mode 2, coeff of x^2
          0   BldFl2Sh(3)   - Blade 1 flap mode 2, coeff of x^3
          0   BldFl2Sh(4)   - Blade 1 flap mode 2, coeff of x^4
          0   BldFl2Sh(5)   - Blade 1 flap mode 2, coeff of x^5
          0   BldFl2Sh(6)   - Blade 1 flap mode 2, coeff of x^6
          1   BldEdgSh(2)   - Blade 1 edge mode 1, coeff of x^2
          0   BldEdgSh(3)   - Blade 1 edge mode 1, coeff of x^3
          0   BldEdgSh(4)   - Blade 1 edge mode 1, coeff of x^4
          0   BldEdgSh(5)   - Blade 1 edge mode 1, coeff of x^5
          0   BldEdgSh(6)   - Blade 1 edge mode 1, coeff of x^6
"""


def _stub_blade_params():
    fit = PolyFitResult(c2=0.4, c3=0.2, c4=0.15, c5=0.15, c6=0.1,
                         rms_residual=0.01, tip_slope=2.5, cond_number=0.0)
    return BladeElastoDynParams(BldFl1Sh=fit, BldFl2Sh=fit, BldEdgSh=fit)


class TestPatchDat:

    def test_blade_patch_roundtrip(self, tmp_path):
        dat = tmp_path / "ElastoDyn.dat"
        dat.write_text(_BLADE_TEMPLATE, encoding="utf-8")
        params = _stub_blade_params()
        patch_dat(dat, params)
        text = dat.read_text(encoding="utf-8")
        d = params.as_dict()
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[1] in d:
                assert float(tokens[0]) == pytest.approx(d[tokens[1]], rel=1e-6)

    def test_missing_param_raises_keyerror(self, tmp_path):
        dat = tmp_path / "incomplete.dat"
        dat.write_text("nothing useful here\n", encoding="utf-8")
        params = _stub_blade_params()
        with pytest.raises(KeyError, match="BldFl1Sh"):
            patch_dat(dat, params)


# ===========================================================================
# Tower / blade distributed-property table row-count validation
# ===========================================================================
#
# Pass-4 added a cross-check that the parsed row count matches the
# declared ``NTwInpSt`` / ``NBlInpSt`` — silent truncation became a
# ``ValueError`` naming the gap.

class TestElastoDynRowCountMismatch:

    def test_tower_short_table_raises(self, tmp_path: pathlib.Path) -> None:
        # NTwInpSt = 5 but only 3 data rows — truncated table.
        tower = tmp_path / "short.dat"
        tower.write_text(
            "------- ELASTODYN V1.00.* TOWER INPUT FILE -------------------------------\n"
            "Synthetic short-table tower for the pyBmodes pass-4 ratchet.\n"
            "---------------------- TOWER PARAMETERS ----------------------------------\n"
            "         5    NTwInpSt    - Number of input stations\n"
            "          1   TwrFADmp(1) - Tower 1st FA damping (%)\n"
            "          1   TwrFADmp(2) - Tower 2nd FA damping (%)\n"
            "          1   TwrSSDmp(1) - Tower 1st SS damping (%)\n"
            "          1   TwrSSDmp(2) - Tower 2nd SS damping (%)\n"
            "---------------------- TOWER ADJUSTMUNT FACTORS --------------------------\n"
            "          1   FAStTunr(1)\n"
            "          1   FAStTunr(2)\n"
            "          1   SSStTunr(1)\n"
            "          1   SSStTunr(2)\n"
            "          1   AdjTwMa\n"
            "          1   AdjFASt\n"
            "          1   AdjSSSt\n"
            "---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------\n"
            "  HtFract       TMassDen         TwFAStif       TwSSStif\n"
            "   (-)           (kg/m)           (Nm^2)         (Nm^2)\n"
            "0.0      5000.0    5.0e10    5.0e10\n"
            "0.25     5000.0    5.0e10    5.0e10\n"
            "0.5      5000.0    5.0e10    5.0e10\n"  # only 3 rows of 5 declared
            "---------------------- TOWER FORE-AFT MODE SHAPES ------------------------\n"
            "1.0   TwFAM1Sh(2)\n"
            "0.0   TwFAM1Sh(3)\n"
            "0.0   TwFAM1Sh(4)\n"
            "0.0   TwFAM1Sh(5)\n"
            "0.0   TwFAM1Sh(6)\n",
            encoding="latin-1",
        )
        from pybmodes.io.elastodyn_reader import read_elastodyn_tower

        with pytest.raises(
            ValueError, match="tower distributed-properties.*NTwInpSt = 5",
        ):
            read_elastodyn_tower(tower)


# ===========================================================================
# compute_rot_mass applies AdjBlMs
# ===========================================================================

def test_compute_rot_mass_applies_adj_bl_ms() -> None:
    """``AdjBlMs`` should multiply through the per-blade mass
    integral. The ``compute_rot_mass`` method must not ignore it, even though the blade
    adapter ``to_pybmodes_blade`` already applied the same scalar."""
    from pybmodes.io._elastodyn.types import ElastoDynBlade, ElastoDynMain

    main = ElastoDynMain(
        header="h", title="t",
        num_bl=3, tip_rad=10.0, hub_rad=0.0,
        hub_mass=1_000.0,
    )
    # Uniform blade mass density: total mass per blade = mass_den * length
    # = 100 * 10 = 1000 kg. With AdjBlMs = 2.0 it should double to 2000 kg.
    blade_base = ElastoDynBlade(
        header="h", title="t",
        bl_fract=np.linspace(0.0, 1.0, 11),
        b_mass_den=np.full(11, 100.0),
    )
    # Baseline: adj_bl_ms = 1.0 (default).
    rot_mass_1 = main.compute_rot_mass(blade_base)
    expected_1 = main.hub_mass + main.num_bl * (100.0 * 10.0) * 1.0
    assert rot_mass_1 == pytest.approx(expected_1)

    # AdjBlMs = 2.0 must scale the per-blade integral.
    blade_scaled = dataclasses.replace(blade_base, adj_bl_ms=2.0)
    rot_mass_2 = main.compute_rot_mass(blade_scaled)
    expected_2 = main.hub_mass + main.num_bl * (100.0 * 10.0) * 2.0
    assert rot_mass_2 == pytest.approx(expected_2)
    assert rot_mass_2 > rot_mass_1


# ===========================================================================
# _tower_element_boundaries — duplicate-station-pair detection
# ===========================================================================

class TestTowerElementBoundaries:
    """The IFE UPSCALE 25MW tower deck (and any other deck produced by a
    preprocessor that emits the same duplicate-pair trick) lists property
    discontinuities as adjacent stations with a HtFract gap on the order
    of 1e-5. Used directly as FEM node locations this produces millimetre
    elements that wreck the bending-stiffness conditioning. The adapter
    detects that pattern and substitutes a uniform mesh; the station list
    itself stays in ``SectionProperties.span_loc`` so the step semantics
    survive the midpoint interpolation.
    """

    def test_well_spaced_stations_pass_through(self):
        from pybmodes.io._elastodyn.adapter import _tower_element_boundaries

        stations = np.linspace(0.0, 1.0, 11)
        out = _tower_element_boundaries(stations)
        np.testing.assert_array_equal(out, stations)

    def test_duplicate_pair_triggers_uniform_mesh(self):
        from pybmodes.io._elastodyn.adapter import _tower_element_boundaries

        # Mirror the UPSCALE 25MW pattern: 10 logical stations encoded as
        # 20 entries with a near-zero gap between each pair.
        eps = 1e-5
        pairs = []
        for i in range(10):
            base = i / 10.0
            pairs.extend([base, base + eps])
        stations = np.array([*pairs[:-1], 1.0], dtype=float)
        assert float(np.diff(stations).min()) < 1e-4

        out = _tower_element_boundaries(stations)

        assert out.size == stations.size
        # Should be a uniform mesh — no near-coincident pairs.
        spacings = np.diff(out)
        assert float(spacings.min()) > 0.01
        np.testing.assert_allclose(spacings, spacings.mean(), atol=1e-12)

    def test_threshold_boundary_well_spaced_preserved(self):
        from pybmodes.io._elastodyn.adapter import _tower_element_boundaries

        # 0.001 gap > 1e-4 tolerance — should pass through unchanged.
        stations = np.array([0.0, 0.25, 0.251, 0.5, 0.75, 1.0])
        out = _tower_element_boundaries(stations)
        np.testing.assert_array_equal(out, stations)
