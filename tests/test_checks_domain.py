"""Domain-aware input validation across the non-naval disciplines (#102).

The floating / naval-architecture gates landed in 1.11.0 (#95) and the
mechanical / units gates already live at the construction layer in
:func:`pybmodes.io.geometry.tubular_section_props`, which sees the user's
raw ``E`` / ``rho`` / ``D`` / ``t``. What this module covers is the rest:

* the **civil-structural** shell band, tightened for a fixed-bottom
  support — the case the construction-layer band explicitly cannot judge,
  because it does not know the boundary condition and a floating tower
  legitimately reaches D/t ~ 1100;
* the **geotechnical** gates — a rigid mudline clamp with no soil, and an
  implausible embedment ratio;
* the **post-solve scale** check, the one gate that looks at the answer
  rather than the inputs.

The standing rule for a domain gate is that it stays silent on the
validated reference turbines. There is no WindIO corpus in the repo (the
independence stance forbids one), so that half is pinned here against the
*published geometry* of the reference towers, entered as literals with
their source named — the same evidence, without vendoring anyone's file.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.checks import check_model, check_solved_frequencies
from pybmodes.models import Tower

# Published tube geometry for the reference towers the thresholds must not
# fire on, base station then top station.
#   NREL 5MW   (Jonkman et al. 2009, NREL/TP-500-38060): 6.000 m OD /
#              27.0 mm wall at the base, 3.870 m / 19.0 mm at the top.
#   IEA-15-240 (Gaertner et al. 2020, NREL/TP-5000-75698): 10.0 m /
#              41.3 mm at the base, 6.5 m / 20.9 mm at the top; its
#              monopile is prismatic at 10.0 m / 55.3 mm.
REFERENCE_TOWERS = {
    "nrel5mw": (np.array([6.000, 3.870]), np.array([0.0270, 0.0190])),
    "iea15": (np.array([10.00, 6.500]), np.array([0.0413, 0.0209])),
    "iea15_monopile": (np.array([10.00, 10.00]), np.array([0.0553, 0.0553])),
}


def _tower(od, wt, *, hub_conn=1, length=100.0, tip_mass=3.0e5) -> Tower:
    od = np.asarray(od, dtype=float)
    grid = np.linspace(0.0, 1.0, od.size)
    return Tower.from_geometry(
        grid, od, np.asarray(wt, dtype=float),
        flexible_length=length, E=2.0e11, rho=7850.0, nu=0.3,
        hub_conn=hub_conn, tip_mass=tip_mass,
    )


def _messages(model, severity=None) -> list[str]:
    return [
        f.message for f in check_model(model)
        if severity is None or f.severity == severity
    ]


class TestSilentOnReferenceTurbines:
    """No domain gate may fire on a published reference tower."""

    @pytest.mark.parametrize("name", sorted(REFERENCE_TOWERS))
    def test_no_warn_or_error(self, name):
        od, wt = REFERENCE_TOWERS[name]
        findings = [
            f for f in check_model(_tower(od, wt))
            if f.severity in ("WARN", "ERROR")
            and f.location.startswith("construction.")
        ]
        assert findings == [], f"{name}: {[str(f) for f in findings]}"

    def test_reference_ratios_sit_inside_the_band(self):
        from pybmodes.options import DEFAULT_CHECK_OPTIONS as opt

        for od, wt in REFERENCE_TOWERS.values():
            ratio = od / wt
            assert opt.diameter_thickness_min < ratio.min()
            assert ratio.max() < opt.diameter_thickness_max


class TestFixedBottomShellBand:
    """The band applies where the support type justifies it, and nowhere
    else — the gap the construction-layer band leaves for this issue."""

    # D/t = 650: legitimate on the IEA-15 VolturnUS-S floating tower,
    # implausible for anything clamped in the seabed.
    THIN = (np.array([6.5, 6.5]), np.array([0.01, 0.01]))

    def test_thin_shell_on_a_clamped_tower_warns(self):
        od, wt = self.THIN
        msgs = _messages(_tower(od, wt, hub_conn=1), "WARN")
        assert any("diameter-to-thickness" in m for m in msgs)

    def test_thin_shell_on_a_soft_monopile_warns(self):
        od, wt = self.THIN
        msgs = _messages(_tower(od, wt, hub_conn=3), "WARN")
        assert any("diameter-to-thickness" in m for m in msgs)

    def test_thin_shell_on_a_floating_tower_is_silent(self):
        """A floating tower carries far less bending, so the fixed-bottom
        band does not describe it and must not be applied to it."""
        od, wt = self.THIN
        msgs = _messages(_tower(od, wt, hub_conn=2))
        assert not any("diameter-to-thickness" in m for m in msgs)

    def test_near_solid_section_warns(self):
        msgs = _messages(_tower(np.array([6.0, 6.0]), np.array([0.5, 0.5])), "WARN")
        assert any("diameter-to-thickness" in m for m in msgs)

    @pytest.mark.parametrize("name", sorted(REFERENCE_TOWERS))
    def test_reference_towers_are_silent(self, name):
        od, wt = REFERENCE_TOWERS[name]
        assert not any(
            "diameter-to-thickness" in m for m in _messages(_tower(od, wt))
        )


class TestGeotechnical:
    """These need a monopile model, so they go through the WindIO splice."""

    YAML = """\
environment:
  water_depth: 30.0
components:
  monopile:
    outer_shape:
      outer_diameter:
        grid: [0.0, 1.0]
        values: [10.0, 10.0]
    structure:
      outfitting_factor: 1.0
      layers:
        - name: wall
          material: steel
          thickness:
            grid: [0.0, 1.0]
            values: [0.055, 0.055]
    reference_axis:
      z:
        grid: [0.0, 1.0]
        values: [{z_base}, 15.0]
  tower:
    outer_shape:
      outer_diameter:
        grid: [0.0, 1.0]
        values: [10.0, 6.5]
    structure:
      outfitting_factor: 1.0
      layers:
        - name: wall
          material: steel
          thickness:
            grid: [0.0, 1.0]
            values: [0.0413, 0.0209]
    reference_axis:
      z:
        grid: [0.0, 1.0]
        values: [15.0, 145.0]
materials:
  - name: steel
    E: 2.0e11
    rho: 7850.0
    nu: 0.3
"""

    def _yaml(self, tmp_path, z_base=-75.0):
        pytest.importorskip("yaml")
        p = tmp_path / "mp.yaml"
        p.write_text(self.YAML.format(z_base=z_base), encoding="utf-8")
        return p

    def test_rigid_clamp_without_soil_is_flagged_as_info(self, tmp_path):
        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6,
        )
        findings = [
            f for f in check_model(model) if "clamped rigidly" in f.message
        ]
        assert len(findings) == 1
        assert findings[0].severity == "INFO"

    def test_soil_model_suppresses_the_note(self, tmp_path):
        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6, soil_E=1.4e8,
        )
        assert not any("clamped rigidly" in m for m in _messages(model))

    def test_plausible_embedment_is_silent(self, tmp_path):
        """45 m embedded on a 10 m pile is L/D = 4.5, a normal design."""
        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6, soil_E=1.4e8,
        )
        assert not any("embedment ratio" in m for m in _messages(model))

    def test_absurd_embedment_is_flagged(self, tmp_path):
        """A pile toe 2 km down is a transcription error, not a design."""
        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path, z_base=-2030.0), tip_mass=1.0e6, soil_E=1.4e8,
        )
        assert any("embedment ratio" in m for m in _messages(model, "WARN"))

    def test_absurd_embedment_is_flagged_on_the_rigid_path_too(self, tmp_path):
        """The rigid path truncates the embedded pile out of the beam, but
        the ontology still states it and an absurd value is still a
        transcription error (Codex review on #138)."""
        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path, z_base=-2030.0), tip_mass=1.0e6,
        )
        assert model._construction.embedded_length == pytest.approx(2000.0)
        assert any("embedment ratio" in m for m in _messages(model, "WARN"))

    def test_rigid_path_records_the_design_embedment(self, tmp_path):
        """Recorded from the pile toe as drawn, not from the truncated beam
        base, so it is the same number on every path."""
        rigid = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6,
        )
        lumped = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6, soil_E=1.4e8,
        )
        distributed = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6, soil_E=1.4e8,
            soil_distributed=True,
        )
        assert rigid._construction.embedded_length == pytest.approx(45.0)
        assert lumped._construction.embedded_length == pytest.approx(45.0)
        assert distributed._construction.embedded_length == pytest.approx(45.0)

    def test_info_does_not_reach_the_solve_path(self, tmp_path):
        """INFO findings stay out of ``.run()`` — the existing contract."""
        import warnings as _w

        model = Tower.from_windio_with_monopile(
            self._yaml(tmp_path), tip_mass=1.0e6, n_nodes=30,
        )
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            model.run(4)
        assert not any("clamped rigidly" in str(w.message) for w in caught)

    def test_land_tower_has_no_geotechnical_findings(self):
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        msgs = _messages(_tower(od, wt))
        assert not any("clamped rigidly" in m or "embedment" in m for m in msgs)


class TestPostSolveFrequencyBand:
    def test_plausible_tower_is_silent(self):
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt, length=87.6)
        result = model.run(4, check_model=False)
        assert check_solved_frequencies(model, result.frequencies) == []
        assert 0.15 < result.frequencies[0] < 1.5

    def test_absurdly_low_frequency_warns(self):
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt)
        findings = check_solved_frequencies(model, np.array([1.0e-4, 2.0e-4]))
        assert len(findings) == 1
        assert findings[0].severity == "WARN"

    def test_absurdly_high_frequency_warns(self):
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt)
        assert check_solved_frequencies(model, np.array([250.0, 900.0]))

    def test_floating_model_is_skipped(self):
        """A floater's rigid-body modes are legitimately far below the
        band, so the check does not apply to them."""
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt, hub_conn=2)
        assert check_solved_frequencies(model, np.array([0.005, 0.008])) == []

    def test_blade_is_skipped(self, tmp_path):
        from pybmodes.models import RotatingBlade
        from tests._synthetic_bmi import write_bmi, write_uniform_sec_props

        write_uniform_sec_props(tmp_path / "secs.dat")
        path = write_bmi(
            tmp_path / "b.bmi", beam_type=1, radius=60.0, hub_conn=1,
            n_elements=8, sec_props_file="secs.dat",
        )
        blade = RotatingBlade(path)
        assert check_solved_frequencies(blade, np.array([1.0e-3])) == []

    def test_non_finite_frequencies_do_not_crash(self):
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt)
        assert check_solved_frequencies(
            model, np.array([np.nan, np.inf, -1.0]),
        ) == []

    def test_it_fires_through_run(self):
        """A tower with a 1000x scale error on its length lands outside
        the band even though each input passes its own gate."""
        od, wt = REFERENCE_TOWERS["nrel5mw"]
        model = _tower(od, wt, length=100000.0, tip_mass=0.0)
        with pytest.warns(UserWarning, match="lowest natural frequency"):
            model.run(4)


class TestDeckModelsSkipTheseGates:
    """A deck carries no raw tube or material, so the input gates are
    skipped rather than guessed at — the axial-rigid placeholder trap
    #102 names."""

    def test_bmi_model_has_no_construction_record(self, tmp_path):
        from tests._synthetic_bmi import write_bmi, write_uniform_sec_props

        write_uniform_sec_props(tmp_path / "secs.dat")
        path = write_bmi(
            tmp_path / "t.bmi", beam_type=2, radius=80.0, hub_conn=1,
            n_elements=10, sec_props_file="secs.dat",
        )
        model = Tower(path)
        assert model._construction is None
        assert not any(
            f.location.startswith("construction.") for f in check_model(model)
        )
