"""Tests for the OpenFAST ElastoDyn reader / writer / adapter.

Three layers are exercised against two real RWT bundles:

  * Parser   — spot-check known scalar/array values from the source file.
  * Writer   — semantic round-trip: parse → emit → re-parse → equal.
  * Adapter  — ``Tower.from_elastodyn`` and ``RotatingBlade.from_elastodyn``
               build a runnable model and the eigenvalue solve produces
               positive, sorted frequencies.

Reference data lives under ``external/OpenFAST_files/`` and is not committed.
The whole module skips at import time if either bundled RWT directory
is missing, which keeps the suite green on contributors who don't have
the local copies.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import numpy as np
import pytest

from pybmodes.io.elastodyn_reader import (
    ElastoDynBlade,
    ElastoDynMain,
    ElastoDynTower,
    read_elastodyn_blade,
    read_elastodyn_main,
    read_elastodyn_tower,
    write_elastodyn_blade,
    write_elastodyn_main,
    write_elastodyn_tower,
)
from pybmodes.models import RotatingBlade, Tower

# ---------------------------------------------------------------------------
# Local-only data location
# ---------------------------------------------------------------------------

# Every test in this module reads an upstream OpenFAST deck. Marked
# integration so the default ``pytest`` run doesn't try to load
# 5 GB of r-test data that lives outside this repo.
pytestmark = pytest.mark.integration

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "external" / "OpenFAST_files"

_M5_LAND = _DOCS / "r-test/glue-codes/openfast/5MW_Land_DLL_WTurb"
_M5_BASE = _DOCS / "r-test/glue-codes/openfast/5MW_Baseline"
_M5_MAIN = _M5_LAND / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
_M5_TOWER = _M5_BASE / "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
_M5_BLADE = _M5_BASE / "NRELOffshrBsline5MW_Blade.dat"

_IEA34 = _DOCS / "IEA-3.4-130-RWT/openfast"
_IEA34_MAIN = _IEA34 / "IEA-3.4-130-RWT_ElastoDyn.dat"
_IEA34_TOWER = _IEA34 / "IEA-3.4-130-RWT_ElastoDyn_tower.dat"
_IEA34_BLADE = _IEA34 / "IEA-3.4-130-RWT_ElastoDyn_blade.dat"

_UPSCALE_MAIN = (
    _DOCS
    / "IFE-UPSCALE-25MW-RWT/input/OpenFAST/CentralTower/UPSCALE-25MW-C_ElastoDyn.dat"
)

_REQUIRED = (_M5_MAIN, _M5_TOWER, _M5_BLADE, _IEA34_MAIN, _IEA34_TOWER, _IEA34_BLADE)
if not all(p.is_file() for p in _REQUIRED):
    missing = [str(p) for p in _REQUIRED if not p.is_file()]
    pytest.skip(
        "ElastoDyn reference inputs not present locally; missing:\n  "
        + "\n  ".join(missing),
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Dataclass equality with numpy fields
# ---------------------------------------------------------------------------

def _eq(a, b) -> bool:
    """Numpy-aware equality used by the round-trip checker.

    Float arrays compare with ``np.allclose`` to absorb the last-ulp drift
    that the writer's ``%.17g`` formatter can introduce on values whose
    decimal representation isn't exactly representable in binary.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            return False
        if np.issubdtype(aa.dtype, np.floating) or np.issubdtype(bb.dtype, np.floating):
            return bool(np.allclose(aa, bb, rtol=1e-12, atol=1e-15, equal_nan=True))
        return bool(np.array_equal(aa, bb))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=1e-12, abs_tol=1e-15
        )
    return a == b


def assert_dataclass_eq(a, b, *, ignore: tuple[str, ...] = ("source_file",)) -> None:
    """Compare two dataclasses field-by-field with numpy-aware equality."""
    assert type(a) is type(b)
    for fld in dataclasses.fields(a):
        if fld.name in ignore:
            continue
        va, vb = getattr(a, fld.name), getattr(b, fld.name)
        assert _eq(va, vb), f"field {fld.name!r}: {va!r} != {vb!r}"


# ---------------------------------------------------------------------------
# Parser spot checks — 5MW
# ---------------------------------------------------------------------------

class TestParse5MWMain:

    @pytest.fixture(scope="class")
    def main(self) -> ElastoDynMain:
        return read_elastodyn_main(_M5_MAIN)

    def test_geometry(self, main):
        assert main.tower_ht == pytest.approx(87.6)
        assert main.tower_bs_ht == 0.0
        assert main.tip_rad == pytest.approx(63.0)
        assert main.hub_rad == pytest.approx(1.5)
        assert main.overhang == pytest.approx(-5.0191)
        assert main.shft_tilt == pytest.approx(-5.0)
        assert main.twr2shft == pytest.approx(1.96256)

    def test_hub_ht_derived_matches_published(self, main):
        # HubHt = TowerHt + Twr2Shft + OverHang*sin(ShftTilt) → 90.0 m for
        # the 5MW deck. The trailing μ-metres reflect ShftTilt being given
        # to whole-degree precision in the source file.
        assert main.hub_ht == pytest.approx(90.0, abs=1e-3)

    def test_masses_and_inertias(self, main):
        assert main.hub_mass == pytest.approx(56780.0)
        assert main.nac_mass == pytest.approx(240000.0)
        assert main.nac_y_iner == pytest.approx(2607890.0)
        assert main.gen_iner == pytest.approx(534.116)
        assert main.hub_iner == pytest.approx(115926.0)

    def test_offsets(self, main):
        assert main.nac_cm_xn == pytest.approx(1.9)
        assert main.nac_cm_yn == 0.0
        assert main.nac_cm_zn == pytest.approx(1.75)
        assert main.hub_cm == 0.0

    def test_initial_rotor_speed(self, main):
        assert main.rot_speed_rpm == pytest.approx(12.1)

    def test_file_refs(self, main):
        assert main.twr_file == "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
        assert main.bld_file[0].endswith("NRELOffshrBsline5MW_Blade.dat")


class TestParse5MWTower:

    @pytest.fixture(scope="class")
    def tower(self) -> ElastoDynTower:
        return read_elastodyn_tower(_M5_TOWER)

    def test_n_stations_and_table_shape(self, tower):
        assert tower.n_tw_inp_st == 11
        assert tower.ht_fract.shape == (11,)
        assert tower.ht_fract[0] == 0.0
        assert tower.ht_fract[-1] == pytest.approx(1.0)

    def test_first_and_last_distributed_row(self, tower):
        # Picked from the source: row 0 and row 10.
        assert tower.t_mass_den[0] == pytest.approx(5590.87)
        assert tower.tw_fa_stif[0] == pytest.approx(6.14343e11, rel=1e-6)
        assert tower.t_mass_den[-1] == pytest.approx(2536.27)
        assert tower.tw_fa_stif[-1] == pytest.approx(1.1582e11, rel=1e-6)

    def test_mode_shape_coefficients(self, tower):
        np.testing.assert_allclose(
            tower.tw_fa_m1_sh, [0.7004, 2.1963, -5.6202, 6.2275, -2.504]
        )
        np.testing.assert_allclose(
            tower.tw_ss_m2_sh, [-121.21, 184.415, -224.904, 298.536, -135.838]
        )


class TestParse5MWBlade:

    @pytest.fixture(scope="class")
    def blade(self) -> ElastoDynBlade:
        return read_elastodyn_blade(_M5_BLADE)

    def test_n_stations_and_5col_format(self, blade):
        assert blade.n_bl_inp_st == 49
        assert blade.bl_fract.shape == (49,)
        # 5MW deck is the legacy 5-column form — no PitchAxis column.
        assert blade.pitch_axis is None

    def test_first_distributed_row(self, blade):
        assert blade.bl_fract[0] == 0.0
        assert blade.strc_twst[0] == pytest.approx(13.308)
        assert blade.b_mass_den[0] == pytest.approx(678.935)
        assert blade.flp_stff[0] == pytest.approx(1.811e10)
        assert blade.edg_stff[0] == pytest.approx(1.81136e10)

    def test_tip_distributed_row(self, blade):
        assert blade.bl_fract[-1] == pytest.approx(1.0)
        assert blade.b_mass_den[-1] == pytest.approx(10.319)

    def test_mode_shape_coefficients(self, blade):
        np.testing.assert_allclose(
            blade.bld_fl1_sh, [0.0622, 1.7254, -3.2452, 4.7131, -2.2555]
        )
        np.testing.assert_allclose(
            blade.bld_edg_sh, [0.3627, 2.5337, -3.5772, 2.376, -0.6952]
        )


# ---------------------------------------------------------------------------
# Parser spot checks — IEA-3.4-130 (newer / 6-col blade format)
# ---------------------------------------------------------------------------

class TestParseIEA34Main:

    @pytest.fixture(scope="class")
    def main(self) -> ElastoDynMain:
        return read_elastodyn_main(_IEA34_MAIN)

    def test_geometry(self, main):
        assert main.tower_ht == pytest.approx(108.0)
        assert main.tip_rad == pytest.approx(64.90852112228899)
        assert main.hub_rad == pytest.approx(2.0)

    def test_masses(self, main):
        assert main.hub_mass == pytest.approx(8239.17392489331)
        assert main.nac_mass == pytest.approx(114022.72257382338)
        assert main.nac_y_iner == pytest.approx(2171344.6678254865)

    def test_bare_digit_bld_file_form(self, main):
        # IEA-3.4 uses ``BldFile1`` (no parens) where 5MW uses ``BldFile(1)``.
        assert main.bld_file[0] == "IEA-3.4-130-RWT_ElastoDyn_blade.dat"
        assert main.bld_file[1] == main.bld_file[0]


class TestParseIEA34Blade:

    @pytest.fixture(scope="class")
    def blade(self) -> ElastoDynBlade:
        return read_elastodyn_blade(_IEA34_BLADE)

    def test_6col_format_with_pitch_axis(self, blade):
        # IEA-3.4 uses the newer 6-column variant with PitchAxis.
        assert blade.n_bl_inp_st == 30
        assert blade.pitch_axis is not None
        assert blade.pitch_axis.shape == (30,)
        assert blade.pitch_axis[0] == pytest.approx(0.5)
        assert blade.pitch_axis[-1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Round-trip: parse → emit → reparse → equal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [_M5_MAIN, _IEA34_MAIN])
def test_main_semantic_roundtrip(path):
    parsed1 = read_elastodyn_main(path)
    text = write_elastodyn_main(parsed1)
    parsed2 = read_elastodyn_main.__wrapped__(path) if hasattr(
        read_elastodyn_main, "__wrapped__"
    ) else None
    # Parse the emitted text directly via the underlying function. We don't
    # have a string-input variant; fall back to writing a tmp and re-reading.
    # See _reparse_text below.
    parsed2 = _reparse_text(read_elastodyn_main, text)
    # Skip ``scalars`` (raw strings vary between source and re-emitted form)
    # and ``out_list`` (writer normalises section header text).
    assert_dataclass_eq(
        parsed1, parsed2,
        ignore=("source_file", "scalars", "out_list", "nodal_out_list",
                "section_dividers", "header"),
    )


@pytest.mark.parametrize("path", [_M5_TOWER, _IEA34_TOWER])
def test_tower_semantic_roundtrip(path):
    parsed1 = read_elastodyn_tower(path)
    text = write_elastodyn_tower(parsed1)
    parsed2 = _reparse_text(read_elastodyn_tower, text)
    assert_dataclass_eq(
        parsed1, parsed2,
        ignore=("source_file", "section_dividers", "distr_header_lines", "header"),
    )


@pytest.mark.parametrize("path", [_M5_BLADE, _IEA34_BLADE])
def test_blade_semantic_roundtrip(path):
    parsed1 = read_elastodyn_blade(path)
    text = write_elastodyn_blade(parsed1)
    parsed2 = _reparse_text(read_elastodyn_blade, text)
    assert_dataclass_eq(
        parsed1, parsed2,
        ignore=("source_file", "section_dividers", "distr_header_lines", "header"),
    )


def _reparse_text(reader_fn, text: str, tmpdir_root: pathlib.Path | None = None):
    """Write text to a temp file and parse it back with ``reader_fn``."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", encoding="latin-1", suffix=".dat", delete=False) as f:
        f.write(text)
        path = pathlib.Path(f.name)
    try:
        return reader_fn(path)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adapter — Tower.from_elastodyn / RotatingBlade.from_elastodyn smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("main_path,label", [
    (_M5_MAIN,    "5MW"),
    (_IEA34_MAIN, "IEA-3.4-130"),
])
def test_tower_from_elastodyn_runs(main_path, label):
    """Build a tower from the ElastoDyn bundle and verify a clean modal solve."""
    tower = Tower.from_elastodyn(main_path)
    result = tower.run(n_modes=6)

    # Frequencies must be positive, finite, and non-decreasing.
    f = result.frequencies
    assert f.shape == (6,)
    assert np.all(np.isfinite(f))
    assert np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9), f"non-monotonic frequencies: {f}"

    # Sanity-check the first FA/SS pair is in a plausible range for a
    # multi-MW utility-scale tower (0.05 Hz – 1.0 Hz).
    assert 0.05 < f[0] < 1.0, f"{label}: tower first mode out of band: {f[0]:.4f} Hz"


@pytest.mark.parametrize("main_path,label", [
    (_M5_MAIN,    "5MW"),
    (_IEA34_MAIN, "IEA-3.4-130"),
])
def test_blade_from_elastodyn_runs(main_path, label):
    """Build a rotating-blade model and verify the modal solve."""
    blade = RotatingBlade.from_elastodyn(main_path)
    result = blade.run(n_modes=6)

    f = result.frequencies
    assert f.shape == (6,)
    assert np.all(np.isfinite(f))
    assert np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)

    # First flap of a multi-MW blade lives in 0.3–1.5 Hz — a wide sanity band.
    assert 0.2 < f[0] < 2.0, f"{label}: blade first mode out of band: {f[0]:.4f} Hz"


# ---------------------------------------------------------------------------
# Published-frequency targets — exercise the RNA tip-mass assembly
# ---------------------------------------------------------------------------

def test_5mw_tower_frequency_target():
    """First FA frequency for the 5MW land-based reference tower.

    Reference: Bir (2010), Table 1 — land-based tower with rigid head
    mass, BModes column reports first tower FA ≈ 0.3324 Hz.

    This is the right peer for our solve: a rigid-RNA tip-mass on a
    cantilevered Euler-Bernoulli tower, which is exactly what
    ``Tower.from_elastodyn`` builds. Coupled-simulation results (e.g.
    Jonkman et al. 2009, Table 9-1, ≈ 0.32 Hz) come from a model with
    blade and drivetrain DOFs that exchange energy with the tower-FA
    mode — not a fair comparison for a standalone modal solve.

    Tolerance is 2 %. Our rigid-RNA assembly lands at 0.3354 Hz (~0.9 %
    high), well within the band.
    """
    tower = Tower.from_elastodyn(_M5_MAIN)
    f = tower.run(n_modes=4).frequencies
    f_fa = float(f[0])
    target = 0.3324
    rel_err = abs(f_fa - target) / target
    assert rel_err < 0.02, (
        f"5MW tower 1st FA = {f_fa:.4f} Hz, target {target} Hz, "
        f"err {100*rel_err:.2f}% (allowed: 2%)"
    )


@pytest.mark.skipif(not _UPSCALE_MAIN.is_file(), reason="UPSCALE 25MW deck not present")
def test_upscale25_tower_duplicate_pair_stations():
    """Regression: the IFE UPSCALE 25MW tower deck (Sandua-Fernández et al.
    2023) lists 20 stations as 10 duplicate-pair entries encoding property
    discontinuities (HtFract gaps ≈ 7.7e-6). Used directly as FEM nodes,
    this produces 9 elements of ≈ 1.3 mm length whose stiffness-to-mass
    ratio collapses the bending spectrum into 4 spurious zero modes plus
    a degenerate 150 Hz pair. The adapter's duplicate-pair detector
    switches to a uniform mesh and recovers a physical spectrum.
    """
    tower = Tower.from_elastodyn(_UPSCALE_MAIN)
    f = tower.run(n_modes=6).frequencies

    assert np.all(np.isfinite(f))
    assert np.all(f > 0.0), f"non-physical zero modes returned: {f}"
    assert np.all(np.diff(f) >= -1e-9)

    # 25 MW on a 167 m flexible tower — 1st bending pair near 0.27 Hz.
    # FA and SS stiffness are identical in the deck so the pair is
    # numerically degenerate.
    assert 0.20 < f[0] < 0.35, f"1st FA/SS pair out of band: {f[0]:.4f} Hz"
    assert abs(f[1] - f[0]) / f[0] < 1e-3, (
        f"1st FA/SS pair should be degenerate: {f[0]:.4f}, {f[1]:.4f} Hz"
    )


def test_iea34_tower_frequency_sanity():
    """First FA frequency for the IEA-3.4MW reference.

    Reference: Bortolotti, Tarrés, Dykes, Merz, Sethuraman, Verelst, Zahle
    (2019), *IEA Wind TCP Task 37: Systems Engineering in Wind Energy —
    WP2.1 Reference Wind Turbines*, Table 5.1 reports tower 1st bending
    near ≈ 0.40 Hz for the land-based 3.4MW configuration.

    Tolerance is 10 % — same rigid-RNA caveat as for 5MW, plus the IEA
    reference modal data come from the pre-release HAWC2 model and the
    OpenFAST conversion shipped here may have slightly different stiffness
    distributions from what produced the original modal numbers.
    """
    tower = Tower.from_elastodyn(_IEA34_MAIN)
    f = tower.run(n_modes=4).frequencies
    f_fa = float(f[0])
    target = 0.40
    rel_err = abs(f_fa - target) / target
    assert rel_err < 0.10, (
        f"IEA-3.4 tower 1st FA = {f_fa:.4f} Hz, target ≈ {target} Hz, "
        f"err {100*rel_err:.2f}% (allowed: 10%)"
    )
