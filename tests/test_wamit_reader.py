"""Integration tests for ``pybmodes.io.wamit_reader``.

Verifies that :class:`WamitReader` and :class:`HydroDynReader` extract the
right dimensional values from the IEA-15-240-RWT-UMaineSemi WAMIT output
files shipped with the upstream IEA-15-240-RWT repo (gitignored under
``external/OpenFAST_files/``). Skips at module level when those files aren't
present on the local machine.

Numeric expectations are derived from a manual WAMIT v7 redimensionalisation
of the upstream ``.1`` and ``.hst`` files:

* ``ρ = 1025 kg/m³``, ``g = 9.80665 m/s²``, ``ULEN = 1 m``.
* Added-mass factor: ``ρ · L^k`` with ``k ∈ {3, 4, 5}`` for
  trans-trans / trans-rot / rot-rot pairs.
* Hydrostatic factor: ``ρ · g · L^k`` with ``k ∈ {2, 3, 4}``.

The 1% tolerance on every numeric expectation absorbs the difference
between published reference values (e.g. Allen et al. 2020) and the exact
re-dimensionalisation under our ρ / g choice; it is *not* an FEM
discretisation tolerance.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.io import HydroDynReader, WamitReader

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DECK_DIR = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-15-240-RWT" / "OpenFAST"
    / "IEA-15-240-RWT-UMaineSemi"
)
HD_PATH = DECK_DIR / "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat"
DOT1 = DECK_DIR / "HydroData" / "IEA-15-240-RWT-UMaineSemi.1"
HST = DECK_DIR / "HydroData" / "IEA-15-240-RWT-UMaineSemi.hst"

if not (HD_PATH.is_file() and DOT1.is_file() and HST.is_file()):
    pytest.skip(
        "IEA-15-240-RWT-UMaineSemi WAMIT data not present under "
        f"{DECK_DIR}; clone the upstream IEA-15-240-RWT repo into "
        "external/OpenFAST_files/ to run these tests.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Module-level fixture — single read shared across the value checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wamit_data():
    return HydroDynReader(HD_PATH).read_platform_matrices()


# ---------------------------------------------------------------------------
# A_inf — infinite-frequency added mass
# ---------------------------------------------------------------------------

def test_wamit_a_inf_surge(wamit_data) -> None:
    """``A_inf[0,0]`` ≈ 1.264e7 kg from 1.233416e4 (nondim) × ρ·L³."""
    assert wamit_data.A_inf[0, 0] == pytest.approx(1.264e7, rel=0.01)


def test_wamit_a_inf_heave(wamit_data) -> None:
    """``A_inf[2,2]`` ≈ 2.693e7 kg from 2.627505e4 (nondim) × ρ·L³."""
    assert wamit_data.A_inf[2, 2] == pytest.approx(2.693e7, rel=0.01)


def test_wamit_a_inf_pitch(wamit_data) -> None:
    """``A_inf[4,4]`` ≈ 1.247e10 kg·m² from 1.216208e7 (nondim) × ρ·L⁵."""
    assert wamit_data.A_inf[4, 4] == pytest.approx(1.247e10, rel=0.01)


# ---------------------------------------------------------------------------
# C_hst — hydrostatic restoring
# ---------------------------------------------------------------------------

def test_wamit_c_hst_heave(wamit_data) -> None:
    """``C_hst[2,2]`` ≈ 4.453e6 N/m from 4.430486e2 (nondim) × ρ·g·L²."""
    assert wamit_data.C_hst[2, 2] == pytest.approx(4.453e6, rel=0.01)


def test_wamit_c_hst_pitch_positive(wamit_data) -> None:
    """``C_hst[3,3]`` is positive for a hydrostatically stable semi.

    UMaine VolturnUS-S is a three-column semi; its pitch / roll restoring
    is positive from the spread-buoyancy waterline alone, *before* mooring
    contributes anything. (The negative expectation is correct for the OC3
    Hywind spar but not for a semi.)
    """
    assert wamit_data.C_hst[3, 3] > 0
    assert wamit_data.C_hst[3, 3] == pytest.approx(2.193e9, rel=0.01)


# ---------------------------------------------------------------------------
# Structural properties of the extracted matrices
# ---------------------------------------------------------------------------

def test_wamit_symmetry(wamit_data) -> None:
    """``A_inf`` and ``C_hst`` are symmetric to <0.1% of their peak value.

    Both matrices should be symmetric in theory (added mass is a generalized
    inertia; hydrostatic restoring is a Hessian of a potential energy).
    WAMIT writes them with small numerical asymmetry from the panel-method
    discretisation; the tolerance below is loose enough to absorb that but
    tight enough to catch a transposed-index bug in the parser.
    """
    rel = 1e-3
    A = wamit_data.A_inf
    assert np.max(np.abs(A - A.T)) < rel * np.max(np.abs(A))
    C = wamit_data.C_hst
    assert np.max(np.abs(C - C.T)) < rel * np.max(np.abs(C))


def test_wamit_shape_and_dtype(wamit_data) -> None:
    """All three returned matrices are 6 × 6 float arrays."""
    for name in ("A_inf", "A_0", "C_hst"):
        m = getattr(wamit_data, name)
        assert m.shape == (6, 6), f"{name} has wrong shape {m.shape}"
        assert np.issubdtype(m.dtype, np.floating)


# ---------------------------------------------------------------------------
# Path-resolution behaviour
# ---------------------------------------------------------------------------

def test_wamit_path_resolution() -> None:
    """All four ``PotFile`` spellings resolve to the same on-disk root."""
    pot = "HydroData/IEA-15-240-RWT-UMaineSemi"
    forms = [
        pot,
        f'"{pot}"',
        pot.replace("/", "\\"),
        f'"{pot}"'.replace("/", "\\"),
    ]
    roots = {
        WamitReader._resolve_pot_path(form, DECK_DIR)
        for form in forms
    }
    assert len(roots) == 1, f"divergent resolutions: {roots}"


def test_wamit_missing_file(tmp_path) -> None:
    """A non-existent ``PotFile`` root raises FileNotFoundError with the
    expected path in the message."""
    rdr = WamitReader(
        "no_such_root", tmp_path,
        rho=1025.0, g=9.80665, ulen=1.0,
    )
    with pytest.raises(FileNotFoundError, match=r"no_such_root\.1"):
        rdr.read()


# ---------------------------------------------------------------------------
# HydroDynReader scalar properties
# ---------------------------------------------------------------------------

def test_hydrodyn_reader_scalars() -> None:
    """``HydroDynReader`` surfaces ULEN, PotMod, PotFile, PtfmRefzt."""
    hd = HydroDynReader(HD_PATH)
    assert hd.ulen == pytest.approx(1.0)
    assert hd.pot_mod == 1
    assert "IEA-15-240-RWT-UMaineSemi" in hd.pot_file
    assert hd.ptfm_ref_zt == pytest.approx(0.0)
    # ρ and g default to ISO sea-water values when not inline.
    assert hd.rho_water == pytest.approx(1025.0)
    assert hd.gravity == pytest.approx(9.80665)


def test_hydrodyn_reader_missing_file(tmp_path) -> None:
    """Constructor surfaces a clear error when the .dat is absent."""
    with pytest.raises(FileNotFoundError, match="HydroDyn .dat not found"):
        HydroDynReader(tmp_path / "nope.dat")
