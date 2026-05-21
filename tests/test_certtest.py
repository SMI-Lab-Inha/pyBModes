"""Cert-test validation against BModes v3.00 reference output.

Each test parses one ``.bmi`` input plus the paired ``.out`` reference file,
runs the pyBModes FEM eigensolver on the same model, and asserts that every
reported frequency matches the reference within tight per-mode tolerances.

The four cases mirror the upstream certification cases:

    Test01  non-uniform blade, rotating at 60 rpm, no tip mass
    Test02  non-uniform blade, rotating at 60 rpm, with tip mass + offsets
    Test03  82.4 m tower, cantilevered base, top mass + c.m. offsets
    Test04  82.4 m tower, cantilevered base, top mass + tension-wire support

The cert-test data files are local-only and are not
committed under ``tests/data/``. This module reads them in place from
``external/BModes/CertTest/`` and skips at module level when that directory
is missing (e.g. on CI for contributors who don't have a local copy).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

from pybmodes.io.out_parser import read_out
from pybmodes.models import RotatingBlade, Tower

# ---------------------------------------------------------------------------
# Local-only data location
# ---------------------------------------------------------------------------

# All tests in this module require external data and run only under
# ``pytest -m integration`` (default ``pytest`` deselects them).
pytestmark = pytest.mark.integration

CERT_DIR = pathlib.Path(__file__).resolve().parents[1] / "external" / "BModes" / "CertTest"
OUT_DIR = CERT_DIR / "TestFiles"

if not CERT_DIR.is_dir() or not OUT_DIR.is_dir():
    pytest.skip(
        f"Cert-test data not present at {CERT_DIR}; "
        "these upstream reference files are local-only (not committed).",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Per-mode tolerances (relative error in percent)
# ---------------------------------------------------------------------------

ERR_LOW_PCT = 1.0   # modes 1..6
ERR_HIGH_PCT = 3.0  # modes 7..N

CASES: dict[str, tuple[str, str, str]] = {
    # case_id: (bmi filename, out filename, kind)
    "Test01": ("Test01_nonunif_blade.bmi",         "Test01_nonunif_blade.out",         "blade"),
    "Test02": ("Test02_blade_with_tip_mass.bmi",   "Test02_blade_with_tip_mass.out",   "blade"),
    "Test03": ("Test03_tower.bmi",                 "Test03_tower.out",                 "tower"),
    "Test04": ("Test04_wires_supported_tower.bmi", "Test04_wires_supported_tower.out", "tower"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve(bmi_path: pathlib.Path, kind: str, n_modes: int):
    if kind == "blade":
        return RotatingBlade(bmi_path).run(n_modes=n_modes)
    return Tower(bmi_path).run(n_modes=n_modes)


def _compare_case(case_id: str) -> None:
    bmi_name, out_name, kind = CASES[case_id]
    bmi_path = CERT_DIR / bmi_name
    out_path = OUT_DIR / out_name
    assert bmi_path.is_file(), f"missing input: {bmi_path}"
    assert out_path.is_file(), f"missing reference: {out_path}"

    # Distributed-hydro warning is irrelevant here and would clutter the table.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = read_out(out_path)
        ref_freqs = ref.frequencies()
        n_modes = int(ref_freqs.size)
        result = _solve(bmi_path, kind, n_modes)

    py_freqs = np.asarray(result.frequencies, dtype=float)
    rel_err_pct = 100.0 * np.abs(py_freqs - ref_freqs) / np.abs(ref_freqs)

    # Full comparison table — visible with `pytest -s`.
    print()
    print(f"=== {case_id}  ({bmi_name}, kind={kind}, n_modes={n_modes}) ===")
    print(f"{'mode':>4}  {'ref Hz':>14}  {'pybmodes Hz':>14}  {'err %':>10}  {'thr %':>6}")
    print(f"{'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*6}")
    for i in range(n_modes):
        thr = ERR_LOW_PCT if i < 6 else ERR_HIGH_PCT
        flag = "" if rel_err_pct[i] <= thr else "  FAIL"
        print(
            f"{i+1:>4}  {ref_freqs[i]:14.6e}  {py_freqs[i]:14.6e}  "
            f"{rel_err_pct[i]:10.4f}  {thr:6.1f}{flag}"
        )

    failures = []
    for i, err in enumerate(rel_err_pct):
        thr = ERR_LOW_PCT if i < 6 else ERR_HIGH_PCT
        if err > thr:
            failures.append(
                f"mode {i+1}: err={err:.3f}% > {thr:.1f}% "
                f"(ref={ref_freqs[i]:.6e} Hz, pybmodes={py_freqs[i]:.6e} Hz)"
            )
    if failures:
        pytest.fail(
            f"{case_id} frequency mismatch ({len(failures)}/{n_modes} modes):\n  "
            + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_certtest_01_blade_rotating_60rpm():
    _compare_case("Test01")


def test_certtest_02_blade_rotating_60rpm_tipmass():
    _compare_case("Test02")


def test_certtest_03_tower_cantilever_topmass_offset():
    _compare_case("Test03")


def test_certtest_04_tower_wire_supported_landbase():
    _compare_case("Test04")


# ---------------------------------------------------------------------------
# Offshore reference cases — CS_Monopile (soft monopile) and OC3Hywind
# (floating spar). Reference outputs come from a local BModes JJ run on
# the bundled examples in ``external/BModes/docs/examples/``. These exercise
# ``hub_conn = 3`` (axial+torsion locked, lateral free) and ``hub_conn = 2``
# (free-free) plus the full ``PlatformSupport`` matrix-assembly path
# (hydro-mass, hydro-stiffness, mooring-stiffness, platform-inertia 6×6
# matrices transformed to FEM tower-base DOFs by ``nondim_platform``).
# ---------------------------------------------------------------------------

OFFSHORE_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "external" / "BModes" / "docs" / "examples"
)
_OFFSHORE_CASES = {
    "CS_Monopile": ("CS_Monopile.bmi", "CS_Monopile.out"),
    "OC3Hywind":   ("OC3Hywind.bmi",   "OC3Hywind.out"),
}


def _offshore_paths(case_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    bmi_name, out_name = _OFFSHORE_CASES[case_id]
    return OFFSHORE_DIR / bmi_name, OFFSHORE_DIR / out_name


def _compare_offshore(case_id: str, n_modes: int, tolerance_pct: float) -> None:
    bmi_path, out_path = _offshore_paths(case_id)
    if not bmi_path.is_file() or not out_path.is_file():
        pytest.skip(
            f"Offshore reference data not present for {case_id} "
            f"(expected at {OFFSHORE_DIR}); local-only files."
        )
    with warnings.catch_warnings():
        # The PlatformSupport parser warns when distr_m is parsed but
        # ignored; both example decks declare zero distr_m stations, so
        # the warning shouldn't fire — silence anyway for clean output.
        warnings.simplefilter("ignore", UserWarning)
        ref = read_out(out_path)
        ref_freqs_full = ref.frequencies()
        if ref_freqs_full.size < n_modes:
            pytest.fail(
                f"{case_id}: reference .out only contains "
                f"{ref_freqs_full.size} modes, need {n_modes}"
            )
        ref_freqs = ref_freqs_full[:n_modes]
        result = Tower.from_bmi(bmi_path).run(n_modes=n_modes)

    py_freqs = np.asarray(result.frequencies, dtype=float)
    rel_err_pct = 100.0 * np.abs(py_freqs - ref_freqs) / np.abs(ref_freqs)

    # Per-mode comparison table — visible with `pytest -s`.
    print()
    print(f"=== {case_id}  (n_modes={n_modes}) ===")
    print(f"{'mode':>4}  {'ref Hz':>14}  {'pybmodes Hz':>14}  "
          f"{'err %':>10}  {'thr %':>6}")
    print(f"{'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*6}")
    for i in range(n_modes):
        thr = tolerance_pct
        flag = "" if rel_err_pct[i] <= thr else "  FAIL"
        print(
            f"{i+1:>4}  {ref_freqs[i]:14.6e}  {py_freqs[i]:14.6e}  "
            f"{rel_err_pct[i]:10.4f}  {thr:6.1f}{flag}"
        )

    failures = [
        f"mode {i+1}: err={rel_err_pct[i]:.3f}% > "
        f"{tolerance_pct:.3f}% "
        f"(ref={ref_freqs[i]:.6e} Hz, pybmodes={py_freqs[i]:.6e} Hz)"
        for i in range(n_modes)
        if rel_err_pct[i] > tolerance_pct
    ]
    if failures:
        pytest.fail(
            f"{case_id} frequency mismatch ({len(failures)}/{n_modes} "
            f"modes):\n  " + "\n  ".join(failures)
        )


def test_certtest_cs_monopile():
    """CS_Monopile.bmi — soft monopile with mooring stiffness, ``hub_conn = 3``.

    The deck is the old offshore dialect: the file says ``tow_support = 1``,
    and the parser normalizes that inline platform block to
    :class:`PlatformSupport`.  It contains non-zero ``mooring_K`` 6×6
    (lateral and rocking stiffness from a soil-equivalent foundation model) plus
    ``i_matrix``, ``hydro_M``, ``hydro_K`` blocks (zero in this case but
    parsed and assembled). The base reaction is "axial and torsion locked,
    lateral and rocking free" — reactions come entirely from the mooring
    stiffness.

    The pyBmodes-vs-BModes match here is essentially exact (typically
    < 0.001 % to 6 digits across the first 7 modes); the 1 % tolerance
    leaves headroom for floating-point rebaselining or mesh-precision
    drift. This test is a strict regression for the mooring-K → tower-base
    matrix transformation in ``nondim_platform``.
    """
    n_modes = 10
    _compare_offshore("CS_Monopile", n_modes, tolerance_pct=0.01)


def test_certtest_oc3hywind():
    """OC3Hywind.bmi — floating spar, free-free base (``hub_conn = 2``).

    The deck is the old offshore dialect: the file says ``tow_support = 1``,
    and the parser normalizes that inline platform block to
    :class:`PlatformSupport`.  It contains non-zero ``hydro_M`` (6×6
    hydrodynamic added mass), ``hydro_K`` (6×6 hydrostatic restoring,
    including negative pitch/roll terms), ``mooring_K``, and ``i_matrix``.
    All 6 base DOFs are free; the platform absorbs all reaction via
    hydrostatics + mooring.

    This is a strict regression for the ``PlatformSupport`` 6×6
    mass/stiffness assembly path.  OC3Hywind's low platform rigid-body
    modes and first tower-bending pair must stay aligned with the local
    BModes JJ reference run; a widened per-mode tolerance would hide the
    very verification issue this example is meant to catch.
    """
    n_modes = 9
    _compare_offshore("OC3Hywind", n_modes, tolerance_pct=0.01)
