"""Round-trip regression for the floating-platform-block emitter.

``src/pybmodes/_examples/sample_inputs/reference_turbines/build.py`` carries
the helpers that generate the floating reference-turbine BMI samples
(sub-cases 07 OC3 Hywind, 08 OC4 DeepCwind semi, 09 IEA-15 UMaineSemi,
10 IEA-22 Semi, 11 UPSCALE 25MW CentralTower). Those helpers serialise a
:class:`pybmodes.io.bmi.PlatformSupport` dataclass into the BMI
``tow_support = 1`` text block via
:func:`_floating_platform_block_from_platform_support`.

This test pins the round-trip: build the PlatformSupport via
:meth:`Tower.from_elastodyn_with_mooring` from upstream OpenFAST decks,
serialise, re-parse, and confirm the 6×6 matrices match the source values
to floating-point precision. Integration-marked because it reads upstream
decks under ``external/OpenFAST_files/`` that aren't bundled with the repo.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pytest

# build.py lives under src/pybmodes/_examples/sample_inputs/reference_turbines/
# and isn't packaged for import; side-load it the same way the existing
# scripts/benchmark_sparse_solver.py does for _synthetic_bmi.
_BUILD_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs" / "reference_turbines"
)
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

import build  # type: ignore[import-not-found]  # noqa: E402

from pybmodes.io.bmi import read_bmi  # noqa: E402
from pybmodes.models import Tower  # noqa: E402

pytestmark = pytest.mark.integration

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "external" / "OpenFAST_files"
_IEA15_UMS = _DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemi"
_IEA15_MAIN = _IEA15_UMS / "IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat"
_IEA15_MOOR = _IEA15_UMS / "IEA-15-240-RWT-UMaineSemi_MoorDyn.dat"
_IEA15_HYDRO = _IEA15_UMS / "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat"

# Upstream floating decks for the rigid-body-mode-naming r-tests.
# (main, moordyn, hydrodyn) tuples — solved straight from the raw
# OpenFAST data via from_elastodyn_with_mooring, NOT the regenerated
# bundled .bmi samples, so the classifier is exercised on real decks.
_IEA22_SEMI = _DOCS / "IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT-Semi"
_OC4 = _DOCS / "r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN"
_FLOATING_DECKS = {
    "IEA-15 UMaineSemi": (
        _IEA15_MAIN, _IEA15_MOOR, _IEA15_HYDRO,
    ),
    "IEA-22 Semi": (
        _IEA22_SEMI / "IEA-22-280-RWT-Semi_ElastoDyn.dat",
        _IEA22_SEMI / "IEA-22-280-RWT-Semi_MoorDyn.dat",
        _IEA22_SEMI / "IEA-22-280-RWT-Semi_HydroDyn_PotMod.dat",
    ),
    "NREL5MW OC4 DeepCwind": (
        _OC4 / "NRELOffshrBsline5MW_OC4DeepCwindSemi_ElastoDyn.dat",
        _OC4 / "NRELOffshrBsline5MW_OC4DeepCwindSemi_MoorDyn.dat",
        _OC4 / "NRELOffshrBsline5MW_OC4DeepCwindSemi_HydroDyn.dat",
    ),
}
_DOF_SET = {"surge", "sway", "heave", "roll", "pitch", "yaw"}


@pytest.mark.skipif(
    not _IEA15_MAIN.is_file(),
    reason="IEA-15-240-RWT UMaineSemi deck not present",
)
def test_floating_platform_block_roundtrip():
    """Serialise → re-parse a PlatformSupport assembled from the IEA-15
    UMaineSemi deck and confirm every field round-trips numerically."""
    tower = Tower.from_elastodyn_with_mooring(_IEA15_MAIN, _IEA15_MOOR, _IEA15_HYDRO)
    bmi = tower._bmi  # type: ignore[attr-defined]
    sp = tower._sp    # type: ignore[attr-defined]
    ps_in = bmi.support

    block_lines = build._floating_platform_block_from_platform_support(
        ps_in, provenance="test round-trip",
    )

    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td)
        sp_path = work / "test_sec_props.dat"
        build._emit_tower_sec_props(
            path=sp_path,
            title="round-trip section properties",
            ht_fract=np.asarray(sp.span_loc, dtype=float),
            t_mass_den=np.asarray(sp.mass_den, dtype=float),
            tw_fa_stif=np.asarray(sp.flp_stff, dtype=float),
        )
        bmi_path = work / "test_tower.bmi"
        inertias = dict(
            cm_loc=0.0, cm_axial=0.0,
            ixx_tip=1.0e7, iyy_tip=1.0e7, izz_tip=1.0e7, izx_tip=0.0,
        )
        build._emit_floating_tower_bmi(
            path=bmi_path,
            title="round-trip probe",
            radius=float(bmi.radius),
            hub_rad=0.0,
            tip_mass=1.0e6,
            inertias=inertias,
            sec_props_filename=sp_path.name,
            el_loc=np.asarray(bmi.el_loc, dtype=float),
            platform_block=block_lines,
        )

        bmi_out = read_bmi(bmi_path)

    ps_out = bmi_out.support
    assert ps_out is not None

    # Scalars
    np.testing.assert_allclose(ps_out.draft,      ps_in.draft,      rtol=1e-6)
    np.testing.assert_allclose(ps_out.cm_pform,   ps_in.cm_pform,   rtol=1e-6)
    np.testing.assert_allclose(ps_out.mass_pform, ps_in.mass_pform, rtol=1e-6)
    np.testing.assert_allclose(ps_out.ref_msl,    ps_in.ref_msl,    rtol=1e-6, atol=1e-9)

    # The 6×6 i_matrix carries the platform mass on the translational
    # diagonal and angular inertias in the 3-6 block; the parser
    # reconstructs the translational diagonal from mass_pform so we
    # only compare the structural (off-diagonal-zero, mass-on-diag)
    # form holistically.
    np.testing.assert_allclose(ps_out.i_matrix, ps_in.i_matrix, rtol=1e-5)
    np.testing.assert_allclose(ps_out.hydro_M,  ps_in.hydro_M,  rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(ps_out.hydro_K,  ps_in.hydro_K,  rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(ps_out.mooring_K, ps_in.mooring_K, rtol=1e-5, atol=1e-2)


@pytest.mark.skipif(
    not _IEA15_MAIN.is_file(),
    reason="IEA-15-240-RWT UMaineSemi deck not present",
)
def test_from_elastodyn_with_mooring_spectrum_is_nmodes_stable():
    """The in-memory ``from_elastodyn_with_mooring`` path must produce
    an ``n_modes``-invariant rigid-body spectrum.

    Companion to ``test_floating_samples_spectra`` (which pins the
    bundled-BMI path). Up to v1.1.1 the bundled-sample fix lived in
    ``build.py``; the ElastoDyn→pyBmodes adapter still synthesised the
    ~5e6×-too-stiff axial proxy, so a user driving their own asymmetric
    spar/semi deck through ``from_elastodyn_with_mooring`` would still
    hit the conditioning collapse (the modes drifted with the requested
    count). The fix threads ``physical_sec_props=True`` through
    ``to_pybmodes_tower`` for the free-base path; this asserts the
    invariant the old code violated."""
    def _solve(nm: int):
        return Tower.from_elastodyn_with_mooring(
            _IEA15_MAIN, _IEA15_MOOR, _IEA15_HYDRO
        ).run(n_modes=nm, check_model=False).frequencies

    f9 = _solve(9)
    f15 = _solve(15)

    drift = float(np.max(np.abs(f9[:6] - f15[:6])))
    assert drift < 1e-4, (
        f"from_elastodyn_with_mooring rigid-body spectrum drifts with "
        f"n_modes ({drift:.3e} Hz, n=9 vs n=15) — ill-conditioned "
        f"floating assembly.\n  n=9 : {np.array2string(f9[:6], precision=5)}"
        f"\n  n=15: {np.array2string(f15[:6], precision=5)}"
    )
    # Physical sanity: the six rigid-body modes must span a real
    # range, not collapse to one degenerate value (surge≈sway is fine;
    # a single six-fold value was the bug signature).
    rigid = f9[:6]
    distinct = 1 + sum(
        1 for i in range(1, 6) if rigid[i] > rigid[i - 1] * 1.01
    )
    assert distinct >= 3, (
        f"rigid-body modes collapsed to {distinct} distinct level(s): "
        f"{np.array2string(rigid, precision=5)}"
    )


@pytest.mark.parametrize("deck_name", list(_FLOATING_DECKS))
def test_rigid_body_modes_named_on_upstream_deck(deck_name: str) -> None:
    """Robustness r-test: solve a real upstream floating deck straight
    from its OpenFAST files (``from_elastodyn_with_mooring``, NOT the
    regenerated bundled .bmi) and confirm the classifier names the six
    platform rigid-body modes correctly.

    This exercises ``ModalResult.mode_labels`` on raw upstream data —
    three platform types (IEA-15 / IEA-22 UMaine-style semis + the
    NREL-5MW OC4 DeepCwind semi) — so the surge/sway/heave/roll/pitch/
    yaw identification is proven robust beyond the bundled samples and
    the synthetic unit cases.
    """
    main, moor, hyd = _FLOATING_DECKS[deck_name]
    if not main.is_file():
        pytest.skip(f"upstream deck not present: {main}")

    res = Tower.from_elastodyn_with_mooring(main, moor, hyd).run(
        n_modes=12, check_model=False
    )
    assert res.mode_labels is not None, deck_name
    first6 = res.mode_labels[:6]
    assert all(lbl is not None for lbl in first6), (deck_name, first6)
    # Exactly one of each platform DOF among the six rigid-body modes.
    assert set(first6) == _DOF_SET, (deck_name, first6)
    # Flexible tower modes above the rigid-body cluster stay unnamed.
    assert all(lbl is None for lbl in res.mode_labels[6:]), (
        deck_name, res.mode_labels[6:],
    )
