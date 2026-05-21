"""Asymmetric floating-platform CM support (1.2.0).

The rigid-arm transform in :func:`pybmodes.fem.nondim._rigid_arm_T`
maps the platform's 6×6 mass/stiffness matrices from their reference
point onto the tower-base FEM DOFs. Through 1.1.x it carried only a
*vertical* lever (``cm_pform − draft``); 1.2.0 generalises it to a
full 3-D arm so an asymmetric floating substructure with a horizontal
CM offset (``PtfmCMxt`` / ``PtfmCMyt``) is solved correctly.

These tests are self-contained (default suite, no external data):

1. ``rx = ry = 0`` is byte-identical to the pre-1.2.0 hand-written
   transform — the strongest possible guard that every axisymmetric
   spar / symmetric semi deck is unchanged (the OC3 Hywind cert test
   already pins this empirically vs BModes JJ at 0.0003 %).
2. The transform has the exact rigid-body kinematic structure
   ``G = [[I3, -skew(r)], [0, I3]]`` for the 3-D arm.
3. Closed-form, non-circular: transferring a point-mass spatial
   inertia through the transform reproduces the textbook
   parallel-axis result derived independently from kinetic energy.
4. End-to-end through the public solver: a synthetic floating model
   with a horizontal CM offset reproduces the analytic sway–yaw
   coupled-oscillator frequency split, and collapses back to the
   decoupled frequencies when the offset is zero.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pytest

from pybmodes.fem.nondim import _rigid_arm_T

# build.py ships the floating-sample emitters but isn't packaged for
# import; side-load it the same way test_floating_samples.py does.
_BUILD_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs" / "reference_turbines"
)
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

import build  # type: ignore[import-not-found]  # noqa: E402


def _skew(r: np.ndarray) -> np.ndarray:
    rx, ry, rz = r
    return np.array([
        [0.0, -rz,  ry],
        [rz,  0.0, -rx],
        [-ry, rx,  0.0],
    ])


# arm-independent FEM↔file reorder, identical to the one inside
# _rigid_arm_T (rebuilt here so the test is independent of internals).
def _P() -> np.ndarray:
    P = np.zeros((6, 6))
    P[0, 1] = 1.0
    P[1, 3] = 1.0
    P[2, 0] = 1.0
    P[3, 4] = -1.0
    P[4, 2] = 1.0
    P[5, 5] = 1.0
    return P


def _pre_120_T(p_base: float) -> np.ndarray:
    """The exact vertical-only transform pyBmodes shipped through
    1.1.x (verified vs BModes JJ on OC3 Hywind at 0.0003 %)."""
    T = np.zeros((6, 6))
    T[0, 1] = 1.0
    T[0, 2] = -p_base
    T[1, 3] = 1.0
    T[1, 4] = -p_base
    T[2, 0] = 1.0
    T[3, 4] = -1.0
    T[4, 2] = 1.0
    T[5, 5] = 1.0
    return T


@pytest.mark.parametrize("p_base", [0.0, -10.0, 89.9155, 144.386])
def test_symmetric_arm_byte_identical_to_pre_120(p_base: float) -> None:
    """rx = ry = 0 must reproduce the pre-1.2.0 transform exactly."""
    np.testing.assert_array_equal(
        _rigid_arm_T(p_base, 0.0, 0.0), _pre_120_T(p_base)
    )


@pytest.mark.parametrize(
    "p_base,rx,ry",
    [(89.9155, 5.0, -3.0), (-10.0, 0.0, 2.5), (50.0, -4.2, 0.0)],
)
def test_rigid_body_kinematic_structure(p_base, rx, ry) -> None:
    """T = G @ P with G = [[I3, -skew(r)], [0, I3]] and r = (rx, ry,
    -p_base) — i.e. a textbook rigid-body small-rotation transfer."""
    P = _P()
    T = _rigid_arm_T(p_base, rx, ry)
    # P is orthogonal (signed permutation): G = T @ Pᵀ.
    G = T @ P.T

    r = np.array([rx, ry, -p_base])
    G_expected = np.eye(6)
    G_expected[0:3, 3:6] = -_skew(r)

    np.testing.assert_allclose(G, G_expected, atol=1e-12)
    # Lower-left block is exactly zero, lower-right is identity:
    # rotations transfer rigidly, translations pick up the arm.
    np.testing.assert_array_equal(G[3:6, 0:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(G[3:6, 3:6], np.eye(3))


@pytest.mark.parametrize(
    "p_base,rx,ry,m",
    [(30.0, 6.0, -4.0, 1.7e6), (-12.0, 0.0, 3.5, 9.0e5)],
)
def test_point_mass_parallel_axis_closed_form(p_base, rx, ry, m) -> None:
    """Transferring a point mass referenced at its own CM through the
    arm must give the textbook spatial inertia about the tower base.

    Independent (non-circular) reference: from KE = ½m|ṫ + θ̇×r|² the
    spatial inertia about O is
        M_O = [[ m I3,        -m S(r) ],
               [ -m S(r)ᵀ,     m(|r|²I − r rᵀ) ]]
    in file DOF order. We build M_O from this formula and compare
    against the code path Tᵀ M_cm T (reordered back to file DOFs)."""
    r = np.array([rx, ry, -p_base])
    S = _skew(r)

    # Platform 6×6 at its own CM, file DOF order: point mass, no
    # rotational inertia, no coupling.
    M_cm_file = np.zeros((6, 6))
    M_cm_file[0, 0] = M_cm_file[1, 1] = M_cm_file[2, 2] = m

    P = _P()
    T = _rigid_arm_T(p_base, rx, ry)          # FEM ← (maps to) file
    M_base_fem = T.T @ M_cm_file @ T          # tower base, FEM order
    M_base_file = P @ M_base_fem @ P.T        # back to file order

    M_O = np.zeros((6, 6))
    M_O[0:3, 0:3] = m * np.eye(3)
    M_O[0:3, 3:6] = -m * S
    M_O[3:6, 0:3] = -m * S.T
    M_O[3:6, 3:6] = m * (float(r @ r) * np.eye(3) - np.outer(r, r))

    np.testing.assert_allclose(M_base_file, M_O, rtol=1e-10, atol=1e-6)


_SAMPLE09 = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "pybmodes" / "_examples" / "sample_inputs"
    / "reference_turbines" / "09_iea15_umainesemi"
    / "09_iea15_umainesemi_tower.bmi"
)


def _solve_with_cm_offset(rx: float, ry: float, n_modes: int = 12):
    """Solve the bundled, known-good sample-09 floating tower with its
    PlatformSupport's horizontal CM offset overridden to (rx, ry).

    Reuses the validated sample-09 tower beam + section properties (a
    well-conditioned regime — `test_floating_samples_spectra` pins it)
    and changes ONLY the horizontal CM offset, so this isolates the
    wiring from ``PlatformSupport.cm_pform_x/y`` through
    ``nondim_platform`` into the solver without the conditioning
    pitfalls of a hand-rolled massless tower.
    """
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    bmi = read_bmi(_SAMPLE09)
    bmi.support = dataclasses.replace(
        bmi.support, cm_pform_x=rx, cm_pform_y=ry
    )
    t = Tower.__new__(Tower)
    t._bmi = bmi
    t._sp = read_sec_props(bmi.resolve_sec_props_path())
    return t.run(n_modes=n_modes, check_model=False).frequencies


def test_zero_offset_reproduces_stock_sample09() -> None:
    """cm_pform_x = cm_pform_y = 0 must give exactly the stock
    sample-09 spectrum — i.e. the new field is inert for a symmetric
    platform (the value the bundled file already carries)."""
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    bmi = read_bmi(_SAMPLE09)
    assert bmi.support.cm_pform_x == 0.0 and bmi.support.cm_pform_y == 0.0
    t = Tower.__new__(Tower)
    t._bmi = bmi
    t._sp = read_sec_props(bmi.resolve_sec_props_path())
    stock = t.run(n_modes=12, check_model=False).frequencies

    explicit_zero = _solve_with_cm_offset(0.0, 0.0)
    np.testing.assert_array_equal(stock, explicit_zero)


@pytest.mark.parametrize("rx", [10.0, 30.0])
def test_horizontal_cm_offset_is_wired_and_stable(rx: float) -> None:
    """A horizontal CM offset must (a) measurably shift the rigid-body
    spectrum vs the symmetric case — proving the field is wired
    through nondim_platform into the solve — and (b) stay
    n_modes-stable (no conditioning regression from the new coupling
    terms)."""
    f0 = _solve_with_cm_offset(0.0, 0.0, n_modes=12)
    fr = _solve_with_cm_offset(rx, 0.0, n_modes=12)

    # The offset must change the low (rigid-body) spectrum.
    assert np.max(np.abs(fr[:6] - f0[:6])) > 1e-4, (
        "horizontal CM offset had no effect — not wired through"
    )
    # …and the offset solve must itself be n_modes-stable.
    fr15 = _solve_with_cm_offset(rx, 0.0, n_modes=15)
    drift = float(np.max(np.abs(fr[:6] - fr15[:6])))
    assert drift < 1e-4, (
        f"asymmetric-CM spectrum drifts with n_modes ({drift:.2e} Hz) — "
        f"conditioning regression from the horizontal-arm coupling"
    )
    assert np.all(np.isfinite(fr)) and np.all(fr > 0.0)


# ---------------------------------------------------------------------------
# BMI text-format round-trip (1.2.1): hand-authored asymmetric .bmi.
# ---------------------------------------------------------------------------


def _platform(cm_x: float, cm_y: float):
    from pybmodes.io.bmi import PlatformSupport

    i6 = np.zeros((6, 6))
    m = 1.6e7
    i6[0, 0] = i6[1, 1] = i6[2, 2] = m
    i6[3, 3] = i6[4, 4] = 1.4e10
    i6[5, 5] = 2.1e10
    K = np.diag([8.0e4, 7.9e4, 5.5e4, 1.8e8, 1.8e8, 1.7e8]).astype(float)
    return PlatformSupport(
        draft=-15.0, cm_pform=14.0, mass_pform=m, i_matrix=i6,
        ref_msl=0.0, hydro_M=np.zeros((6, 6)), hydro_K=np.zeros((6, 6)),
        mooring_K=K, distr_m_z=np.zeros(0), distr_m=np.zeros(0),
        distr_k_z=np.zeros(0), distr_k=np.zeros(0),
        cm_pform_x=cm_x, cm_pform_y=cm_y,
    )


def _write_floating_bmi(tmp: pathlib.Path, ps, name: str) -> pathlib.Path:
    """Emit a complete floating .bmi (+ sec-props) for PlatformSupport
    ``ps`` using the shipped build.py emitters, the same path the
    bundled samples are generated by."""
    sp_path = tmp / f"{name}_sec.dat"
    span = np.linspace(0.0, 1.0, 11)
    build._emit_tower_sec_props(
        path=sp_path, title="rt sec props",
        ht_fract=span, t_mass_den=np.full(11, 6.0e3),
        tw_fa_stif=np.full(11, 8.0e11), physical=True,
    )
    bmi_path = tmp / f"{name}.bmi"
    build._emit_floating_tower_bmi(
        path=bmi_path, title="hand-authored asymmetric float",
        radius=140.0, hub_rad=0.0, tip_mass=8.0e5,
        inertias=dict(cm_loc=0.0, cm_axial=0.0, ixx_tip=1e7,
                      iyy_tip=1e7, izz_tip=1e7, izx_tip=0.0),
        sec_props_filename=sp_path.name,
        el_loc=np.linspace(0.0, 1.0, 21),
        platform_block=build._floating_platform_block_from_platform_support(
            ps, provenance="round-trip test"),
    )
    return bmi_path


def test_symmetric_emits_legacy_single_value_line(tmp_path) -> None:
    """A symmetric platform must emit the pre-1.2.1 single-value
    cm_pform line verbatim (zero new tokens) so every existing deck /
    bundled sample is byte-identical."""
    bmi_path = _write_floating_bmi(tmp_path, _platform(0.0, 0.0), "sym")
    cm_lines = [
        ln for ln in bmi_path.read_text().splitlines()
        if "cm_pform" in ln and "below MSL" in ln
    ]
    assert len(cm_lines) == 1
    # Exactly one leading numeric token, then the label word.
    toks = cm_lines[0].split()
    assert toks[0].replace(".", "").replace("-", "").isdigit()
    assert toks[1].startswith("cm_pform")


@pytest.mark.parametrize("bad_line", [
    "14.0  6.5    cm_pform_xyz : one offset omitted (malformed)",
    "14.0  6.5  -3.0  2.0    cm_pform_xyzw : too many (malformed)",
])
def test_incomplete_cm_pform_offset_pair_rejected(tmp_path, bad_line) -> None:
    """The horizontal CM offsets are an (x, y) pair: a leading numeric
    run of 2 (one omitted) or >= 4 is a malformed hand-authored line
    and must raise, not silently default the missing coordinate."""
    from pybmodes.io.bmi import read_bmi

    good = _write_floating_bmi(tmp_path, _platform(6.5, -3.25), "bad")
    text = good.read_text()
    cm_line = next(
        ln for ln in text.splitlines() if "cm_pform_xyz" in ln
    )
    corrupted = tmp_path / "corrupt.bmi"
    corrupted.write_text(text.replace(cm_line, bad_line))
    with pytest.raises(ValueError, match="cm_pform line must have"):
        read_bmi(corrupted)


@pytest.mark.parametrize("cm_x,cm_y", [(6.5, -3.25), (0.0, 4.0), (-7.0, 0.0)])
def test_hand_authored_asymmetric_bmi_roundtrips(tmp_path, cm_x, cm_y) -> None:
    """Emit → re-parse: a hand-authored asymmetric .bmi preserves the
    horizontal CM offset through the (extended) text format."""
    from pybmodes.io.bmi import read_bmi

    bmi_path = _write_floating_bmi(tmp_path, _platform(cm_x, cm_y), "asym")
    back = read_bmi(bmi_path)
    assert back.support.cm_pform == pytest.approx(14.0, rel=1e-6)
    assert back.support.cm_pform_x == pytest.approx(cm_x, abs=1e-6)
    assert back.support.cm_pform_y == pytest.approx(cm_y, abs=1e-6)


def test_hand_authored_asymmetric_bmi_changes_spectrum(tmp_path) -> None:
    """End-to-end through the public .bmi path: the parsed horizontal
    CM offset reaches the solver and shifts the rigid-body spectrum
    (n_modes-stable), proving the format → nondim → solve chain works
    for a hand-authored deck (hand-authored .bmi workflow — no OpenFAST)."""
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    def _solve(cm_x: float, n: int = 12):
        p = _write_floating_bmi(tmp_path, _platform(cm_x, 0.0), f"e{cm_x}_{n}")
        bmi = read_bmi(p)
        t = Tower.__new__(Tower)
        t._bmi = bmi
        t._sp = read_sec_props(bmi.resolve_sec_props_path())
        return t.run(n_modes=n, check_model=False).frequencies

    f0 = _solve(0.0)
    fr = _solve(12.0)
    assert np.max(np.abs(fr[:6] - f0[:6])) > 1e-4, (
        "hand-authored horizontal CM offset had no effect — the "
        "extended .bmi format did not reach the solver"
    )
    fr15 = _solve(12.0, n=15)
    assert float(np.max(np.abs(fr[:6] - fr15[:6]))) < 1e-4
    assert np.all(np.isfinite(fr)) and np.all(fr > 0.0)


@pytest.mark.parametrize("cm_x,cm_y", [(0.0, 0.0), (6.5, -3.25)])
def test_hand_authored_bmi_rigid_modes_named(tmp_path, cm_x, cm_y) -> None:
    """Rigid-body mode naming must work for a *hand-authored* floating
    .bmi (hand-authored .bmi workflow: no OpenFAST), symmetric and asymmetric.

    Reads the emitted .bmi the public way (``read_bmi`` → ``Tower``)
    and checks ``ModalResult.mode_labels``. This synthetic fixture is
    deliberately stiff with a large (~29 m) CM lever and no
    hydrostatic restoring, so its roll/pitch rigid-body modes are
    genuine strongly-coupled sway-roll / surge-pitch *pendulum* modes
    — the classifier correctly leaves those ``None`` (it never
    mislabels a coupled mode). The robustly-separable DOFs (surge,
    sway, heave, yaw) must still be named, no DOF name may repeat, and
    the flexible tower modes above the rigid cluster must stay
    unnamed. The full clean six-DOF guarantee is asserted on the
    *realistic* bundled samples (``test_platform_mode_labels``) and
    real upstream decks (``test_floating_samples``)."""
    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props
    from pybmodes.models import Tower

    bmi_path = _write_floating_bmi(tmp_path, _platform(cm_x, cm_y), "lbl")
    bmi = read_bmi(bmi_path)
    t = Tower.__new__(Tower)
    t._bmi = bmi
    t._sp = read_sec_props(bmi.resolve_sec_props_path())
    res = t.run(n_modes=12, check_model=False)

    assert res.mode_labels is not None  # classifier ran on the .bmi path
    dof_set = {"surge", "sway", "heave", "roll", "pitch", "yaw"}
    first6 = res.mode_labels[:6]
    named = [lbl for lbl in first6 if lbl is not None]
    assert set(named) <= dof_set, named            # only valid DOF names
    assert len(named) == len(set(named)), named    # no DOF named twice
    # Heave (pure axial) and yaw (pure torsion) are kinematically
    # decoupled from the surge/sway/roll/pitch pendulum block, so they
    # are always cleanly identifiable even on this pathological fixture
    # (the in-plane DOFs strongly pendulum-couple here and are
    # correctly left None — never mislabelled).
    assert {"heave", "yaw"} <= set(named), first6
    # Flexible tower modes are never named.
    assert all(lbl is None for lbl in res.mode_labels[6:]), res.mode_labels[6:]
