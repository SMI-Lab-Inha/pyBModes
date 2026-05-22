"""Analytic round-trip tests for the 6×6 cross-sectional decoupling
(issue #50). Self-contained — every reference is constructed in the
test, no external data.

The contract: a 6×6 built *from* known decoupled properties at a
known elastic-/shear-centre offset and principal-axis rotation must
decouple back to exactly those properties. This pins the offset
congruence transform, the tension/shear-centre solve, and the
principal-axis eigen-decomposition without trusting any reference
code.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.io._precomp.decouple import (
    _offset_transform,
    decouple_inertia,
    decouple_stiffness,
)

# Section-frame indices (WindIO/BeamDyn order).
_AX, _B1, _B2, _TOR = 2, 3, 4, 5


def _diag_K(EA, EIf, EIe, GJ, shear=1.0e9):
    """A perfectly decoupled 6×6 at the reference point: diagonal,
    principal axes aligned with the section frame."""
    return np.diag([shear, shear, EA, EIf, EIe, GJ]).astype(float)


def test_already_decoupled_passes_through() -> None:
    K = _diag_K(2.0e9, 3.0e8, 5.0e8, 1.0e8)
    d = decouple_stiffness(K)
    assert pytest.approx(2.0e9) == d.EA
    assert {round(d.EI_flap), round(d.EI_edge)} == {300000000, 500000000}
    assert pytest.approx(1.0e8, rel=1e-6) == d.GJ
    assert d.x_tc == pytest.approx(0.0, abs=1e-9)
    assert d.y_tc == pytest.approx(0.0, abs=1e-9)


def test_recovers_known_tension_centre_offset() -> None:
    """Build a diagonal section, move the reference point by a known
    offset via the congruence transform, and check the decoupling
    recovers that offset and the original EA / principal EI."""
    EA, EIf, EIe, GJ = 2.0e9, 3.0e8, 5.0e8, 1.0e8
    K0 = _diag_K(EA, EIf, EIe, GJ)
    d1, d2 = 0.37, -0.21                      # metres
    K = _offset_transform(d1, d2).T @ K0 @ _offset_transform(d1, d2)
    # Off-diagonal axial↔bending coupling must now be non-zero...
    assert abs(K[_AX, _B1]) > 0.0 and abs(K[_AX, _B2]) > 0.0

    dec = decouple_stiffness(K)
    assert pytest.approx(EA, rel=1e-9) == dec.EA
    # Re-expressing about a reference displaced by (d1,d2) ⇒ the
    # decoupling translation is −(d1,d2) (documented sign convention).
    assert dec.x_tc == pytest.approx(-d1, rel=1e-6)
    assert dec.y_tc == pytest.approx(-d2, rel=1e-6)
    got = sorted([dec.EI_flap, dec.EI_edge])
    np.testing.assert_allclose(got, [EIf, EIe], rtol=1e-6)
    # Defining property: applying the reported offset decouples it.
    T = _offset_transform(dec.x_tc, dec.y_tc)
    Kd = T.T @ K @ T
    assert abs(Kd[_AX, _B1]) < 1e-3 * Kd[_AX, _AX]
    assert abs(Kd[_AX, _B2]) < 1e-3 * Kd[_AX, _AX]


def test_recovers_principal_axis_rotation() -> None:
    """A section rotated by θ relative to the reference axes has a
    non-zero K45; decoupling must recover the principal EI and the
    rotation angle."""
    EIf, EIe = 3.0e8, 9.0e8
    theta = np.deg2rad(20.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    B_ref = R @ np.diag([EIf, EIe]) @ R.T          # bending in ref axes
    K = _diag_K(2.0e9, 0.0, 0.0, 1.0e8)
    K[np.ix_((_B1, _B2), (_B1, _B2))] = B_ref
    assert abs(K[_B1, _B2]) > 0.0                  # coupled in ref frame

    dec = decouple_stiffness(K)
    np.testing.assert_allclose(
        sorted([dec.EI_flap, dec.EI_edge]), [EIf, EIe], rtol=1e-9)
    # Convention-robust angle check: rotating the principal diagonal
    # back by the reported angle must reproduce the reference-frame B.
    a = dec.principal_angle
    Rr = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    lam = np.linalg.eigvalsh(B_ref)            # ascending principal EI
    B_rebuilt = Rr @ np.diag(lam) @ Rr.T
    np.testing.assert_allclose(B_rebuilt, B_ref, rtol=1e-6, atol=1.0)


def test_shear_centre_and_GJ() -> None:
    """Offsetting a diagonal section couples shear↔torsion; the
    decoupling must recover GJ at the shear centre."""
    GJ = 4.2e7
    K0 = _diag_K(2.0e9, 3.0e8, 5.0e8, GJ, shear=8.0e8)
    s1, s2 = 0.15, 0.05
    K = _offset_transform(s1, s2).T @ K0 @ _offset_transform(s1, s2)
    dec = decouple_stiffness(K)
    assert pytest.approx(GJ, rel=1e-6) == dec.GJ
    assert dec.x_sc == pytest.approx(-s1, rel=1e-5)
    assert dec.y_sc == pytest.approx(-s2, rel=1e-5)


def test_decoupled_differs_from_raw_diagonal_when_offset() -> None:
    """The whole point of #50: with a real offset the raw diagonal
    K44/K55 is *not* the principal EI — they must differ materially."""
    K0 = _diag_K(2.0e9, 3.0e8, 5.0e8, 1.0e8)
    K = _offset_transform(0.5, 0.3).T @ K0 @ _offset_transform(0.5, 0.3)
    dec = decouple_stiffness(K)
    raw_flap, raw_edge = K[_B1, _B1], K[_B2, _B2]
    # Raw diagonal is inflated by the parallel-axis term; decoupled
    # recovers the true (smaller) principal values.
    assert dec.EI_flap < raw_flap and dec.EI_edge < raw_edge
    assert sorted([round(dec.EI_flap), round(dec.EI_edge)]) == [
        300000000, 500000000]


def test_inertia_decoupling_recovers_mass_centre() -> None:
    mass, i_f, i_e = 120.0, 5.0, 7.0
    M0 = np.diag([mass, mass, mass, i_f, i_e, i_f + i_e]).astype(float)
    d1, d2 = 0.22, -0.1
    M = _offset_transform(d1, d2).T @ M0 @ _offset_transform(d1, d2)
    di = decouple_inertia(M)
    assert di.mass == pytest.approx(mass, rel=1e-9)
    assert di.x_cg == pytest.approx(-d1, rel=1e-6)
    assert di.y_cg == pytest.approx(-d2, rel=1e-6)
    np.testing.assert_allclose(
        sorted([di.i_flap, di.i_edge]), [i_f, i_e], rtol=1e-6)


def test_rejects_bad_shapes_and_nonpositive() -> None:
    with pytest.raises(ValueError, match="6×6"):
        decouple_stiffness(np.eye(4))
    with pytest.raises(ValueError, match="axial"):
        decouple_stiffness(_diag_K(0.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="6×6"):
        decouple_inertia(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="mass"):
        decouple_inertia(np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
