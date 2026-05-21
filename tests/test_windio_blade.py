"""WindIO composite-blade cross-section reduction (1.4.0, issue #35).

`RotatingBlade.from_windio` will reduce a WindIO blade composite layup
to the 1-D distributed beam properties the FEM consumes, via a
PreComp-class thin-wall classical-lamination-theory (CLT) shear-flow
reduction (Bir 2006, NREL/TP-500-38929).

Validation ladder (default-suite rungs are self-contained;
integration rungs anchor on the companion RWT BeamDyn 6×6 decks):

* CLT laminate primitives vs closed-form (Jones, *Mechanics of
  Composite Materials*): reduced stiffness, ply transformation, ABD
  assembly, membrane condensation. **(this file, below)**
* section geometry: airfoil arc parameterisation, spanwise
  blend, region arc-band resolution across both WindIO dialects.
* single- then multi-cell reduction vs closed-form tube /
  box (exact) and the IEA-3.4/10/15/22 + NREL-5MW BeamDyn 6×6
  diagonals (integration; PreComp-class tolerances).
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

from pybmodes.io._precomp.arc_resolver import (
    resolve_blade_structure,
)
from pybmodes.io._precomp.laminate import (
    PlyElastic,
    abd_matrices,
    material_plane_stress,
    membrane_condensed,
    reduced_stiffness,
    transform_reduced_stiffness,
)
from pybmodes.io._precomp.profile import Profile
from pybmodes.io._precomp.reduction import (
    LayerStation,
    WebStation,
    reduce_section,
)


def _circle_coords(n: int = 241, diameter: float = 1.0):
    """WindIO 'circular' airfoil: TE→suction→LE→pressure→TE, the exact
    convention IEA-15's `circular` airfoil uses (x = ½(1+cos α),
    y = ½ sin α scaled by diameter). Odd `n` puts the LE exactly on a
    vertex (α = π) so a symmetric shape's landmarks are exact."""
    a = np.linspace(0.0, 2.0 * np.pi, n)
    x = 0.5 * (1.0 + np.cos(a))
    y = 0.5 * np.sin(a) * diameter
    return x, y


def _ellipse_coords(n: int = 241, tc: float = 0.4):
    a = np.linspace(0.0, 2.0 * np.pi, n)
    return 0.5 * (1.0 + np.cos(a)), 0.5 * np.sin(a) * tc


def _rect_coords(b: float, h: float):
    """Closed rectangle loop, width b (x) × height h (y), centred on y."""
    x = np.array([b, b, 0.0, 0.0, b])
    y = np.array([-h / 2, h / 2, h / 2, -h / 2, -h / 2])
    return x, y


def _iso_ply(E=70.0e9, nu=0.33, rho=2700.0):
    from pybmodes.io._precomp.laminate import PlyElastic
    return PlyElastic(E1=E, E2=E, G12=E / (2.0 * (1.0 + nu)), nu12=nu,
                      rho=rho)

# ---------------------------------------------------------------------------
# CLT laminate primitives vs closed form (default; no external data)
# ---------------------------------------------------------------------------


def test_material_plane_stress_isotropic() -> None:
    """Isotropic: E1 == E2 == E, G defaults to E/2(1+nu), rho scalar."""
    p = material_plane_stress({"name": "steel", "E": 2.0e11, "nu": 0.3,
                               "rho": 7800.0})
    assert p.E1 == p.E2 == 2.0e11
    assert p.nu12 == 0.3
    assert p.G12 == pytest.approx(2.0e11 / (2.0 * 1.3))
    assert p.rho == 7800.0
    # Explicit G overrides the isotropic default.
    p2 = material_plane_stress({"name": "s", "E": 2.0e11, "nu": 0.3,
                                "G": 7.7e10, "rho": 7800.0})
    assert p2.G12 == 7.7e10


def test_material_plane_stress_orthotropic() -> None:
    """Orthotropic vectors → in-plane (E1,E2,G12,nu12) subset; rho scalar."""
    p = material_plane_stress({
        "name": "CarbonUD",
        "E": [1.14e11, 8.0e9, 8.0e9],
        "G": [5.0e9, 5.0e9, 3.0e9],
        "nu": [0.3, 0.3, 0.4],
        "rho": 1600.0,
    })
    assert p.E1 == 1.14e11 and p.E2 == 8.0e9
    assert p.G12 == 5.0e9 and p.nu12 == 0.3 and p.rho == 1600.0


def test_material_plane_stress_density_synonym() -> None:
    p = material_plane_stress({"name": "x", "E": 1.0e10, "density": 1200.0})
    assert p.rho == 1200.0


def test_reduced_stiffness_isotropic_closed_form() -> None:
    """Q reduces to the isotropic E/(1-ν²) form (Jones eq. 2.66)."""
    E, nu = 70.0e9, 0.33
    p = PlyElastic(E1=E, E2=E, G12=E / (2.0 * (1.0 + nu)), nu12=nu, rho=2700.0)
    Q = reduced_stiffness(p)
    assert Q[0, 0] == pytest.approx(E / (1.0 - nu**2))
    assert Q[1, 1] == pytest.approx(E / (1.0 - nu**2))
    assert Q[0, 1] == pytest.approx(nu * E / (1.0 - nu**2))
    assert Q[2, 2] == pytest.approx(E / (2.0 * (1.0 + nu)))
    assert Q[0, 2] == 0.0 and Q[1, 2] == 0.0


def test_transform_zero_angle_is_identity() -> None:
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    np.testing.assert_allclose(transform_reduced_stiffness(Q, 0.0), Q)


def test_transform_ninety_degrees_swaps_axes() -> None:
    """A 90° ply swaps the 11/22 stiffnesses; no shear coupling at 90°."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    Qb = transform_reduced_stiffness(Q, np.pi / 2.0)
    assert Qb[0, 0] == pytest.approx(Q[1, 1])
    assert Qb[1, 1] == pytest.approx(Q[0, 0])
    assert Qb[2, 2] == pytest.approx(Q[2, 2])
    assert Qb[0, 2] == pytest.approx(0.0, abs=1e-3)
    assert Qb[1, 2] == pytest.approx(0.0, abs=1e-3)


def test_transform_pm45_shear_coupling_cancels_in_balanced_pair() -> None:
    """+θ and −θ plies have equal/opposite Qbar16 (Jones §2.6) — the
    basis for a *balanced* laminate having A16 = 0."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    qp = transform_reduced_stiffness(Q, np.deg2rad(45.0))
    qm = transform_reduced_stiffness(Q, np.deg2rad(-45.0))
    assert qp[0, 2] == pytest.approx(-qm[0, 2])
    assert qp[0, 0] == pytest.approx(qm[0, 0])  # even in θ


def test_abd_single_ply_closed_form() -> None:
    """Single centred ply: B = 0, A = Qbar·t, D = Qbar·t³/12."""
    p = PlyElastic(E1=70.0e9, E2=70.0e9, G12=70.0e9 / 2.6, nu12=0.3,
                   rho=2700.0)
    Q = reduced_stiffness(p)
    t = 0.004
    A, B, D = abd_matrices([(Q, t)])
    np.testing.assert_allclose(A, Q * t, rtol=1e-12)
    np.testing.assert_allclose(B, np.zeros((3, 3)), atol=1e-9)
    np.testing.assert_allclose(D, Q * t**3 / 12.0, rtol=1e-12)


def test_abd_symmetric_stack_has_zero_B() -> None:
    """A mid-plane-symmetric [0/90/0] stack → B ≈ 0 (extension-bending
    decoupled, Jones §4.3)."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    _, B, _ = abd_matrices([(q0, 0.001), (q90, 0.002), (q0, 0.001)])
    assert np.max(np.abs(B)) < 1e-6 * 1.4e11 * 0.001


def test_abd_balanced_symmetric_has_zero_A16() -> None:
    """A balanced symmetric [+45/−45]_s laminate → A16 = A26 = 0."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    qp = transform_reduced_stiffness(Q, np.deg2rad(45.0))
    qm = transform_reduced_stiffness(Q, np.deg2rad(-45.0))
    t = 0.001
    A, B, _ = abd_matrices([(qp, t), (qm, t), (qm, t), (qp, t)])
    assert abs(A[0, 2]) < 1e-6 * A[0, 0]
    assert abs(A[1, 2]) < 1e-6 * A[1, 1]
    assert np.max(np.abs(B)) < 1e-6 * A[0, 0]


def test_membrane_condensed_symmetric_returns_A() -> None:
    """B = 0 ⇒ Atilde = A exactly (no wall-bending knock-down)."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    A, B, D = abd_matrices([(q0, 0.001), (q90, 0.002), (q0, 0.001)])
    np.testing.assert_array_equal(membrane_condensed(A, B, D), A)


def test_membrane_condensed_unsymmetric_knocks_A_down() -> None:
    """An unsymmetric [0/90] laminate (B ≠ 0): Atilde = A − B D⁻¹ B,
    symmetric and strictly softer than A in the axial term."""
    p = PlyElastic(E1=1.4e11, E2=9.0e9, G12=5.0e9, nu12=0.3, rho=1600.0)
    Q = reduced_stiffness(p)
    q0 = transform_reduced_stiffness(Q, 0.0)
    q90 = transform_reduced_stiffness(Q, np.pi / 2.0)
    A, B, D = abd_matrices([(q0, 0.002), (q90, 0.002)])
    assert np.max(np.abs(B)) > 0.0
    At = membrane_condensed(A, B, D)
    np.testing.assert_allclose(At, A - B @ np.linalg.solve(D, B), rtol=1e-12)
    np.testing.assert_allclose(At, At.T, rtol=1e-10)
    assert At[0, 0] < A[0, 0]


# ---------------------------------------------------------------------------
# Airfoil profile geometry / nd_arc spine (default; no external data)
# ---------------------------------------------------------------------------


def test_profile_circle_normalisation_and_arc_spine() -> None:
    """A unit circle: chord-normalised, the nd_arc spine spans [0,1]
    with the LE at the half-perimeter, t/c = 1."""
    p = Profile.from_windio_coords(*_circle_coords())
    assert p.s[0] == pytest.approx(0.0)
    assert p.s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(p.s) > 0.0)            # strictly monotone spine
    assert p.s_le == pytest.approx(0.5, abs=2e-3)
    assert p.tc == pytest.approx(1.0, abs=5e-3)
    assert p.xc.min() == pytest.approx(0.0, abs=1e-9)
    assert p.xc.max() == pytest.approx(1.0, abs=1e-9)


def test_profile_arc_to_xy_landmarks() -> None:
    """s=0 and s=1 are the TE; s=s_le is the LE (x≈0)."""
    p = Profile.from_windio_coords(*_circle_coords())
    x0, y0 = p.arc_to_xy(0.0)
    x1, y1 = p.arc_to_xy(1.0)
    xle, yle = p.arc_to_xy(p.s_le)
    assert (x0, x1) == (pytest.approx(1.0, abs=1e-6),
                        pytest.approx(1.0, abs=1e-6))
    assert y0 == pytest.approx(0.0, abs=1e-6)
    assert xle == pytest.approx(0.0, abs=2e-3)
    assert yle == pytest.approx(0.0, abs=2e-3)


def test_profile_arc_of_chord_anchors() -> None:
    """fixed:LE → x/c=0, fixed:TE → x/c=1 map to the right nd_arc on
    each surface (the resolver's anchor primitives)."""
    p = Profile.from_windio_coords(*_circle_coords())
    assert p.arc_of_chord(1.0, side="suction") == pytest.approx(0.0, abs=2e-3)
    assert p.arc_of_chord(0.0, side="suction") == pytest.approx(
        p.s_le, abs=2e-3)
    assert p.arc_of_chord(0.0, side="pressure") == pytest.approx(
        p.s_le, abs=2e-3)
    assert p.arc_of_chord(1.0, side="pressure") == pytest.approx(
        1.0, abs=2e-3)
    with pytest.raises(ValueError, match="suction.*pressure"):
        p.arc_of_chord(0.5, side="sideways")


def test_profile_blend_weight_semantics() -> None:
    """weight 0 → self, 1 → other, ½ → midway t/c (spanwise airfoil
    interpolation)."""
    circ = Profile.from_windio_coords(*_circle_coords())          # t/c 1.0
    ell = Profile.from_windio_coords(*_ellipse_coords(tc=0.4))     # t/c 0.4
    assert circ.blend(ell, 0.0).tc == pytest.approx(circ.tc, abs=5e-3)
    assert circ.blend(ell, 1.0).tc == pytest.approx(ell.tc, abs=5e-3)
    mid = circ.blend(ell, 0.5)
    assert mid.tc == pytest.approx(0.5 * (circ.tc + ell.tc), abs=5e-3)
    # The blended profile keeps a valid, monotone nd_arc spine.
    assert mid.s[0] == pytest.approx(0.0) and mid.s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(mid.s) > 0.0)


def test_profile_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="zero chord|zero perimeter"):
        Profile.from_windio_coords(np.zeros(10), np.zeros(10))


# ---------------------------------------------------------------------------
# WindIO web/layer nd_arc resolver — dual-dialect (default; no data)
# ---------------------------------------------------------------------------

# Numerically identical structure in the two WindIO dialects: the older
# explicit-curve form (IEA-3.4/10/22) and the modern anchor-registry
# indirection (IEA-15 WT_Ontology, every WISDEM FOWT). They must resolve
# to byte-identical bands.
_OLDER_STRUCT = {
    "webs": [
        {"name": "w0",
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.40, 0.42]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [0.70, 0.68]}},
    ],
    "layers": [
        {"name": "shell", "material": "glass",
         "thickness": {"grid": [0.0, 1.0], "values": [0.05, 0.02]},
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [1.0, 1.0]}},
        {"name": "spar", "material": "carbon",
         "thickness": {"grid": [0.0, 1.0], "values": [0.10, 0.03]},
         "fiber_orientation": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.45, 0.46]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [0.55, 0.54]}},
    ],
}

_MODERN_STRUCT = {
    "anchors": [
        {"name": "w0",
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.40, 0.42]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [0.70, 0.68]}},
        {"name": "shell",
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [1.0, 1.0]}},
        {"name": "spar",
         "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.45, 0.46]},
         "end_nd_arc": {"grid": [0.0, 1.0], "values": [0.55, 0.54]},
         "plane_intersection": {"side": "both"}},   # recipe present but unused
    ],
    "webs": [
        {"name": "w0",
         "start_nd_arc": {"anchor": {"name": "w0", "handle": "start_nd_arc"}},
         "end_nd_arc": {"anchor": {"name": "w0", "handle": "end_nd_arc"}}},
    ],
    "layers": [
        {"name": "shell", "material": "glass",
         "thickness": {"grid": [0.0, 1.0], "values": [0.05, 0.02]},
         "start_nd_arc": {"anchor": {"name": "shell",
                                     "handle": "start_nd_arc"}},
         "end_nd_arc": {"anchor": {"name": "shell",
                                   "handle": "end_nd_arc"}}},
        {"name": "spar", "material": "carbon",
         "thickness": {"grid": [0.0, 1.0], "values": [0.10, 0.03]},
         "fiber_orientation": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
         "start_nd_arc": {"anchor": {"name": "spar",
                                     "handle": "start_nd_arc"}},
         "end_nd_arc": {"anchor": {"name": "spar",
                                   "handle": "end_nd_arc"}}},
    ],
}


def test_resolver_older_explicit_interpolates_onto_span() -> None:
    s = np.linspace(0.0, 1.0, 11)
    r = resolve_blade_structure(_OLDER_STRUCT, s)
    assert [w.name for w in r.webs] == ["w0"]
    assert [ly.name for ly in r.layers] == ["shell", "spar"]
    np.testing.assert_allclose(r.webs[0].start_nd, 0.40 + 0.02 * s)
    np.testing.assert_allclose(r.webs[0].end_nd, 0.70 - 0.02 * s)
    spar = r.layers[1]
    np.testing.assert_allclose(spar.start_nd, 0.45 + 0.01 * s)
    np.testing.assert_allclose(spar.end_nd, 0.55 - 0.01 * s)
    np.testing.assert_allclose(spar.thickness, 0.10 - 0.07 * s)
    assert spar.web is None and spar.material == "carbon"


def test_resolver_modern_anchor_dereference() -> None:
    s = np.linspace(0.0, 1.0, 7)
    r = resolve_blade_structure(_MODERN_STRUCT, s)
    np.testing.assert_allclose(r.layers[1].start_nd, 0.45 + 0.01 * s)
    np.testing.assert_allclose(r.webs[0].end_nd, 0.70 - 0.02 * s)


def test_resolver_dialect_equivalence() -> None:
    """The older explicit form and the modern anchor-registry
    indirection resolve to identical bands."""
    s = np.linspace(0.0, 1.0, 23)
    a = resolve_blade_structure(_OLDER_STRUCT, s)
    b = resolve_blade_structure(_MODERN_STRUCT, s)
    assert len(a.webs) == len(b.webs) and len(a.layers) == len(b.layers)
    for wa, wb in zip(a.webs, b.webs):
        np.testing.assert_allclose(wa.start_nd, wb.start_nd)
        np.testing.assert_allclose(wa.end_nd, wb.end_nd)
    for la, lb in zip(a.layers, b.layers):
        assert (la.name, la.material, la.web) == (lb.name, lb.material,
                                                  lb.web)
        np.testing.assert_allclose(la.thickness, lb.thickness)
        np.testing.assert_allclose(la.fiber_orientation, lb.fiber_orientation)
        np.testing.assert_allclose(la.start_nd, lb.start_nd)
        np.testing.assert_allclose(la.end_nd, lb.end_nd)


def test_resolver_on_web_layer_via_key_and_via_anchor() -> None:
    """An on-web layer is flagged via the `web:` key, or via an arc
    anchor that points at a web's name; its shell band is zeroed."""
    s = np.linspace(0.0, 1.0, 5)
    struct = {
        "webs": [{"name": "w0",
                  "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.4, 0.4]},
                  "end_nd_arc": {"grid": [0.0, 1.0], "values": [0.6, 0.6]}}],
        "layers": [
            {"name": "core", "material": "balsa", "web": "w0",
             "thickness": {"grid": [0.0, 1.0], "values": [0.02, 0.02]},
             "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
             "end_nd_arc": {"grid": [0.0, 1.0], "values": [1.0, 1.0]}},
            {"name": "skin", "material": "glass",
             "thickness": {"grid": [0.0, 1.0], "values": [0.001, 0.001]},
             "start_nd_arc": {"anchor": {"name": "w0",
                                         "handle": "start_nd_arc"}},
             "end_nd_arc": {"anchor": {"name": "w0",
                                       "handle": "end_nd_arc"}}},
        ],
    }
    r = resolve_blade_structure(struct, s)
    assert r.layers[0].web == "w0"
    np.testing.assert_array_equal(r.layers[0].start_nd, np.zeros(5))
    assert r.layers[1].web == "w0"          # detected via web-named anchor


def test_resolver_parametric_only_anchor_raises() -> None:
    """An anchor with only a parametric recipe (no resolved
    grid/values) raises an actionable error, not a silent guess."""
    struct = {
        "webs": [],
        "anchors": [{"name": "sc",
                     "plane_intersection": {"side": "suction"}}],
        "layers": [
            {"name": "spar", "material": "carbon",
             "thickness": {"grid": [0.0, 1.0], "values": [0.1, 0.1]},
             "start_nd_arc": {"anchor": {"name": "sc",
                                         "handle": "start_nd_arc"}},
             "end_nd_arc": {"anchor": {"name": "sc",
                                       "handle": "end_nd_arc"}}},
        ],
    }
    with pytest.raises(NotImplementedError, match="WISDEM-resolved"):
        resolve_blade_structure(struct, np.linspace(0.0, 1.0, 4))


def test_resolver_missing_anchor_raises() -> None:
    struct = {
        "webs": [],
        "anchors": [{"name": "real",
                     "start_nd_arc": {"grid": [0.0, 1.0], "values": [0.0,
                                                                     0.0]},
                     "end_nd_arc": {"grid": [0.0, 1.0], "values": [1.0,
                                                                   1.0]}}],
        "layers": [
            {"name": "x", "material": "g",
             "thickness": {"grid": [0.0, 1.0], "values": [0.01, 0.01]},
             "start_nd_arc": {"anchor": {"name": "ghost",
                                         "handle": "start_nd_arc"}},
             "end_nd_arc": {"anchor": {"name": "ghost",
                                       "handle": "end_nd_arc"}}},
        ],
    }
    with pytest.raises(KeyError, match="registry"):
        resolve_blade_structure(struct, np.linspace(0.0, 1.0, 4))


def test_resolver_zero_outside_defined_grid() -> None:
    """A region defined only over part of span (WISDEM
    extrapolate=False + nan_to_num) is zero elsewhere."""
    s = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    struct = {
        "webs": [{"name": "w",
                  "start_nd_arc": {"grid": [0.1, 0.9], "values": [0.4, 0.4]},
                  "end_nd_arc": {"grid": [0.1, 0.9], "values": [0.6, 0.6]}}],
        "layers": [],
    }
    r = resolve_blade_structure(struct, s)
    assert r.webs[0].start_nd[0] == 0.0          # below grid → 0
    assert r.webs[0].start_nd[-1] == 0.0         # above grid → 0
    assert r.webs[0].start_nd[2] == pytest.approx(0.4)   # interior


# ---------------------------------------------------------------------------
# Single-cell thin-wall reduction vs closed form (default; no data)
# ---------------------------------------------------------------------------


def test_reduce_isotropic_tube_matches_thin_ring_closed_form() -> None:
    """A single-ply isotropic circular tube reduces to the exact
    thin-ring formulae (material on the outer perimeter of radius R):
    EA=E·2πR·t, EI_flap=EI_edge=E·πR³·t, GJ=G·2πR³·t, m=ρ·2πR·t,
    centred (TC≡CG≡SC), EI_flap==EI_edge."""
    chord, t = 2.0, 0.01
    R = chord / 2.0
    p = Profile.from_windio_coords(*_circle_coords(n=401))
    ply = _iso_ply()
    G = ply.G12
    res = reduce_section(p, chord, 0.5,
                         [LayerStation(ply, t, 0.0, 0.0, 1.0)],
                         n_perim=600)

    EA = ply.E1 * (2.0 * np.pi * R) * t
    EI = ply.E1 * np.pi * R**3 * t
    GJ = G * 2.0 * np.pi * R**3 * t
    m = ply.rho * (2.0 * np.pi * R) * t
    assert res.EA == pytest.approx(EA, rel=5e-3)
    assert res.EI_flap == pytest.approx(EI, rel=1e-2)
    assert res.EI_edge == pytest.approx(EI, rel=1e-2)
    assert res.EI_flap == pytest.approx(res.EI_edge, rel=1e-3)  # symmetry
    assert res.GJ == pytest.approx(GJ, rel=1e-2)
    assert res.mass == pytest.approx(m, rel=5e-3)
    assert res.x_tc == pytest.approx(0.0, abs=1e-6 * chord)
    assert res.x_cg == pytest.approx(0.0, abs=1e-6 * chord)
    assert res.x_sc == pytest.approx(res.x_tc)            # SC≈TC


def test_reduce_isotropic_box_matches_thin_wall_closed_form() -> None:
    """A single-ply isotropic rectangular box vs the exact thin-wall
    box formulae (translation-invariant: EA, EI about the centroid,
    GJ, mass)."""
    b, h, t = 1.2, 0.4, 0.005
    p = Profile.from_windio_coords(*_rect_coords(b, h))
    ply = _iso_ply()
    E, G, rho = ply.E1, ply.G12, ply.rho
    res = reduce_section(p, b, 0.5,
                         [LayerStation(ply, t, 0.0, 0.0, 1.0)],
                         n_perim=2000)

    P = 2.0 * (b + h)
    assert res.EA == pytest.approx(E * P * t, rel=1e-2)
    assert res.mass == pytest.approx(rho * P * t, rel=1e-2)
    # Flap = bending about the horizontal centroidal axis (Y deflection).
    EI_flap = E * t * (b * h**2 / 2.0 + h**3 / 6.0)
    EI_edge = E * t * (h * b**2 / 2.0 + b**3 / 6.0)
    assert res.EI_flap == pytest.approx(EI_flap, rel=2e-2)
    assert res.EI_edge == pytest.approx(EI_edge, rel=2e-2)
    GJ = 2.0 * G * t * b**2 * h**2 / (b + h)
    assert res.GJ == pytest.approx(GJ, rel=2e-2)
    assert res.x_tc == pytest.approx(res.x_cg, abs=1e-9)   # uniform wall


def test_reduce_thickness_scales_axial_and_mass_linearly() -> None:
    """Doubling a single isotropic ply's thickness doubles EA / mass
    (membrane), ~doubles GJ; EI grows faster (wall-offset) — sanity
    that the segment assembly is linear in t."""
    p = Profile.from_windio_coords(*_circle_coords(n=201))
    ply = _iso_ply()
    a = reduce_section(p, 1.0, 0.5, [LayerStation(ply, 0.01, 0.0, 0.0, 1.0)])
    b = reduce_section(p, 1.0, 0.5, [LayerStation(ply, 0.02, 0.0, 0.0, 1.0)])
    assert b.EA == pytest.approx(2.0 * a.EA, rel=1e-9)
    assert b.mass == pytest.approx(2.0 * a.mass, rel=1e-9)
    assert b.GJ == pytest.approx(2.0 * a.GJ, rel=1e-9)


def test_reduce_two_layers_stack_through_wall() -> None:
    """Two full-perimeter isotropic plies of equal ν stack linearly:
    EA grows by exactly E₂·t₂·perimeter (the laminate ``A11−A12²/A22``
    reduction is additive only when ν matches — differing ν couples,
    which is correct physics, so the linearity check uses equal ν)."""
    from pybmodes.io._precomp.laminate import PlyElastic
    p = Profile.from_windio_coords(*_circle_coords(n=201))
    p1 = _iso_ply(E=70e9, nu=0.33, rho=2700.0)
    p2 = PlyElastic(E1=140e9, E2=140e9, G12=140e9 / (2.0 * 1.33),
                    nu12=0.33, rho=1600.0)
    one = reduce_section(p, 1.0, 0.5,
                         [LayerStation(p1, 0.01, 0.0, 0.0, 1.0)])
    two = reduce_section(p, 1.0, 0.5, [
        LayerStation(p1, 0.01, 0.0, 0.0, 1.0),
        LayerStation(p2, 0.02, 0.0, 0.0, 1.0),
    ])
    perim = np.pi * 1.0          # circle diameter 1 → perimeter πd
    assert two.EA - one.EA == pytest.approx(
        p2.E1 * 0.02 * perim, rel=3e-3)
    assert two.mass > one.mass


def test_reduce_uncovered_perimeter_raises() -> None:
    """No load-bearing material anywhere on the perimeter is rejected
    (an empty shell stack → zero EA / mass)."""
    p = Profile.from_windio_coords(*_circle_coords(n=101))
    with pytest.raises(ValueError, match="no load-bearing|zero"):
        reduce_section(p, 1.0, 0.5, [], n_perim=50)


# ---------------------------------------------------------------------------
# Webs + multi-cell Bredt–Batho torsion (default; no external data)
# ---------------------------------------------------------------------------


def _tube(chord=2.0, t=0.01, n=401):
    p = Profile.from_windio_coords(*_circle_coords(n=n))
    return p, _iso_ply(), chord, t


def test_no_web_is_single_cell_regression() -> None:
    """A webless section is one cell and matches the single-cell thin-ring
    GJ (multi-cell must not regress the single-cell path)."""
    p, ply, chord, t = _tube()
    R = chord / 2.0
    res = reduce_section(p, chord, 0.5,
                         [LayerStation(ply, t, 0.0, 0.0, 1.0)],
                         n_perim=600)
    assert res.n_cells == 1
    assert res.GJ == pytest.approx(ply.G12 * 2.0 * np.pi * R**3 * t,
                                   rel=1e-2)


def test_symmetric_diametral_web_leaves_GJ_unchanged() -> None:
    """Exact closed-form anchor for the multi-cell machinery: a
    symmetric web on the vertical diameter of a circular tube carries
    *zero* torsional shear flow, so GJ stays at the webless
    2πR³·G·t — independent of the web — while n_cells becomes 2."""
    p, ply, chord, t = _tube()
    R = chord / 2.0
    shell = [LayerStation(ply, t, 0.0, 0.0, 1.0)]
    # Vertical diameter: suction foot at s=0.25 (top), pressure foot at
    # s=0.75 (bottom); s_LE = 0.5. Web is a shear-bearing iso ply.
    web = WebStation(0.25, 0.75, [(ply, 0.02, 0.0)])
    base = reduce_section(p, chord, 0.5, shell, n_perim=600)
    withw = reduce_section(p, chord, 0.5, shell, [web], n_perim=600)
    assert withw.n_cells == 2 and base.n_cells == 1
    assert withw.GJ == pytest.approx(base.GJ, rel=2e-2)
    assert withw.GJ == pytest.approx(
        ply.G12 * 2.0 * np.pi * R**3 * t, rel=2e-2)


def test_web_adds_mass_and_axial_exactly() -> None:
    """A web's straight wall adds its material to the section: mass by
    ρ·L_web·t_w and EA by E·t_w·L_web (L_web = 2R for a diametral
    web), on top of the shell."""
    p, ply, chord, t = _tube()
    R = chord / 2.0
    shell = [LayerStation(ply, t, 0.0, 0.0, 1.0)]
    tw = 0.03
    web = WebStation(0.25, 0.75, [(ply, tw, 0.0)])
    base = reduce_section(p, chord, 0.5, shell, n_perim=800)
    withw = reduce_section(p, chord, 0.5, shell, [web], n_perim=800,
                           n_web=400)
    L_web = 2.0 * R
    assert withw.mass - base.mass == pytest.approx(
        ply.rho * tw * L_web, rel=5e-3)
    assert withw.EA - base.EA == pytest.approx(
        ply.E1 * tw * L_web, rel=5e-3)


def test_offcentre_web_changes_GJ_two_cells() -> None:
    """An asymmetric (off-centre chord) web makes the two cells unequal
    so it *does* carry shear flow → GJ differs from the single-cell
    value, finite and positive (qualitative; the symmetric case pins
    correctness)."""
    p, ply, chord, t = _tube()
    shell = [LayerStation(ply, t, 0.0, 0.0, 1.0)]
    base = reduce_section(p, chord, 0.5, shell, n_perim=600)
    # Off-centre chord web (feet not antipodal): smaller TE cell.
    web = WebStation(0.15, 0.70, [(ply, 0.02, 0.0)])
    res = reduce_section(p, chord, 0.5, shell, [web], n_perim=600)
    assert res.n_cells == 2
    assert np.isfinite(res.GJ) and res.GJ > 0.0
    assert abs(res.GJ - base.GJ) / base.GJ > 1e-3


def test_two_webs_three_cells_smoke() -> None:
    """Two webs → a 3-cell chain; the coupled solve stays finite,
    positive, and the assembly reports n_cells = 3."""
    p, ply, chord, t = _tube()
    shell = [LayerStation(ply, t, 0.0, 0.0, 1.0)]
    webs = [
        WebStation(0.30, 0.70, [(ply, 0.02, 0.0)]),
        WebStation(0.18, 0.82, [(ply, 0.02, 0.0)]),
    ]
    res = reduce_section(p, chord, 0.5, shell, webs, n_perim=600)
    assert res.n_cells == 3
    assert np.isfinite(res.GJ) and res.GJ > 0.0


# ---------------------------------------------------------------------------
# windio_blade public glue — dual-dialect end-to-end (no data)
# ---------------------------------------------------------------------------


def _circ_xy_yaml(n: int = 61) -> tuple[str, str]:
    x, y = _circle_coords(n=n)
    fx = "[" + ", ".join(f"{v:.6f}" for v in x) + "]"
    fy = "[" + ", ".join(f"{v:.6f}" for v in y) + "]"
    return fx, fy


def _modern_blade_yaml() -> str:
    fx, fy = _circ_xy_yaml()
    return textwrap.dedent(f"""\
        components:
          blade:
            reference_axis:
              z: {{grid: [0.0, 1.0], values: [0.0, 50.0]}}
            outer_shape:
              chord: {{grid: [0.0, 1.0], values: [2.0, 2.0]}}
              twist: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
              section_offset_y: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
              airfoils:
                - {{name: circ, spanwise_position: 0.0}}
                - {{name: circ, spanwise_position: 1.0}}
            structure:
              anchors:
                - name: skin
                  start_nd_arc: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  end_nd_arc: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
              webs: []
              layers:
                - name: skin
                  material: glass
                  thickness: {{grid: [0.0, 1.0], values: [0.01, 0.01]}}
                  fiber_orientation: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  start_nd_arc: {{anchor: {{name: skin, handle: start_nd_arc}}}}
                  end_nd_arc: {{anchor: {{name: skin, handle: end_nd_arc}}}}
        airfoils:
          - name: circ
            coordinates: {{x: {fx}, y: {fy}}}
        materials:
          - {{name: glass, E: 7.0e10, nu: 0.3, G: 2.6923e10, rho: 2600.0}}
        """)


def _older_blade_yaml() -> str:
    fx, fy = _circ_xy_yaml()
    return textwrap.dedent(f"""\
        components:
          blade:
            outer_shape_bem:
              reference_axis:
                z: {{grid: [0.0, 1.0], values: [0.0, 50.0]}}
              airfoil_position: {{grid: [0.0, 1.0], labels: [circ, circ]}}
              chord: {{grid: [0.0, 1.0], values: [2.0, 2.0]}}
              twist: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
              pitch_axis: {{grid: [0.0, 1.0], values: [0.5, 0.5]}}
            internal_structure_2d_fem:
              webs: []
              layers:
                - name: skin
                  material: glass
                  thickness: {{grid: [0.0, 1.0], values: [0.01, 0.01]}}
                  fiber_orientation: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  start_nd_arc: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  end_nd_arc: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
        airfoils:
          - name: circ
            coordinates: {{x: {fx}, y: {fy}}}
        materials:
          - {{name: glass, E: 7.0e10, nu: 0.3, G: 2.6923e10, rho: 2600.0}}
        """)


def test_windio_blade_dialects_equivalent_and_physical(tmp_path) -> None:
    """End-to-end glue gate: a tube-section blade specified in the
    modern (anchor-registry) and older (explicit) WindIO dialects
    reduces to *identical* SectionProperties, and they are physically
    sane (tube ⇒ flap ≈ edge, centred, positive)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    pm = tmp_path / "modern_blade.yaml"
    po = tmp_path / "older_blade.yaml"
    pm.write_text(_modern_blade_yaml(), encoding="utf-8")
    po.write_text(_older_blade_yaml(), encoding="utf-8")

    bm = read_windio_blade(pm, n_span=5)
    bo = read_windio_blade(po, n_span=5)
    assert bm.flexible_length == pytest.approx(50.0)
    np.testing.assert_allclose(bm.ref_axis_xc, 0.5)
    np.testing.assert_allclose(bo.ref_axis_xc, 0.5)

    sp_m = windio_blade_section_props(bm, n_perim=240)
    sp_o = windio_blade_section_props(bo, n_perim=240)

    for col in ("mass_den", "flp_stff", "edge_stff", "tor_stff",
                "axial_stff", "cg_offst", "tc_offst", "str_tw"):
        np.testing.assert_allclose(getattr(sp_m, col), getattr(sp_o, col),
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"dialect mismatch in {col}")

    np.testing.assert_array_less(0.0, sp_m.mass_den)
    np.testing.assert_array_less(0.0, sp_m.flp_stff)
    np.testing.assert_array_less(0.0, sp_m.axial_stff)
    assert np.all(np.diff(sp_m.span_loc) > 0.0)
    # Circular section: flap ≈ edge, tension/area centred on the axis.
    np.testing.assert_allclose(sp_m.flp_stff, sp_m.edge_stff, rtol=2e-2)
    np.testing.assert_allclose(sp_m.tc_offst, 0.0, atol=1e-6 * 2.0)


# ---------------------------------------------------------------------------
# Published elastic properties preferred over PreComp (issue #48)
# ---------------------------------------------------------------------------

def _blade_yaml_with_elastic() -> str:
    """Older-dialect tube blade *plus* a modern ``elastic_properties``
    block nested inside ``internal_structure_2d_fem`` (the IEA-15
    nesting). Constant 2-point grids ⇒ the published values map onto
    any span grid exactly."""
    fx, fy = _circ_xy_yaml()
    return textwrap.dedent(f"""\
        components:
          blade:
            outer_shape_bem:
              reference_axis:
                z: {{grid: [0.0, 1.0], values: [0.0, 50.0]}}
              airfoil_position: {{grid: [0.0, 1.0], labels: [circ, circ]}}
              chord: {{grid: [0.0, 1.0], values: [2.0, 2.0]}}
              twist: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
              pitch_axis: {{grid: [0.0, 1.0], values: [0.5, 0.5]}}
            internal_structure_2d_fem:
              webs: []
              layers:
                - name: skin
                  material: glass
                  thickness: {{grid: [0.0, 1.0], values: [0.01, 0.01]}}
                  fiber_orientation: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  start_nd_arc: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  end_nd_arc: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
              elastic_properties:
                inertia_matrix:
                  grid: [0.0, 1.0]
                  mass: [100.0, 100.0]
                  i_flap: [5.0, 5.0]
                  i_edge: [7.0, 7.0]
                  cm_x: [0.1, 0.1]
                stiffness_matrix:
                  grid: [0.0, 1.0]
                  K33: [2.0e9, 2.0e9]
                  K44: [3.0e8, 3.0e8]
                  K55: [5.0e8, 5.0e8]
                  K66: [1.0e8, 1.0e8]
        airfoils:
          - name: circ
            coordinates: {{x: {fx}, y: {fy}}}
        materials:
          - {{name: glass, E: 7.0e10, nu: 0.3, G: 2.6923e10, rho: 2600.0}}
        """)


def _blade_yaml_with_mb_sixxsix() -> str:
    """Older-dialect tube blade plus the ``elastic_properties_mb.
    six_x_six`` 21-vector dialect (IEA-22 style), as a direct
    ``components.blade`` child."""
    fx, fy = _circ_xy_yaml()
    # 21-element upper-triangular flatten of a symmetric 6×6; only the
    # diagonal (idx 0,6,11,15,18,20) is read.
    k = [0.0] * 21
    k[0] = k[6] = 9.0e9          # K11, K22 (shear — unused)
    k[11], k[15], k[18], k[20] = 2.0e9, 3.0e8, 5.0e8, 1.0e8  # EA,EIf,EIe,GJ
    m = [0.0] * 21
    m[0] = m[6] = m[11] = 100.0  # mass/length
    m[15], m[18], m[20] = 5.0, 7.0, 12.0  # i_flap, i_edge, i_polar
    return textwrap.dedent(f"""\
        components:
          blade:
            elastic_properties_mb:
              six_x_six:
                stiff_matrix:
                  grid: [0.0, 1.0]
                  values: [{k}, {k}]
                inertia_matrix:
                  grid: [0.0, 1.0]
                  values: [{m}, {m}]
            outer_shape_bem:
              reference_axis:
                z: {{grid: [0.0, 1.0], values: [0.0, 50.0]}}
              airfoil_position: {{grid: [0.0, 1.0], labels: [circ, circ]}}
              chord: {{grid: [0.0, 1.0], values: [2.0, 2.0]}}
              twist: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
              pitch_axis: {{grid: [0.0, 1.0], values: [0.5, 0.5]}}
            internal_structure_2d_fem:
              webs: []
              layers:
                - name: skin
                  material: glass
                  thickness: {{grid: [0.0, 1.0], values: [0.01, 0.01]}}
                  fiber_orientation: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  start_nd_arc: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  end_nd_arc: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
        airfoils:
          - name: circ
            coordinates: {{x: {fx}, y: {fy}}}
        materials:
          - {{name: glass, E: 7.0e10, nu: 0.3, G: 2.6923e10, rho: 2600.0}}
        """)


def test_windio_blade_prefers_published_elastic_properties(tmp_path) -> None:
    """issue #48: when the WindIO file carries published distributed
    beam properties, ``elastic='auto'`` / ``'file'`` use them verbatim
    (so the model matches the reference exactly); ``'precomp'`` still
    reduces the layup; the published path differs from PreComp."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    p = tmp_path / "blade_ep.yaml"
    p.write_text(_blade_yaml_with_elastic(), encoding="utf-8")
    blade = read_windio_blade(p, n_span=6)
    assert blade.elastic is not None

    for mode in ("auto", "file"):
        sp = windio_blade_section_props(blade, n_perim=120, elastic=mode)
        np.testing.assert_allclose(sp.mass_den, 100.0)
        np.testing.assert_allclose(sp.flp_iner, 5.0)
        np.testing.assert_allclose(sp.edge_iner, 7.0)
        np.testing.assert_allclose(sp.axial_stff, 2.0e9)   # K33
        np.testing.assert_allclose(sp.flp_stff, 3.0e8)     # K44
        np.testing.assert_allclose(sp.edge_stff, 5.0e8)    # K55
        np.testing.assert_allclose(sp.tor_stff, 1.0e8)     # K66
        np.testing.assert_allclose(sp.cg_offst, 0.1)
        np.testing.assert_allclose(sp.sc_offst, 0.0)       # decoupled
        np.testing.assert_allclose(sp.tc_offst, 0.0)

    pc = windio_blade_section_props(blade, n_perim=120, elastic="precomp")
    # PreComp reduces the actual glass tube — nowhere near the
    # synthetic published numbers.
    assert not np.allclose(pc.axial_stff, 2.0e9)
    assert not np.allclose(pc.mass_den, 100.0)


def test_windio_blade_mb_sixxsix_diagonal(tmp_path) -> None:
    """The ``elastic_properties_mb.six_x_six`` 21-vector dialect:
    only the symmetric-6×6 diagonal is taken (EA=K33, EI_flap=K44,
    EI_edge=K55, GJ=K66; mass=M11, flap/edge inertia=M44/M55)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    p = tmp_path / "blade_mb.yaml"
    p.write_text(_blade_yaml_with_mb_sixxsix(), encoding="utf-8")
    blade = read_windio_blade(p, n_span=5)
    assert blade.elastic is not None
    sp = windio_blade_section_props(blade, elastic="file")
    np.testing.assert_allclose(sp.mass_den, 100.0)
    np.testing.assert_allclose(sp.flp_iner, 5.0)
    np.testing.assert_allclose(sp.edge_iner, 7.0)
    np.testing.assert_allclose(sp.axial_stff, 2.0e9)
    np.testing.assert_allclose(sp.flp_stff, 3.0e8)
    np.testing.assert_allclose(sp.edge_stff, 5.0e8)
    np.testing.assert_allclose(sp.tor_stff, 1.0e8)


def test_windio_blade_elastic_mode_guards(tmp_path) -> None:
    """``elastic='file'`` on a layup-only blade raises; an invalid
    mode raises; ``'auto'`` on a layup-only blade falls back to
    PreComp (no published block ⇒ runs, positive)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    p = tmp_path / "layup_only.yaml"
    p.write_text(_older_blade_yaml(), encoding="utf-8")
    blade = read_windio_blade(p, n_span=5)
    assert blade.elastic is None

    with pytest.raises(ValueError, match="elastic='file'"):
        windio_blade_section_props(blade, elastic="file")
    with pytest.raises(ValueError, match="elastic must be"):
        windio_blade_section_props(blade, elastic="bogus")

    sp = windio_blade_section_props(blade, n_perim=120, elastic="auto")
    np.testing.assert_array_less(0.0, sp.mass_den)
    np.testing.assert_array_less(0.0, sp.axial_stff)


# ---------------------------------------------------------------------------
# Static-review hardening: twist units, malformed published block,
# single-airfoil schedule
# ---------------------------------------------------------------------------

def test_twist_unit_autodetect_radians_vs_degrees() -> None:
    """``_twist_to_degrees`` converts a radian-convention table
    (IEA-3.4: max ≈ 0.35) but leaves a degree-convention one
    (IEA-15: max ≈ 15.6) untouched — the old unconditional
    ``np.degrees`` turned 15.6° into ≈ 894°."""
    from pybmodes.io.windio_blade import _twist_to_degrees

    rad = np.array([0.349, 0.20, 0.0, -0.081])      # IEA-3.4 style
    np.testing.assert_allclose(_twist_to_degrees(rad), np.degrees(rad))

    deg = np.array([15.594, 8.55, 0.04, -1.24])     # IEA-15 style
    np.testing.assert_allclose(_twist_to_degrees(deg), deg)

    assert _twist_to_degrees(np.array([])).size == 0


def _blade_yaml(*, twist_vals="[0.0, 0.0]", labels="[circ, circ]",
                af_grid="[0.0, 1.0]", elastic_block="") -> str:
    fx, fy = _circ_xy_yaml()
    return textwrap.dedent(f"""\
        components:
          blade:
            outer_shape_bem:
              reference_axis:
                z: {{grid: [0.0, 1.0], values: [0.0, 50.0]}}
              airfoil_position: {{grid: {af_grid}, labels: {labels}}}
              chord: {{grid: [0.0, 1.0], values: [2.0, 2.0]}}
              twist: {{grid: [0.0, 1.0], values: {twist_vals}}}
              pitch_axis: {{grid: [0.0, 1.0], values: [0.5, 0.5]}}
            internal_structure_2d_fem:
              webs: []
              layers:
                - name: skin
                  material: glass
                  thickness: {{grid: [0.0, 1.0], values: [0.01, 0.01]}}
                  fiber_orientation: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  start_nd_arc: {{grid: [0.0, 1.0], values: [0.0, 0.0]}}
                  end_nd_arc: {{grid: [0.0, 1.0], values: [1.0, 1.0]}}
{elastic_block}
        airfoils:
          - name: circ
            coordinates: {{x: {fx}, y: {fy}}}
        materials:
          - {{name: glass, E: 7.0e10, nu: 0.3, G: 2.6923e10, rho: 2600.0}}
        """)


def test_windio_blade_twist_degrees_not_double_converted(tmp_path) -> None:
    """A degree-convention twist table (15.6° root) round-trips to
    ~15.6° structural twist, not ~894° (issue #47 follow-up)."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import read_windio_blade

    p = tmp_path / "deg.yaml"
    p.write_text(_blade_yaml(twist_vals="[15.594, -1.242]"),
                 encoding="utf-8")
    blade = read_windio_blade(p, n_span=5)
    assert float(np.max(np.abs(blade.twist_deg))) < 20.0   # not ~894

    p2 = tmp_path / "rad.yaml"
    p2.write_text(_blade_yaml(twist_vals="[0.349, -0.081]"),
                  encoding="utf-8")
    blade2 = read_windio_blade(p2, n_span=5)
    np.testing.assert_allclose(np.max(blade2.twist_deg),
                               np.degrees(0.349), rtol=1e-6)


def test_windio_blade_single_airfoil_schedule(tmp_path) -> None:
    """A constant-profile blade with ONE airfoil definition parses
    (the old ``len(af_grid) - 2`` indexing crashed); every station
    reuses the single profile."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import read_windio_blade

    p = tmp_path / "one_af.yaml"
    p.write_text(_blade_yaml(labels="[circ]", af_grid="[0.0]"),
                 encoding="utf-8")
    blade = read_windio_blade(p, n_span=6)
    assert len(blade.profiles) == 6
    first = blade.profiles[0]
    # The guard caches and reuses the single profile for every station.
    for prof in blade.profiles[1:]:
        assert prof is first


def test_windio_blade_malformed_elastic_block_not_silent(tmp_path) -> None:
    """A *present but unparseable* published block must not silently
    degrade to PreComp (issue #47 follow-up): ``read_windio_blade``
    records the parse error, ``auto`` warns then falls back, ``file``
    raises, and a genuinely absent block stays silent."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    # stiffness_matrix present but K33 missing -> KeyError on parse.
    bad = textwrap.indent(textwrap.dedent("""\
        elastic_properties:
          inertia_matrix:
            grid: [0.0, 1.0]
            mass: [100.0, 100.0]
            i_flap: [5.0, 5.0]
            i_edge: [7.0, 7.0]
          stiffness_matrix:
            grid: [0.0, 1.0]
            K44: [3.0e8, 3.0e8]
        """), " " * 14)
    p = tmp_path / "bad_ep.yaml"
    p.write_text(_blade_yaml(elastic_block=bad), encoding="utf-8")
    blade = read_windio_blade(p, n_span=5)
    assert blade.elastic is None
    assert blade.elastic_parse_error is not None
    assert "could not be parsed" in blade.elastic_parse_error

    with pytest.warns(UserWarning, match="could not be parsed"):
        sp = windio_blade_section_props(blade, n_perim=120,
                                        elastic="auto")
    np.testing.assert_array_less(0.0, sp.mass_den)   # PreComp fallback

    with pytest.raises(ValueError, match="unusable"):
        windio_blade_section_props(blade, elastic="file")

    # A genuinely absent block: no error recorded, no warning on auto.
    p2 = tmp_path / "no_ep.yaml"
    p2.write_text(_blade_yaml(), encoding="utf-8")
    blade2 = read_windio_blade(p2, n_span=5)
    assert blade2.elastic is None
    assert blade2.elastic_parse_error is None
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")            # any warning => failure
        windio_blade_section_props(blade2, n_perim=120, elastic="auto")


def test_windio_blade_named_inertia_labels_not_swapped(tmp_path) -> None:
    """issue #50 follow-up: a modern ``elastic_properties`` block with
    ``i_flap > i_edge`` and no ``i_cp`` must keep the schema's named
    flap/edge inertias — the old min/max eigen-sort silently swapped
    them, mislabelling rotary inertia and shifting modal results."""
    pytest.importorskip("yaml")
    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )

    # i_flap (12) deliberately > i_edge (4); no i_cp ⇒ already
    # diagonal in the named frame. Diagonal stiffness so the EI/EA
    # mapping is trivial and the test isolates the inertia labelling.
    ep = textwrap.indent(textwrap.dedent("""\
        elastic_properties:
          inertia_matrix:
            grid: [0.0, 1.0]
            mass: [80.0, 80.0]
            i_flap: [12.0, 12.0]
            i_edge: [4.0, 4.0]
          stiffness_matrix:
            grid: [0.0, 1.0]
            K33: [2.0e9, 2.0e9]
            K44: [3.0e8, 3.0e8]
            K55: [5.0e8, 5.0e8]
            K66: [1.0e8, 1.0e8]
        """), " " * 14)
    p = tmp_path / "named_iner.yaml"
    p.write_text(_blade_yaml(elastic_block=ep), encoding="utf-8")
    blade = read_windio_blade(p, n_span=5)
    sp = windio_blade_section_props(blade, elastic="file")
    np.testing.assert_allclose(sp.flp_iner, 12.0)   # NOT min(12,4)
    np.testing.assert_allclose(sp.edge_iner, 4.0)   # NOT max(12,4)


# ---------------------------------------------------------------------------
# Integration — real WindIO blades vs companion BeamDyn 6×6
#
# The companion `*_BeamDyn_blade.dat` tables were WISDEM-PreComp-
# generated from the same geometry, so they are the diagonal-property
# oracle. Honest, *measured* bounds (NOT reverse-fit): the diagonal
# thin-wall reduction reproduces mass / EA / GJ / the stiff bending
# axis to PreComp-class accuracy; the *weak* bending axis is a known
# limitation of a diagonal (uncoupled) reduction — it omits the
# spar-cap-offset / bend-twist coupling terms a full PreComp keeps —
# so it is gated only to order-of-magnitude and documented as such
# here and in VALIDATION.md. IEA-3.4 / IEA-10 blades use the
# parametric `start_nd_arc:{fixed:…}` / `midpoint+width` layer forms
# and need the geometric resolver — explicitly
# skipped so the gap is visible, not silent.
# ---------------------------------------------------------------------------

_DOCS = (pathlib.Path(__file__).resolve().parents[1]
         / "external" / "OpenFAST_files")
_IEA15_BL_Y = _DOCS / "IEA-15-240-RWT/WT_Ontology/IEA-15-240-RWT.yaml"
_IEA15_BL_BD = (_DOCS / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT/"
                "IEA-15-240-RWT_BeamDyn_blade.dat")
_IEA22_BL_Y = _DOCS / "IEA-22-280-RWT/windIO/IEA-22-280-RWT.yaml"
_IEA22_BL_BD = (_DOCS / "IEA-22-280-RWT/OpenFAST/IEA-22-280-RWT/"
                "IEA-22-280-RWT_BeamDyn_Blade.dat")
_IEA34_BL_Y = _DOCS / "IEA-3.4-130-RWT/yaml/IEA-3.4-130-RWT.yaml"
_IEA34_BL_BD = (_DOCS / "IEA-3.4-130-RWT/openfast/"
                "IEA-3.4-130-RWT_BeamDyn_Blade.dat")
_IEA10_BL_Y = _DOCS / "IEA-10.0-198-RWT/yaml/IEA-10-198-RWT.yaml"
_IEA10_BL_BD = (_DOCS / "IEA-10.0-198-RWT/openfast/"
                "IEA-10.0-198-RWT_BeamDyn_blade.dat")


def _relerr(a, b):
    b = np.where(np.abs(b) < 1e-9, 1e-9, b)
    return np.abs(np.asarray(a) - np.asarray(b)) / np.abs(b)


@pytest.mark.integration
@pytest.mark.parametrize(
    "yaml_p,bd_p,ident",
    [
        (_IEA15_BL_Y, _IEA15_BL_BD, "iea15-modern"),
        (_IEA22_BL_Y, _IEA22_BL_BD, "iea22-older"),
        (_IEA34_BL_Y, _IEA34_BL_BD, "iea3.4-older-parametric"),
        (_IEA10_BL_Y, _IEA10_BL_BD, "iea10-older-parametric"),
    ],
    ids=["iea15", "iea22", "iea3.4", "iea10"],
)
def test_windio_blade_vs_beamdyn_precomp_class(yaml_p, bd_p, ident) -> None:
    """WindIO composite reduction vs each turbine's own BeamDyn 6×6
    (itself WISDEM-PreComp-generated), structural span 0.15–0.90, for
    all four RWTs incl. the IEA-3.4/10 parametric-layer blades
    (geometric resolver). The bounds are the **honestly measured**
    cross-turbine envelope (worst-observed across IEA-3.4/10/15/22) +
    a small margin — they are regression + honest-disclosure bounds,
    NOT a precision claim. Mass / EA are PreComp-class; GJ and the
    bending stiffnesses carry the documented limitations of a
    *diagonal* (uncoupled) thin-wall reduction — composite multi-cell
    torsion (IEA-10) and weak-axis EI are its weakest metrics. The
    per-quantity worst-observed values are tabulated openly in
    VALIDATION.md."""
    pytest.importorskip("yaml")
    if not (yaml_p.is_file() and bd_p.is_file()):
        pytest.skip(f"{ident}: WindIO yaml / BeamDyn blade deck absent")

    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )
    from tests._beamdyn_blade import read_beamdyn_blade

    blade = read_windio_blade(yaml_p, n_span=40)
    # This is the *PreComp-class* validation — it must exercise the
    # PreComp layup reduction explicitly, not whatever the 1.5
    # ``elastic='auto'`` default now prefers (which would, for the
    # turbines that ship published properties, validate a different
    # path entirely; the published path is covered separately by
    # test_windio_blade_published_matches_beamdyn_far_tighter).
    sp = windio_blade_section_props(blade, n_perim=240, elastic="precomp")
    bd = read_beamdyn_blade(bd_p)

    sel = (bd.eta >= 0.15) & (bd.eta <= 0.90)
    et = bd.eta[sel]
    z = sp.span_loc
    mass = np.interp(et, z, sp.mass_den)
    EA = np.interp(et, z, sp.axial_stff)
    GJ = np.interp(et, z, sp.tor_stff)
    ei_lo = np.minimum(np.interp(et, z, sp.flp_stff),
                       np.interp(et, z, sp.edge_stff))
    ei_hi = np.maximum(np.interp(et, z, sp.flp_stff),
                       np.interp(et, z, sp.edge_stff))
    bd_lo = np.minimum(bd.EI_a[sel], bd.EI_b[sel])
    bd_hi = np.maximum(bd.EI_a[sel], bd.EI_b[sel])

    # Honest cross-turbine envelope (measured worst med/max over
    # IEA-3.4/10/15/22): mass med≤4 % max≤12 %; EA med≤8 % max≤12 %.
    assert np.median(_relerr(mass, bd.mpl[sel])) < 0.06
    assert np.max(_relerr(mass, bd.mpl[sel])) < 0.15
    assert np.median(_relerr(EA, bd.EA[sel])) < 0.10
    assert np.max(_relerr(EA, bd.EA[sel])) < 0.18
    # GJ — composite multi-cell torsion, a known PreComp/diagonal weak
    # metric (IEA-10 med≈18 % max≈41 %): regression bound, documented.
    assert np.median(_relerr(GJ, bd.GJ[sel])) < 0.22
    assert np.max(_relerr(GJ, bd.GJ[sel])) < 0.47
    # Bending stiffnesses — diagonal-reduction limitation (omits the
    # spar-cap-offset / bend-twist coupling a full PreComp keeps);
    # measured med up to ≈22 %, max up to ≈40 %.
    assert np.median(_relerr(ei_hi, bd_hi)) < 0.28
    assert np.max(_relerr(ei_hi, bd_hi)) < 0.45
    assert np.median(_relerr(ei_lo, bd_lo)) < 0.35
    assert np.max(_relerr(ei_lo, bd_lo)) < 0.55


@pytest.mark.integration
@pytest.mark.parametrize(
    "yaml_p,ident",
    [
        (_IEA15_BL_Y, "iea15"),
        (_IEA22_BL_Y, "iea22"),
        (_IEA34_BL_Y, "iea3.4"),
        (_IEA10_BL_Y, "iea10"),
    ],
    ids=["iea15", "iea22", "iea3.4", "iea10"],
)
def test_rotatingblade_from_windio_modal_smoke(yaml_p, ident) -> None:
    """`RotatingBlade.from_windio` drives the full FEM to a physical
    parked-blade spectrum (finite, positive, ascending) for all four
    RWTs — both WindIO dialects and the IEA-3.4/10 parametric-layer
    blades (geometric resolver)."""
    pytest.importorskip("yaml")
    if not yaml_p.is_file():
        pytest.skip(f"{ident}: WindIO yaml absent")
    from pybmodes.models import RotatingBlade

    f = RotatingBlade.from_windio(yaml_p, n_span=24).run(
        n_modes=6, check_model=False
    ).frequencies
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)
    assert np.all(np.diff(f) >= -1e-9)


@pytest.mark.integration
def test_windio_blade_published_matches_beamdyn_far_tighter(tmp_path) -> None:
    """issue #48 value + orientation pin: IEA-15 carries
    ``elastic_properties``. ``elastic='file'`` must reproduce the
    turbine's own BeamDyn 6×6 diagonal *far* tighter than the PreComp
    reduction (both derive from the same WISDEM source), and the
    flap/edge axes must not be swapped (K44↔EI_a, K55↔EI_b)."""
    pytest.importorskip("yaml")
    if not (_IEA15_BL_Y.is_file() and _IEA15_BL_BD.is_file()):
        pytest.skip("IEA-15 WindIO yaml / BeamDyn blade deck absent")

    from pybmodes.io.windio_blade import (
        read_windio_blade,
        windio_blade_section_props,
    )
    from tests._beamdyn_blade import read_beamdyn_blade

    blade = read_windio_blade(_IEA15_BL_Y, n_span=40)
    assert blade.elastic is not None, "IEA-15 should carry elastic_properties"
    bd = read_beamdyn_blade(_IEA15_BL_BD)
    sel = (bd.eta >= 0.15) & (bd.eta <= 0.90)
    et = bd.eta[sel]

    def _med(sp, attr, ref):
        z = sp.span_loc
        return float(np.median(_relerr(np.interp(et, z, getattr(sp, attr)),
                                       ref[sel])))

    pub = windio_blade_section_props(blade, n_perim=240, elastic="file")
    pc = windio_blade_section_props(blade, n_perim=240, elastic="precomp")

    # issue #50: both the WindIO 6×6 and the companion BeamDyn 6×6 are
    # referenced at the blade axis, so the apples-to-apples oracle is
    # the *decoupled* BeamDyn (elastic/shear centre + principal axes),
    # not its raw diagonal. The two matrices share a WISDEM-PreComp
    # origin, so decoupled-vs-decoupled is tight (grid/interp only).
    assert _med(pub, "mass_den", bd.mpl) < 0.03
    assert _med(pub, "axial_stff", bd.EA) < 0.03      # EA is invariant
    assert _med(pub, "flp_stff", bd.EI_flap_dec) < 0.05
    assert _med(pub, "edge_stff", bd.EI_edge_dec) < 0.05
    assert _med(pub, "tor_stff", bd.GJ_dec) < 0.08

    # The #50 fix must matter: the decoupled published EI is materially
    # different from (and closer to the decoupled reference than) the
    # raw reference-axis diagonal it used to copy.
    raw_vs_dec = float(np.median(_relerr(bd.EI_flap_dec[sel],
                                         bd.EI_a[sel])))
    assert raw_vs_dec > 0.05, (
        "raw vs decoupled BeamDyn EI should differ materially — "
        "otherwise this turbine wouldn't exercise #50"
    )
    # And the published path stays far tighter than PreComp.
    assert (_med(pub, "flp_stff", bd.EI_flap_dec)
            < _med(pc, "flp_stff", bd.EI_flap_dec))
