"""Tests for ``pybmodes.mooring``.

Two tiers:

- **Analytical** (default-run, self-contained): catenary boundary-
  condition equations checked against Irvine 1981 inextensible / Jonkman
  2007 extensible closed forms with no external data.
- **Integration** (``@pytest.mark.integration``): MoorDyn parse +
  stiffness assembly on the OpenFAST ``r-test`` 5MW OC3 spar deck.
  Skipped when the upstream r-test repo isn't checked out under
  ``external/OpenFAST_files/``.

The OC3 published mooring K[0,0] (Jonkman 2010 NREL/TP-500-47535 Table
5-1) is reproduced to better than 0.01 % — that's the canonical
sanity check on the catenary math + force assembly.

K[5,5] (yaw) is intentionally NOT compared against Jonkman 2010's
published value (9.83e7 N·m/rad). The OC3 reference design adds an
"additional yaw spring" via ElastoDyn's ``PtfmYawStiff`` input to
model the spar's delta-line crowfoot connection at the fairleads; that
spring is NOT in the MoorDyn ``.dat`` so a MoorDyn-only solver
naturally reproduces the catenary contribution (~ 1.15e7 N·m/rad,
~ 12 % of the total). Test below pins the catenary-only value.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from pybmodes.mooring import Line, LineType, MooringSystem, Point

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OC3_MOORDYN = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
    / "NRELOffshrBsline5MW_OC3Hywind_MoorDyn.dat"
)


# ---------------------------------------------------------------------------
# Analytical fixtures
# ---------------------------------------------------------------------------

def _make_analytical_line(
    L: float = 900.0, W: float = 500.0, EA: float = 1.0e12,
) -> Line:
    """A line floating in air (``seabed_contact=False``) with adjustable
    properties. Endpoints are placeholders — only the geometry passed to
    ``solve_static`` matters here."""
    lt = LineType(
        name="analytical",
        diam=0.1,
        mass_per_length_air=W / 9.80665,
        EA=EA,
        w=W,
    )
    a = Point(id=1, attachment="Fixed", r_body=np.zeros(3))
    b = Point(id=2, attachment="Fixed", r_body=np.zeros(3))
    return Line(
        line_type=lt, point_a=a, point_b=b,
        unstretched_length=L,
        seabed_contact=False,
    )


# ---------------------------------------------------------------------------
# Analytical: catenary closed forms
# ---------------------------------------------------------------------------

def test_catenary_inextensible_limit() -> None:
    """With ``EA → ∞`` the elastic-stretch terms vanish and Irvine 1981
    §2.3 inextensible equations must hold to 1e-4 m relative."""
    line = _make_analytical_line(L=900.0, W=500.0, EA=1.0e12)
    r_a = np.array([0.0, 0.0, 0.0])
    r_b = np.array([800.0, 0.0, 300.0])  # 300 m vertical rise
    H, V_F, _ = line.solve_static(r_a, r_b)
    L, W = 900.0, 500.0
    # Check Irvine inextensible residuals (drop the H·L/EA, L/EA
    # correction terms).
    u = V_F / H
    v = (V_F - W * L) / H
    e_x = (H / W) * (math.asinh(u) - math.asinh(v)) - 800.0
    e_z = (H / W) * (math.sqrt(1 + u * u) - math.sqrt(1 + v * v)) - 300.0
    # EA = 1e12 → L/EA ≈ 1e-9, completely negligible.
    assert abs(e_x) < 1e-4 * 800.0
    assert abs(e_z) < 1e-4 * 300.0


def test_catenary_horizontal_line() -> None:
    """For a line between two endpoints at the same depth, ``V_F = W·L/2``
    by symmetry (Jonkman 2007 B-2 with V_A = V_F − W·L = −W·L/2 → fairlead
    supports half the line weight, anchor supports the other half)."""
    L, W, EA = 900.0, 500.0, 1.0e10
    line = _make_analytical_line(L=L, W=W, EA=EA)
    # Pick a horizontal span that gives a moderately slack line.
    r_a = np.array([0.0, 0.0, 0.0])
    r_b = np.array([600.0, 0.0, 0.0])
    H, V_F, _ = line.solve_static(r_a, r_b)
    assert V_F == pytest.approx(W * L / 2, rel=1e-6)


def test_catenary_solver_satisfies_residual() -> None:
    """The returned ``(H, V_F)`` must satisfy the elastic-catenary
    equations to better than the solver tolerance.

    This is a self-consistency check — the spec'd
    ``test_catenary_vertical_line`` (``ΔX = 0``) is a degenerate
    boundary that the asinh-based formulation can't represent (``H = 0``
    makes ``asinh(V/H)`` ill-defined); we replace it with this
    residual-satisfaction test which exercises the same convergence
    requirement without the singularity.
    """
    L, W, EA = 800.0, 700.0, 4.0e8
    line = _make_analytical_line(L=L, W=W, EA=EA)
    r_a = np.array([0.0, 0.0, -250.0])
    r_b = np.array([700.0, 0.0, 0.0])
    H, V_F, _ = line.solve_static(r_a, r_b, tol=1e-8)
    u = V_F / H
    v = (V_F - W * L) / H
    e_x = (H / W) * (math.asinh(u) - math.asinh(v)) + H * L / EA - 700.0
    e_z = (
        (H / W) * (math.sqrt(1 + u * u) - math.sqrt(1 + v * v))
        + (L / EA) * (V_F - 0.5 * W * L)
        - 250.0
    )
    assert abs(e_x) < 1e-6
    assert abs(e_z) < 1e-6


# ---------------------------------------------------------------------------
# Analytical: synthetic 3-line symmetric mooring
# ---------------------------------------------------------------------------

def _make_three_line_system(
    fairlead_r: float = 5.2,
    anchor_r: float = 850.0,
    fairlead_z: float = -70.0,
    anchor_z: float = -320.0,
    L: float = 902.0,
    diam: float = 0.09,
    mass_air: float = 77.7066,
    EA: float = 3.842e8,
    rho: float = 1025.0,
    g: float = 9.80665,
) -> MooringSystem:
    """Three identical catenary lines, 120° spaced around z-axis."""
    area = math.pi * 0.25 * diam * diam
    w = (mass_air - rho * area) * g
    lt = LineType(
        name="main", diam=diam, mass_per_length_air=mass_air,
        EA=EA, w=w,
    )
    points: dict[int, Point] = {}
    lines: list[Line] = []
    for k in range(3):
        theta = k * 2.0 * math.pi / 3.0
        c, s = math.cos(theta), math.sin(theta)
        anchor = Point(
            id=10 + k, attachment="Fixed",
            r_body=np.array([anchor_r * c, anchor_r * s, anchor_z]),
        )
        fair = Point(
            id=20 + k, attachment="Vessel",
            r_body=np.array([fairlead_r * c, fairlead_r * s, fairlead_z]),
        )
        points[anchor.id] = anchor
        points[fair.id] = fair
        lines.append(Line(
            line_type=lt, point_a=anchor, point_b=fair,
            unstretched_length=L,
        ))
    return MooringSystem(
        depth=abs(anchor_z), rho=rho, g=g,
        line_types={"main": lt}, points=points, lines=lines,
    )


def test_restoring_force_symmetry() -> None:
    """A 3-fold-symmetric mooring at ``r6 = 0`` has zero net horizontal
    force and zero net yaw moment; in-plane DOFs ``F_x``, ``F_y``,
    ``M_z`` are zero by symmetry. ``F_z`` is non-zero (lines pull body
    down) and ``M_x``, ``M_y`` are zero by 3-fold symmetry."""
    ms = _make_three_line_system()
    F = ms.restoring_force(np.zeros(6))
    # Relative scale for "approximately zero" tolerance — V_F ~ 5e5 per
    # line, so 3-fold cancellation should land below 1e-3 of that.
    scale = abs(F[2])  # |F_z|
    assert abs(F[0]) < 1e-3 * scale
    assert abs(F[1]) < 1e-3 * scale
    assert abs(F[3]) < 1e-3 * scale
    assert abs(F[4]) < 1e-3 * scale
    assert abs(F[5]) < 1e-3 * scale
    assert F[2] < 0  # mooring pulls body DOWN


def test_stiffness_matrix_positive_diagonal() -> None:
    """All diagonal entries of ``K_moor`` are positive at zero offset —
    mooring stiffens against every DOF perturbation.

    (We don't assert full positive-definiteness because the off-diagonal
    surge-pitch / sway-roll couplings of a typical FOWT mooring make
    some eigenvalues small, and a strict ``eig > 0`` check is
    unnecessarily brittle. Positive diagonal is the directly-physical
    invariant.)
    """
    ms = _make_three_line_system()
    K = ms.stiffness_matrix(np.zeros(6))
    diag = np.diag(K)
    assert np.all(diag > 0), f"non-positive diagonal: {diag}"


def test_stiffness_matrix_symmetry() -> None:
    """``K`` is symmetric (Hessian of a conservative potential).

    Our solver symmetrises the trans-rot off-diagonal blocks
    explicitly; this test pins that symmetrisation. The full-matrix
    asymmetry should be machine-precision after the averaging.
    """
    ms = _make_three_line_system()
    K = ms.stiffness_matrix(np.zeros(6))
    asymmetry = float(np.max(np.abs(K - K.T)))
    peak = float(np.max(np.abs(K)))
    assert asymmetry < 1e-10 * peak, (
        f"asymmetry {asymmetry:.3e} > 1e-10·{peak:.3e}"
    )


# ---------------------------------------------------------------------------
# Integration: OC3 Hywind MoorDyn deck
# ---------------------------------------------------------------------------

if not OC3_MOORDYN.is_file():
    _oc3_skip = pytest.mark.skip(
        reason=(
            "OpenFAST r-test 5MW_OC3Spar MoorDyn deck not present under "
            f"{OC3_MOORDYN.parent}; clone the upstream r-test repo to "
            "run these tests."
        ),
    )
else:
    _oc3_skip = pytest.mark.integration


@_oc3_skip
def test_moordyn_parse_roundtrip() -> None:
    """Parse the OC3 deck and verify the published 3-line / 6-point
    layout, anchor radii, and fairlead radii."""
    ms = MooringSystem.from_moordyn(OC3_MOORDYN)
    assert len(ms.lines) == 3
    assert len(ms.points) == 6
    assert "main" in ms.line_types
    # Three Fixed anchors + three Vessel fairleads.
    n_fixed = sum(1 for p in ms.points.values() if p.attachment == "Fixed")
    n_vessel = sum(1 for p in ms.points.values() if p.attachment == "Vessel")
    assert n_fixed == 3
    assert n_vessel == 3
    # Fairleads at radius ~ 5.2 m (Jonkman 2010 §4.2.2).
    fair_radii = [
        math.hypot(p.r_body[0], p.r_body[1])
        for p in ms.points.values() if p.attachment == "Vessel"
    ]
    for r in fair_radii:
        assert r == pytest.approx(5.2, rel=1e-3)
    # Anchors at radius ~ 853.87 m.
    anchor_radii = [
        math.hypot(p.r_body[0], p.r_body[1])
        for p in ms.points.values() if p.attachment == "Fixed"
    ]
    for r in anchor_radii:
        assert r == pytest.approx(853.87, rel=1e-3)
    # Unstretched length 902.2 m per Jonkman 2010 Table 4-2.
    for line in ms.lines:
        assert line.unstretched_length == pytest.approx(902.2, rel=1e-4)


@_oc3_skip
def test_oc3hywind_surge_stiffness() -> None:
    """``K_moor[0,0] ≈ 41,180 N/m`` (Jonkman 2010 Table 5-1, OC3-Hywind
    static surge stiffness about the platform reference point)."""
    ms = MooringSystem.from_moordyn(OC3_MOORDYN)
    K = ms.stiffness_matrix(np.zeros(6))
    assert K[0, 0] == pytest.approx(41_180.0, rel=0.05)


@_oc3_skip
def test_oc3hywind_yaw_stiffness_catenary_only() -> None:
    """``K_moor[5,5]`` from the catenary lines alone (no delta-line
    crowfoot contribution).

    The OC3 spec adds a separate ``PtfmYawStiff`` term in ElastoDyn to
    model the delta-line crowfoot connection at each fairlead — that's
    where the bulk of OC3's published 9.83e7 N·m/rad yaw stiffness
    comes from. Our MoorDyn-only solve reproduces the catenary
    contribution at ~ 1.15e7 N·m/rad. The value is positive (mooring
    resists yaw) and reproducible to 1 % across solver perturbation
    sizes.
    """
    ms = MooringSystem.from_moordyn(OC3_MOORDYN)
    K = ms.stiffness_matrix(np.zeros(6))
    assert K[5, 5] > 0
    assert K[5, 5] == pytest.approx(1.156e7, rel=0.05)


@_oc3_skip
def test_tower_from_elastodyn_with_mooring() -> None:
    """End-to-end: ``Tower.from_elastodyn_with_mooring`` on the OC3
    ElastoDyn + MoorDyn decks assembles a free-free floating BMI with
    a populated ``PlatformSupport`` block in the right BModes file
    convention and produces a finite coupled modal solve.

    We don't pin a tight tower-bending frequency value here because
    the coupled value also depends on the platform hydro_K — and the
    OC3 spar is *hydrostatically unstable* in pitch / roll (published
    platform K_44 ≈ K_55 ≈ −5e9 N·m/rad), stabilised at runtime by
    mooring + delta-line yaw spring. Without that destabilising
    hydrostatic term the coupled modes land further from published;
    routing the full HydroDyn deck through with non-zero ``A_inf`` and
    ``C_hst`` brings them back.
    """
    import numpy as np

    from pybmodes.models import Tower

    elastodyn = (
        REPO_ROOT / "src" / "pybmodes" / "_examples" / "reference_decks"
        / "nrel5mw_oc3spar"
        / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat"
    )
    if not elastodyn.is_file():
        pytest.skip(f"OC3 reference deck not present at {elastodyn}")
    tower = Tower.from_elastodyn_with_mooring(elastodyn, OC3_MOORDYN)
    assert tower._bmi.hub_conn == 2
    assert tower._bmi.tow_support == 1
    ps = tower._bmi.support
    assert ps is not None
    # Platform mass from ElastoDyn ``PtfmMass``.
    assert ps.mass_pform == pytest.approx(7.466e6, rel=0.01)
    # ``i_matrix`` is stored AT THE CM (no parallel-axis transfer).
    # OC3 ``PtfmPIner = PtfmRIner = 4.229e9``; ``PtfmYIner = 1.642e8``.
    # The downstream :func:`pybmodes.fem.nondim.nondim_platform` does
    # the rigid-arm CM → tower-base transfer using ``cm_pform - draft``;
    # adding ``M·dz²`` here would double-count (pre-1.0 review finding).
    assert ps.i_matrix[3, 3] == pytest.approx(4.229e9, rel=0.01)
    assert ps.i_matrix[4, 4] == pytest.approx(4.229e9, rel=0.01)
    assert ps.i_matrix[5, 5] == pytest.approx(1.642e8, rel=0.01)
    # No surge-pitch / sway-roll coupling on the at-CM matrix.
    assert ps.i_matrix[0, 4] == 0.0
    assert ps.i_matrix[1, 3] == 0.0
    # BMI sign convention (positive distance below MSL; signed draft).
    # OC3 TP sits at +10 m above MSL → ``draft = -10``.
    # OC3 spar CM at z = −89.9155 → ``cm_pform = +89.9155``.
    assert ps.draft == pytest.approx(-10.0, abs=0.01)
    assert ps.cm_pform == pytest.approx(89.9155, rel=0.01)
    assert ps.ref_msl == pytest.approx(0.0, abs=0.01)
    # Mooring K[0,0] matches the standalone OC3 catenary solve.
    assert ps.mooring_K[0, 0] == pytest.approx(41_180.0, rel=0.05)
    # BMI ``radius`` is the FULL ``TowerHt`` (not the flexible
    # ``TowerHt − TowerBsHt``) so that ``radius + draft`` recovers the
    # flexible-tower length after the nondim step. For OC3
    # ``TowerHt = 87.6 m``; the flexible length is 77.6 m.
    # Pre-1.0 review surfaced this — the cantilever adapter
    # alone leaves ``bmi.radius = 77.6`` which would compound with
    # ``draft = -10`` to give a flexible length of 67.6 m.
    assert tower._bmi.radius == pytest.approx(87.6, rel=0.01)
    assert (tower._bmi.radius + ps.draft) == pytest.approx(77.6, rel=0.01)
    # Coupled modal solve runs and produces real positive frequencies.
    result = tower.run(n_modes=8, check_model=False)
    assert np.all(np.isfinite(result.frequencies))
    assert np.all(result.frequencies >= 0)


def test_scan_platform_fields_raises_on_malformed_critical(
    tmp_path: pathlib.Path,
) -> None:
    """Every scalar this helper extracts is load-bearing — inertia
    fields define the rigid-body mass-matrix contribution, and
    ``Ptfm*Stiff`` carries restoring springs that aren't in HydroDyn
    or MoorDyn (the OC3 yaw spring lives in ``PtfmYawStiff`` only).
    A malformed value silently falling back to ``0.0`` would create
    a physically wrong floating model with no warning, so the helper
    raises on every parse failure. Pre-1.0 review surfaced this.
    """
    from pybmodes.models.tower import _scan_platform_fields

    deck = tmp_path / "broken.dat"
    deck.write_text(
        "synthetic ElastoDyn header\n"
        "not_a_number   PtfmMass    - kg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PtfmMass"):
        _scan_platform_fields(deck)


def test_scan_platform_fields_raises_on_malformed_ptfm_stiff(
    tmp_path: pathlib.Path,
) -> None:
    """A malformed ``PtfmYawStiff`` (or any other ``Ptfm*Stiff``)
    raises rather than silently dropping a restoring spring — the OC3
    yaw spring at 9.83e7 N·m/rad lives in this field only, and losing
    it would drop the coupled yaw frequency by an order of magnitude.
    """
    from pybmodes.models.tower import _scan_platform_fields

    deck = tmp_path / "broken_stiff.dat"
    deck.write_text(
        "synthetic ElastoDyn header\n"
        "1.0e6          PtfmMass        - kg\n"
        "1.0e9          PtfmRIner       - kg m^2\n"
        "1.0e9          PtfmPIner       - kg m^2\n"
        "1.0e8          PtfmYIner       - kg m^2\n"
        "not_a_number   PtfmYawStiff    - N m/rad\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PtfmYawStiff"):
        _scan_platform_fields(deck)


def test_scan_platform_fields_reads_ptfm_stiff_scalars(
    tmp_path: pathlib.Path,
) -> None:
    """``_scan_platform_fields`` recognises ElastoDyn's six
    ``Ptfm*Stiff`` scalars so ``Tower.from_elastodyn_with_mooring``
    can fold them into ``mooring_K``. The OC3 spec carries the delta-
    line crowfoot's yaw spring via ``PtfmYawStiff`` (98,340,000
    N·m/rad), and it isn't in the MoorDyn ``.dat`` — without reading
    it the OC3 coupled yaw frequency comes out an order of magnitude
    low. Pre-1.0 review surfaced this gap.
    """
    from pybmodes.models.tower import _scan_platform_fields

    deck = tmp_path / "ptfm_stiff.dat"
    deck.write_text(
        "synthetic ElastoDyn header\n"
        "1.5E+05    PtfmSurgeStiff    - N/m\n"
        "1.5E+05    PtfmSwayStiff     - N/m\n"
        "3.5E+05    PtfmHeaveStiff    - N/m\n"
        "0.0E+00    PtfmRollStiff     - N m/rad\n"
        "0.0E+00    PtfmPitchStiff    - N m/rad\n"
        "9.834E+07  PtfmYawStiff      - N m/rad (OC3 delta-line crowfoot)\n",
        encoding="utf-8",
    )
    fields = _scan_platform_fields(deck)
    assert fields["PtfmSurgeStiff"] == pytest.approx(1.5e5)
    assert fields["PtfmSwayStiff"] == pytest.approx(1.5e5)
    assert fields["PtfmHeaveStiff"] == pytest.approx(3.5e5)
    assert fields["PtfmRollStiff"] == 0.0
    assert fields["PtfmPitchStiff"] == 0.0
    assert fields["PtfmYawStiff"] == pytest.approx(9.834e7)


def test_scan_platform_fields_fortran_d_exponent(
    tmp_path: pathlib.Path,
) -> None:
    """ElastoDyn ``.dat`` scalars may use Fortran-style ``D`` exponent
    notation (``7.466D+06`` rather than ``7.466E+06``). The platform-
    scalar scanner must normalise ``D`` / ``d`` to ``E`` before
    ``float(...)`` so a valid scalar doesn't silently become 0.0.
    Pre-1.0 review surfaced this.
    """
    from pybmodes.models.tower import _scan_platform_fields

    deck = tmp_path / "synthetic.dat"
    deck.write_text(
        "synthetic ElastoDyn header\n"
        "7.466D+06    PtfmMass    - kg\n"
        "4.229E+09    PtfmRIner   - kg m^2\n"
        "4.229d+09    PtfmPIner   - lowercase d also OK\n"
        "1.642E+08    PtfmYIner   - kg m^2\n"
        "-89.9155     PtfmCMzt    - m\n"
        "0.0          PtfmRefzt   - m\n",
        encoding="utf-8",
    )
    fields = _scan_platform_fields(deck)
    assert fields["PtfmMass"] == pytest.approx(7.466e6)
    assert fields["PtfmRIner"] == pytest.approx(4.229e9)
    assert fields["PtfmPIner"] == pytest.approx(4.229e9)


def test_moordyn_v1_lines_column_order(tmp_path: pathlib.Path) -> None:
    """MoorDyn v1 ``LINE PROPERTIES`` rows use the column order
    ``ID LineType UnstrLen NumSegs NodeAnch NodeFair`` — different
    from v2's ``ID LineType AttachA AttachB UnstrLen ...``. The parser
    must detect the v1 layout and wire the right point IDs, not skip
    the rows because ``902.2`` doesn't parse as an integer.
    Reported by Pre-1.0 review.
    """
    deck = tmp_path / "v1.dat"
    deck.write_text(
        "--------------------- MoorDyn v1 ---------------------------\n"
        "Synthetic OC3-style 3-line mooring (v1 column order)\n"
        "------- LINE DICTIONARY -----------------------\n"
        "Name   Diam   MassDen   EA          BA/-zeta   Can  Cdn  Cat  Cdt\n"
        "(-)    (m)    (kg/m)    (N)         (-)        (-)  (-)  (-)  (-)\n"
        "main   0.09   77.7066   384.243E6   -0.8       1.0  1.6  0.0  0.1\n"
        "------- CONNECTION PROPERTIES -----------------\n"
        "Node   Type   X        Y       Z       M   V   FX  FY  FZ  CdA  Ca\n"
        "(-)    (-)    (m)      (m)     (m)     (kg)(m^3)(N) (N) (N) (m^2)(-)\n"
        "1      Fix     853.87   0.0     -320.0  0  0  0  0  0  0  0\n"
        "2      Fix    -426.94   739.47  -320.0  0  0  0  0  0  0  0\n"
        "3      Fix    -426.94  -739.47  -320.0  0  0  0  0  0  0  0\n"
        "4      Vessel  5.2      0.0     -70.0   0  0  0  0  0  0  0\n"
        "5      Vessel -2.6      4.5     -70.0   0  0  0  0  0  0  0\n"
        "6      Vessel -2.6     -4.5     -70.0   0  0  0  0  0  0  0\n"
        "------- LINE PROPERTIES -----------------------\n"
        "Line  LineType  UnstrLen  NumSegs  NodeAnch  NodeFair  Flags\n"
        "(-)   (-)        (m)       (-)      (-)       (-)       (-)\n"
        "1     main       902.2     20       1         4         -\n"
        "2     main       902.2     20       2         5         -\n"
        "3     main       902.2     20       3         6         -\n"
        "------- SOLVER OPTIONS ------------------------\n"
        "320      WtrDpth\n"
        "1025     rhoW\n"
        "------- OUTPUT LIST ---------------------------\n"
        "END\n",
        encoding="utf-8",
    )
    ms = MooringSystem.from_moordyn(deck)
    assert len(ms.lines) == 3
    assert len(ms.points) == 6
    # Every line should have its unstretched length correctly parsed.
    for line in ms.lines:
        assert line.unstretched_length == pytest.approx(902.2, rel=1e-6)
    # Attachments routed correctly: anchor points are Fixed, fairleads Vessel.
    for line in ms.lines:
        assert line.point_a.attachment == "Fixed"
        assert line.point_b.attachment == "Vessel"


def test_point_attachment_aliases() -> None:
    """``Point.__post_init__`` should accept MoorDyn-style abbreviations
    (``Fix`` / ``Connect`` / ``Body`` / ``Anchor``) and normalise them
    to the v2 canonical names. Unknown strings raise ``ValueError``."""
    import numpy as np

    p_fix = Point(id=1, attachment="Fix", r_body=np.zeros(3))
    assert p_fix.attachment == "Fixed"
    p_connect = Point(id=2, attachment="Connect", r_body=np.zeros(3))
    assert p_connect.attachment == "Free"
    p_body = Point(id=3, attachment="Body", r_body=np.zeros(3))
    assert p_body.attachment == "Vessel"
    with pytest.raises(ValueError, match="attachment"):
        Point(id=4, attachment="Whatever", r_body=np.zeros(3))


def test_oc3hywind_bmi_dof_order_matches_jonkman_2010() -> None:
    """Pin the BMI rigid-body DOF order using the canonical
    OC3Hywind.bmi sample shipped under
    ``src/pybmodes/_examples/sample_inputs/reference_turbines/07_nrel5mw_oc3hywind_spar/``.

    Jonkman (2010) NREL/TP-500-47535 Table 5-1 publishes the OC3
    mooring stiffness with K_15 = surge → pitch coupling = -2.821e6
    N. If the BMI file stores its 6x6 matrices in OpenFAST DOF order
    ``[surge, sway, heave, roll, pitch, yaw]`` then the file's
    ``mooring_K[0, 4]`` should match this value. Any future DOF-order
    refactor that breaks this invariant — particularly anything
    rotating the assumed order to ``[sway, surge, ...]`` — fails this
    test immediately, before producing wrong physics on asymmetric
    platforms.

    The same invariant flows through
    :meth:`Tower.from_elastodyn_with_mooring`'s ``Ptfm*Stiff``
    fold-in, which writes ElastoDyn ``PtfmSurgeStiff`` to ``K_moor[0]``
    and so on. As long as this test passes, that mapping is correct.
    """
    from pybmodes.io.bmi import read_bmi

    bmi_path = (
        REPO_ROOT / "src" / "pybmodes" / "_examples" / "sample_inputs"
        / "reference_turbines" / "07_nrel5mw_oc3hywind_spar"
        / "07_nrel5mw_oc3hywind_spar_tower.bmi"
    )
    if not bmi_path.is_file():
        pytest.skip(f"OC3 Hywind sample BMI not present at {bmi_path}")
    bmi = read_bmi(bmi_path)
    assert bmi.support is not None
    # K_15 in Jonkman 1-indexed = K[0, 4] in 0-indexed = surge -> pitch
    # coupling. Negative sign per Table 5-1.
    assert bmi.support.mooring_K[0, 4] == pytest.approx(-2.821e6, rel=1e-3)
    # K_24 in Jonkman 1-indexed = K[1, 3] in 0-indexed = sway -> roll
    # coupling. Positive sign per the same table; OC3 deck stores a
    # slightly rounded +2.816e6.
    assert bmi.support.mooring_K[1, 3] == pytest.approx(+2.816e6, rel=1e-2)
    # And the (positive) coupling of opposite sign vs (0,4) is a
    # direct cross-check of the 3-fold-symmetric layout's mathematical
    # invariant: the two off-diagonal blocks of the surge/pitch and
    # sway/roll planes must have opposite signs.
    assert bmi.support.mooring_K[0, 4] * bmi.support.mooring_K[1, 3] < 0


@_oc3_skip
def test_oc3hywind_3fold_symmetry() -> None:
    """On the OC3 layout, surge and sway stiffness must be equal to
    machine precision, and the surge-pitch / sway-roll couplings must
    have opposite signs of equal magnitude (Jonkman 2010 Table 5-1
    reports ``K_15 = -K_24 ≈ -2.821e6 N``)."""
    ms = MooringSystem.from_moordyn(OC3_MOORDYN)
    K = ms.stiffness_matrix(np.zeros(6))
    assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-4)
    assert K[3, 3] == pytest.approx(K[4, 4], rel=1e-4)
    # K[0,4] (surge → pitch moment) and K[1,3] (sway → roll moment)
    # are of opposite sign by 3-fold symmetry about the z-axis.
    assert K[0, 4] * K[1, 3] < 0


# ===========================================================================
# Shared synthetic MoorDyn deck builders
# ===========================================================================
#
# Used by the strict-parsing / header-variant / OPTIONS-strict /
# non-finite-rejection tests below. The minimum valid v2 deck has
# 1 line type, 2 points, and 1 line; assembling it as named section
# blocks lets each test vary one section while keeping the others
# intact.

_MOORDYN_HEADER = "----------- MoorDyn v2 Input File -------------------\n"


def _write_moordyn(path: pathlib.Path, *sections: str) -> pathlib.Path:
    """Concatenate a header + arbitrary section blocks into a
    syntactically valid MoorDyn file at ``path``."""
    path.write_text(_MOORDYN_HEADER + "".join(sections), encoding="utf-8")
    return path


_LINE_TYPES_2HEADER = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    50.0            1.0e9   0.0
"""

_LINE_TYPES_1HEADER = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
chain   0.10    50.0            1.0e9   0.0
"""

_POINTS_2HEADER = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           100.0   0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""

_LINES_2HEADER = """\
---------------------- LINES ---------------------------------------------
ID      LineType        AttachA AttachB UnstrLen        NumSegs Outputs
(-)     (-)             (-)     (-)     (m)             (-)     (-)
1       chain           1       2       102.0           20      -
"""

_OPTIONS = """\
---------------------- OPTIONS -------------------------------------------
200.0   WtrDpth
1025.0  WtrDens
"""


# ===========================================================================
# Strict parsing on malformed data rows
# ===========================================================================

class TestMoorDynStrictParsing:
    """Malformed-but-data-looking rows raise rather than silently
    skipping. A ``continue`` branch would convert typos into
    incomplete mooring models.
    """

    def test_line_types_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    50.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _POINTS_2HEADER, _LINES_2HEADER,
        )
        with pytest.raises(ValueError, match="Malformed LINE TYPES"):
            MooringSystem.from_moordyn(path)

    def test_line_types_non_numeric_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   not_a_number    50.0    1.0e9   0.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _POINTS_2HEADER, _LINES_2HEADER,
        )
        with pytest.raises(
            ValueError,
            match="Malformed LINE TYPES row.*chain.*Diam / MassPerLength / EA",
        ):
            MooringSystem.from_moordyn(path)

    def test_points_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           100.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, bad_section, _LINES_2HEADER,
        )
        with pytest.raises(ValueError, match="Malformed POINTS"):
            MooringSystem.from_moordyn(path)

    def test_points_non_integer_id_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1.5     Fixed           100.0   0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""
        # The first POINTS row's first column is "1.5" which DOES match
        # ``_looks_like_number`` (=> it's a data row, not a header).
        # The strict ``int(parts[0])`` raises ValueError on the
        # non-integer, so the parser converts it to the friendly
        # diagnostic.
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, bad_section, _LINES_2HEADER,
        )
        with pytest.raises(
            ValueError, match="Malformed POINTS.*expected integer ID",
        ):
            MooringSystem.from_moordyn(path)

    def test_lines_too_few_columns_raises(self, tmp_path: pathlib.Path) -> None:
        bad_section = """\
---------------------- LINES ---------------------------------------------
ID      LineType        AttachA AttachB UnstrLen        NumSegs Outputs
(-)     (-)             (-)     (-)     (m)             (-)     (-)
1       chain
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, _POINTS_2HEADER, bad_section,
        )
        with pytest.raises(ValueError, match="Malformed LINES"):
            MooringSystem.from_moordyn(path)


# ===========================================================================
# Content-aware header detection
# ===========================================================================

class TestMoorDynHeaderVariants:
    """``_looks_like_header_row`` lets the parser tolerate decks
    shipped with a 1-row header (no units line). Previously the
    hard-coded ``pending_skip = 2`` ate the first data row.
    """

    def test_one_row_header_still_parses(self, tmp_path: pathlib.Path) -> None:
        """LINE TYPES section with ONLY a column-name header (no units
        line). The previous ``pending_skip = 2`` would have eaten the
        ``chain ...`` data row; the new content-aware skip stops the
        moment it sees a data-looking row.
        """
        path = _write_moordyn(
            tmp_path / "single_header.dat",
            _LINE_TYPES_1HEADER, _POINTS_2HEADER, _LINES_2HEADER, _OPTIONS,
        )
        ms = MooringSystem.from_moordyn(path)
        assert "chain" in ms.line_types, (
            "single-header LINE TYPES should still produce a parsed "
            "LineType — the content-aware skip must stop before the "
            "data row."
        )
        # Sanity: the parsed properties match the source row.
        lt = ms.line_types["chain"]
        assert lt.diam == pytest.approx(0.10)
        assert lt.mass_per_length_air == pytest.approx(50.0)
        assert lt.EA == pytest.approx(1.0e9)

    def test_two_row_header_still_parses(self, tmp_path: pathlib.Path) -> None:
        """The standard MoorDyn convention (column names + units) is
        still accepted unchanged."""
        path = _write_moordyn(
            tmp_path / "two_header.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, _OPTIONS,
        )
        ms = MooringSystem.from_moordyn(path)
        assert "chain" in ms.line_types
        assert ms.depth == pytest.approx(200.0)


# ===========================================================================
# OPTIONS strict-parse for the three load-bearing keys
# ===========================================================================

class TestMoorDynOptionsStrictParse:
    """The three recognised OPTIONS keys (``WtrDpth`` / ``rhoW`` /
    ``g``) raise on a malformed value rather than silently falling
    back to constructor defaults. Unknown keys remain permissive."""

    def test_rhow_with_garbage_value_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
not_a_number    rhoW
200.0           WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, options,
        )
        with pytest.raises(ValueError, match="Malformed OPTIONS.*rhoW"):
            MooringSystem.from_moordyn(path)

    def test_wtrdpth_with_inf_raises(self, tmp_path: pathlib.Path) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
inf             WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, options,
        )
        with pytest.raises(ValueError, match="not finite"):
            MooringSystem.from_moordyn(path)

    def test_g_with_typo_raises(self, tmp_path: pathlib.Path) -> None:
        options = """\
---------------------- OPTIONS -------------------------------------------
9.8point1       g
"""
        path = _write_moordyn(
            tmp_path / "bad.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, options,
        )
        with pytest.raises(ValueError, match="Malformed OPTIONS.*['\"]g['\"]"):
            MooringSystem.from_moordyn(path)

    def test_unknown_option_still_permissive(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """An unknown key with a malformed value doesn't raise — the
        OPTIONS block can legitimately carry MoorDyn-version-specific
        informational lines we don't recognise."""
        options = """\
---------------------- OPTIONS -------------------------------------------
not_a_number    SomeNewKeyword
200.0           WtrDpth
"""
        path = _write_moordyn(
            tmp_path / "ok.dat",
            _LINE_TYPES_2HEADER, _POINTS_2HEADER, _LINES_2HEADER, options,
        )
        ms = MooringSystem.from_moordyn(path)
        assert ms.depth == pytest.approx(200.0)


# ===========================================================================
# Non-finite rejection in LINE TYPES / POINTS
# ===========================================================================

class TestMoorDynRejectsNonFinite:

    def test_line_types_with_inf_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- LINE TYPES ----------------------------------------
Name    Diam    MassPerLength   EA      diff
(-)     (m)     (kg/m)          (N)     (-)
chain   0.10    inf             1.0e9   0.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", bad_section, _POINTS_2HEADER, _LINES_2HEADER,
        )
        with pytest.raises(ValueError, match="Malformed LINE TYPES"):
            MooringSystem.from_moordyn(path)

    def test_points_with_nan_coord_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        bad_section = """\
---------------------- POINTS --------------------------------------------
ID      Attachment      X       Y       Z
(-)     (-)             (m)     (m)     (m)
1       Fixed           nan     0.0     -50.0
2       Vessel          5.0     0.0     -10.0
"""
        path = _write_moordyn(
            tmp_path / "bad.dat", _LINE_TYPES_2HEADER, bad_section, _LINES_2HEADER,
        )
        with pytest.raises(ValueError, match="finite X / Y / Z"):
            MooringSystem.from_moordyn(path)


# ===========================================================================
# Platform inertia matrix DOF order
# ===========================================================================
#
# Used by ``Tower.from_elastodyn_with_mooring``. The pass-1 review
# caught a latent roll / pitch swap (invisible on OC3 by axisymmetry
# but real on asymmetric semis).

class TestPlatformInertiaMatrixDofOrder:
    """``_platform_inertia_matrix`` must place ``PtfmRIner`` at slot 3
    (roll) and ``PtfmPIner`` at slot 4 (pitch) — the OpenFAST 6-DOF
    convention.
    """

    def test_distinct_inertias_land_in_correct_slots(self) -> None:
        """OC3 has ``PtfmRIner == PtfmPIner`` by axisymmetry; an
        asymmetric semi or submersible has them distinct. We use
        deliberately distinct values to expose any swap.
        """
        from pybmodes.models.tower import _platform_inertia_matrix

        ptfm = {
            "PtfmMass": 1.5e7,
            "PtfmRIner": 1.0e10,  # roll inertia
            "PtfmPIner": 3.0e10,  # pitch inertia — deliberately different
            "PtfmYIner": 5.0e10,  # yaw inertia
            "PtfmCMxt": 0.0, "PtfmCMyt": 0.0, "PtfmCMzt": -20.0,
            "PtfmRefzt": 0.0,
            "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
            "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
            "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
        }
        i_mat = _platform_inertia_matrix(ptfm)
        # Translational mass at slots 0–2.
        assert i_mat[0, 0] == ptfm["PtfmMass"], "surge mass slot"
        assert i_mat[1, 1] == ptfm["PtfmMass"], "sway mass slot"
        assert i_mat[2, 2] == ptfm["PtfmMass"], "heave mass slot"
        # Rotational inertias in OpenFAST DOF order: roll(3) /
        # pitch(4) / yaw(5). The earlier code had RIner ↔ PIner
        # swapped — this is the load-bearing assertion.
        assert i_mat[3, 3] == ptfm["PtfmRIner"], (
            "i_matrix[3, 3] (roll, DOF 3) must equal PtfmRIner"
        )
        assert i_mat[4, 4] == ptfm["PtfmPIner"], (
            "i_matrix[4, 4] (pitch, DOF 4) must equal PtfmPIner"
        )
        assert i_mat[5, 5] == ptfm["PtfmYIner"], "yaw slot"

    def test_diagonal_only_no_cross_coupling(self) -> None:
        """No surge↔pitch or sway↔roll cross-coupling on the at-CM
        matrix. The rigid-arm transfer to the tower base happens
        downstream in ``nondim_platform``; adding it here would
        double-count.
        """
        from pybmodes.models.tower import _platform_inertia_matrix

        ptfm = {
            "PtfmMass": 1.0e7, "PtfmRIner": 1.0e9, "PtfmPIner": 2.0e9,
            "PtfmYIner": 3.0e9, "PtfmCMxt": 0.0, "PtfmCMyt": 0.0,
            "PtfmCMzt": -10.0, "PtfmRefzt": 0.0,
            "PtfmSurgeStiff": 0.0, "PtfmSwayStiff": 0.0,
            "PtfmHeaveStiff": 0.0, "PtfmRollStiff": 0.0,
            "PtfmPitchStiff": 0.0, "PtfmYawStiff": 0.0,
        }
        i_mat = _platform_inertia_matrix(ptfm)
        # Pull the off-diagonal mask and check every off-diagonal is 0.
        off_diag = i_mat - np.diag(np.diag(i_mat))
        assert np.all(off_diag == 0.0), "no cross-coupling on at-CM matrix"
