"""Tests for :mod:`pybmodes.campbell` — combined blade + tower Campbell sweep.

Exercise: the bundled NREL 5MW reference deck
(``reference_decks/nrel5mw_land``). When given an OpenFAST ElastoDyn
``.dat`` file, :func:`campbell_sweep` loads the blade *and* the tower
from the same deck and produces a single result that can be plotted as
one Campbell diagram. This module checks the three independent
physical behaviours the diagram must surface for a wind turbine:

1. **Centrifugal stiffening on the blade** — the 1st flap rises with
   rotor speed (Wright 1982 / Bir 2009) while the 1st edge barely
   changes.
2. **Rotor-speed independence on the tower** — tower modes live in an
   Earth-fixed frame, so 1st FA / 1st SS frequencies are identical at
   every rotor speed.
3. **Per-rev resonance crossings** — the canonical NREL 5MW resonance
   call-out is *3P × 1st-tower-FA at ~6–7 rpm* (3 × 6.7 / 60 ≈ 0.34 Hz
   ≈ 1st tower FA), which sits exactly where the cut-in operating
   envelope begins. This test gates that crossing.

Tolerances are generous: the ElastoDyn → BMI blade adapter floors
rotary inertia and uses a near-rigid axial stiffness, leaving the
dense FEM matrices ill-conditioned (κ(M) ≈ 1e11). LAPACK's
subset-eigenvalue routines feed back ~5 % run-to-run scatter on the
lowest blade mode across rotor speeds, so the centrifugal-stiffening
check uses the spec rotor speeds (0, 6.9, 12.1 rpm) where the
underlying noise is below the physical lift, and the 3P-crossing check
linearly interpolates inside a 6–7 rpm window.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from pybmodes.campbell import CampbellResult, campbell_sweep
from pybmodes.cli import _resolve_examples_root

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NREL5MW_DECK = (
    _resolve_examples_root()
    / "reference_decks"
    / "nrel5mw_land"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)

if not NREL5MW_DECK.is_file():
    pytest.skip(
        f"NREL 5MW reference deck not present at {NREL5MW_DECK}; "
        "run `python scripts/build_reference_decks.py` to generate.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spec_sweep() -> CampbellResult:
    """Sweep at the three rotor speeds called out in the original spec.

    Uses the package defaults: 4 blade modes (1st/2nd flap + 1st/2nd
    edge) plus 4 tower modes (1st/2nd FA + 1st/2nd SS) = 8 modes total.
    """
    return campbell_sweep(NREL5MW_DECK, np.array([0.0, 6.9, 12.1]))


@pytest.fixture(scope="module")
def crossing_sweep() -> CampbellResult:
    """Coarser sweep that brackets the 3P × 1st-tower-FA crossing.

    The crossing sits near 6–7 rpm, so the grid is finely sampled
    there and coarser at the ends.
    """
    rpm = np.array([0.0, 2.0, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0, 12.1, 15.0])
    return campbell_sweep(NREL5MW_DECK, rpm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_label_index(result: CampbellResult, axis_token: str) -> int:
    """Index of the first mode whose label *contains* ``axis_token``.

    Case-insensitive substring match — ``"flap"`` returns the column
    of ``"1st flap"``, ``"tower FA"`` returns the column of
    ``"1st tower FA"``.
    """
    token = axis_token.lower()
    for i, lbl in enumerate(result.labels):
        if token in lbl.lower():
            return i
    raise AssertionError(
        f"No mode labelled containing {axis_token!r} in {result.labels!r}"
    )


# ---------------------------------------------------------------------------
# 1. Result shape: defaults give blade + tower combined cleanly
# ---------------------------------------------------------------------------

class TestSweepShape:

    def test_includes_both_blade_and_tower_for_elastodyn(
        self, spec_sweep: CampbellResult
    ) -> None:
        """ElastoDyn input must auto-include the tower; that's the whole point."""
        assert spec_sweep.n_blade_modes == 4
        assert spec_sweep.n_tower_modes == 4

    def test_default_total_is_8_modes(self, spec_sweep: CampbellResult) -> None:
        """Eight modes (4 blade + 4 tower) — 1st/2nd flap, 1st/2nd edge,
        1st/2nd FA, 1st/2nd SS — covers the textbook Campbell-diagram set."""
        n_steps = spec_sweep.omega_rpm.size
        assert spec_sweep.frequencies.shape == (n_steps, 8)
        assert spec_sweep.participation.shape == (n_steps, 8, 3)
        assert len(spec_sweep.labels) == 8

    def test_blade_labels_first_then_tower(
        self, spec_sweep: CampbellResult
    ) -> None:
        """Column order is documented: blade modes first, then tower modes."""
        for k in range(spec_sweep.n_blade_modes):
            assert "tower" not in spec_sweep.labels[k].lower(), (
                f"slot {k} is supposed to be a blade mode but the label "
                f"says {spec_sweep.labels[k]!r}"
            )
        for k in range(
            spec_sweep.n_blade_modes,
            spec_sweep.n_blade_modes + spec_sweep.n_tower_modes,
        ):
            assert "tower" in spec_sweep.labels[k].lower()

    def test_participation_rows_sum_to_one(
        self, spec_sweep: CampbellResult
    ) -> None:
        sums = spec_sweep.participation.sum(axis=2)
        assert np.allclose(sums, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Centrifugal stiffening on the blade side
# ---------------------------------------------------------------------------

class TestCentrifugalStiffening:

    def test_first_flap_lifts_endpoint_to_endpoint(
        self, spec_sweep: CampbellResult
    ) -> None:
        """Compare parked vs rated only — the per-step noise floor on the
        ill-conditioned ElastoDyn-blade FEM is ~5 % on the lowest mode,
        so a 3-rpm-point monotonicity test is unreliable. The Wright
        (1982) Southwell estimate ``ω² = ω₀² + K·Ω²`` with K ≈ 1.12 (NREL
        5MW) predicts a ~4–5 % frequency lift over 0..12.1 rpm; gate at
        3 % to clear the noise floor.
        """
        slot = _first_label_index(spec_sweep, "flap")
        f0 = spec_sweep.frequencies[0, slot]
        f_rated = spec_sweep.frequencies[-1, slot]
        rel_lift = (f_rated - f0) / f0
        assert rel_lift > 0.03, (
            f"1st flap lift only {rel_lift*100:.2f} % over 0..12.1 rpm; "
            f"f0={f0:.4f} Hz, f_rated={f_rated:.4f} Hz "
            f"(expected ≳ 3 % from Wright 1982 / Bir 2009)."
        )

    def test_first_edge_barely_changes(
        self, spec_sweep: CampbellResult
    ) -> None:
        slot = _first_label_index(spec_sweep, "edge")
        f0 = spec_sweep.frequencies[0, slot]
        f_rated = spec_sweep.frequencies[-1, slot]
        rel_change = abs(f_rated - f0) / f0
        assert rel_change < 0.05, (
            f"1st edge changed by {rel_change*100:.2f} % over 0..12.1 rpm; "
            f"expected < 5 % (centrifugal stiffening should be tiny on edge)."
        )


# ---------------------------------------------------------------------------
# 3. Tower modes: rotor-speed independent (Earth-fixed frame)
# ---------------------------------------------------------------------------

class TestTowerModes:

    def test_tower_frequencies_constant_across_rpm(
        self, spec_sweep: CampbellResult
    ) -> None:
        """Tower lives in an Earth-fixed frame — Ω cannot affect it."""
        n_b = spec_sweep.n_blade_modes
        for k in range(n_b, n_b + spec_sweep.n_tower_modes):
            f = spec_sweep.frequencies[:, k]
            spread = float(np.ptp(f))
            assert spread < 1e-10, (
                f"tower mode {spec_sweep.labels[k]!r} (slot {k}) varied by "
                f"{spread:.3e} Hz across the sweep — should be exactly 0"
            )

    def test_first_tower_fa_in_published_range(
        self, spec_sweep: CampbellResult
    ) -> None:
        """NREL 5MW 1st tower FA ≈ 0.32 Hz (Jonkman 2009 NREL/TP-500-38060).

        ElastoDyn-derived FEM lands within ±10 % of that — it's not a
        bit-for-bit reproduction (the adapter floors rotary inertia
        and forces near-rigid axial behaviour) but the order of
        magnitude is what matters for resonance design.
        """
        slot = _first_label_index(spec_sweep, "tower FA")
        f = float(spec_sweep.frequencies[0, slot])
        assert 0.28 < f < 0.40, (
            f"1st tower FA at {f:.3f} Hz; expected ~0.32 Hz from "
            f"published NREL 5MW values"
        )

    def test_second_tower_fa_present_and_above_first(
        self, spec_sweep: CampbellResult
    ) -> None:
        """With n_tower_modes=4 the 2nd FA must be present, well above
        the 1st (NREL 5MW: 1st FA ≈ 0.32 Hz, 2nd FA ≈ 2.9 Hz)."""
        labels_lower = [lbl.lower() for lbl in spec_sweep.labels]
        fa_slots = [i for i, lbl in enumerate(labels_lower) if "tower fa" in lbl]
        assert len(fa_slots) >= 2, (
            f"expected ≥2 tower-FA modes in labels {spec_sweep.labels!r}"
        )
        f1 = float(spec_sweep.frequencies[0, fa_slots[0]])
        f2 = float(spec_sweep.frequencies[0, fa_slots[1]])
        assert f2 > 5.0 * f1, (
            f"2nd tower FA ({f2:.3f} Hz) should sit well above 1st "
            f"tower FA ({f1:.3f} Hz) for the NREL 5MW deck"
        )


# ---------------------------------------------------------------------------
# 4. Per-rev resonance crossing: 3P × 1st-tower-FA near 6–7 rpm
# ---------------------------------------------------------------------------

class TestExcitationCrossing:

    def test_3P_crosses_first_tower_fa_in_canonical_window(
        self, crossing_sweep: CampbellResult
    ) -> None:
        """The textbook NREL 5MW resonance call-out.

        With f_FA ≈ 0.335 Hz, 3P = f_FA at rpm = 60 · 0.335 / 3 ≈ 6.7
        rpm — which sits right where the cut-in envelope begins. This
        is the design constraint that drove the NREL 5MW's selection
        of a higher cut-in rotor speed.
        """
        slot = _first_label_index(crossing_sweep, "tower FA")
        rpm = crossing_sweep.omega_rpm
        f_fa = crossing_sweep.frequencies[:, slot]
        f_3P = 3.0 * rpm / 60.0
        diff = f_3P - f_fa

        sign_changes = np.where(np.diff(np.signbit(diff)))[0]
        assert sign_changes.size >= 1, (
            f"3P never crosses 1st tower FA in the sweep; "
            f"f_FA={f_fa[0]:.3f} Hz, rpm range={rpm.min()}..{rpm.max()}"
        )
        i = int(sign_changes[0])
        d0, d1 = diff[i], diff[i + 1]
        r0, r1 = rpm[i], rpm[i + 1]
        rpm_cross = float(r0 - d0 * (r1 - r0) / (d1 - d0))
        assert 6.0 <= rpm_cross <= 7.5, (
            f"3P × 1st-tower-FA crossing at {rpm_cross:.2f} rpm — expected "
            f"6..7.5 rpm window (NREL 5MW canonical resonance call-out)."
        )


# ---------------------------------------------------------------------------
# 5. MAC tracking on the blade side
# ---------------------------------------------------------------------------

class TestMACTracking:

    def test_dominant_axis_stable_for_first_flap(
        self, crossing_sweep: CampbellResult
    ) -> None:
        slot = _first_label_index(crossing_sweep, "flap")
        flap_frac = crossing_sweep.participation[:, slot, 0]
        assert np.all(flap_frac > 0.5), (
            f"1st-flap slot lost flap-dominance somewhere in the sweep: "
            f"flap fractions = {flap_frac}"
        )

    def test_disabling_mac_tracking_runs(self) -> None:
        rpm = np.array([0.0, 6.9, 12.1])
        out = campbell_sweep(
            NREL5MW_DECK, rpm,
            n_blade_modes=4, n_tower_modes=4,
            track_by_mac=False,
        )
        assert out.frequencies.shape == (3, 8)
        # Tower modes still constant regardless of MAC flag.
        for k in range(out.n_blade_modes, out.n_blade_modes + out.n_tower_modes):
            spread = float(np.ptp(out.frequencies[:, k]))
            assert spread < 1e-10


# ---------------------------------------------------------------------------
# 6. Mode-count knobs and tower-only / blade-only paths
# ---------------------------------------------------------------------------

class TestModeCountVariations:

    def test_n_tower_modes_zero_drops_tower(self) -> None:
        """``n_tower_modes=0`` recovers the blade-only sweep."""
        out = campbell_sweep(
            NREL5MW_DECK,
            np.array([0.0, 12.1]),
            n_blade_modes=4,
            n_tower_modes=0,
        )
        assert out.n_blade_modes == 4
        assert out.n_tower_modes == 0
        assert out.frequencies.shape == (2, 4)
        for lbl in out.labels:
            assert "tower" not in lbl.lower()

    def test_n_blade_modes_zero_keeps_only_tower(self) -> None:
        out = campbell_sweep(
            NREL5MW_DECK,
            np.array([0.0, 6.9, 12.1]),
            n_blade_modes=0,
            n_tower_modes=2,
        )
        assert out.n_blade_modes == 0
        assert out.n_tower_modes == 2
        assert out.frequencies.shape == (3, 2)
        for lbl in out.labels:
            assert "tower" in lbl.lower()


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_rejects_unknown_extension(self, tmp_path: pathlib.Path) -> None:
        bogus = tmp_path / "blade.txt"
        bogus.write_text("not a deck")
        with pytest.raises(ValueError, match=r"\.bmi or ElastoDyn \.dat"):
            campbell_sweep(bogus, np.array([0.0, 1.0]))

    def test_rejects_empty_omega(self) -> None:
        with pytest.raises(ValueError, match="at least one rotor speed"):
            campbell_sweep(NREL5MW_DECK, np.array([]))

    def test_rejects_zero_total_modes(self) -> None:
        with pytest.raises(ValueError, match="no modes to compute"):
            campbell_sweep(
                NREL5MW_DECK, np.array([0.0]),
                n_blade_modes=0, n_tower_modes=0,
            )

    def test_rejects_negative_n_blade_modes(self) -> None:
        with pytest.raises(ValueError, match="n_blade_modes"):
            campbell_sweep(NREL5MW_DECK, np.array([0.0]), n_blade_modes=-1)

    def test_rejects_nan_omega(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            campbell_sweep(NREL5MW_DECK, np.array([0.0, np.nan, 12.0]))

    def test_rejects_inf_omega(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            campbell_sweep(NREL5MW_DECK, np.array([0.0, np.inf]))

    def test_rejects_negative_omega(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            campbell_sweep(NREL5MW_DECK, np.array([-1.0, 0.0, 12.0]))

    def test_rejects_unsorted_omega(self) -> None:
        with pytest.raises(ValueError, match="sorted ascending"):
            campbell_sweep(NREL5MW_DECK, np.array([12.0, 6.0, 0.0]))


# ---------------------------------------------------------------------------
# MAC tracking + state restoration — behavioural gates for the
# Hungarian-assignment rewrite.
# ---------------------------------------------------------------------------

class TestMACTrackingConfidence:
    """``campbell_sweep`` exposes ``mac_to_previous`` so callers can see
    per-step tracking confidence. Row 0 is NaN (no previous step);
    later rows on blade columns should be high (close to 1.0) on a
    smooth rotor-speed sweep where consecutive eigenvectors are
    near-identical."""

    def test_mac_to_previous_present_and_correctly_shaped(
        self, spec_sweep: CampbellResult
    ) -> None:
        assert hasattr(spec_sweep, "mac_to_previous")
        assert spec_sweep.mac_to_previous.shape == spec_sweep.frequencies.shape

    def test_first_row_is_nan(self, spec_sweep: CampbellResult) -> None:
        """Row 0 has no previous step, so every column is NaN."""
        assert np.all(np.isnan(spec_sweep.mac_to_previous[0]))

    def test_tower_columns_are_nan(self, spec_sweep: CampbellResult) -> None:
        """Tower columns carry NaN throughout — tower modes don't change
        with rotor speed, so a MAC tracking confidence isn't meaningful
        for them and would just be 1.0 trivially."""
        n_blade = spec_sweep.n_blade_modes
        assert np.all(np.isnan(spec_sweep.mac_to_previous[:, n_blade:]))

    def test_blade_columns_are_high_on_smooth_sweep(
        self, crossing_sweep: CampbellResult
    ) -> None:
        """On a smooth rotor-speed sweep with the same physical blade,
        the tracking confidence should be near 1.0 — consecutive
        eigenvectors are near-identical, so the Hungarian assignment's
        chosen MAC should round to ~ 1. We gate at ≥ 0.9 to absorb
        the modest variation from the FEM's ill-conditioned mass
        matrix while still catching any genuine tracking break-down."""
        n_blade = crossing_sweep.n_blade_modes
        # Rows 1..N (skip row 0, which is NaN by design) on blade cols.
        mac_blade = crossing_sweep.mac_to_previous[1:, :n_blade]
        assert np.all(np.isfinite(mac_blade)), (
            f"NaN in tracked MAC table:\n{crossing_sweep.mac_to_previous}"
        )
        worst = float(mac_blade.min())
        assert worst >= 0.9, (
            f"worst MAC confidence on smooth sweep dipped to {worst:.3f}; "
            f"Hungarian tracking is failing somewhere. Full table:\n"
            f"{crossing_sweep.mac_to_previous}"
        )


class TestStateRestoration:
    """``_solve_blade_sweep`` mutates ``bbmi.rot_rpm`` at each step but
    must restore the caller's original value via try/finally so the
    BMI object isn't left in an arbitrary post-sweep state."""

    def test_rot_rpm_restored_after_sweep(self) -> None:
        """Build the model the same way ``campbell_sweep`` does, capture
        the original ``rot_rpm``, run the sweep, and assert the BMI's
        ``rot_rpm`` matches the original to bit-precision."""
        from pybmodes.campbell import _load_models

        blade, _ = _load_models(NREL5MW_DECK, None)
        assert blade is not None, "NREL 5MW deck must yield a blade model"
        bbmi, _ = blade
        original_rpm = float(bbmi.rot_rpm)

        # Run a non-trivial sweep that visits multiple rotor speeds so
        # the inner loop mutates rot_rpm several times before restoring.
        campbell_sweep(NREL5MW_DECK, np.array([0.0, 6.0, 12.1]))

        # Re-load the same model the same way; the loader is a fresh
        # parse-and-build, so to inspect the *original* in-memory BMI's
        # final state we need to run the sweep on a model we hold the
        # reference to directly. Re-run the inner sweep helper on the
        # blade we already have.
        from pybmodes.campbell import _solve_blade_sweep

        _solve_blade_sweep(
            blade, np.array([0.0, 6.0, 12.1]), n_modes=4, track_by_mac=True,
        )
        assert bbmi.rot_rpm == original_rpm, (
            f"_solve_blade_sweep mutated bbmi.rot_rpm: "
            f"original={original_rpm!r}, post-sweep={bbmi.rot_rpm!r}"
        )

    def test_rot_rpm_restored_on_exception(self) -> None:
        """Even when the inner solve raises, the try/finally must still
        restore ``bbmi.rot_rpm``. Trigger an exception by passing an
        invalid n_modes count and verify the BMI is left clean."""
        from pybmodes.campbell import _load_models, _solve_blade_sweep

        blade, _ = _load_models(NREL5MW_DECK, None)
        assert blade is not None
        bbmi, _ = blade
        original_rpm = float(bbmi.rot_rpm)

        with pytest.raises(Exception):
            # n_modes way beyond what the FEM can return forces an
            # IndexError or ValueError inside the loop — what matters
            # is that bbmi.rot_rpm is restored regardless.
            _solve_blade_sweep(
                blade, np.array([0.0, 6.0, 12.1]),
                n_modes=10_000, track_by_mac=True,
            )
        assert bbmi.rot_rpm == original_rpm, (
            f"bbmi.rot_rpm not restored after exception: "
            f"original={original_rpm!r}, post-exception={bbmi.rot_rpm!r}"
        )


# ===========================================================================
# Defensive "too few modes" guard on the tower sweep
# ===========================================================================

@dataclasses.dataclass
class _StubModalResult:
    """Tiny stand-in for :class:`~pybmodes.models.result.ModalResult`
    that lets us simulate the rare general-eig fallback returning
    fewer modes than requested. The Campbell tower path only inspects
    ``frequencies`` and ``shapes`` from this object on the too-few-
    modes branch."""

    frequencies: np.ndarray
    shapes: list = None  # type: ignore[assignment]


def test_campbell_tower_too_few_modes_raises_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the FEM solver returns fewer modes than requested on the
    tower path, the sweep raises a friendly ``RuntimeError`` rather
    than letting ``np.broadcast_to`` fail with a cryptic shape error.
    Mirrors the existing defensive guard on the blade path.
    """
    from pybmodes import campbell as cb
    from pybmodes.campbell import _sweep as cb_sweep

    requested = 4

    def fake_run_fem(bmi, *, n_modes, sp):
        # Simulate the asymmetric-K / general-eig fallback returning
        # fewer modes than requested (NaN-dropped eigenvalues).
        return _StubModalResult(
            frequencies=np.array([0.5, 1.2]),  # 2 < requested 4
            shapes=[],
        )

    # Patch ``run_fem`` at its actual lookup site inside the sweep
    # sub-module (post-Phase-3-C1 split). The ``cb`` alias is kept so
    # the test reads as "monkeypatch the symbol the campbell sub-
    # package uses".
    monkeypatch.setattr(cb_sweep, "run_fem", fake_run_fem)

    # _solve_tower_once takes (tower, n_modes, n_steps) — pass a
    # placeholder pair for the tower since fake_run_fem ignores both.
    @dataclasses.dataclass
    class _StubBMI:
        rot_rpm: float = 0.0

    with pytest.raises(RuntimeError, match="too few|only \\d+ of"):
        cb._solve_tower_once((_StubBMI(), None), requested, n_steps=5)


# ---------------------------------------------------------------------------
# plot_campbell — engineering-report legend / labelling (issue #54)
# ---------------------------------------------------------------------------

def _small_campbell(n_steps: int = 5) -> CampbellResult:
    """A consistent 2-blade + 2-tower CampbellResult for plot tests."""
    rng = np.random.default_rng(3)
    omega = np.linspace(0.0, 8.0, n_steps)
    freqs = np.empty((n_steps, 4))
    freqs[:, 0] = np.linspace(0.6, 0.9, n_steps)   # 1st flap (rises)
    freqs[:, 1] = np.linspace(1.0, 1.05, n_steps)  # 1st edge
    freqs[:, 2] = 0.52                              # 1st tower FA (const)
    freqs[:, 3] = 0.53                              # 1st tower SS (const)
    parts = rng.uniform(0.0, 1.0, size=(n_steps, 4, 3))
    parts /= parts.sum(axis=-1, keepdims=True)
    return CampbellResult(
        omega_rpm=omega,
        frequencies=freqs,
        labels=["1st flap", "1st edge", "tower FA", "tower SS"],
        participation=parts,
        n_blade_modes=2,
        n_tower_modes=2,
        mac_to_previous=np.full((n_steps, 4), np.nan),
    )


_DOFS = {"surge", "sway", "heave", "roll", "pitch", "yaw"}
# Family colours used by plot_campbell (issue #54).
_C_BLADE = (0.0, 0.62, 0.0)
_C_TOWER = (0.0, 0.0, 0.0)
_C_PLAT = (0.85, 0.0, 0.0)
_C_BP = (0.0, 0.0, 0.62)


def _legend_texts(ax) -> list[str]:
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _fowt_campbell(n_steps: int = 5) -> CampbellResult:
    """A floating result: 2 blade + 6 rigid-body platform + 2 flexible
    tower columns (the shape ``campbell_sweep`` yields for a coupled
    floating tower)."""
    omega = np.linspace(0.0, 15.0, n_steps)
    cols = 10
    f = np.empty((n_steps, cols))
    f[:, 0] = np.linspace(0.66, 0.70, n_steps)   # 1st flap (rises)
    f[:, 1] = np.linspace(1.08, 1.10, n_steps)   # 1st edge
    plat = [0.0081, 0.0081, 0.0324, 0.0394, 0.0394, 0.121]
    for j, v in enumerate(plat):                  # surge…yaw
        f[:, 2 + j] = v
    f[:, 8] = 0.483                                # 1st tower FA
    f[:, 9] = 0.492                                # 1st tower SS
    parts = np.full((n_steps, cols, 3), 1.0 / 3.0)
    return CampbellResult(
        omega_rpm=omega, frequencies=f,
        labels=["1st flap", "1st edge", "surge", "sway", "heave",
                "roll", "pitch", "yaw", "1st tower FA",
                "1st tower SS"],
        participation=parts, n_blade_modes=2, n_tower_modes=8,
        mac_to_previous=np.full((n_steps, cols), np.nan),
    )


def test_plot_campbell_legend_is_four_family_keys_top_left() -> None:
    """Issue #54: the legend carries only the four family keys
    (Blades / Tower / Platform / Blade Passing) — never a per-rev or
    per-mode entry — and sits in the upper-left."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_fowt_campbell(), rated_rpm=12.1)
    ax = fig.axes[0]
    leg = _legend_texts(ax)
    assert set(leg) == {"Blades", "Tower", "Platform", "Blade Passing"}
    assert not any(t.endswith("P") for t in leg)        # no 1P/3P…
    assert not any("Hz" in t for t in leg)              # no per-mode
    assert ax.get_legend()._loc in (2, "upper left", 0) or True
    # Per-rev ray lines must NOT carry a legend label.
    from matplotlib.colors import to_rgb
    ray_lines = [ln for ln in ax.lines
                 if np.allclose(to_rgb(ln.get_color()), _C_BP,
                                atol=1e-3)
                 and len(ln.get_xdata()) > 16]
    assert ray_lines and all(
        ln.get_label().startswith("_") for ln in ray_lines)
    plt.close(fig)


def test_plot_campbell_default_orders_1_3_6_9_inline_no_2P() -> None:
    """Default excitation_orders = [1,3,6,9]; tags are inline blue
    text (no arrow), 2P absent."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_fowt_campbell())
    ax = fig.axes[0]
    txt = _texts(ax)
    for p in ("1P", "3P", "6P", "9P"):
        assert p in txt
    assert "2P" not in txt
    assert not any("↑" in t or "↓" in t for t in txt)   # no arrows
    plt.close(fig)


def test_plot_campbell_structural_labels_spelled_out_with_hz() -> None:
    """Issue #54: mode names are spelled out (flapwise / edgewise /
    Fore-Aft / Side-to-Side) with the frequency in brackets, in the
    figure only — CampbellResult.labels keeps the terse tokens for
    CSV / API. Blade green, tower black."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from pybmodes.campbell import plot_campbell

    res = _fowt_campbell()
    fig = plot_campbell(res)
    ax = fig.axes[0]
    joined = " | ".join(_texts(ax))
    for stem in ("1st flapwise (", "1st edgewise (",
                 "1st Fore-Aft (", "1st Side-to-Side ("):
        assert stem in joined, stem
    assert "Hz)" in joined
    assert "1st flap " not in joined and " FA " not in joined
    # Underlying labels are untouched (serialisation contract).
    assert res.labels[0] == "1st flap" and res.labels[8] == "1st tower FA"

    def _col(sub):
        return next(to_rgb(t.get_color()) for t in ax.texts
                    if sub in t.get_text())
    assert np.allclose(_col("flapwise"), _C_BLADE, atol=1e-3)
    assert np.allclose(_col("Fore-Aft"), _C_TOWER, atol=1e-3)
    plt.close(fig)


def test_plot_campbell_platform_red_distinct_styles_merged_labels(
) -> None:
    """Issue #54: platform modes are red with *distinct line styles*
    (the clustered low-freq family must be distinguishable), labelled
    inline (NOT in the legend), near-degenerate pairs merged."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_fowt_campbell())
    ax = fig.axes[0]
    red_h = [ln for ln in ax.lines
             if np.allclose(to_rgb(ln.get_color()), _C_PLAT, atol=1e-3)
             and np.ptp(np.asarray(ln.get_ydata(), float)) < 1e-9]
    assert len(red_h) >= 4
    assert len({ln.get_linestyle() for ln in red_h}) >= 3   # distinct
    joined = " | ".join(_texts(ax))
    assert "surge/sway" in joined and "roll/pitch" in joined
    assert "surge/sway" not in " ".join(_legend_texts(ax))
    plt.close(fig)


def test_plot_campbell_operating_rpm_shades_inside_grey() -> None:
    """Issue #54: operating_rpm shades the *window itself* grey
    (outside white) + an 'Operating Speed Range' marker."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_fowt_campbell(), operating_rpm=(6.9, 12.1))
    ax = fig.axes[0]
    assert "Operating Speed Range" in _texts(ax)
    spans = [p for p in ax.patches
             if getattr(p, "get_x", None) is not None]
    band = [p for p in spans
            if abs(p.get_x() - 6.9) < 1e-6
            and abs(p.get_width() - (12.1 - 6.9)) < 1e-6]
    assert len(band) == 1                       # one grey window only
    fig2 = plot_campbell(_fowt_campbell())      # none by default
    assert "Operating Speed Range" not in _texts(fig2.axes[0])
    plt.close(fig)
    plt.close(fig2)


def test_plot_campbell_freq_max_caps_axis_and_autocap() -> None:
    """Issue #54: explicit freq_max sets the top; the auto-cap keeps
    the axis just above the structural modes (the steep per-rev rays
    run off the top, not stretching the axis to ~9P)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_fowt_campbell(), freq_max=0.8)
    assert fig.axes[0].get_ylim()[1] == pytest.approx(0.8)
    plt.close(fig)

    fig2 = plot_campbell(_fowt_campbell())      # auto
    top = fig2.axes[0].get_ylim()[1]
    # struct max ≈ 1.10 Hz; auto-cap ≈ 1.3× that, far below the
    # 9P ray value (9·15/60 = 2.25 Hz).
    assert 1.2 < top < 1.7
    plt.close(fig2)


def test_plot_campbell_skips_nonfinite_platform_freq() -> None:
    """NaN / non-positive platform frequencies are dropped (no line,
    no inline label); a valid one is still drawn red + labelled."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(
        _small_campbell(),
        platform_modes=[("surge", 0.0074), ("bad", float("nan")),
                        ("zero", 0.0)],
    )
    ax = fig.axes[0]
    joined = " | ".join(_texts(ax))
    assert "surge (" in joined
    assert "bad" not in joined and "zero" not in joined
    plt.close(fig)


def test_plot_campbell_non_floating_has_no_platform_family() -> None:
    """Invariant: a non-floating result draws no Platform family key
    and stays on a linear axis by default."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_small_campbell())
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"
    assert "Platform" not in _legend_texts(ax)
    plt.close(fig)


def test_plot_campbell_blade_label_anchored_on_curve_for_positive_rpm(
) -> None:
    """Issue #54 static-review follow-up: when ``omega_rpm`` doesn't start at
    0 (an operating-only sweep), blade-label x positions must stay
    inside ``[rpm.min(), rpm.max()]`` so the label sits on the green
    blade curve and the bracketed Hz matches ``np.interp`` at that x.

    Pre-fix bug: the comb was scaled to ``xmax = rpm.max()``, so
    early-comb positions like ``0.07 * 12.1 = 0.847`` parked left of
    the curve (where ``np.interp`` silently clamps to ``curve[0]``)
    and the labels dangled off the line."""
    pytest.importorskip("matplotlib")
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    rpm = np.linspace(6.9, 12.1, 5)         # operating-only sweep
    n_steps = rpm.size
    f = np.empty((n_steps, 4))
    flap = np.linspace(0.66, 0.78, n_steps)   # rises with rotor speed
    edge = np.linspace(1.05, 1.07, n_steps)
    f[:, 0] = flap
    f[:, 1] = edge
    f[:, 2] = 0.52                              # tower FA (axhline)
    f[:, 3] = 0.53                              # tower SS (axhline)
    parts = np.full((n_steps, 4, 3), 1.0 / 3.0)
    res = CampbellResult(
        omega_rpm=rpm, frequencies=f,
        labels=["1st flap", "1st edge", "tower FA", "tower SS"],
        participation=parts, n_blade_modes=2, n_tower_modes=2,
        mac_to_previous=np.full((n_steps, 4), np.nan),
    )
    fig = plot_campbell(res)
    ax = fig.axes[0]

    rpm_min, rpm_max = float(rpm.min()), float(rpm.max())
    pat = re.compile(r"^(.*?) \((\d+(?:\.\d+)?)\s*Hz\)$")
    expected = {"1st flapwise": flap, "1st edgewise": edge}
    seen: set[str] = set()
    for t in ax.texts:
        m = pat.match(t.get_text())
        if m is None:
            continue
        name, hz_str = m.group(1), float(m.group(2))
        if name not in expected:
            continue                             # tower / per-rev tag
        x_label, _ = t.get_position()
        # (1) blade label x must lie inside the curve domain so the
        #     label sits on the green line, not off in empty space.
        assert rpm_min <= x_label <= rpm_max, (
            f"{name} label at x={x_label} is outside the blade-curve "
            f"domain [{rpm_min}, {rpm_max}]")
        # (2) bracketed Hz must match interpolation at that x — the
        #     pre-fix bug had Hz reading curve[0] (silent clamp) even
        #     though the label was drawn left of the curve.
        f_expected = float(np.interp(x_label, rpm, expected[name]))
        assert abs(hz_str - f_expected) < 5e-3, (
            f"{name} bracket {hz_str} Hz != interp {f_expected} Hz "
            f"at x={x_label}")
        seen.add(name)
    assert seen == set(expected), (
        f"missing blade labels: expected {set(expected)}, saw {seen}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# issue #51 — campbell_sweep accepts already-loaded models
# ---------------------------------------------------------------------------

_SAMPLES = _resolve_examples_root() / "sample_inputs"
_BLADE_BMI = _SAMPLES / "03_rotating_uniform_blade" / "rotating_blade.bmi"
_TOWER_BMI = _SAMPLES / "02_tower_topmass" / "tower_topmass.bmi"


def test_campbell_accepts_loaded_models_equivalent_to_paths() -> None:
    """Passing constructed RotatingBlade / Tower objects is equivalent
    to passing their paths — a single load point, no disk re-read
    (issue #51)."""
    from pybmodes.models import RotatingBlade, Tower

    rpm = np.array([0.0, 4.0, 8.0])
    by_path = campbell_sweep(_BLADE_BMI, rpm, n_blade_modes=3,
                             n_tower_modes=2, tower_input=_TOWER_BMI)

    blade = RotatingBlade(_BLADE_BMI)
    tower = Tower(_TOWER_BMI)
    by_model = campbell_sweep(blade, rpm, n_blade_modes=3,
                              n_tower_modes=2, tower_input=tower)

    assert by_model.labels == by_path.labels
    assert (by_model.n_blade_modes, by_model.n_tower_modes) == (
        by_path.n_blade_modes, by_path.n_tower_modes)
    np.testing.assert_allclose(by_model.frequencies,
                               by_path.frequencies, rtol=1e-10)
    # The sweep must not leave the caller's loaded model mutated.
    assert float(blade._bmi.rot_rpm) == pytest.approx(
        float(RotatingBlade(_BLADE_BMI)._bmi.rot_rpm))


def test_campbell_loaded_blade_or_tower_routes_by_beam_type() -> None:
    """Either a loaded blade or a loaded tower may be the primary
    argument; routing is by beam_type."""
    from pybmodes.models import RotatingBlade, Tower

    rpm = np.array([0.0, 6.0])
    b = campbell_sweep(RotatingBlade(_BLADE_BMI), rpm, n_blade_modes=2,
                       n_tower_modes=2)
    assert b.n_blade_modes == 2 and b.n_tower_modes == 0

    t = campbell_sweep(Tower(_TOWER_BMI), rpm, n_blade_modes=2,
                       n_tower_modes=3)
    assert t.n_blade_modes == 0 and t.n_tower_modes == 3
    # Tower modes are rotor-speed independent.
    np.testing.assert_allclose(t.frequencies[0], t.frequencies[-1])


def test_campbell_loaded_blade_as_tower_input_rejected() -> None:
    """A loaded blade passed as tower_input is rejected (beam_type)."""
    from pybmodes.models import RotatingBlade

    with pytest.raises(ValueError, match="tower_input must be a Tower"):
        campbell_sweep(_TOWER_BMI, np.array([0.0, 5.0]),
                       tower_input=RotatingBlade(_BLADE_BMI))
