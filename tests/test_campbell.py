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

    requested = 4

    def fake_run_fem(bmi, *, n_modes, sp):  # noqa: ARG001
        # Simulate the asymmetric-K / general-eig fallback returning
        # fewer modes than requested (NaN-dropped eigenvalues).
        return _StubModalResult(
            frequencies=np.array([0.5, 1.2]),  # 2 < requested 4
            shapes=[],
        )

    monkeypatch.setattr(cb, "run_fem", fake_run_fem)

    # _solve_tower_once takes (tower, n_modes, n_steps) — pass a
    # placeholder pair for the tower since fake_run_fem ignores both.
    @dataclasses.dataclass
    class _StubBMI:
        rot_rpm: float = 0.0

    with pytest.raises(RuntimeError, match="too few|only \\d+ of"):
        cb._solve_tower_once((_StubBMI(), None), requested, n_steps=5)


# ---------------------------------------------------------------------------
# plot_campbell — 6-DOF platform rigid-body overlay (issue #39)
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


def _legend_texts(ax) -> list[str]:
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def _platform_lines(ax):
    return [ln for ln in ax.lines
            if ln.get_label().split(" ")[0] in _DOFS]


def test_plot_campbell_platform_modes_in_legend_distinct_styles() -> None:
    """Issue #47: each floating-platform rigid-body mode is a LEGEND
    entry (frequency + period) with its own colour and line style —
    not merged, not a right-margin annotation. The per-rev rays stay
    in the legend too."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from pybmodes.campbell import plot_campbell

    res = _small_campbell()
    plat = [
        ("surge", 0.0074), ("sway", 0.0081), ("heave", 0.0492),
        ("roll", 0.0338), ("pitch", 0.0345), ("yaw", 0.0114),
    ]
    fig = plot_campbell(res, excitation_orders=[1, 3, 6],
                        platform_modes=plat, log_freq=True)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    leg = _legend_texts(ax)
    # Every DOF: its own legend entry with Hz + period; none merged.
    for dof in _DOFS:
        assert any(t.startswith(dof + " (") and "Hz," in t and " s)" in t
                   for t in leg), dof
    assert not any("/" in t for t in leg)        # no merged pair labels
    for p in ("1P", "3P", "6P"):                 # rays kept in legend
        assert p in leg
    # Distinct (colour, linestyle) per platform line; nothing in the
    # right margin for them.
    pls = _platform_lines(ax)
    assert len(pls) == 6
    styles = {
        (tuple(np.round(to_rgb(ln.get_color()), 3)), ln.get_linestyle())
        for ln in pls
    }
    assert len(styles) == 6
    assert not any(t.get_text().strip().split(" ")[0] in _DOFS
                   for t in ax.texts)
    plt.close(fig)


def test_plot_campbell_default_unchanged_without_platform_modes() -> None:
    """Regression invariant: omitting platform_modes leaves the
    diagram on a linear axis with no platform overlay (byte-identical
    to the pre-existing behaviour)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_small_campbell())
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"
    # No platform lines / legend entries for a non-floating result.
    assert _platform_lines(ax) == []
    leg = _legend_texts(ax)
    assert not any(t.split(" ")[0] in _DOFS for t in leg)
    assert not any(" s)" in t for t in leg)   # no floater period labels
    plt.close(fig)


def test_plot_campbell_log_freq_excitation_rays_render(
) -> None:
    """Issue #47: under log_freq the per-rev rays must still render as
    the correct curve (the old two-point [0, rpm_max] sample collapsed
    on a log axis). The 1P line is now densely sampled, strictly
    positive, and tracks f = 1·rpm/60."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.campbell import plot_campbell

    fig = plot_campbell(_small_campbell(), excitation_orders=[1, 3],
                        log_freq=True)
    ax = fig.axes[0]
    one_p = [ln for ln in ax.lines if ln.get_label() == "1P"]
    assert len(one_p) == 1
    x = np.asarray(one_p[0].get_xdata(), dtype=float)
    y = np.asarray(one_p[0].get_ydata(), dtype=float)
    # Densely sampled (not the old 2-point segment) and log-safe.
    assert x.size >= 64
    assert np.all(y > 0.0)
    np.testing.assert_allclose(y, x / 60.0, rtol=1e-9)
    plt.close(fig)


def test_plot_campbell_native_platform_columns_go_to_legend() -> None:
    """Issue #47: tower columns the FEM classified as platform DOFs
    (surge/…/yaw, carried through CampbellResult.labels) become styled
    LEGEND lines with a period label, *without* the caller passing
    platform_modes by hand — and no flexible tower columns remain."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from pybmodes.campbell import plot_campbell

    res = _small_campbell()
    res.frequencies[:, 2] = 0.0074   # surge
    res.frequencies[:, 3] = 0.0492   # heave
    res.labels = ["1st flap", "1st edge", "surge", "heave"]

    fig = plot_campbell(res)   # no platform_modes passed
    ax = fig.axes[0]
    leg = _legend_texts(ax)
    assert any(t.startswith("surge (") for t in leg)
    assert any(t.startswith("heave (") for t in leg)
    assert all("Hz," in t and " s)" in t
               for t in leg if t.split(" ")[0] in _DOFS)
    pls = _platform_lines(ax)
    assert len(pls) == 2
    assert len({
        (tuple(np.round(to_rgb(ln.get_color()), 3)), ln.get_linestyle())
        for ln in pls
    }) == 2                          # distinct styles
    # No platform right-margin text; no flexible tower (grey) lines.
    assert not any(t.get_text().strip().split(" ")[0] in _DOFS
                   for t in ax.texts)
    grey = [
        ln for ln in ax.lines
        if np.allclose(to_rgb(ln.get_color()), (0.25, 0.25, 0.25),
                       atol=1e-3)
    ]
    assert grey == []
    plt.close(fig)


def test_plot_campbell_tower_labels_decluttered() -> None:
    """Issue #47 plot-iteration: the remaining *tower-bending*
    right-margin labels are spread apart (>= ~5 % of axis height
    between consecutive labels) so close modes don't stack."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from pybmodes.campbell import CampbellResult, plot_campbell

    n_steps = 4
    omega = np.linspace(0.0, 12.0, n_steps)
    freqs = np.empty((n_steps, 4))
    freqs[:, 0] = np.linspace(1.9, 2.0, n_steps)   # blade -> axis ~2 Hz
    freqs[:, 1] = 0.30                              # close tower modes
    freqs[:, 2] = 0.33
    freqs[:, 3] = 0.36
    parts = np.full((n_steps, 4, 3), 1.0 / 3.0)
    res = CampbellResult(
        omega_rpm=omega, frequencies=freqs,
        labels=["1st flap", "1st tower FA", "1st tower SS",
                "2nd tower FA"],
        participation=parts, n_blade_modes=1, n_tower_modes=3,
        mac_to_previous=np.full((n_steps, 4), np.nan),
    )
    fig = plot_campbell(res)        # linear axis
    ax = fig.axes[0]
    ymin, ymax = ax.get_ylim()
    grey = to_rgb((0.20, 0.20, 0.20))
    ys = sorted(
        t.get_position()[1] for t in ax.texts
        if np.allclose(to_rgb(t.get_color()), grey, atol=1e-3)
    )
    assert len(ys) >= 2
    fracs = [(y - ymin) / (ymax - ymin) for y in ys]
    assert np.all(np.diff(fracs) >= 0.045)   # decluttered, not stacked
    plt.close(fig)


def test_plot_campbell_skips_nonfinite_platform_freq() -> None:
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
    leg = _legend_texts(ax)
    assert any(t.startswith("surge (") for t in leg)
    assert not any(t.startswith(("bad", "zero")) for t in leg)
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
