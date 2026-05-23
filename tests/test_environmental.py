"""Environmental-spectra closed forms + frequency-placement diagram.

The two spectrum helpers are validated against their analytic
properties (Kaimal low-frequency plateau + monotonicity; JONSWAP
peak location + the ``m0 = Hs**2 / 16`` spectral-moment identity);
the diagram itself gets a structural smoke test (skipped when
matplotlib is absent).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.plots.environmental import jonswap_spectrum, kaimal_spectrum

# ---------------------------------------------------------------------------
# Kaimal wind spectrum
# ---------------------------------------------------------------------------


def test_kaimal_low_frequency_plateau_and_monotone() -> None:
    U, L, sig = 11.0, 340.2, 1.5
    f = np.linspace(0.0, 0.6, 400)
    s = kaimal_spectrum(f, mean_speed=U, length_scale=L, sigma=sig)
    # S(0) = 4 sigma^2 L / U exactly.
    assert s[0] == pytest.approx(4.0 * sig**2 * L / U, rel=1e-12)
    # Strictly decreasing in f (turbulence energy bleeds off).
    assert np.all(np.diff(s) < 0.0)


def test_kaimal_default_sigma_from_turbulence_intensity() -> None:
    s_default = kaimal_spectrum(
        np.array([0.0]), mean_speed=10.0, length_scale=300.0,
        turbulence_intensity=0.14,
    )
    s_explicit = kaimal_spectrum(
        np.array([0.0]), mean_speed=10.0, length_scale=300.0, sigma=1.4,
    )
    assert s_default[0] == pytest.approx(s_explicit[0], rel=1e-12)


def test_kaimal_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError, match="mean_speed"):
        kaimal_spectrum(np.array([0.1]), mean_speed=0.0, length_scale=300.0)
    with pytest.raises(ValueError, match="length_scale"):
        kaimal_spectrum(np.array([0.1]), mean_speed=10.0, length_scale=-1.0)


# ---------------------------------------------------------------------------
# JONSWAP wave spectrum
# ---------------------------------------------------------------------------


def test_jonswap_peak_at_inverse_tp() -> None:
    hs, tp = 6.0, 10.0
    f = np.linspace(1e-4, 0.6, 6000)
    s = jonswap_spectrum(f, hs=hs, tp=tp)
    f_peak = f[int(np.argmax(s))]
    assert f_peak == pytest.approx(1.0 / tp, abs=2e-3)
    assert np.all(s >= 0.0)


def test_jonswap_zeroth_moment_matches_hs() -> None:
    """Integral of S(f) df = m0 = Hs**2 / 16 (the JONSWAP Hs
    identity) by construction, to grid resolution."""
    hs, tp = 6.0, 10.0
    f = np.linspace(1e-4, 2.0, 40000)
    s = jonswap_spectrum(f, hs=hs, tp=tp)
    m0 = float(0.5 * np.sum((s[1:] + s[:-1]) * np.diff(f)))
    assert m0 == pytest.approx(hs**2 / 16.0, rel=0.02)


def test_jonswap_zero_below_dc_and_input_guards() -> None:
    s = jonswap_spectrum(np.array([-0.1, 0.0, 0.1]), hs=4.0, tp=8.0)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] > 0.0
    with pytest.raises(ValueError, match="hs"):
        jonswap_spectrum(np.array([0.1]), hs=0.0, tp=8.0)
    with pytest.raises(ValueError, match="gamma"):
        jonswap_spectrum(np.array([0.1]), hs=4.0, tp=8.0, gamma=0.5)


# ---------------------------------------------------------------------------
# Frequency-placement diagram (matplotlib smoke)
# ---------------------------------------------------------------------------


def test_plot_environmental_spectra_structure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.plots import plot_environmental_spectra

    fig = plot_environmental_spectra(
        tower_fa_hz=0.50,
        tower_ss_hz=0.48,
        rpm_design=(5.0, 7.5),
        harmonics=(1, 3),
        wind={"mean_speed": 11.0, "length_scale": 340.2},
        wave={"hs": 6.0, "tp": 10.0},
        title="placement",
    )
    ax = fig.axes[0]
    # 1P+3P x (design hatched + constraint) = 4 axvspans; 2 tower lines.
    assert len(list(ax.patches)) >= 4
    assert len(ax.lines) >= 2 + 2  # 2 tower vlines + wind + wave curves
    leg = ax.get_legend()
    txt = {t.get_text() for t in leg.get_texts()}
    assert {"1P Design", "3P Design", "Tower, 1st Fore-Aft Bending",
            "Tower, 1st Side-to-Side Bending", "Waves, JONSWAP Spec.",
            "Wind, Kaimal Spec."} <= txt
    plt.close(fig)


def test_plot_environmental_spectra_constraint_false_suppresses_band() -> None:
    """rpm_constraint=False draws only the operating *design* band —
    no wider constraint band and no Constraint legend entries (issue
    #47 plot-iteration). None keeps the auto +/-15 % band."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.plots import plot_environmental_spectra

    fig = plot_environmental_spectra(
        rpm_design=(6.9, 12.1), rpm_constraint=False, harmonics=(1, 3),
    )
    ax = fig.axes[0]
    txt = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"1P Design", "3P Design"} <= txt
    assert not any("Constraint" in t for t in txt)
    # 1P + 3P design spans only (no constraint spans).
    assert len(list(ax.patches)) == 2
    plt.close(fig)

    # Default (None) still auto-draws the constraint band + legend.
    fig2 = plot_environmental_spectra(
        rpm_design=(6.9, 12.1), harmonics=(1, 3),
    )
    txt2 = {t.get_text() for t in fig2.axes[0].get_legend().get_texts()}
    assert any("Constraint" in t for t in txt2)
    assert len(list(fig2.axes[0].patches)) == 4
    plt.close(fig2)


def test_plot_environmental_spectra_optional_layers_and_guards() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pybmodes.plots import plot_environmental_spectra

    # Wind/wave omitted, only tower lines - still a valid figure.
    fig = plot_environmental_spectra(tower_fa_hz=0.5)
    assert fig.axes
    plt.close(fig)

    with pytest.raises(ValueError, match="freq_max"):
        plot_environmental_spectra(freq_max=0.0)
    with pytest.raises(ValueError, match="harmonics"):
        plot_environmental_spectra(rpm_design=(5.0, 7.0), harmonics=(0,))


def test_spectra_reject_nonfinite_and_invalid_inputs() -> None:
    """Public engineering functions must reject NaN / inf / physically
    invalid inputs, not propagate them into the figure (review
    Medium #6)."""
    inf = float("inf")
    nan = float("nan")
    # Kaimal: NaN/inf mean_speed/length_scale, negative sigma / TI.
    for kw in (
        {"mean_speed": nan, "length_scale": 300.0},
        {"mean_speed": 10.0, "length_scale": inf},
        {"mean_speed": 10.0, "length_scale": 300.0, "sigma": -1.0},
        {"mean_speed": 10.0, "length_scale": 300.0,
         "turbulence_intensity": -0.1},
    ):
        with pytest.raises(ValueError):
            kaimal_spectrum(np.array([0.1]), **kw)
    # JONSWAP: NaN/inf hs/tp, inf gamma.
    for kw in (
        {"hs": nan, "tp": 10.0},
        {"hs": 6.0, "tp": inf},
        {"hs": 6.0, "tp": 10.0, "gamma": inf},
    ):
        with pytest.raises(ValueError):
            jonswap_spectrum(np.array([0.1]), **kw)
    # Round-3 review #6: the frequency array itself must be finite.
    with pytest.raises(ValueError, match="f contains non-finite"):
        kaimal_spectrum(np.array([0.1, nan]), mean_speed=10.0,
                        length_scale=300.0)
    with pytest.raises(ValueError, match="f contains non-finite"):
        jonswap_spectrum(np.array([0.1, inf]), hs=6.0, tp=10.0)


def test_plot_rejects_nonfinite_and_invalid_inputs() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    from pybmodes.plots import plot_environmental_spectra

    inf = float("inf")
    nan = float("nan")
    with pytest.raises(ValueError, match="freq_max"):
        plot_environmental_spectra(freq_max=inf)
    with pytest.raises(ValueError, match="n_points"):
        plot_environmental_spectra(n_points=1)
    with pytest.raises(ValueError, match="rpm_design"):
        plot_environmental_spectra(rpm_design=(-1.0, 7.0))
    with pytest.raises(ValueError, match="rpm_design"):
        plot_environmental_spectra(rpm_design=(nan, 7.0))
    with pytest.raises(ValueError, match="tower_fa_hz"):
        plot_environmental_spectra(tower_fa_hz=nan)
    with pytest.raises(ValueError, match="tower_ss_hz"):
        plot_environmental_spectra(tower_ss_hz=-0.5)
    # Round-2 review E: no silent truncation / incidental exceptions.
    with pytest.raises(ValueError, match="n_points must be an integer"):
        plot_environmental_spectra(n_points=2.9)
    with pytest.raises(ValueError, match="n_points must be an integer"):
        plot_environmental_spectra(n_points=nan)
    with pytest.raises(ValueError, match="harmonics must be positive"):
        plot_environmental_spectra(rpm_design=(5.0, 7.0),
                                   harmonics=(2.5,))
    with pytest.raises(ValueError, match="harmonics must be positive"):
        plot_environmental_spectra(rpm_design=(5.0, 7.0),
                                   harmonics=(inf,))
