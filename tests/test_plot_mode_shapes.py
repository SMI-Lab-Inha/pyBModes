"""Regression tests for the mode-shape plot fixes (issue #47).

Two distinct user-reported defects:

* the styled palette is 7 colours, so 8+ modes wrapped the cycle and
  mode 8 reused mode 1's hue (``_mode_colors``);
* ``plot_mode_shapes`` normalised each panel to its own peak, so a
  predominantly-flap mode looked the same height in the lag panel —
  you could not tell FA from SS (``normalize`` convention).

These stay independent of any bundled deck (synthetic shapes only).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models.result import ModalResult
from pybmodes.plots.mode_shapes import _mode_colors, plot_mode_shapes


def _shape(mode_number: int, flap: np.ndarray, lag: np.ndarray) -> NodeModeShape:
    flap = np.asarray(flap, dtype=float)
    lag = np.asarray(lag, dtype=float)
    n = flap.size
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=0.1 * mode_number,
        span_loc=np.linspace(0.0, 1.0, n),
        flap_disp=flap,
        flap_slope=np.zeros(n),
        lag_disp=lag,
        lag_slope=np.zeros(n),
        twist=np.zeros(n),
    )


def _result(n_modes: int) -> ModalResult:
    s = np.linspace(0.0, 1.0, 20)
    shapes = [
        _shape(i + 1, flap=np.sin((i + 1) * s), lag=0.3 * np.sin((i + 1) * s))
        for i in range(n_modes)
    ]
    return ModalResult(
        frequencies=np.array([sh.freq_hz for sh in shapes]),
        shapes=shapes,
    )


# ---------------------------------------------------------------------------
# Fix 3 — _mode_colors must not repeat a hue when n > palette
# ---------------------------------------------------------------------------

def test_mode_colors_distinct_when_more_modes_than_styled_palette() -> None:
    """With apply_style()'s 7-colour palette active and 8 modes, the
    old ``i % len`` wrap made colour 0 == colour 7. They must now be
    distinct (issue #47)."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib.colors import to_rgba

    from pybmodes.plots import apply_style

    apply_style()
    cols = _mode_colors(8)
    keys = {tuple(np.round(to_rgba(c), 5)) for c in cols}
    assert len(keys) == 8
    assert to_rgba(cols[0]) != to_rgba(cols[7])


def test_mode_colors_small_n_keeps_styled_palette() -> None:
    """n within the palette still returns the deliberate engineering-
    paper colours (black-first), unchanged."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgb

    from pybmodes.plots import apply_style
    from pybmodes.plots.style import STANDARD_LINES

    apply_style()
    cols = _mode_colors(4)
    for got, want in zip(cols, STANDARD_LINES[:4]):
        assert np.allclose(to_rgb(got), want, atol=1e-6)


def test_mode_colors_override_colormap_and_list() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba

    cmap_cols = _mode_colors(5, override="turbo")
    assert len({tuple(np.round(to_rgba(c), 5)) for c in cmap_cols}) == 5

    listed = _mode_colors(3, override=["red", "green"])
    assert [to_rgba(c) for c in listed] == [
        to_rgba("red"), to_rgba("green"), to_rgba("red"),
    ]

    with pytest.raises(ValueError, match="non-empty"):
        _mode_colors(3, override=[])


# ---------------------------------------------------------------------------
# Fix 4 — shared per-mode normalisation makes FA vs SS readable
# ---------------------------------------------------------------------------

def _line_peak(ax, mode_number: int) -> float:
    for ln in ax.lines:
        if ln.get_label().startswith(f"Mode {mode_number} "):
            return float(np.max(np.abs(np.asarray(ln.get_ydata()))))
    raise AssertionError(f"no line for mode {mode_number}")


def test_plot_mode_shapes_mode_normalisation_keeps_relative_amplitude() -> None:
    """A 95 % flap / 5 % lag mode must read as full height in the flap
    panel and ~0.05 in the lag panel under the default normalize='mode'
    (issue #47) — not 1.0 in both."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.linspace(0.0, 1.0, 20)
    shape = _shape(1, flap=np.sin(np.pi * s), lag=0.05 * np.sin(np.pi * s))
    res = ModalResult(frequencies=np.array([0.1]), shapes=[shape])

    fig = plot_mode_shapes(res, n_modes=1, component="both")
    flap_ax, lag_ax = fig.axes[0], fig.axes[1]
    assert _line_peak(flap_ax, 1) == pytest.approx(1.0, rel=1e-6)
    assert _line_peak(lag_ax, 1) == pytest.approx(0.05, rel=1e-3)
    plt.close(fig)


def test_plot_mode_shapes_component_normalisation_is_legacy_behaviour() -> None:
    """normalize='component' reproduces the pre-1.5 figure: each panel
    independently peaks at 1.0 (kept as an explicit opt-in)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.linspace(0.0, 1.0, 20)
    shape = _shape(1, flap=np.sin(np.pi * s), lag=0.05 * np.sin(np.pi * s))
    res = ModalResult(frequencies=np.array([0.1]), shapes=[shape])

    fig = plot_mode_shapes(res, n_modes=1, component="both",
                           normalize="component")
    flap_ax, lag_ax = fig.axes[0], fig.axes[1]
    assert _line_peak(flap_ax, 1) == pytest.approx(1.0, rel=1e-6)
    assert _line_peak(lag_ax, 1) == pytest.approx(1.0, rel=1e-6)
    plt.close(fig)


def test_plot_mode_shapes_rejects_bad_normalize() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with pytest.raises(ValueError, match="normalize must be"):
        plot_mode_shapes(_result(2), n_modes=2, normalize="bogus")


def test_plot_mode_shapes_colors_override_threads_through() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    fig = plot_mode_shapes(_result(3), n_modes=3, component="flap",
                           colors=["red", "blue", "green"])
    ax = fig.axes[0]
    line_colors = [
        to_rgba(ln.get_color())
        for ln in ax.lines
        if ln.get_label().startswith("Mode ")
    ]
    assert line_colors == [to_rgba("red"), to_rgba("blue"), to_rgba("green")]
    plt.close(fig)
