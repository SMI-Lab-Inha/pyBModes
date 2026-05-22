# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Professional plotting functions for mode shapes and polynomial fit quality.

All public functions require *matplotlib* (optional dependency).  They return
:class:`~matplotlib.figure.Figure` objects; the caller decides whether to
display or save them.

Typical usage::

    from pybmodes.models import RotatingBlade
    from pybmodes.elastodyn import compute_blade_params
    from pybmodes.plots import plot_mode_shapes, plot_fit_quality, blade_fit_pairs

    result = RotatingBlade("blade.bmi").run(n_modes=10)
    fig1 = plot_mode_shapes(result, n_modes=6)
    fig1.savefig("mode_shapes.png", dpi=150)

    params = compute_blade_params(result)
    fig2 = plot_fit_quality(blade_fit_pairs(result, params))
    fig2.savefig("fit_quality.png", dpi=150)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from pybmodes.elastodyn.params import BladeElastoDynParams, TowerElastoDynParams
    from pybmodes.fitting.poly_fit import PolyFitResult
    from pybmodes.models.result import ModalResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  "
            'Install it with: pip install "pybmodes[plots]"'
        ) from exc


def _apply_style(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _mode_colors(n: int, override: list | str | None = None):
    """Pick *n* visually distinct line colours.

    ``override`` lets the caller take full control: a colormap name
    (e.g. ``"turbo"``, ``"viridis"``) sampled into *n* colours, or an
    explicit list of matplotlib colours (used verbatim, cycled if
    shorter than *n*).

    Otherwise the active rcParams ``prop_cycle`` is used (honouring
    :func:`pybmodes.plots.apply_style`'s engineering-paper palette) —
    but **only while it has at least *n* distinct entries**. The
    styled palette is 7 colours; with ``apply_style()`` active and 8+
    modes, wrapping the cycle made mode 8 reuse mode 1's colour (the
    issue #47 "first and last mode are the same colour" collision).
    Once *n* exceeds the palette, a perceptually-ordered continuous
    colormap is sampled instead so every mode gets a unique hue. Falls
    back to ``tab10`` if no cycle is configured.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if override is not None:
        if isinstance(override, str):
            cmap = plt.get_cmap(override)
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        ov = list(override)
        if not ov:
            raise ValueError(
                "colors override must be a non-empty list of colours "
                "or a matplotlib colormap name"
            )
        return [ov[i % len(ov)] for i in range(n)]

    cycle = mpl.rcParams.get("axes.prop_cycle")
    palette: list = []
    if cycle is not None:
        palette = [entry.get("color") for entry in cycle if entry.get("color")]
    if not palette:
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i) for i in range(10)]
    if n <= len(palette):
        return [palette[i] for i in range(n)]
    # More modes than distinct palette colours — wrapping the cycle
    # would repeat a hue. Sample a perceptually-ordered continuous map
    # so all n modes stay distinguishable (issue #47).
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _smooth_curve(
    y_nodes: np.ndarray,
    x_nodes: np.ndarray,
    n_dense: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-spline-interpolate (x_nodes, y_nodes) onto an evenly-spaced
    grid of length *n_dense* in *y_nodes*, returning (y_dense, x_dense).

    Used by the Bir-style mode-shape plots so the mass-normalised
    eigenvector samples (50-60 nodes for offshore decks) render as a
    smooth curve rather than piecewise-linear segments. Falls back to
    the raw nodal arrays if scipy is unavailable.
    """
    if y_nodes.size < 4:
        return y_nodes, x_nodes
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return y_nodes, x_nodes
    # The natural BC matches a free-end / pinned-end mode shape well at
    # the extremes (zero curvature) and avoids overshoot.
    cs = CubicSpline(y_nodes, x_nodes, bc_type="natural")
    y_dense = np.linspace(y_nodes[0], y_nodes[-1], n_dense)
    return y_dense, cs(y_dense)


# ---------------------------------------------------------------------------
# plot_mode_shapes
# ---------------------------------------------------------------------------

def plot_mode_shapes(
    result: ModalResult,
    n_modes: int = 6,
    component: str = "both",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    normalize: str = "mode",
    colors: list | str | None = None,
) -> Figure:
    """Plot normalised mode shape displacements vs normalised span.

    Parameters
    ----------
    result :
        Output from :meth:`RotatingBlade.run` or :meth:`Tower.run`.
    n_modes :
        Number of modes to overlay (at most ``len(result.shapes)``).
    component :
        ``"flap"`` — fore-aft (w) panel only.
        ``"lag"``  — side-side (v) panel only.
        ``"both"`` — two side-by-side panels (default).
    title :
        Overall figure title.  Uses a sensible default when *None*.
    figsize :
        Matplotlib figure size ``(width_in, height_in)``.  Defaults to
        ``(9, 4)`` for one panel and ``(14, 4)`` for two panels.
    normalize :
        How each mode's two curves are scaled (issue #47):

        - ``"mode"`` (default) — both panels share **one** scale per
          mode, the peak ``|displacement|`` across flap *and* lag. The
          dominant direction then reaches ±1 and the minor direction
          stays proportionally small, so a fore-aft mode reads as a
          full-amplitude curve in the flap panel and a flat near-zero
          line in the lag panel. This is what makes FA vs SS (and a
          rigid platform DOF vs a real bending mode) visually
          distinguishable.
        - ``"component"`` — the legacy behaviour: each panel is
          normalised independently to its own peak, so even a 1 %
          cross-coupling component is blown up to full height and a
          predominantly-flap mode looks identical in both panels. Kept
          for reproducing pre-1.5 figures.
    colors :
        Optional colour control passed to the internal palette picker:
        a matplotlib colormap name (sampled into ``n_modes`` distinct
        colours) or an explicit list of colours. ``None`` (default)
        uses the styled ``prop_cycle``, automatically switching to a
        continuous colormap when there are more modes than distinct
        palette colours so no two modes collide on the same hue.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    component = component.lower()
    if component not in ("flap", "lag", "both"):
        raise ValueError(f"component must be 'flap', 'lag', or 'both'; got {component!r}")
    if normalize not in ("mode", "component"):
        raise ValueError(
            f"normalize must be 'mode' or 'component'; got {normalize!r}"
        )

    shapes = result.shapes[: min(n_modes, len(result.shapes))]
    n = len(shapes)
    colors = _mode_colors(n, override=colors)

    # One shared scale per mode for the "mode" convention: the peak
    # |displacement| over both transverse components. Falls back to 1
    # for a null shape so the division is safe.
    def _shared_scale(shape) -> float:
        peak = max(
            float(np.max(np.abs(shape.flap_disp))) if shape.flap_disp.size else 0.0,
            float(np.max(np.abs(shape.lag_disp))) if shape.lag_disp.size else 0.0,
        )
        return peak if peak > 0.0 else 1.0

    mode_scales = [_shared_scale(s) for s in shapes]

    two_panels = component == "both"
    if figsize is None:
        figsize = (14.0, 4.5) if two_panels else (7.0, 4.5)

    fig, axes = plt.subplots(
        1, 2 if two_panels else 1,
        figsize=figsize,
        constrained_layout=True,
    )
    if not two_panels:
        axes = [axes]

    def _normalise(arr: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(arr))
        return arr / peak if peak > 0 else arr

    def _disp(shape, comp: str, i: int) -> np.ndarray:
        raw = shape.flap_disp if comp == "flap" else shape.lag_disp
        if normalize == "mode":
            return raw / mode_scales[i]
        return _normalise(raw)

    panel_specs = []
    if component in ("flap", "both"):
        panel_specs.append(("flap", "Flap (fore-aft) displacement"))
    if component in ("lag", "both"):
        panel_specs.append(("lag", "Lag (side-side) displacement"))

    # Floating models carry per-mode platform rigid-body names; append
    # them to the legend (e.g. "Mode 1  (0.0079 Hz) — surge"). None for
    # an unattributed mode, and the whole list is None for a
    # non-floating model — both leave the legend as it was.
    mode_labels = result.mode_labels

    for ax, (comp, panel_title) in zip(axes, panel_specs):
        for i, shape in enumerate(shapes):
            disp = _disp(shape, comp, i)
            label = f"Mode {shape.mode_number}  ({shape.freq_hz:.4f} Hz)"
            if mode_labels is not None and i < len(mode_labels):
                dof = mode_labels[i]
                if dof is not None:
                    label += f" — {dof}"
            ax.plot(shape.span_loc, disp, color=colors[i], linewidth=1.8,
                    label=label)
            ax.plot(shape.span_loc, disp, "o", color=colors[i],
                    markersize=3, markeredgewidth=0)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="-")
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="-")
        _apply_style(ax, "Normalised span  [−]",
                     "Normalised displacement  [−]", panel_title)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
                  edgecolor="0.8", handlelength=1.5)

    default_title = title or f"Mode shapes — {n} modes"
    fig.suptitle(default_title, fontsize=12, fontweight="bold", y=1.02)

    return fig


# ---------------------------------------------------------------------------
# plot_fit_quality
# ---------------------------------------------------------------------------

FitEntry = tuple[str, np.ndarray, np.ndarray, "PolyFitResult"]


def plot_fit_quality(
    fits: list[FitEntry],
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot polynomial fit vs FEM data with residuals for each mode.

    Each subplot shows:
    * FEM nodal values (circles)
    * Fitted polynomial (solid line over a fine grid)
    * Residual band (shaded region between FEM and fit)
    * RMS residual and tip-slope annotation

    Parameters
    ----------
    fits :
        List of ``(label, span_loc, fem_disp, fit)`` tuples as returned by
        :func:`blade_fit_pairs` or :func:`tower_fit_pairs`.
    title :
        Overall figure title.
    figsize :
        Matplotlib figure size.  Defaults to ``(4.5 * n_cols, 4.0 * n_rows)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(fits)
    if n == 0:
        raise ValueError("fits list is empty")

    n_cols = min(n, 3)
    n_rows = math.ceil(n / n_cols)
    if figsize is None:
        figsize = (4.8 * n_cols, 4.4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    # Flatten to 1-D list regardless of grid shape
    if n == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(np.asarray(axes).ravel())
    else:
        axes_flat = [ax for row in axes for ax in row]

    x_fine = np.linspace(0.0, 1.0, 300)

    for idx, (label, span_loc, fem_disp, fit) in enumerate(fits):
        ax = axes_flat[idx]

        # Normalise FEM data so tip = 1
        tip = fem_disp[-1]
        if abs(tip) < 1e-30:
            tip = np.max(np.abs(fem_disp)) or 1.0
        y_fem = fem_disp / tip

        # Polynomial on fine grid
        y_poly_fine = fit.evaluate(x_fine)
        # Polynomial at FEM stations
        y_poly_fem = fit.evaluate(np.asarray(span_loc, dtype=float))

        # Residual fill between FEM scatter and polynomial (at FEM stations)
        ax.fill_between(span_loc, y_fem, y_poly_fem,
                        alpha=0.25, color="#d80000", label="Residual")

        # Polynomial line
        ax.plot(x_fine, y_poly_fine, color="#0000d8", linewidth=2.0,
                label="Polynomial fit")

        # FEM data
        ax.scatter(span_loc, y_fem, s=28, color="#009900", zorder=5,
                   label="FEM data")

        # Reference lines
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")

        # Coefficient table inset
        coeffs = fit.coefficients()
        coeff_text = "\n".join(
            f"C{k+2} = {c:+.4f}" for k, c in enumerate(coeffs)
        )
        ax.text(0.03, 0.97, coeff_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=7.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.75", alpha=0.9))

        # RMS and tip-slope annotation (bottom-right)
        metrics_text = (
            f"RMS = {fit.rms_residual:.4f}\n"
            f"tip slope = {fit.tip_slope:.3f}"
        )
        ax.text(0.97, 0.04, metrics_text,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=8,
                color="#d80000",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.75", alpha=0.9))

        _apply_style(ax, "Normalised span  [−]",
                     "Normalised displacement  [−]", label)
        ax.set_xlim(-0.02, 1.02)

        if idx == 0:
            handles = [
                plt.Line2D([0], [0], color="#0000d8", linewidth=2),
                plt.scatter([], [], s=28, color="#009900"),
                Patch(facecolor="#d80000", alpha=0.35),
            ]
            labels_ = ["Polynomial fit", "FEM data", "Residual"]
            ax.legend(handles, labels_, fontsize=8, loc="upper right",
                      framealpha=0.9, edgecolor="0.8")

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title or "Polynomial fit quality",
                 fontsize=12, fontweight="bold", y=1.01)
    return fig


# ---------------------------------------------------------------------------
# bir_mode_shape_plot — Bir 2010 figure convention
# ---------------------------------------------------------------------------
#
# Bir's figures (Bir 2010, NREL/CP-500-47953, Figs 4, 5a, 5b, 6a-6c, 8) plot
# *modal displacement* on the x-axis (mass-normalised, i.e. straight from the
# eigenvector — NOT scaled to unit tip) and *normalised height* (z / H) on the
# y-axis, with 0 at the tower base and 1 at the tower top.
#
# Each mode is drawn as a single curve with a vertical zero line representing
# the undeformed tower position. Optional horizontal annotation lines mark
# physical interfaces (Mean Sea Level, Mud Line) for offshore configurations.
# A coupled-mode overlay (e.g. the small twist component plotted alongside
# the dominant S-S component in Fig 5b / 6b) is supported via the dashed
# ``coupling_overlay`` argument.

ModeSpec = tuple[int, str, str]   # (mode_index_1based, component, label)


def bir_mode_shape_plot(
    result: ModalResult,
    mode_specs: list[ModeSpec],
    *,
    title: str | None = None,
    height_label: str = "Tower section height / H",
    x_label: str = "Modal displacement",
    annotations: dict[str, float] | None = None,
    coupling_overlay: list[ModeSpec] | None = None,
    figsize: tuple[float, float] = (5.5, 6.5),
    xlim: tuple[float, float] | None = None,
    colors: list | str | None = None,
) -> Figure:
    """Plot mode shapes in the Bir 2010 figure convention.

    Parameters
    ----------
    result :
        ``ModalResult`` from ``Tower.run()`` or ``RotatingBlade.run()``.
    mode_specs :
        List of ``(mode_index_1based, component, label)`` tuples. ``component``
        is one of ``"flap"`` (fore-aft / F-A), ``"lag"`` (side-side / S-S),
        ``"twist"``, or ``"axial"``.  ``label`` appears in the legend.
    title :
        Optional figure title.
    height_label :
        Y-axis label. Default matches Bir's notation; pass
        ``"Span fraction"`` for blade plots.
    x_label :
        X-axis label.  Default ``"Modal displacement"`` matches the paper.
    annotations :
        Optional ``{label: y_fraction}`` dict drawing horizontal markers at
        the given normalised heights (e.g. ``{"Mean Sea Level": 0.40, "Mud
        Line": 0.25}`` for Bir Fig 8).
    coupling_overlay :
        Optional list of ``(mode_index, component, label)`` plotted as dashed
        lines on the same axes; used to show e.g. the twist component of an
        S-S mode (Bir Fig 5b, 6b).
    figsize :
        Matplotlib figure size in inches.
    xlim :
        Optional ``(xmin, xmax)``; auto-fits with a small pad if omitted.
    colors :
        Optional colour control (issue #47): a matplotlib colormap
        name or an explicit list of colours. ``None`` (default) uses
        the styled palette, switching to a continuous colormap when
        there are more curves than distinct palette colours so no two
        modes share a hue.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    n_solid = len(mode_specs)
    colors = _mode_colors(max(n_solid, 1), override=colors)

    def _component(shape, comp: str) -> np.ndarray:
        if comp == "flap":
            return shape.flap_disp
        if comp == "lag":
            return shape.lag_disp
        if comp == "twist":
            return shape.twist
        if comp == "axial":
            # Axial DOFs aren't surfaced in NodeModeShape; warn-and-skip.
            raise ValueError(
                "Axial component is not exposed by NodeModeShape; pass "
                "'flap' / 'lag' / 'twist' instead."
            )
        raise ValueError(
            f"component must be 'flap', 'lag', or 'twist'; got {comp!r}"
        )

    def _resolve_mode(idx_1b: int):
        for shape in result.shapes:
            if shape.mode_number == idx_1b:
                return shape
        raise IndexError(
            f"Mode {idx_1b} not in result (have modes "
            f"{[s.mode_number for s in result.shapes]})."
        )

    all_x: list[np.ndarray] = []

    for i, (mode_idx, comp, label) in enumerate(mode_specs):
        shape = _resolve_mode(mode_idx)
        y_nodes = shape.span_loc
        x_nodes = _component(shape, comp)
        y_smooth, x_smooth = _smooth_curve(y_nodes, x_nodes)
        full_label = f"{label}  ({shape.freq_hz:.4f} Hz)"
        ax.plot(x_smooth, y_smooth, color=colors[i % len(colors)],
                linewidth=1.8, label=full_label)
        all_x.append(x_nodes)

    if coupling_overlay:
        for i, (mode_idx, comp, label) in enumerate(coupling_overlay):
            shape = _resolve_mode(mode_idx)
            y_nodes = shape.span_loc
            x_nodes = _component(shape, comp)
            y_smooth, x_smooth = _smooth_curve(y_nodes, x_nodes)
            color = colors[i % len(colors)]
            ax.plot(x_smooth, y_smooth, color=color, linewidth=1.4,
                    linestyle="--", alpha=0.85, label=label)
            all_x.append(x_nodes)

    # Vertical "undeformed" line — slightly thicker than the grid.
    ax.axvline(0.0, color="black", linewidth=0.7, zorder=1)

    # Horizontal annotation lines (MSL / Mud Line for monopile cases).
    if annotations:
        for ann_label, y_frac in annotations.items():
            ax.axhline(y_frac, color="0.45", linewidth=0.8,
                       linestyle=":", zorder=1)
            ax.text(0.98, y_frac + 0.012, ann_label,
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom",
                    fontsize=8, color="0.30")

    ax.set_ylim(0.0, 1.0)
    if xlim is None and all_x:
        xs = np.concatenate(all_x)
        xmax = float(np.max(np.abs(xs)))
        pad = 0.10 * xmax if xmax > 0 else 0.05
        ax.set_xlim(-xmax - pad, xmax + pad)
    elif xlim is not None:
        ax.set_xlim(*xlim)

    _apply_style(ax, x_label, height_label, title)
    ax.legend(fontsize=8, loc="best", framealpha=0.9, edgecolor="0.8",
              handlelength=2.0)

    return fig


def bir_mode_shape_subplot(
    result: ModalResult,
    panels: list[tuple[str, list[ModeSpec]]],
    *,
    suptitle: str | None = None,
    height_label: str = "Tower section height / H",
    x_label: str = "Modal displacement",
    annotations: dict[str, float] | None = None,
    figsize: tuple[float, float] | None = None,
    colors: list | str | None = None,
) -> Figure:
    """Multi-panel Bir-convention plot (matches Bir Fig 8 layout).

    Parameters
    ----------
    panels :
        List of ``(panel_title, mode_specs)`` tuples; one subplot per entry.
    annotations :
        Drawn on every panel (e.g. MSL + Mud Line).
    colors :
        Optional colour control (issue #47): a matplotlib colormap
        name or an explicit list of colours, applied per panel.
        ``None`` (default) uses the styled palette, switching to a
        continuous colormap when a panel has more curves than distinct
        palette colours.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        raise ValueError("panels must contain at least one entry")
    if figsize is None:
        figsize = (4.2 * n, 6.5)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True,
                             sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (panel_title, mode_specs) in zip(axes, panels):
        panel_colors = _mode_colors(max(len(mode_specs), 1), override=colors)
        all_x: list[np.ndarray] = []
        for i, (mode_idx, comp, label) in enumerate(mode_specs):
            shape = next(
                s for s in result.shapes if s.mode_number == mode_idx
            )
            y_nodes = shape.span_loc
            if comp == "flap":
                x_nodes = shape.flap_disp
            elif comp == "lag":
                x_nodes = shape.lag_disp
            elif comp == "twist":
                x_nodes = shape.twist
            else:
                raise ValueError(f"unsupported component {comp!r}")
            y_smooth, x_smooth = _smooth_curve(y_nodes, x_nodes)
            ax.plot(x_smooth, y_smooth,
                    color=panel_colors[i % len(panel_colors)],
                    linewidth=1.8,
                    label=f"{label}  ({shape.freq_hz:.4f} Hz)")
            all_x.append(x_nodes)

        ax.axvline(0.0, color="black", linewidth=0.7, zorder=1)

        if annotations:
            for ann_label, y_frac in annotations.items():
                ax.axhline(y_frac, color="0.45", linewidth=0.8,
                           linestyle=":", zorder=1)
                ax.text(0.98, y_frac + 0.012, ann_label,
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom",
                        fontsize=7, color="0.30")

        ax.set_ylim(0.0, 1.0)
        if all_x:
            xs = np.concatenate(all_x)
            xmax = float(np.max(np.abs(xs)))
            pad = 0.10 * xmax if xmax > 0 else 0.05
            ax.set_xlim(-xmax - pad, xmax + pad)

        _apply_style(ax, x_label, height_label, panel_title)
        ax.legend(fontsize=8, loc="best", framealpha=0.9, edgecolor="0.8",
                  handlelength=2.0)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# blade_fit_pairs / tower_fit_pairs
# ---------------------------------------------------------------------------

def blade_fit_pairs(
    result: ModalResult,
    params: BladeElastoDynParams,
) -> list[FitEntry]:
    """Build ``(label, span_loc, fem_disp, fit)`` entries for blade modes.

    Returns entries for the 1st flap, 2nd flap, and 1st edge modes, matching
    the order in *params*.

    Parameters
    ----------
    result :
        ``ModalResult`` from ``RotatingBlade.run()``.
    params :
        ``BladeElastoDynParams`` from ``compute_blade_params(result)``.
    """
    from pybmodes.elastodyn.params import _is_fa_dominated

    flap_shapes = [s for s in result.shapes if _is_fa_dominated(s)]
    edge_shapes = [s for s in result.shapes if not _is_fa_dominated(s)]

    entries: list[FitEntry] = []

    if len(flap_shapes) >= 1:
        s = flap_shapes[0]
        entries.append((
            f"1st flap  ({s.freq_hz:.4f} Hz)",
            s.span_loc, s.flap_disp, params.BldFl1Sh,
        ))
    if len(flap_shapes) >= 2:
        s = flap_shapes[1]
        entries.append((
            f"2nd flap  ({s.freq_hz:.4f} Hz)",
            s.span_loc, s.flap_disp, params.BldFl2Sh,
        ))
    if len(edge_shapes) >= 1:
        s = edge_shapes[0]
        entries.append((
            f"1st edge  ({s.freq_hz:.4f} Hz)",
            s.span_loc, s.lag_disp, params.BldEdgSh,
        ))

    return entries


def tower_fit_pairs(
    result: ModalResult,
    params: TowerElastoDynParams,
) -> list[FitEntry]:
    """Build ``(label, span_loc, fem_disp, fit)`` entries for tower modes.

    Returns entries for FA1, FA2, SS1, SS2, matching the order in *params*.
    The rigid-body root component is removed from displacements before fitting
    (same as :func:`~pybmodes.elastodyn.params.compute_tower_params`).

    Parameters
    ----------
    result :
        ``ModalResult`` from ``Tower.run()``.
    params :
        ``TowerElastoDynParams`` from ``compute_tower_params(result)``.
    """
    from pybmodes.elastodyn.params import (
        _remove_root_rigid_motion,
        compute_tower_params_report,
    )

    _, report = compute_tower_params_report(result)
    by_mode = {shape.mode_number: shape for shape in result.shapes}
    fa1 = by_mode[report.selected_fa_modes[0]]
    fa2 = by_mode[report.selected_fa_modes[1]]
    ss1 = by_mode[report.selected_ss_modes[0]]
    ss2 = by_mode[report.selected_ss_modes[1]]

    return [
        (
            f"FA mode 1  ({fa1.freq_hz:.4f} Hz)",
            fa1.span_loc,
            _remove_root_rigid_motion(fa1.span_loc, fa1.flap_disp, fa1.flap_slope),
            params.TwFAM1Sh,
        ),
        (
            f"FA mode 2  ({fa2.freq_hz:.4f} Hz)",
            fa2.span_loc,
            _remove_root_rigid_motion(fa2.span_loc, fa2.flap_disp, fa2.flap_slope),
            params.TwFAM2Sh,
        ),
        (
            f"SS mode 1  ({ss1.freq_hz:.4f} Hz)",
            ss1.span_loc,
            _remove_root_rigid_motion(ss1.span_loc, ss1.lag_disp, ss1.lag_slope),
            params.TwSSM1Sh,
        ),
        (
            f"SS mode 2  ({ss2.freq_hz:.4f} Hz)",
            ss2.span_loc,
            _remove_root_rigid_motion(ss2.span_loc, ss2.lag_disp, ss2.lag_slope),
            params.TwSSM2Sh,
        ),
    ]
