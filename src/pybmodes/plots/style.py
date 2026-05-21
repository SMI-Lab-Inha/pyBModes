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

"""Unified pyBmodes plot style — standard engineering-paper defaults.

A single :func:`apply_style` call configures matplotlib with the
black/red/blue/green/magenta/orange/cyan colour order conventional
in academic and engineering publications: a Helvetica/Arial
sans-serif font, boxed axes with inward ticks, a soft dark-grey
frame, and dashed grid lines (off by default; opt in per axes).

Typical usage at the top of a case script::

    import matplotlib
    matplotlib.use("Agg")
    from pybmodes.plots.style import apply_style
    apply_style()

Once :func:`apply_style` has run, every subsequent ``plt.subplots``
call inherits the defaults. The function only mutates
``matplotlib.rcParams`` so it composes cleanly with per-axes overrides.
"""

from __future__ import annotations

#: Standard engineering-paper line colour order. Black first so the
#: dominant curve in single-line plots reads as line art rather than
#: as a coloured emphasis; saturated red / blue / green next to give
#: maximum discrimination on white backgrounds and to remain
#: distinguishable when converted to grayscale (black / red / blue
#: land at distinct luminance values). RGB triples in [0, 1] so
#: callers can do ``ax.plot(x, y, color=STANDARD_LINES[0])`` directly.
STANDARD_LINES: list[tuple[float, float, float]] = [
    (0.000, 0.000, 0.000),  # black
    (0.850, 0.000, 0.000),  # red
    (0.000, 0.000, 0.850),  # blue
    (0.000, 0.600, 0.000),  # green
    (0.800, 0.000, 0.800),  # magenta
    (1.000, 0.500, 0.000),  # orange
    (0.000, 0.700, 0.700),  # cyan
]

#: Backwards-compatibility alias. The previous palette name pointed
#: at MATLAB's RGB triples; kept as a name so existing callers that
#: imported ``MATLAB_LINES`` keep working while the values move to
#: the new standard order.
MATLAB_LINES: list[tuple[float, float, float]] = STANDARD_LINES

#: Default matplotlib ``axes.prop_cycle`` colour list. Aliased to
#: :data:`STANDARD_LINES`; existing callers that slice ``PALETTE``
#: keep working with the new colours.
PALETTE: list[tuple[float, float, float]] = STANDARD_LINES

#: Soft dark grey MATLAB uses for axis spines and tick labels — gives
#: the "almost black" frame contrast without the harshness of pure
#: ``#000000`` on white backgrounds.
_FRAME_GREY = (0.15, 0.15, 0.15)


def apply_style() -> None:
    """Apply the MATLAB-style plot defaults.

    Mutates ``matplotlib.rcParams`` in place. Safe to call multiple
    times — the second call simply re-applies the same values. Raises
    :class:`ImportError` if matplotlib isn't installed (the optional
    ``[plots]`` extra).
    """
    try:
        import matplotlib as mpl
        from cycler import cycler
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for pybmodes.plots.style; "
            'install it with: pip install "pybmodes[plots]"'
        ) from exc

    mpl.rcParams.update({
        # --- Fonts: MATLAB ships Helvetica on Mac/Linux and Arial on
        #     Windows; both render almost identically. Fall back to
        #     Liberation Sans / DejaVu Sans where neither is installed.
        "font.family":        "sans-serif",
        "font.sans-serif":    [
            "Helvetica", "Helvetica Neue", "Arial",
            "Liberation Sans", "DejaVu Sans",
        ],
        "mathtext.fontset":   "dejavusans",
        "mathtext.default":   "regular",         # math glyphs match text weight
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",          # MATLAB titles are not bold
        "axes.labelsize":     10,
        "axes.labelweight":   "normal",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   11,
        "figure.titleweight": "normal",
        # --- Colour cycle ------------------------------------------------
        "axes.prop_cycle":    cycler(color=PALETTE),
        # --- Lines: MATLAB defaults to 0.5pt; bump slightly so plots
        #     remain legible at print sizes without losing the thin look.
        "lines.linewidth":    1.0,
        "lines.markersize":   5.0,
        "lines.solid_capstyle": "round",
        # --- Frame: boxed axes (all four spines), soft dark-grey edge,
        #     no top-or-right hiding.
        "axes.linewidth":     0.75,
        "axes.edgecolor":     _FRAME_GREY,
        "axes.labelcolor":    _FRAME_GREY,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        # --- Grid: MATLAB default is OFF, but plots that opt in via
        #     ``ax.grid(True)`` get the familiar light-grey dashed look.
        "axes.grid":          False,
        "grid.color":         (0.65, 0.65, 0.65),
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.5,
        # --- Ticks: inside the box, mirrored on all four sides for the
        #     classic MATLAB engineering look.
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        True,
        "xtick.color":        _FRAME_GREY,
        "ytick.color":        _FRAME_GREY,
        "xtick.major.size":   4.0,
        "ytick.major.size":   4.0,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "xtick.major.width":  0.75,
        "ytick.major.width":  0.75,
        "xtick.minor.width":  0.5,
        "ytick.minor.width":  0.5,
        # --- Legend: MATLAB-style thin grey border, opaque white fill.
        "legend.frameon":     True,
        "legend.framealpha":  1.0,
        "legend.edgecolor":   _FRAME_GREY,
        "legend.facecolor":   "white",
        "legend.fancybox":    False,
        "legend.borderpad":   0.4,
        # --- Output: 100 DPI inline (matches MATLAB's default figure
        #     resolution), 300 DPI for saved figures.
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })
