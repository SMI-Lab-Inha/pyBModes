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

""":func:`plot_campbell` — engineering-report-style Campbell diagram.

Issue #54 — structural modes coloured by family (Blades / Tower /
Platform / Blade Passing) with mode names written inline on the
curves, per-rev rays tagged ``↑ nP``, optional operating-rpm shading,
and a four-key legend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .result import CampbellResult

if TYPE_CHECKING:
    import matplotlib


def plot_campbell(
    result: CampbellResult,
    excitation_orders: list[int] | None = None,
    rated_rpm: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    platform_modes: list[tuple[str, float]] | None = None,
    log_freq: bool = False,
    *,
    operating_rpm: tuple[float, float] | None = None,
    freq_max: float | None = None,
) -> matplotlib.figure.Figure:
    """Render a Campbell diagram from a :class:`CampbellResult`.

    Engineering-report style (issue #54): structural modes are
    coloured by **family**, the legend carries only those four family
    keys, mode names are written inline next to their lines, and the
    per-rev family is the only thing the legend enumerates as a group:

    * **Blades** — green; per-blade curves, name written at the line.
    * **Tower** — black; horizontal lines, name at the line.
    * **Platform** — red; floating-platform rigid-body modes (surge /
      sway / heave / roll / pitch / yaw), near-degenerate symmetric
      pairs merged (``surge/sway``) to keep the figure clean.
    * **Blade Passing** — blue; the per-rev rays (default 1P / 3P /
      6P / 9P), each tagged ``↑ nP`` inline (no legend clutter).

    ``operating_rpm=(lo, hi)`` shades the operating rotor-speed window
    grey (outside it stays white) and draws a ``↔ Operating Speed
    Range`` marker.

    Parameters
    ----------
    result :
        Output of :func:`campbell_sweep`.
    excitation_orders :
        Per-rev orders. Default ``[1, 3, 6, 9]``.
    rated_rpm :
        If given, a thin reference line at the operating rotor speed.
    ax :
        Existing Axes to draw into; a fresh figure if ``None``.
    platform_modes :
        Optional ``[(dof, freq_hz), ...]`` floating rigid-body modes
        for the *screening* path (when the result has no platform
        columns). Drawn in the red **Platform** family. The
        coupled-tower path classifies these natively — see
        :func:`campbell_sweep`.
    log_freq :
        Log-scaled frequency axis (the per-rev rays are densely
        sampled so they render correctly). Default ``False``.
    operating_rpm :
        ``(lo, hi)`` rotor-speed operating window (rpm) — shaded grey
        with an ``Operating Speed Range`` marker. ``None`` (default)
        draws no band — backward compatible.
    freq_max :
        Upper frequency-axis limit (Hz). ``None`` (default) auto-caps
        the axis just above the highest *structural* mode so the
        modes of interest fill the figure (the steep per-rev rays run
        off the top, as in a standard Campbell report) instead of the
        axis stretching to the highest ray. Ignored when
        ``log_freq=True``.

    Note on blade-line jitter
    -------------------------
    For ElastoDyn-derived blade FEMs the 1st-flap line typically shows
    ~5 % step-to-step scatter — *not* real Southwell dynamics. The
    BMI adapter floors rotary inertia and forces near-rigid axial
    behaviour (``EA / EI ≈ 1e6``), leaving the dense FEM matrices
    ill-conditioned (κ(M) ≈ 1e11), which makes LAPACK's subset
    eigenvalue routines wobble on the lowest mode even when the
    underlying eigenvector is identical step to step. The MAC tracker
    catches this — the participation array stays > 98 % flap-dominant
    in the 1st-flap slot — so the mode *identity* is correct, only
    the eigenvalue precision suffers. Centrifugal stiffening is
    monotonic in physics (Wright 1982); endpoint-to-endpoint
    comparisons (parked vs rated) are reliable, individual-step
    monotonicity is not.

    Returns
    -------
    :class:`matplotlib.figure.Figure` for the rendered axes.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_campbell; install with "
            'pip install "pybmodes[plots]"'
        ) from exc

    if excitation_orders is None:
        excitation_orders = [1, 3, 6, 9]

    owns_fig = ax is None
    if ax is None:
        # Wider than tall: the extra width is the right-margin label
        # column (issue #57).
        fig, ax = plt.subplots(figsize=(11.0, 6.0))
    else:
        # ``ax.figure`` is typed as ``Figure | SubFigure`` upstream
        # but in every realistic embedding the caller passes an Axes
        # from a top-level Figure, never a SubFigure. Cast accordingly.
        from typing import cast

        from matplotlib.figure import Figure

        fig = cast(Figure, ax.figure)

    # Family colours (issue #54 — engineering-report convention):
    # Blades green, Tower black, Platform red, Blade Passing blue.
    C_BLADE = (0.0, 0.62, 0.0)
    C_TOWER = (0.0, 0.0, 0.0)
    C_PLAT = (0.85, 0.0, 0.0)
    C_BP = (0.0, 0.0, 0.62)
    C_OPR = (0.85, 0.85, 0.85)        # operating-speed-range shade

    # Per-line dash cycle within a colour band (issue #57): same-family
    # lines share a colour, so a distinct dash lets adjacent lines be
    # told apart and traced to their right-margin labels.
    _LS_CYCLE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)),
                 (0, (1, 1))]

    rpm = result.omega_rpm
    rpm_max = float(rpm.max()) if rpm.size > 0 else 0.0
    xmax = rpm_max if rpm_max > 0.0 else 1.0
    n_blade = result.n_blade_modes
    n_tower = result.n_tower_modes

    from pybmodes.campbell._classify import _COUPLED_PLATFORM_LABEL
    from pybmodes.fem.platform_modes import _PLATFORM_DOF_NAMES

    plat_name_set = set(_PLATFORM_DOF_NAMES)
    # Split tower columns into flexible-tower (black) vs rigid-body
    # platform (red). A named platform DOF (surge through yaw) and the
    # coupled-platform sentinel both belong to the Platform family, and a
    # ``None`` label defensively falls into that sentinel too, so a
    # rigid-body mode the FEM left unclassified is never drawn as a
    # flexible tower line. Named DOFs are deduped and degeneracy-merged;
    # coupled modes are kept verbatim (each is a distinct physical mode).
    flex_modes: list[tuple[str, float]] = []
    plat_pairs: list[tuple[str, float]] = []
    plat_coupled: list[tuple[str, float]] = []
    for k in range(n_blade, n_blade + n_tower):
        f = float(result.frequencies[0, k])
        lbl = result.labels[k]
        if lbl is None:
            lbl = _COUPLED_PLATFORM_LABEL
        if lbl in plat_name_set:
            plat_pairs.append((lbl, f))
        elif lbl == _COUPLED_PLATFORM_LABEL:
            if np.isfinite(f) and f > 0.0:
                plat_coupled.append((lbl, f))
        else:
            flex_modes.append((lbl.replace("tower ", ""), f))
    if platform_modes:
        plat_pairs += [(str(nm), float(fv)) for nm, fv in platform_modes]
    seen: set[str] = set()
    plat_clean: list[tuple[str, float]] = []
    for nm, f in plat_pairs:
        if nm in seen or not np.isfinite(f) or f <= 0.0:
            continue
        seen.add(nm)
        plat_clean.append((nm, f))
    # Merge near-degenerate symmetric pairs (surge≈sway, roll≈pitch)
    # within 2 % into one "a/b" label — matches the reference and
    # keeps a symmetric floater's labels from stacking.
    plat_clean.sort(key=lambda t: t[1])
    plat_merged: list[tuple[str, float]] = []
    for nm, f in plat_clean:
        if (plat_merged
                and abs(plat_merged[-1][1] - f) <= 0.02 * max(f, 1e-9)):
            pn, pf = plat_merged[-1]
            plat_merged[-1] = (f"{pn}/{nm}", 0.5 * (pf + f))
        else:
            plat_merged.append((nm, f))
    # Coupled or unclassified rigid-body modes join the Platform family
    # without dedup or degeneracy-merge, because their shared generic
    # label would otherwise collapse several distinct modes into one.
    plat_merged.extend(plat_coupled)
    plat_merged.sort(key=lambda t: t[1])

    # Operating-speed-range shading: the *window itself* is grey,
    # outside stays white (behind everything).
    op: tuple[float, float] | None = None
    if operating_rpm is not None:
        lo, hi = sorted(float(v) for v in operating_rpm)
        lo, hi = max(0.0, lo), min(xmax, hi)
        if hi > lo:
            op = (lo, hi)
            ax.axvspan(lo, hi, color=C_OPR, lw=0, zorder=0)

    # Per-rev "Blade Passing" rays — uniform blue. Dense grid so the
    # rays render correctly on a log axis too.
    n_ray = 256
    if log_freq and rpm_max > 0.0:
        ray = np.linspace(rpm_max * 1.0e-3, rpm_max, n_ray)
    else:
        ray = np.linspace(0.0, xmax, n_ray)
    for order in excitation_orders:
        ax.plot(ray, order * ray / 60.0, "-", color=C_BP,
                linewidth=1.8, zorder=2)

    # Blade modes — green curves ("Blades" family), each a distinct dash.
    for k in range(n_blade):
        ax.plot(rpm, result.frequencies[:, k], color=C_BLADE,
                linewidth=1.8, linestyle=_LS_CYCLE[k % len(_LS_CYCLE)],
                zorder=3)

    # Flexible tower modes — black horizontal lines (distinct dashes);
    # platform modes — red horizontal lines.
    for i, (_nm, f) in enumerate(flex_modes):
        ax.axhline(f, color=C_TOWER, linewidth=1.6,
                   linestyle=_LS_CYCLE[i % len(_LS_CYCLE)], zorder=2)
    # Platform modes cluster within a narrow low-frequency band, so
    # one red colour alone makes them indistinguishable — give each a
    # distinct line *style* (still the red family) so they can be told
    # apart and traced to their labels.
    _PLAT_LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, (_nm, f) in enumerate(plat_merged):
        ax.axhline(f, color=C_PLAT, linewidth=1.6,
                   linestyle=_PLAT_LS[i % len(_PLAT_LS)], zorder=2)

    # Rated-speed reference (thin; the operating-range band is the
    # primary speed annotation, so this stays out of the legend).
    if rated_rpm is not None:
        ax.axvline(float(rated_rpm), color=(0.4, 0.4, 0.4),
                   linestyle=":", linewidth=0.9, zorder=1)

    ax.set_xlabel("Rotor Speed [RPM]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Campbell Diagram")
    ax.set_xlim(0.0, xmax)
    if log_freq:
        cand = [float(v) for v in np.asarray(result.frequencies).ravel()
                if np.isfinite(v) and v > 0.0]
        cand += [f for _, f in plat_merged + flex_modes]
        floor = max(1.0e-4, 0.5 * min(cand)) if cand else 1.0e-3
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
    else:
        # Cap the axis just above the highest *structural* mode so the
        # modes of interest fill the figure; the steep per-rev rays
        # simply run off the top (standard Campbell-report framing).
        struct_max = 0.0
        if n_blade > 0:
            struct_max = float(np.nanmax(
                result.frequencies[:, :n_blade]))
        for _nm, f in flex_modes + plat_merged:
            struct_max = max(struct_max, f)
        if freq_max is not None:
            top = float(freq_max)
        elif struct_max > 0.0:
            top = 1.30 * struct_max     # headroom for the op-range bar
        else:
            top = None        # nothing structural — let matplotlib pick
        ax.set_ylim(0.0, top)
    ymin, ymax = ax.get_ylim()
    log_scale = log_freq and ymin > 0.0 and ymax > ymin

    def _to_frac(yv: float) -> float:
        if log_scale:
            return float(
                (np.log10(max(yv, ymin)) - np.log10(ymin))
                / (np.log10(ymax) - np.log10(ymin))
            )
        return (yv - ymin) / (ymax - ymin) if ymax > ymin else 0.0

    def _from_frac(fr: float) -> float:
        if log_scale:
            return float(10.0 ** (np.log10(ymin) + fr * (
                np.log10(ymax) - np.log10(ymin))))
        return float(ymin + fr * (ymax - ymin))

    # Operating-speed-range marker — set just below the top so it
    # clears the legend / frame.
    if op is not None:
        tr = ax.get_xaxis_transform()        # x = data, y = axes frac
        ax.annotate("", xy=(op[1], 0.92), xytext=(op[0], 0.92),
                    xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="<->", color="0.30",
                                    lw=1.3), zorder=5)
        ax.text(0.5 * (op[0] + op[1]), 0.93, "Operating Speed Range",
                transform=tr, ha="center", va="bottom", fontsize=9,
                color="0.20", zorder=5)

    # Inline nP tags along the blue rays, heights staggered so
    # successive orders don't collide. A white backing box keeps the
    # tag from sitting on top of (overlapping) its ray.
    no = len(excitation_orders)
    for i, order in enumerate(excitation_orders):
        ty = _from_frac(0.28 + 0.52 * (i / max(no - 1, 1)))
        tx = ty * 60.0 / order
        if tx > 0.90 * xmax:
            tx = 0.55 * xmax
            ty = order * tx / 60.0
        ax.text(tx, ty, f"{order}P", color=C_BP, fontsize=9,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.80))

    # Structural mode names + frequencies in a clean column down the
    # **right margin** (issue #57): every mode is labelled at the right
    # edge at its frequency height, de-overlapped vertically, with a
    # thin leader from the line's right end to its label. This reads far
    # more clearly than labels scattered inline along the curves —
    # especially for a FOWT whose modes cluster in a narrow band.
    # Engineering terms are spelled out in full for the figure. These are
    # bending modes, so the figure says "flapwise bending" and so on. The
    # terse tokens in CampbellResult.labels stay as-is for CSV and API.
    _pretty_tok = {"flap": "flapwise bending", "edge": "edgewise bending",
                   "FA": "fore-aft bending", "SS": "side-to-side bending"}

    def _pretty(name: str) -> str:
        return " ".join(_pretty_tok.get(t, t) for t in name.split(" "))

    # (label, y at the line's right end, x of that right end, colour).
    # Blade modes are rotor-speed dependent, so anchor at the swept
    # end (rpm_max); tower / platform lines are constant.
    right_labels: list[tuple[str, float, float, tuple]] = []
    for k in range(n_blade):
        curve = np.asarray(result.frequencies[:, k], dtype=float)
        yend = float(curve[-1]) if curve.size else float(
            result.frequencies[0, k])
        right_labels.append(
            (f"{_pretty(result.labels[k])} ({yend:.3g} Hz)", yend,
             rpm_max, C_BLADE))
    for _nm, f in flex_modes:
        right_labels.append((f"{_pretty(_nm)} ({f:.3g} Hz)", f, xmax,
                             C_TOWER))
    for _nm, f in plat_merged:
        right_labels.append((f"{_pretty(_nm)} ({f:.3g} Hz)", f, xmax,
                             C_PLAT))

    # Where the label column sits depends on who owns the figure:
    #
    # * We own it → reserve a right margin and place labels *outside* the
    #   axes (a clean column past the right spine).
    # * Caller supplied the Axes (subplots / gridspec) → we must NOT touch
    #   their layout, and there's no reserved margin, so outside-the-axes
    #   text would be clipped off the canvas. Place labels just *inside*
    #   the right edge instead, so they stay visible in embedded use
    #   (Codex review P2 — a regression vs the old inline labels).
    if owns_fig:
        x_anchor = xmax * 1.02          # leader endpoint, outside the axes
        x_text = x_anchor + 0.012 * xmax
        ha = "left"
    else:
        x_anchor = xmax                 # leader endpoint, on the right spine
        x_text = xmax - 0.012 * xmax    # text reads leftwards, inside
        ha = "right"

    # De-overlap the *label* y-positions in axes-fraction space (works
    # for linear and log axes alike); the lines themselves don't move —
    # only the text is nudged, with a leader back to the true height.
    right_labels.sort(key=lambda e: e[1])
    _min_gap_fr = 0.042
    _prev_fr = -1.0
    for text, yline, xend, col in right_labels:
        fr = _to_frac(yline)
        fr = min(max(fr, _prev_fr + _min_gap_fr), 1.0)
        _prev_fr = fr
        ly = _from_frac(fr)
        ax.plot([xend, x_anchor], [yline, ly], color=col, linewidth=0.6,
                alpha=0.55, zorder=2, clip_on=False)   # leader
        ax.text(x_text, ly, text, color=col, fontsize=8.0,
                ha=ha, va="center", zorder=5, clip_on=False)

    # Reserve room in the right margin for the label column (only when
    # this function owns the figure; a caller-supplied Axes keeps its own
    # layout and gets the inside-the-edge placement above).
    if owns_fig:
        fig.subplots_adjust(right=0.74)

    # Four family keys only (those that are present).
    handles = []
    if n_blade > 0:
        handles.append(Line2D([], [], color=C_BLADE, lw=2.0,
                              label="Blades"))
    if flex_modes:
        handles.append(Line2D([], [], color=C_TOWER, lw=2.0,
                              label="Tower"))
    if plat_merged:
        handles.append(Line2D([], [], color=C_PLAT, lw=2.0,
                              label="Platform"))
    handles.append(Line2D([], [], color=C_BP, lw=2.0,
                          label="Blade Passing"))
    ax.legend(handles=handles, loc="upper left", frameon=True,
              fontsize=9)
    return fig
