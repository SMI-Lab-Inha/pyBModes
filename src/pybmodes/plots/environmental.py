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

"""Environmental-loading frequency-placement diagram for floating
(and fixed) offshore wind turbines.

Reproduces the standard soft-stiff / frequency-separation figure used
in reference-turbine design reports: normalised power spectral density
versus frequency, overlaying

* the **wind** turbulence spectrum (Kaimal, IEC 61400-1 longitudinal),
* the **wave** spectrum (JONSWAP),
* the **1P / 3P rotor-excitation bands** — a darker *design* band
  (the actual operating rotor-speed range) inside a lighter
  *constraint* band (the allowable placement window), and
* the **tower 1st fore-aft / side-side** natural frequencies as
  vertical reference lines.

The two spectrum closed forms are exposed separately
(:func:`kaimal_spectrum`, :func:`jonswap_spectrum`) so they can be
unit-tested against their analytic properties independently of the
plot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np


def _pos_finite(name: str, value: float) -> float:
    """Reject NaN / inf / non-positive engineering inputs."""
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number; "
                          f"got {value!r}")
    return v


def _nonneg_finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a finite non-negative "
                          f"number; got {value!r}")
    return v

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _require_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  "
            'Install it with: pip install "pybmodes[plots]"'
        ) from exc


# ---------------------------------------------------------------------------
# Closed-form environmental spectra
# ---------------------------------------------------------------------------

def kaimal_spectrum(
    f: np.ndarray,
    *,
    mean_speed: float,
    length_scale: float,
    sigma: float | None = None,
    turbulence_intensity: float = 0.14,
) -> np.ndarray:
    """One-sided longitudinal Kaimal turbulence spectrum ``S_u(f)``.

    ``S_u(f) = 4 sigma_u^2 (L/U) / (1 + 6 f L/U)^(5/3)`` (IEC 61400-1
    form), monotonically decreasing in ``f`` with the finite
    low-frequency plateau ``S_u(0) = 4 sigma_u^2 L / U``. Units are
    ``m^2/s`` (PSD of wind speed) when ``mean_speed`` is in ``m/s`` and
    ``length_scale`` in ``m``; the plot normalises it, so only the
    *shape* matters there.

    ``sigma`` (the longitudinal standard deviation) defaults to
    ``turbulence_intensity * mean_speed`` when not given.
    """
    mean_speed = _pos_finite("mean_speed", mean_speed)
    length_scale = _pos_finite("length_scale", length_scale)
    turbulence_intensity = _nonneg_finite(
        "turbulence_intensity", turbulence_intensity
    )
    if sigma is not None:
        sigma = _nonneg_finite("sigma", sigma)
    sig = sigma if sigma is not None else turbulence_intensity * mean_speed
    f = np.asarray(f, dtype=float)
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains non-finite (NaN / inf) values")
    n = length_scale / mean_speed
    return 4.0 * sig**2 * n / np.power(1.0 + 6.0 * np.abs(f) * n, 5.0 / 3.0)


def jonswap_spectrum(
    f: np.ndarray,
    *,
    hs: float,
    tp: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """JONSWAP wave elevation spectrum ``S(f)`` (frequency in Hz).

    Standard Pierson-Moskowitz core times the peak-enhancement factor
    ``gamma``. The shape is scaled so the zeroth spectral moment is
    *exactly* the significant-wave-height identity ``m0 = Hs**2 / 16``
    (``Hs = 4 sqrt(m0)``). The peak sits at ``f_p = 1/Tp``. Returns
    ``0`` for ``f <= 0``.
    """
    hs = _pos_finite("hs", hs)
    tp = _pos_finite("tp", tp)
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 1.0:
        raise ValueError(f"gamma must be a finite number >= 1; "
                         f"got {gamma!r}")
    f = np.asarray(f, dtype=float)
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains non-finite (NaN / inf) values")
    fp = 1.0 / tp

    def _shape(x: np.ndarray) -> np.ndarray:
        """Un-scaled JONSWAP shape; zero at / below DC."""
        x = np.asarray(x, dtype=float)
        sig = np.where(x <= fp, 0.07, 0.09)
        pos = np.where(x > 0.0, x, np.nan)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            r = np.exp(-((x - fp) ** 2) / (2.0 * sig**2 * fp**2))
            core = np.power(pos, -5.0) * np.exp(
                -1.25 * np.power(pos / fp, -4.0)
            )
            out = core * np.power(gamma, r)
        return np.where(x > 0.0, np.nan_to_num(out, nan=0.0), 0.0)

    # Scale the shape so the zeroth spectral moment is exactly the
    # JONSWAP significant-wave-height identity m0 = Hs**2 / 16. The
    # normalising integral is taken on a fixed dense grid spanning the
    # energetic band so the result is independent of the caller's
    # frequency sampling.
    grid = np.linspace(1.0e-4, 12.0 * fp, 20000)
    sg = _shape(grid)
    # Trapezoidal rule written out so it is independent of the
    # numpy-version churn around trapz / trapezoid.
    m0_shape = float(0.5 * np.sum((sg[1:] + sg[:-1]) * np.diff(grid)))
    scale = (hs**2 / 16.0) / m0_shape if m0_shape > 0.0 else 0.0
    return scale * _shape(f)


# ---------------------------------------------------------------------------
# Frequency-placement diagram
# ---------------------------------------------------------------------------

def _rev_band(rpm_lo: float, rpm_hi: float, order: int) -> tuple[float, float]:
    """Per-rev excitation band in Hz for a rotor-speed range (rpm)."""
    lo, hi = sorted((rpm_lo, rpm_hi))
    return order * lo / 60.0, order * hi / 60.0


def plot_environmental_spectra(
    *,
    tower_fa_hz: float | None = None,
    tower_ss_hz: float | None = None,
    rpm_design: tuple[float, float] | None = None,
    rpm_constraint: tuple[float, float] | bool | None = None,
    harmonics: Sequence[int] = (1, 3),
    wind: dict | None = None,
    wave: dict | None = None,
    freq_max: float = 0.6,
    n_points: int = 2000,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Draw the environmental-loading frequency-placement diagram.

    Parameters
    ----------
    tower_fa_hz, tower_ss_hz :
        Tower 1st fore-aft / side-side natural frequencies (Hz),
        drawn as a solid / dashed black vertical line. Either may be
        ``None`` to omit.
    rpm_design :
        ``(rpm_min, rpm_max)`` of the actual rotor operating range —
        the darker hatched *design* band for each requested harmonic.
    rpm_constraint :
        ``(rpm_min, rpm_max)`` of the allowable placement window — the
        lighter solid *constraint* band drawn behind the design band.
        Defaults to the design range widened by +/-15 % when omitted
        (``None``). Pass ``False`` to suppress the constraint band (and
        its legend entries) entirely and draw only the operating
        *design* band — for callers who have a fixed operating range
        but no separate placement envelope.
    harmonics :
        Per-rev orders to draw (default ``(1, 3)`` -> 1P and 3P).
    wind :
        ``dict`` of :func:`kaimal_spectrum` keyword arguments
        (``mean_speed``, ``length_scale``, optionally ``sigma`` /
        ``turbulence_intensity``). ``None`` skips the wind curve.
    wave :
        ``dict`` of :func:`jonswap_spectrum` keyword arguments
        (``hs``, ``tp``, optionally ``gamma``). ``None`` skips the
        wave curve.
    freq_max, n_points :
        Frequency-axis upper bound (Hz) and sample count.
    ax :
        Existing matplotlib ``Axes`` to draw into; a new figure is
        created when ``None``.
    title :
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_matplotlib()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    freq_max = _pos_finite("freq_max", freq_max)
    npf = float(n_points)
    if not math.isfinite(npf) or npf != int(npf) or int(npf) < 2:
        raise ValueError(
            f"n_points must be an integer >= 2 (no silent truncation "
            f"of e.g. 2.9); got {n_points!r}"
        )
    n_points = int(npf)
    for h in harmonics:
        hf = float(h)
        if not math.isfinite(hf) or hf != int(hf) or int(hf) <= 0:
            raise ValueError(
                f"harmonics must be positive integers; got {h!r}"
            )
    for nm, band in (("rpm_design", rpm_design),
                     ("rpm_constraint", rpm_constraint)):
        # ``rpm_constraint`` also accepts a bool: False = suppress the
        # band, True is treated like the None auto-default. Bools carry
        # no (rpm_min, rpm_max) to validate, so skip them here.
        if band is not None and not isinstance(band, bool):
            if len(band) != 2:
                raise ValueError(f"{nm} must be a (rpm_min, rpm_max) "
                                 f"pair; got {band!r}")
            for v in band:
                _nonneg_finite(f"{nm} entry", v)
    for nm, fhz in (("tower_fa_hz", tower_fa_hz),
                    ("tower_ss_hz", tower_ss_hz)):
        if fhz is not None:
            _pos_finite(nm, fhz)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.2))
    else:
        fig = cast("Figure", ax.figure)

    freq = np.linspace(0.0, freq_max, int(n_points))

    # Per-rev design / constraint bands (drawn first, behind).
    dark = (0.45, 0.45, 0.45)
    light = (0.78, 0.78, 0.78)
    legend: list[tuple[object, str]] = []
    if rpm_design is not None:
        # rpm_constraint: False suppresses the band entirely; None/True
        # auto-widen the design range by +/-15 %; a tuple is used
        # as-is. Resolve to a concrete (lo, hi) tuple or None so the
        # draw loop has no bool to unpack.
        constraint_band: tuple[float, float] | None
        if rpm_constraint is False:
            constraint_band = None
        elif rpm_constraint is None or rpm_constraint is True:
            lo, hi = sorted(rpm_design)
            constraint_band = (lo * 0.85, hi * 1.15)
        else:
            constraint_band = (float(rpm_constraint[0]),
                               float(rpm_constraint[1]))
        for idx, order in enumerate(harmonics):
            face = dark if idx == 0 else light
            d_lo, d_hi = _rev_band(rpm_design[0], rpm_design[1], order)
            if constraint_band is not None:
                c_lo, c_hi = _rev_band(constraint_band[0],
                                       constraint_band[1], order)
                ax.axvspan(c_lo, c_hi, color=face, alpha=0.45, lw=0,
                           zorder=1)
            ax.axvspan(
                d_lo, d_hi, facecolor=face, edgecolor=(0.2, 0.2, 0.2),
                hatch="//", alpha=0.85, lw=0.0, zorder=2,
            )
            legend.append((
                mpatches.Patch(facecolor=face, edgecolor=(0.2, 0.2, 0.2),
                               hatch="//"),
                f"{order}P Design",
            ))
            if constraint_band is not None:
                legend.append((
                    mpatches.Patch(facecolor=face, alpha=0.45),
                    f"{order}P Constraint",
                ))

    # Tower natural-frequency reference lines.
    if tower_ss_hz is not None:
        ax.axvline(tower_ss_hz, color="k", ls="--", lw=1.6, zorder=5)
        legend.append((
            Line2D([0], [0], color="k", ls="--", lw=1.6),
            "Tower, 1st Side-Side",
        ))
    if tower_fa_hz is not None:
        ax.axvline(tower_fa_hz, color="k", ls="-", lw=1.6, zorder=5)
        legend.append((
            Line2D([0], [0], color="k", ls="-", lw=1.6),
            "Tower, 1st Fore-Aft",
        ))

    # Normalised environmental spectra.
    if wave is not None:
        sw = jonswap_spectrum(freq, **wave)
        peak = float(np.max(sw)) or 1.0
        ax.plot(freq, sw / peak, color=(0.0, 0.0, 0.85), lw=2.0, zorder=4)
        legend.append((
            Line2D([0], [0], color=(0.0, 0.0, 0.85), lw=2.0),
            "Waves, JONSWAP Spec.",
        ))
    if wind is not None:
        su = kaimal_spectrum(freq, **wind)
        peak = float(np.max(su)) or 1.0
        ax.plot(freq, su / peak, color=(0.85, 0.0, 0.0), lw=2.0, zorder=4)
        legend.append((
            Line2D([0], [0], color=(0.85, 0.0, 0.0), lw=2.0),
            "Wind, Kaimal Spec.",
        ))

    ax.set_xlim(0.0, freq_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("Normalised\nPower Spectral Density", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    if legend:
        handles, labels = zip(*legend)
        ax.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8,
            frameon=False, handlelength=2.2,
        )
    fig.tight_layout()
    return fig
