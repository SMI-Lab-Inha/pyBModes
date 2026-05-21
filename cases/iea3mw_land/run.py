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

"""Exploratory pyBmodes run for the IEA-3.4 MW land-based reference turbine.

Reads the bundled ElastoDyn deck under
``external/OpenFAST_files/IEA-3.4-130-RWT/openfast``, solves the standalone tower
modal problem (RNA lumped at the tower top) and the blade modal problem at
both 0 rpm and the rated 10.5 rpm, fits 6th-order ElastoDyn polynomials to
the resulting mode shapes, and compares the fitted coefficients against the
ones already present in the .dat files.

No assertions — this is a side-by-side print + plot exploration. The
comparison table is the primary output of interest.

Outputs land under ``cases/iea3mw_land/outputs/``:

  * ``coefficients.txt``  — full coefficient comparison
  * ``tower_modes.png``   — FA1, SS1, FA2, SS2 mode shapes
  * ``blade_modes.png``   — flap1, edge1, flap2 at 0 rpm and 10.5 rpm

Run from the repo root with ``PYTHONPATH=src``::

    set PYTHONPATH=%CD%\\src
    python cases/iea3mw_land/run.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Headless matplotlib backend so the script runs cleanly in CI / over SSH.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pybmodes.plots import apply_style  # noqa: E402

apply_style()

from pybmodes.elastodyn import compute_blade_params, compute_tower_params_report  # noqa: E402
# Private helpers — used here for diagnostics only (not part of the public API).
from pybmodes.elastodyn.params import (  # noqa: E402
    _is_degenerate_pair,
    _remove_root_rigid_motion,
    _rotate_degenerate_pairs,
    _shape_participation,
    _sorted_modes,
)
from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_blade,
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.models import RotatingBlade, Tower  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
OUT_DIR = HERE.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ED_DIR = ROOT / "external" / "OpenFAST_files" / "IEA-3.4-130-RWT" / "openfast"
MAIN_DAT = ED_DIR / "IEA-3.4-130-RWT_ElastoDyn.dat"

RATED_RPM = 10.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_tower_coeffs(tower_ed) -> dict[str, float]:
    """Pull the polynomial coefficients (degrees 2..6) out of the tower .dat."""
    out: dict[str, float] = {}
    for name, arr in [
        ("TwFAM1Sh", tower_ed.tw_fa_m1_sh),
        ("TwFAM2Sh", tower_ed.tw_fa_m2_sh),
        ("TwSSM1Sh", tower_ed.tw_ss_m1_sh),
        ("TwSSM2Sh", tower_ed.tw_ss_m2_sh),
    ]:
        for k in range(2, 7):
            out[f"{name}({k})"] = float(arr[k - 2])
    return out


def file_blade_coeffs(blade_ed) -> dict[str, float]:
    """Pull the polynomial coefficients (degrees 2..6) out of the blade .dat."""
    out: dict[str, float] = {}
    for name, arr in [
        ("BldFl1Sh", blade_ed.bld_fl1_sh),
        ("BldFl2Sh", blade_ed.bld_fl2_sh),
        ("BldEdgSh", blade_ed.bld_edg_sh),
    ]:
        for k in range(2, 7):
            out[f"{name}({k})"] = float(arr[k - 2])
    return out


def diff_table(
    file_dict: dict[str, float],
    py_dict: dict[str, float],
    names,
    py_residuals: dict[str, float] | None = None,
    file_residuals: dict[str, float] | None = None,
    py_cond: dict[str, float] | None = None,
) -> str:
    """Format a side-by-side coefficient comparison as a multi-line string.

    ``py_residuals`` and ``file_residuals`` are keyed by the *prefix* of the
    coefficient name (e.g. ``"TwFAM1Sh"``) — one residual per mode, repeated
    across that mode's five coefficient rows. ``file_residuals`` is the RMS
    of the *file's* polynomial evaluated against *our* mode shape, telling
    you how well the bundled coefficients describe the shape we computed
    (a large gap implies the file polynomial was fit to a different shape).

    ``py_cond`` similarly holds the polynomial-fit design-matrix condition
    number per mode prefix. Same value across all five rows of a given
    mode (it depends on the spanwise sampling, not on the data being fit).
    Useful for distinguishing conditioning-driven coefficient scatter
    from genuine shape-driven coefficient differences.
    """
    have_extras = (
        py_residuals is not None or file_residuals is not None or py_cond is not None
    )
    lines = []
    if not have_extras:
        header = f"{'Parameter':<14} {'File':>12} {'pyBmodes':>12} {'Diff':>12} {'Diff %':>10}"
    else:
        header = (
            f"{'Parameter':<14} {'File':>12} {'pyBmodes':>12} {'Diff':>12} {'Diff %':>9}"
            f"  {'pyB rms*':>9} {'file rms*':>9}  {'cond':>9}"
        )
    lines.append(header)
    lines.append("-" * len(header))
    for n in names:
        f = file_dict[n]
        p = py_dict[n]
        d = p - f
        pct = (100 * d / f) if abs(f) > 1e-9 else float("nan")
        pct_s = f"{pct:>+8.1f}%" if np.isfinite(pct) else f"{'n/a':>9}"
        prefix = n.split("(")[0]
        if not have_extras:
            lines.append(f"{n:<14} {f:>12.4f} {p:>12.4f} {d:>+12.4f} {pct_s}")
        else:
            pyr = py_residuals.get(prefix, float("nan")) if py_residuals else float("nan")
            fir = file_residuals.get(prefix, float("nan")) if file_residuals else float("nan")
            cnd = py_cond.get(prefix, float("nan")) if py_cond else float("nan")
            cnd_s = f"{cnd:>9.2e}" if np.isfinite(cnd) else f"{'n/a':>9}"
            lines.append(
                f"{n:<14} {f:>12.4f} {p:>12.4f} {d:>+12.4f} {pct_s}  "
                f"{pyr:>9.4f} {fir:>9.4f}  {cnd_s}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode-level diagnostics: participation, classification, file-poly residuals
# ---------------------------------------------------------------------------

def participation(shape) -> tuple[float, float, float]:
    """Return ``(fa, ss, torsion)`` participation fractions for one mode.

    Each fraction is ``Σ component² / Σ_all_components²``. They sum to 1.

    Note that twist is in radians and translational displacements in
    eigenvector units; the ratio is therefore not a strict energy share
    but a relative-magnitude comparison. Useful for separating bending
    modes (FA/SS dominant) from torsion-coupled or axial-coupled modes.
    """
    fa = float(np.sum(shape.flap_disp ** 2))
    ss = float(np.sum(shape.lag_disp ** 2))
    tw = float(np.sum(shape.twist ** 2))
    total = fa + ss + tw
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return (fa / total, ss / total, tw / total)


def classify_label(p_fa: float, p_ss: float, p_tw: float) -> str:
    """Compact label describing which DOF dominates a mode."""
    if p_tw >= 0.50:
        return "TORSION"
    if p_fa >= 0.70:
        return "FA"
    if p_ss >= 0.70:
        return "SS"
    if p_fa >= 0.40 and p_ss >= 0.40:
        return "FA+SS mixed"
    if p_fa > p_ss:
        return f"FA-leaning ({p_fa:.2f})"
    return f"SS-leaning ({p_ss:.2f})"


def file_poly_residual_for_tower(
    shape,
    file_coeffs: tuple[float, ...],
    is_fa: bool,
) -> float:
    """RMS residual of the FILE polynomial evaluated against pyBmodes shape.

    Mirrors the same shape-preprocessing pyBmodes does internally before
    its own fit (``_remove_root_rigid_motion`` then tip-normalise), so the
    two residual columns are directly comparable.
    """
    if is_fa:
        disp = _remove_root_rigid_motion(shape.span_loc, shape.flap_disp, shape.flap_slope)
    else:
        disp = _remove_root_rigid_motion(shape.span_loc, shape.lag_disp, shape.lag_slope)
    tip = disp[-1]
    if abs(tip) < 1e-30:
        return float("nan")
    y_norm = disp / tip
    x = shape.span_loc
    c2, c3, c4, c5, c6 = file_coeffs
    y_poly = c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4 + c5 * x ** 5 + c6 * x ** 6
    return float(np.sqrt(np.mean((y_poly - y_norm) ** 2)))


def file_poly_residual_for_blade(
    shape,
    file_coeffs: tuple[float, ...],
    is_flap: bool,
) -> float:
    """Same as :func:`file_poly_residual_for_tower` but for the blade path.

    Blade fits don't subtract a root rigid-motion term (the cantilever
    BC is enforced directly in the FEM), so we only tip-normalise.
    """
    disp = shape.flap_disp if is_flap else shape.lag_disp
    tip = disp[-1]
    if abs(tip) < 1e-30:
        return float("nan")
    y_norm = disp / tip
    x = shape.span_loc
    c2, c3, c4, c5, c6 = file_coeffs
    y_poly = c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4 + c5 * x ** 5 + c6 * x ** 6
    return float(np.sqrt(np.mean((y_poly - y_norm) ** 2)))


def tower_diagnostic_table(modal) -> str:
    """One row per extracted tower mode with FA/SS/torsion participation."""
    lines = []
    header = (
        f"{'Mode':>4} {'Freq (Hz)':>10}  {'p_FA':>6}  {'p_SS':>6}  "
        f"{'p_tor':>6}  {'Label':<14}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for shape in modal.shapes:
        p_fa, p_ss, p_tw = participation(shape)
        label = classify_label(p_fa, p_ss, p_tw)
        lines.append(
            f"{shape.mode_number:>4} {shape.freq_hz:>10.4f}  "
            f"{p_fa:>6.3f}  {p_ss:>6.3f}  {p_tw:>6.3f}  {label:<14}"
        )
    return "\n".join(lines)


def classifier_rationale(report) -> str:
    """Print why each tower mode was (or wasn't) selected for FA1/FA2/SS1/SS2.

    Mirrors the criterion in ``_select_tower_family``:

      * lowest-frequency candidate of each direction wins family rank 1.
      * for rank 2: walk up by frequency and accept the first mode whose
        clamped-base polynomial fit has ``rms_residual <= 0.09``.
      * fallback: if no mode in the rank-2+ tail meets that threshold,
        the candidate with the smallest fit residual is taken.
    """
    lines = []
    lines.append("Tower FA family (sorted by frequency):")
    lines.append(
        f"  {'mode':>4} {'freq Hz':>9} {'rank':>4} "
        f"{'fa_rms':>8} {'ss_rms':>8} {'fit_rms':>8} {'good?':>6} {'sel?':>5}"
    )
    for m in report.fa_family:
        lines.append(
            f"  {m.mode_number:>4} {m.frequency_hz:>9.4f} {m.family_rank:>4} "
            f"{m.fa_rms:>8.4f} {m.ss_rms:>8.4f} {m.fit_rms:>8.4f} "
            f"{('yes' if m.fit_is_good else 'no'):>6} "
            f"{('YES' if m.selected else '-'):>5}"
        )
    lines.append("")
    lines.append("Tower SS family (sorted by frequency):")
    lines.append(
        f"  {'mode':>4} {'freq Hz':>9} {'rank':>4} "
        f"{'fa_rms':>8} {'ss_rms':>8} {'fit_rms':>8} {'good?':>6} {'sel?':>5}"
    )
    for m in report.ss_family:
        lines.append(
            f"  {m.mode_number:>4} {m.frequency_hz:>9.4f} {m.family_rank:>4} "
            f"{m.fa_rms:>8.4f} {m.ss_rms:>8.4f} {m.fit_rms:>8.4f} "
            f"{('yes' if m.fit_is_good else 'no'):>6} "
            f"{('YES' if m.selected else '-'):>5}"
        )
    lines.append("")
    lines.append(f"Selected: FA1=mode {report.selected_fa_modes[0]}, "
                 f"FA2=mode {report.selected_fa_modes[1]}, "
                 f"SS1=mode {report.selected_ss_modes[0]}, "
                 f"SS2=mode {report.selected_ss_modes[1]}")
    lines.append("Decision rule: rank-1 by lowest frequency in family; "
                 "rank-2 by first fit_rms <= 0.09 walking up in frequency.")
    return "\n".join(lines)


def tower_mode_names() -> list[str]:
    return [f"{m}({k})" for m in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh")
            for k in range(2, 7)]


def blade_mode_names() -> list[str]:
    return [f"{m}({k})" for m in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh")
            for k in range(2, 7)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tower_modes(modal, report, params, out_path: pathlib.Path) -> None:
    """Plot tower FA1, SS1, FA2, SS2 displacement shapes vs span fraction."""
    fa1_mode, fa2_mode = report.selected_fa_modes
    ss1_mode, ss2_mode = report.selected_ss_modes

    # NodeModeShape stores both FA and SS components per mode (col labels for
    # tower: col1 = SS, col3 = FA). We pull the *dominant* component for each
    # selected mode rather than guessing column.
    def get(mode_no: int):
        s = modal.shapes[mode_no - 1]
        return s

    fa1, fa2 = get(fa1_mode), get(fa2_mode)
    ss1, ss2 = get(ss1_mode), get(ss2_mode)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False)
    f = modal.frequencies

    ax = axs[0]
    ax.plot(fa1.span_loc, fa1.flap_disp / fa1.flap_disp[-1],
            "-o", label=f"FA1 (mode {fa1_mode}, {f[fa1_mode-1]:.3f} Hz)")
    ax.plot(fa2.span_loc, fa2.flap_disp / fa2.flap_disp[-1],
            "-s", label=f"FA2 (mode {fa2_mode}, {f[fa2_mode-1]:.3f} Hz)")
    # Overlay polynomial fits for FA modes for visual cross-check.
    x = np.linspace(0, 1, 51)
    ax.plot(x, params.TwFAM1Sh.evaluate(x), "k--", alpha=0.5, label="poly fit")
    ax.plot(x, params.TwFAM2Sh.evaluate(x), "k--", alpha=0.5)
    ax.set_title("Tower fore-aft modes")
    ax.set_xlabel("span fraction")
    ax.set_ylabel("normalised tip displacement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.plot(ss1.span_loc, ss1.lag_disp / ss1.lag_disp[-1],
            "-o", label=f"SS1 (mode {ss1_mode}, {f[ss1_mode-1]:.3f} Hz)")
    ax.plot(ss2.span_loc, ss2.lag_disp / ss2.lag_disp[-1],
            "-s", label=f"SS2 (mode {ss2_mode}, {f[ss2_mode-1]:.3f} Hz)")
    ax.plot(x, params.TwSSM1Sh.evaluate(x), "k--", alpha=0.5, label="poly fit")
    ax.plot(x, params.TwSSM2Sh.evaluate(x), "k--", alpha=0.5)
    ax.set_title("Tower side-side modes")
    ax.set_xlabel("span fraction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("IEA-3.4MW tower mode shapes (pyBmodes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_all_tower_modes(modal, n_show: int, out_path: pathlib.Path) -> None:
    """Small-multiples grid of every extracted tower mode shape.

    For each mode plots the FA (flap) and SS (lag) displacement components
    overlaid on the same axes. Twist is annotated in the title via the
    participation fraction so torsion-coupled modes stand out.
    """
    n = min(n_show, len(modal.shapes))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows),
                            sharex=True, sharey=False)
    axs = np.atleast_2d(axs).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axs[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue
        shape = modal.shapes[i]
        # Normalise each component by its own maximum-magnitude value so the
        # smaller component stays visible. (Normalising both by FA's tip
        # value would squash modes where SS dominates.)
        def norm(arr):
            m = np.max(np.abs(arr))
            return arr / m if m > 0 else arr

        ax.plot(shape.span_loc, norm(shape.flap_disp), "-o", ms=3,
                label="FA (flap)")
        ax.plot(shape.span_loc, norm(shape.lag_disp), "-s", ms=3,
                label="SS (lag)")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.grid(alpha=0.3)
        p_fa, p_ss, p_tw = participation(shape)
        label = classify_label(p_fa, p_ss, p_tw)
        ax.set_title(
            f"Mode {shape.mode_number}: {shape.freq_hz:.3f} Hz\n"
            f"{label}  (twist={p_tw:.2f})",
            fontsize=9,
        )
        if i // cols == rows - 1:
            ax.set_xlabel("span fraction")
        if i % cols == 0:
            ax.set_ylabel("normalised disp")
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("IEA-3.4MW tower mode shapes — all extracted modes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_blade_modes(modal_0, modal_rated, params_rated, out_path: pathlib.Path) -> None:
    """Plot flap1, edge1, flap2 mode shapes at 0 rpm and 10.5 rpm."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # We read the flap-dominated and edge-dominated *first* couple of modes
    # via the same classifier compute_blade_params uses. Fall back to using
    # mode index in modal.shapes after fitting is done — the params already
    # carry the polynomials, but we want the raw FEM curves to show the fit
    # quality. Match by sorting flap-dominant vs edge-dominant.
    from pybmodes.elastodyn.params import _sorted_modes
    flap_0 = _sorted_modes(modal_0.shapes, fa_dominated=True)
    edge_0 = _sorted_modes(modal_0.shapes, fa_dominated=False)
    flap_r = _sorted_modes(modal_rated.shapes, fa_dominated=True)
    edge_r = _sorted_modes(modal_rated.shapes, fa_dominated=False)

    ax = axs[0]
    f0 = modal_0.frequencies
    ax.plot(flap_0[0].span_loc, flap_0[0].flap_disp / flap_0[0].flap_disp[-1],
            "-o", label=f"flap1 ({flap_0[0].freq_hz:.3f} Hz)")
    ax.plot(flap_0[1].span_loc, flap_0[1].flap_disp / flap_0[1].flap_disp[-1],
            "-s", label=f"flap2 ({flap_0[1].freq_hz:.3f} Hz)")
    ax.plot(edge_0[0].span_loc, edge_0[0].lag_disp / edge_0[0].lag_disp[-1],
            "-^", label=f"edge1 ({edge_0[0].freq_hz:.3f} Hz)")
    ax.set_title("Blade modes @ 0 rpm")
    ax.set_xlabel("span fraction")
    ax.set_ylabel("normalised tip displacement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axs[1]
    x = np.linspace(0, 1, 51)
    ax.plot(flap_r[0].span_loc, flap_r[0].flap_disp / flap_r[0].flap_disp[-1],
            "-o", label=f"flap1 ({flap_r[0].freq_hz:.3f} Hz)")
    ax.plot(flap_r[1].span_loc, flap_r[1].flap_disp / flap_r[1].flap_disp[-1],
            "-s", label=f"flap2 ({flap_r[1].freq_hz:.3f} Hz)")
    ax.plot(edge_r[0].span_loc, edge_r[0].lag_disp / edge_r[0].lag_disp[-1],
            "-^", label=f"edge1 ({edge_r[0].freq_hz:.3f} Hz)")
    # Overlay fits at rated rpm.
    ax.plot(x, params_rated.BldFl1Sh.evaluate(x), "k--", alpha=0.4, label="poly fit")
    ax.plot(x, params_rated.BldFl2Sh.evaluate(x), "k--", alpha=0.4)
    ax.plot(x, params_rated.BldEdgSh.evaluate(x), "k--", alpha=0.4)
    ax.set_title(f"Blade modes @ {RATED_RPM} rpm (rated)")
    ax.set_xlabel("span fraction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("IEA-3.4MW blade mode shapes (pyBmodes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not MAIN_DAT.is_file():
        print(f"ERROR: missing ElastoDyn deck at {MAIN_DAT}", file=sys.stderr)
        return 1

    # 1. Verbatim file coefficients (pulled by the reader, not re-fit).
    main_ed = read_elastodyn_main(MAIN_DAT)
    tower_ed = read_elastodyn_tower(MAIN_DAT.parent / main_ed.twr_file)
    blade_ed = read_elastodyn_blade(MAIN_DAT.parent / main_ed.bld_file[0])

    file_tow = file_tower_coeffs(tower_ed)
    file_bld = file_blade_coeffs(blade_ed)

    # 2. Tower solve (RNA at top via Tower.from_elastodyn).
    tower = Tower.from_elastodyn(MAIN_DAT)
    tower_modal = tower.run(n_modes=10)
    tower_params, tower_report = compute_tower_params_report(tower_modal)
    py_tow = tower_params.as_dict()

    # 3. Blade solves at 0 rpm and rated 10.5 rpm (override RotSpeed in-place).
    blade_static = RotatingBlade.from_elastodyn(MAIN_DAT)
    blade_static._bmi.rot_rpm = 0.0
    modal_0 = blade_static.run(n_modes=8)
    params_0 = compute_blade_params(modal_0)

    blade_rated = RotatingBlade.from_elastodyn(MAIN_DAT)
    blade_rated._bmi.rot_rpm = RATED_RPM
    modal_rated = blade_rated.run(n_modes=8)
    params_rated = compute_blade_params(modal_rated)

    # 4. Print summary
    out = []
    out.append("=" * 76)
    out.append("IEA-3.4MW land-based ElastoDyn deck — pyBmodes exploratory comparison")
    out.append("=" * 76)
    out.append("")
    out.append(f"Source: {MAIN_DAT.relative_to(ROOT).as_posix()}")
    out.append(f"Tower file:  {main_ed.twr_file}")
    out.append(f"Blade file:  {main_ed.bld_file[0]}")
    out.append("")
    out.append(f"TipRad = {main_ed.tip_rad:.3f} m, HubRad = {main_ed.hub_rad:.3f} m")
    out.append(f"TowerHt = {main_ed.tower_ht:.3f} m, HubHt(derived) = {main_ed.hub_ht:.3f} m")
    out.append(f"NacMass = {main_ed.nac_mass:.0f} kg, HubMass = {main_ed.hub_mass:.0f} kg")
    out.append("")
    out.append("Computed natural frequencies (Hz)")
    out.append(f"  Tower (n=10):           "
               + ", ".join(f"{f:.4f}" for f in tower_modal.frequencies))
    out.append(f"  Blade @ 0 rpm (n=8):    "
               + ", ".join(f"{f:.4f}" for f in modal_0.frequencies))
    out.append(f"  Blade @ {RATED_RPM} rpm (n=8):  "
               + ", ".join(f"{f:.4f}" for f in modal_rated.frequencies))
    out.append("")
    out.append(f"Tower FA family selected modes: {tower_report.selected_fa_modes}")
    out.append(f"Tower SS family selected modes: {tower_report.selected_ss_modes}")
    out.append("")

    # ----- Mode-level diagnostics -----------------------------------------
    out.append("Tower modes — DOF participation and classification (raw eigensolver output)")
    out.append(tower_diagnostic_table(tower_modal))
    out.append("")

    # Show post-rotation participation for any degenerate pair the resolver
    # would touch. compute_tower_params_report applies this rotation
    # internally before fitting; we replay it here just for display so the
    # raw eigensolver output above can stay untouched for inspection.
    rotated_for_view = _rotate_degenerate_pairs(tower_modal.shapes)
    rotation_lines = []
    for i in range(len(tower_modal.shapes) - 1):
        if _is_degenerate_pair(tower_modal.shapes[i], tower_modal.shapes[i + 1]):
            for k in (i, i + 1):
                p_fa_pre, p_ss_pre = _shape_participation(tower_modal.shapes[k])
                p_fa_post, p_ss_post = _shape_participation(rotated_for_view[k])
                rotation_lines.append(
                    f"  mode {tower_modal.shapes[k].mode_number}: "
                    f"pre p_FA={p_fa_pre:.3f}/p_SS={p_ss_pre:.3f}  ->  "
                    f"post p_FA={p_fa_post:.3f}/p_SS={p_ss_post:.3f}"
                )
    if rotation_lines:
        out.append("Degenerate-pair rotation (applied internally before fitting):")
        out.extend(rotation_lines)
        out.append("")

    out.append("Tower FA/SS classifier rationale")
    out.append(classifier_rationale(tower_report))
    out.append("")

    # ----- Per-mode RMS residuals for the residual columns ----------------
    # pyBmodes residuals come straight from the fit objects.
    py_tow_resid = {
        "TwFAM1Sh": tower_params.TwFAM1Sh.rms_residual,
        "TwFAM2Sh": tower_params.TwFAM2Sh.rms_residual,
        "TwSSM1Sh": tower_params.TwSSM1Sh.rms_residual,
        "TwSSM2Sh": tower_params.TwSSM2Sh.rms_residual,
    }
    # File residuals — file polynomial evaluated against pyBmodes shape.
    fa1_idx, fa2_idx = tower_report.selected_fa_modes
    ss1_idx, ss2_idx = tower_report.selected_ss_modes
    file_tow_resid = {
        "TwFAM1Sh": file_poly_residual_for_tower(
            tower_modal.shapes[fa1_idx - 1], tuple(tower_ed.tw_fa_m1_sh), is_fa=True),
        "TwFAM2Sh": file_poly_residual_for_tower(
            tower_modal.shapes[fa2_idx - 1], tuple(tower_ed.tw_fa_m2_sh), is_fa=True),
        "TwSSM1Sh": file_poly_residual_for_tower(
            tower_modal.shapes[ss1_idx - 1], tuple(tower_ed.tw_ss_m1_sh), is_fa=False),
        "TwSSM2Sh": file_poly_residual_for_tower(
            tower_modal.shapes[ss2_idx - 1], tuple(tower_ed.tw_ss_m2_sh), is_fa=False),
    }

    # Blade residuals: pull the shapes the same way compute_blade_params does.
    flap_rated = _sorted_modes(modal_rated.shapes, fa_dominated=True)
    edge_rated = _sorted_modes(modal_rated.shapes, fa_dominated=False)
    flap_0 = _sorted_modes(modal_0.shapes, fa_dominated=True)
    edge_0 = _sorted_modes(modal_0.shapes, fa_dominated=False)

    py_bld_rated_resid = {
        "BldFl1Sh": params_rated.BldFl1Sh.rms_residual,
        "BldFl2Sh": params_rated.BldFl2Sh.rms_residual,
        "BldEdgSh": params_rated.BldEdgSh.rms_residual,
    }
    file_bld_rated_resid = {
        "BldFl1Sh": file_poly_residual_for_blade(
            flap_rated[0], tuple(blade_ed.bld_fl1_sh), is_flap=True),
        "BldFl2Sh": file_poly_residual_for_blade(
            flap_rated[1], tuple(blade_ed.bld_fl2_sh), is_flap=True),
        "BldEdgSh": file_poly_residual_for_blade(
            edge_rated[0], tuple(blade_ed.bld_edg_sh), is_flap=False),
    }
    py_bld_0_resid = {
        "BldFl1Sh": params_0.BldFl1Sh.rms_residual,
        "BldFl2Sh": params_0.BldFl2Sh.rms_residual,
        "BldEdgSh": params_0.BldEdgSh.rms_residual,
    }
    file_bld_0_resid = {
        "BldFl1Sh": file_poly_residual_for_blade(
            flap_0[0], tuple(blade_ed.bld_fl1_sh), is_flap=True),
        "BldFl2Sh": file_poly_residual_for_blade(
            flap_0[1], tuple(blade_ed.bld_fl2_sh), is_flap=True),
        "BldEdgSh": file_poly_residual_for_blade(
            edge_0[0], tuple(blade_ed.bld_edg_sh), is_flap=False),
    }

    # Per-mode polynomial-fit condition numbers (same value across all five
    # rows of a given coefficient block — depends only on spanwise sampling).
    py_tow_cond = {
        "TwFAM1Sh": tower_params.TwFAM1Sh.cond_number,
        "TwFAM2Sh": tower_params.TwFAM2Sh.cond_number,
        "TwSSM1Sh": tower_params.TwSSM1Sh.cond_number,
        "TwSSM2Sh": tower_params.TwSSM2Sh.cond_number,
    }
    py_bld_rated_cond = {
        "BldFl1Sh": params_rated.BldFl1Sh.cond_number,
        "BldFl2Sh": params_rated.BldFl2Sh.cond_number,
        "BldEdgSh": params_rated.BldEdgSh.cond_number,
    }
    py_bld_0_cond = {
        "BldFl1Sh": params_0.BldFl1Sh.cond_number,
        "BldFl2Sh": params_0.BldFl2Sh.cond_number,
        "BldEdgSh": params_0.BldEdgSh.cond_number,
    }

    out.append("Tower polynomial coefficients (file vs pyBmodes fit) — with RMS residuals + cond")
    out.append("rms* columns: per-mode polynomial residual against pyBmodes shape")
    out.append("  pyB rms*  = pyBmodes fit residual  (PolyFitResult.rms_residual)")
    out.append("  file rms* = file polynomial evaluated against pyBmodes shape")
    out.append("  cond      = 2-norm condition number of the reduced design matrix")
    out.append(diff_table(file_tow, py_tow, tower_mode_names(),
                          py_residuals=py_tow_resid,
                          file_residuals=file_tow_resid,
                          py_cond=py_tow_cond))
    out.append("")

    out.append(f"Blade polynomial coefficients @ {RATED_RPM} rpm (file vs pyBmodes fit) — with RMS residuals + cond")
    out.append(diff_table(file_bld, params_rated.as_dict(), blade_mode_names(),
                          py_residuals=py_bld_rated_resid,
                          file_residuals=file_bld_rated_resid,
                          py_cond=py_bld_rated_cond))
    out.append("")

    out.append("Blade polynomial coefficients @ 0 rpm (no centrifugal) — with RMS residuals + cond")
    out.append(diff_table(file_bld, params_0.as_dict(), blade_mode_names(),
                          py_residuals=py_bld_0_resid,
                          file_residuals=file_bld_0_resid,
                          py_cond=py_bld_0_cond))
    out.append("")

    text = "\n".join(out)
    print(text)
    (OUT_DIR / "coefficients.txt").write_text(text + "\n", encoding="utf-8")

    # 5. Plots
    plot_tower_modes(tower_modal, tower_report, tower_params,
                     OUT_DIR / "tower_modes.png")
    plot_all_tower_modes(tower_modal, n_show=8,
                         out_path=OUT_DIR / "tower_modes_all.png")
    plot_blade_modes(modal_0, modal_rated, params_rated,
                     OUT_DIR / "blade_modes.png")

    print()
    print(f"Wrote outputs to {OUT_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  - coefficients.txt")
    print(f"  - tower_modes.png         (FA1/FA2/SS1/SS2 with poly-fit overlay)")
    print(f"  - tower_modes_all.png     (all 8 extracted modes, small multiples)")
    print(f"  - blade_modes.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
