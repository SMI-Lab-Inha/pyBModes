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

"""Polynomial-vs-FEM comparison for the IEA-3.4-130-RWT land-based tower.

Mirror of :mod:`visualise_polynomial_comparison` but for the
IEA-3.4-130-RWT (Bortolotti et al. 2019, IEA Wind Task 37) deck and
plotting *both* the 1st and 2nd tower bending modes in separate
figures — the IEA-3.4 1st-mode coefficients are ~ 49 × off the
pyBmodes baseline (vs ~ 300 × for NREL 5MW), so the 1st modes are
worth a visual too.

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\visualise_polynomial_comparison_iea34.py

Outputs ``scripts/outputs/polynomial_comparison_iea34_TwFA2_TwSS2.png``
and ``polynomial_comparison_iea34_TwFA1_TwSS1.png``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from visualise_polynomial_comparison import (  # noqa: E402
    _eval_poly,
    _has_inflection,
    _print_diagnostic_table,
)

from pybmodes.elastodyn.params import (  # noqa: E402
    _remove_root_rigid_motion,
    compute_tower_params_report,
)
from pybmodes.fitting.poly_fit import PolyFitResult  # noqa: E402
from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.models import Tower  # noqa: E402

_IEA34_DECK = (
    REPO_ROOT
    / "external" / "OpenFAST_files" / "IEA-3.4-130-RWT" / "openfast"
    / "IEA-3.4-130-RWT_ElastoDyn.dat"
)


def _coeffs_from_fit(fit: PolyFitResult) -> np.ndarray:
    return np.asarray(fit.coefficients(), dtype=float)


def _amplitude_ratio(pyb_coeffs: np.ndarray, ref_coeffs: np.ndarray) -> float:
    """``max |phi_ref|`` / ``max |phi_pyB|`` on a 1001-point grid."""
    fine_x = np.linspace(0.0, 1.0, 1001)
    pyb_max = float(np.max(np.abs(_eval_poly(pyb_coeffs, fine_x))))
    ref_max = float(np.max(np.abs(_eval_poly(ref_coeffs, fine_x))))
    if pyb_max <= 0.0:
        return float("inf")
    return ref_max / pyb_max


def _plot_pair(
    ax_left,
    ax_right,
    *,
    left_title: str,
    right_title: str,
    span_loc: np.ndarray,
    disp_left: np.ndarray,
    disp_right: np.ndarray,
    pyb_coeffs_left: np.ndarray,
    pyb_coeffs_right: np.ndarray,
    ref_coeffs_left: np.ndarray,
    ref_coeffs_right: np.ndarray,
) -> None:
    x = np.linspace(0.0, 1.0, 100)
    panels = [
        (ax_left, left_title, span_loc, disp_left,
         pyb_coeffs_left, ref_coeffs_left),
        (ax_right, right_title, span_loc, disp_right,
         pyb_coeffs_right, ref_coeffs_right),
    ]
    for ax, title, span, disp, pyb_c, ref_c in panels:
        ax.plot(x, _eval_poly(ref_c, x), "-",
                color=(0.85, 0.0, 0.0),  # standard red
                linewidth=1.6, label="IEA-3.4 repo polynomial")
        ax.plot(x, _eval_poly(pyb_c, x), "-",
                color=(0.0, 0.0, 0.85),  # standard blue
                linewidth=1.6, label="pyBmodes polynomial")
        ax.scatter(span, disp, s=22, facecolors="none",
                   edgecolors=(0.20, 0.20, 0.20), linewidths=0.9,
                   label="pyBmodes FEM (raw)")
        ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=0.5, zorder=0)
        ax.axhline(1.0, color=(0.6, 0.6, 0.6), linewidth=0.5,
                   linestyle=":", zorder=0)
        ax.set_xlabel("Normalised tower height  z / H")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True)
        ax.legend(loc="best", fontsize=9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=_IEA34_DECK,
        help="ElastoDyn main .dat (default: IEA-3.4 land-based deck under "
             "external/OpenFAST_files/)",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input deck not found: {args.input}", file=sys.stderr)
        print(
            "  IEA-3.4-130-RWT data is gitignored under external/OpenFAST_files/;\n"
            "  see VALIDATION.md for how to clone it.",
            file=sys.stderr,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input}")
    main_deck = read_elastodyn_main(args.input)
    tower = read_elastodyn_tower(args.input.parent / main_deck.twr_file)

    print("  building pyBmodes Tower model + solving FEM ...")
    model = Tower.from_elastodyn(args.input)
    modal = model.run(n_modes=10)
    params, report = compute_tower_params_report(modal)

    by_mode = {s.mode_number: s for s in modal.shapes}
    fa1_n, fa2_n = report.selected_fa_modes
    ss1_n, ss2_n = report.selected_ss_modes
    fa1_shape = by_mode[fa1_n]
    fa2_shape = by_mode[fa2_n]
    ss1_shape = by_mode[ss1_n]
    ss2_shape = by_mode[ss2_n]
    print(f"  selected FA1 = mode {fa1_n} ({fa1_shape.freq_hz:.4f} Hz)")
    print(f"  selected FA2 = mode {fa2_n} ({fa2_shape.freq_hz:.4f} Hz)")
    print(f"  selected SS1 = mode {ss1_n} ({ss1_shape.freq_hz:.4f} Hz)")
    print(f"  selected SS2 = mode {ss2_n} ({ss2_shape.freq_hz:.4f} Hz)")

    def _normalised(shape, is_fa: bool) -> np.ndarray:
        if is_fa:
            disp = _remove_root_rigid_motion(
                shape.span_loc, shape.flap_disp, shape.flap_slope,
            )
        else:
            disp = _remove_root_rigid_motion(
                shape.span_loc, shape.lag_disp, shape.lag_slope,
            )
        return disp / disp[-1]

    fa1_disp = _normalised(fa1_shape, is_fa=True)
    fa2_disp = _normalised(fa2_shape, is_fa=True)
    ss1_disp = _normalised(ss1_shape, is_fa=False)
    ss2_disp = _normalised(ss2_shape, is_fa=False)

    pyb_fa1 = _coeffs_from_fit(params.TwFAM1Sh)
    pyb_fa2 = _coeffs_from_fit(params.TwFAM2Sh)
    pyb_ss1 = _coeffs_from_fit(params.TwSSM1Sh)
    pyb_ss2 = _coeffs_from_fit(params.TwSSM2Sh)
    ref_fa1 = np.asarray(tower.tw_fa_m1_sh, dtype=float)
    ref_fa2 = np.asarray(tower.tw_fa_m2_sh, dtype=float)
    ref_ss1 = np.asarray(tower.tw_ss_m1_sh, dtype=float)
    ref_ss2 = np.asarray(tower.tw_ss_m2_sh, dtype=float)

    # Diagnostics — same format as the 5MW script for direct comparison.
    _print_diagnostic_table("TwFAM1Sh", pyb_fa1, ref_fa1)
    _print_diagnostic_table("TwSSM1Sh", pyb_ss1, ref_ss1)
    _print_diagnostic_table("TwFAM2Sh", pyb_fa2, ref_fa2)
    _print_diagnostic_table("TwSSM2Sh", pyb_ss2, ref_ss2)

    print()
    print("=== Amplitude ratios (max|phi_rtest| / max|phi_pyB| on [0, 1]) ===")
    print(f"  TwFAM1Sh: {_amplitude_ratio(pyb_fa1, ref_fa1):>5.2f} x")
    print(f"  TwSSM1Sh: {_amplitude_ratio(pyb_ss1, ref_ss1):>5.2f} x")
    print(f"  TwFAM2Sh: {_amplitude_ratio(pyb_fa2, ref_fa2):>5.2f} x")
    print(f"  TwSSM2Sh: {_amplitude_ratio(pyb_ss2, ref_ss2):>5.2f} x")

    # Plot.
    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")
    import matplotlib.pyplot as plt

    # ---- Figure 2: 2nd modes ----------------------------------------
    out_2nd = args.out_dir / "polynomial_comparison_iea34_TwFA2_TwSS2.png"
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    _plot_pair(
        axes2[0], axes2[1],
        left_title="TwFAM2Sh — fore-aft 2nd bending",
        right_title="TwSSM2Sh — side-side 2nd bending",
        span_loc=fa2_shape.span_loc,  # FA and SS share span_loc grid
        disp_left=fa2_disp, disp_right=ss2_disp,
        pyb_coeffs_left=pyb_fa2, pyb_coeffs_right=pyb_ss2,
        ref_coeffs_left=ref_fa2, ref_coeffs_right=ref_ss2,
    )
    axes2[0].set_ylabel("Modal displacement  phi(z)")
    fig2.suptitle(
        f"IEA-3.4-130-RWT tower: 2nd-mode polynomial vs FEM mode shape — "
        f"{args.input.name}",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(out_2nd)
    print()
    print(f"Wrote {out_2nd}")

    # ---- Figure 1: 1st modes ----------------------------------------
    out_1st = args.out_dir / "polynomial_comparison_iea34_TwFA1_TwSS1.png"
    fig1, axes1 = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    _plot_pair(
        axes1[0], axes1[1],
        left_title="TwFAM1Sh — fore-aft 1st bending",
        right_title="TwSSM1Sh — side-side 1st bending",
        span_loc=fa1_shape.span_loc,
        disp_left=fa1_disp, disp_right=ss1_disp,
        pyb_coeffs_left=pyb_fa1, pyb_coeffs_right=pyb_ss1,
        ref_coeffs_left=ref_fa1, ref_coeffs_right=ref_ss1,
    )
    axes1[0].set_ylabel("Modal displacement  phi(z)")
    fig1.suptitle(
        f"IEA-3.4-130-RWT tower: 1st-mode polynomial vs FEM mode shape — "
        f"{args.input.name}",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(out_1st)
    print(f"Wrote {out_1st}")

    # Quick interpretation hint based on the 2nd-mode amplitude ratio.
    fa2_ratio = _amplitude_ratio(pyb_fa2, ref_fa2)
    ss2_ratio = _amplitude_ratio(pyb_ss2, ref_ss2)
    fa2_inflect = _has_inflection(ref_fa2)
    ss2_inflect = _has_inflection(ref_ss2)
    avg_ratio = 0.5 * (fa2_ratio + ss2_ratio)

    print()
    print("=== Interpretation vs NREL 5MW (avg 2nd-mode ratio ~ 2.3 x) ===")
    if not (fa2_inflect and ss2_inflect):
        print("  Outcome C: at least one IEA-3.4 reference 2nd-mode polynomial")
        print("  has no inflection in (0, 1) — non-physical. Worse than 5MW.")
    elif avg_ratio < 1.7:
        print(f"  Outcome A: average 2nd-mode amplitude ratio {avg_ratio:.2f} x")
        print("  is well below 5MW's 2.3 x. The IEA-3.4 upstream pipeline is")
        print("  measurably more internally consistent than NREL 5MW's legacy")
        print("  Modes-tool polynomials.")
    elif avg_ratio < 3.0:
        print(f"  Outcome B: average 2nd-mode amplitude ratio {avg_ratio:.2f} x")
        print("  is comparable to NREL 5MW's 2.3 x — same magnitude of issue,")
        print("  different upstream source. Suggests a systematic problem")
        print("  across reference-turbine pipelines rather than a 5MW-specific")
        print("  legacy artefact.")
    else:
        print(f"  Outcome C: average 2nd-mode amplitude ratio {avg_ratio:.2f} x")
        print("  is far worse than NREL 5MW's 2.3 x. Coefficients describe a")
        print("  mode that differs from the FEM eigenvector by a factor of")
        print("  several in curvature magnitude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
