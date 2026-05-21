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

"""Plot pyBmodes' fit, the upstream r-test polynomial, and the raw FEM
mode shape on the same axes for the NREL 5MW 2nd-bending tower modes
(TwFAM2Sh and TwSSM2Sh).

The :func:`pybmodes.elastodyn.validate_dat_coefficients` validator
reports the upstream r-test deck has TwFAM2Sh and TwSSM2Sh ``file_RMS``
in the 5–6 range (against pyBmodes' fit at ~0.002), but a single
ratio doesn't communicate *what's wrong*. This script visualises the
gap so the question "is the reference polynomial garbage, or is
pyBmodes' fit?" is answered at a glance — by looking at whether each
polynomial actually traces a recognisable 2nd-bending mode shape (zero
at base, monotone slope, one inflection point, peak at the tip).

Each subplot draws three curves over the normalised tower height
``x ∈ [0, 1]``:

  * **pyBmodes raw eigenvector** — the FEM nodal displacements with
    base rigid-body motion subtracted, normalised so ``phi(1) = 1``
    (scatter, the ground truth the polynomials are supposed to
    approximate).
  * **pyBmodes polynomial** — the constrained 6th-order fit to the
    eigenvector, evaluated on a 100-point grid.
  * **Reference polynomial** — the coefficients shipped in the
    upstream OpenFAST r-test deck, evaluated on the same grid.

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\visualise_polynomial_comparison.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

# Default to the upstream r-test deck (gitignored under docs/). When that
# isn't available, fall back to the bundled reference deck — note that
# the reference deck has been patched, so the "reference polynomial"
# series there is pyBmodes' own fit re-saved, and the comparison only
# really demonstrates the gap when the upstream deck is present.
_UPSTREAM_DECK = (
    REPO_ROOT
    / "external" / "OpenFAST_files" / "r-test" / "glue-codes" / "openfast"
    / "5MW_Land_DLL_WTurb" / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)
_REFERENCE_DECK = (
    REPO_ROOT
    / "src" / "pybmodes" / "_examples"
    / "reference_decks" / "nrel5mw_land"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)


def _eval_poly(coeffs: np.ndarray | Iterable[float], x: np.ndarray) -> np.ndarray:
    """Evaluate the ElastoDyn 6th-order polynomial ``[C2..C6]`` at *x*."""
    c2, c3, c4, c5, c6 = (float(c) for c in coeffs)
    return c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4 + c5 * x ** 5 + c6 * x ** 6


def _eval_poly_second_derivative(
    coeffs: np.ndarray | Iterable[float], x: np.ndarray,
) -> np.ndarray:
    """``phi''(x) = 2·C2 + 6·C3·x + 12·C4·x² + 20·C5·x³ + 30·C6·x⁴``."""
    c2, c3, c4, c5, c6 = (float(c) for c in coeffs)
    return (
        2.0 * c2
        + 6.0 * c3 * x
        + 12.0 * c4 * x ** 2
        + 20.0 * c5 * x ** 3
        + 30.0 * c6 * x ** 4
    )


def _has_inflection(coeffs: np.ndarray | Iterable[float]) -> bool:
    """``True`` iff ``phi''(x)`` changes sign somewhere in the open interval ``(0, 1)``.

    The sign-change bracket is found on a fine grid; an exact polynomial-
    root analysis is overkill for the diagnostic question this script
    asks ("is the curve unimodal or does it wiggle?").
    """
    x = np.linspace(1e-3, 1.0 - 1e-3, 1001)
    yp2 = _eval_poly_second_derivative(coeffs, x)
    return bool(np.any(np.diff(np.signbit(yp2))))


def _coeffs_from_fit(fit: PolyFitResult) -> np.ndarray:
    return np.asarray(fit.coefficients(), dtype=float)


def _print_diagnostic_table(
    name: str,
    pyb_coeffs: np.ndarray,
    ref_coeffs: np.ndarray,
) -> None:
    sample_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    fine_x = np.linspace(0.0, 1.0, 1001)

    pyb_sample = _eval_poly(pyb_coeffs, sample_x)
    ref_sample = _eval_poly(ref_coeffs, sample_x)
    pyb_max = float(np.max(np.abs(_eval_poly(pyb_coeffs, fine_x))))
    ref_max = float(np.max(np.abs(_eval_poly(ref_coeffs, fine_x))))

    print()
    print(f"=== {name} ==="
          f"  (pyBmodes coeffs: {pyb_coeffs.round(3).tolist()};"
          f"  reference: {ref_coeffs.round(3).tolist()})")
    print(f"  {'x':>5}   {'pyBmodes phi(x)':>14}   {'reference phi(x)':>14}")
    for x, py, ref in zip(sample_x, pyb_sample, ref_sample):
        print(f"  {x:5.2f}   {py:14.4f}   {ref:14.4f}")
    print(f"  max |phi(x)| on [0, 1]:"
          f"  pyBmodes={pyb_max:.4f}   reference={ref_max:.4f}")
    pyb_inflect = _has_inflection(pyb_coeffs)
    ref_inflect = _has_inflection(ref_coeffs)
    print(f"  inflection in (0, 1)? "
          f"pyBmodes={'YES' if pyb_inflect else 'no'}   "
          f"reference={'YES' if ref_inflect else 'no'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=_UPSTREAM_DECK if _UPSTREAM_DECK.is_file() else _REFERENCE_DECK,
        help="ElastoDyn main .dat file (default: upstream r-test if present, "
             "else the bundled reference deck)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs"
                          / "polynomial_comparison_TwFA2_TwSS2.png",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input deck not found: {args.input}", file=sys.stderr)
        print(
            "  The upstream r-test deck under external/OpenFAST_files/ is "
            "gitignored;\n"
            "  see VALIDATION.md for how to clone it "
            "locally,\n"
            "  or pass --input to point at a different ElastoDyn .dat.",
            file=sys.stderr,
        )
        return 2

    print(f"Reading {args.input}")
    if args.input == _REFERENCE_DECK:
        print("  NOTE: this is the patched reference deck. Its polynomial")
        print("  blocks were regenerated by pyBmodes — the comparison")
        print("  will show pyBmodes-vs-pyBmodes (i.e. exact agreement).")
        print("  For the upstream-vs-pyBmodes contrast that motivates this")
        print("  visualisation, supply the unpatched r-test deck via --input.")

    main_deck = read_elastodyn_main(args.input)
    tower = read_elastodyn_tower(args.input.parent / main_deck.twr_file)

    print("  building pyBmodes Tower model + solving FEM ...")
    model = Tower.from_elastodyn(args.input)
    modal = model.run(n_modes=10)
    params, report = compute_tower_params_report(modal)

    by_mode = {s.mode_number: s for s in modal.shapes}
    fa2_mode_n = report.selected_fa_modes[1]
    ss2_mode_n = report.selected_ss_modes[1]
    fa2_shape = by_mode[fa2_mode_n]
    ss2_shape = by_mode[ss2_mode_n]
    print(f"  selected FA2 = mode {fa2_mode_n} "
          f"({fa2_shape.freq_hz:.4f} Hz)")
    print(f"  selected SS2 = mode {ss2_mode_n} "
          f"({ss2_shape.freq_hz:.4f} Hz)")

    # FEM raw eigenvector (rigid-body root motion subtracted, then
    # tip-normalised so all three curves share the same y-scale).
    fa2_disp = _remove_root_rigid_motion(
        fa2_shape.span_loc, fa2_shape.flap_disp, fa2_shape.flap_slope,
    )
    fa2_disp = fa2_disp / fa2_disp[-1]
    ss2_disp = _remove_root_rigid_motion(
        ss2_shape.span_loc, ss2_shape.lag_disp, ss2_shape.lag_slope,
    )
    ss2_disp = ss2_disp / ss2_disp[-1]

    # Polynomial coefficients.
    pyb_fa2 = _coeffs_from_fit(params.TwFAM2Sh)
    pyb_ss2 = _coeffs_from_fit(params.TwSSM2Sh)
    ref_fa2 = np.asarray(tower.tw_fa_m2_sh, dtype=float)
    ref_ss2 = np.asarray(tower.tw_ss_m2_sh, dtype=float)

    # Diagnostics to stdout.
    _print_diagnostic_table("TwFAM2Sh", pyb_fa2, ref_fa2)
    _print_diagnostic_table("TwSSM2Sh", pyb_ss2, ref_ss2)

    # Plot.
    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")
    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    x = np.linspace(0.0, 1.0, 100)

    panels = [
        (axes[0], "TwFAM2Sh — fore-aft 2nd bending",
         fa2_shape.span_loc, fa2_disp, pyb_fa2, ref_fa2),
        (axes[1], "TwSSM2Sh — side-side 2nd bending",
         ss2_shape.span_loc, ss2_disp, pyb_ss2, ref_ss2),
    ]
    for ax, title, span, disp, pyb_coeffs, ref_coeffs in panels:
        ax.plot(x, _eval_poly(ref_coeffs, x), "-",
                color=(0.85, 0.0, 0.0),  # standard red
                linewidth=1.6, label="r-test polynomial")
        ax.plot(x, _eval_poly(pyb_coeffs, x), "-",
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
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Modal displacement  phi(z)")

    fig.suptitle(
        f"NREL 5MW tower: 2nd-mode polynomial vs FEM mode shape — "
        f"{args.input.name}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out)
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
