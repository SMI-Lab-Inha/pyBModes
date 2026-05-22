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

"""Polynomial-vs-FEM comparison for the NREL 5MW OC3 Monopile tower.

Mirror of :mod:`visualise_polynomial_comparison` (NREL 5MW land-based)
and :mod:`visualise_polynomial_comparison_iea34` (IEA-3.4-130-RWT
land-based) — same plot format, same diagnostic table, but for the
OC3-style monopile-supported variant of the NREL 5MW shipped under
``r-test/glue-codes/openfast/5MW_OC3Mnpl_DLL_WTurb_WavesIrr/``.

The polynomial blocks shipped in the monopile ElastoDyn deck describe
the **tower-only** 2nd bending shape (over ``z ∈ [TowerBsHt, TowerHt]``,
77.6 m on the OC3 deck), while pyBmodes' :meth:`Tower.from_elastodyn_with_subdyn`
solves the **combined pile + tower** cantilever (107.6 m, clamped at
the SubDyn reaction joint at z = −20 m). To compare apples to apples
this script:

1. Solves the combined-cantilever FEM via the validated SubDyn-splice
   path and identifies the 1st and 2nd FA / SS modes of the *system*
   by participation.
2. Extracts the tower segment of each mode shape (the portion of the
   nodal data with ``z >= TowerBsHt``).
3. Subtracts the rigid-body translation and rotation at the
   transition piece (TP) so what remains is the tower-bending part
   alone — same convention the validator uses for offshore towers via
   :func:`pybmodes.elastodyn.params._remove_root_rigid_motion`.
4. Re-bases the tower segment to local coordinates ``x ∈ [0, 1]`` and
   tip-normalises so ``φ(1) = 1``.
5. Fits a constrained 6th-order polynomial to that tower-segment
   shape and compares it to the polynomial coefficients shipped in
   the deck.

Outputs ``scripts/outputs/polynomial_comparison_5mw_monopile_TwFA2_TwSS2.png``.
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
    _print_diagnostic_table,
)

from pybmodes.fitting.poly_fit import PolyFitResult, fit_mode_shape  # noqa: E402
from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.io.subdyn_reader import read_subdyn  # noqa: E402
from pybmodes.models import Tower  # noqa: E402

_OC3_DECK_DIR = (
    REPO_ROOT
    / "external" / "OpenFAST_files" / "r-test" / "glue-codes" / "openfast"
    / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
)
_OC3_MAIN = _OC3_DECK_DIR / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat"
_OC3_SUBDYN = _OC3_DECK_DIR / "NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat"


def _participation(shape) -> tuple[float, float, float]:
    """``(p_FA, p_SS, p_torsion)`` energy fractions for one mode."""
    fa = float(np.sum(shape.flap_disp ** 2))
    ss = float(np.sum(shape.lag_disp ** 2))
    tw = float(np.sum(shape.twist ** 2))
    total = fa + ss + tw
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return (fa / total, ss / total, tw / total)


def _select_first_two(shapes, *, want_fa: bool) -> tuple[int, int]:
    """Return the indices of the lowest-frequency 1st and 2nd FA-dominated
    (or SS-dominated) modes from a sorted ``shapes`` list.

    Combined-cantilever spectra interleave pile-bending, tower-bending,
    and torsion modes. We scan in ascending-frequency order and accept
    the first two modes whose dominant axis matches the requested
    direction (``p_axis ≥ 0.6``). Bare-minimum classifier; sufficient
    on the OC3 monopile where the relevant modes are well-separated in
    participation from torsion-dominated ones.
    """
    out: list[int] = []
    for idx, shape in enumerate(shapes):
        p_fa, p_ss, _ = _participation(shape)
        is_fa = p_fa >= 0.6
        is_ss = p_ss >= 0.6
        if (want_fa and is_fa) or ((not want_fa) and is_ss):
            out.append(idx)
        if len(out) == 2:
            return (out[0], out[1])
    raise RuntimeError(
        f"could not find two {'FA' if want_fa else 'SS'}-dominated modes "
        f"in the spectrum (got indices {out!r})"
    )


def _extract_tower_segment_bending(
    shape, *, tp_frac: float, is_fa: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_local, phi_bending)`` over the tower-only segment.

    ``x_local`` runs from 0 at the TP (``span_loc == tp_frac``) to 1 at
    the tower top (``span_loc == 1``). ``phi_bending`` is the tower's
    deflection with the rigid-body translation and rotation at the TP
    subtracted, then tip-normalised so ``phi(1) = 1``.
    """
    span = np.asarray(shape.span_loc, dtype=float)
    if is_fa:
        disp = np.asarray(shape.flap_disp, dtype=float)
        slope = np.asarray(shape.flap_slope, dtype=float)
    else:
        disp = np.asarray(shape.lag_disp, dtype=float)
        slope = np.asarray(shape.lag_slope, dtype=float)

    keep = span >= tp_frac - 1.0e-9
    span_t = span[keep]
    disp_t = disp[keep]
    slope_t = slope[keep]

    # Subtract the tangent line at the TP using combined-coord slope:
    #   phi_bending(s) = phi(s) − phi(tp_frac) − phi'(tp_frac) · (s − tp_frac).
    # The combined-coord and tower-local slope agree by chain rule on
    # the resulting bending deflection, so we use combined-coord values
    # directly.
    phi_bend = disp_t - disp_t[0] - slope_t[0] * (span_t - tp_frac)

    # Re-base span to tower-local [0, 1].
    seg_length = 1.0 - tp_frac
    x_local = (span_t - tp_frac) / seg_length

    # Tip-normalise.
    tip = phi_bend[-1]
    if abs(tip) < 1.0e-30:
        raise RuntimeError(
            "tower-segment tip displacement is essentially zero — "
            "mode-classification likely picked a pile-dominated mode"
        )
    phi_norm = phi_bend / tip
    return x_local, phi_norm


def _amplitude_ratio(pyb_coeffs: np.ndarray, ref_coeffs: np.ndarray) -> float:
    fine_x = np.linspace(0.0, 1.0, 1001)
    pyb_max = float(np.max(np.abs(_eval_poly(pyb_coeffs, fine_x))))
    ref_max = float(np.max(np.abs(_eval_poly(ref_coeffs, fine_x))))
    if pyb_max <= 0.0:
        return float("inf")
    return ref_max / pyb_max


def _rms_residuals(
    coeffs: np.ndarray,
    x_local: np.ndarray,
    phi_local: np.ndarray,
    pyb_coeffs: np.ndarray,
) -> tuple[float, float]:
    """``(file_rms, pyB_rms)`` — RMS of polynomial-vs-FEM at the FEM stations.

    Same metric the standard validator uses: the polynomial is
    evaluated at the FEM span_loc and subtracted from the
    tip-normalised FEM mode shape. Captures shape disagreement
    even when the amplitude metric reads 1.0 ×, since the latter
    only sees the constrained tip.
    """
    file_rms = float(np.sqrt(np.mean(
        (_eval_poly(coeffs, x_local) - phi_local) ** 2
    )))
    pyb_rms = float(np.sqrt(np.mean(
        (_eval_poly(pyb_coeffs, x_local) - phi_local) ** 2
    )))
    return file_rms, pyb_rms


def _coeffs_from_fit(fit: PolyFitResult) -> np.ndarray:
    return np.asarray(fit.coefficients(), dtype=float)


def _plot_one(
    ax, *, title: str, x_local: np.ndarray, phi_local: np.ndarray,
    pyb_coeffs: np.ndarray, ref_coeffs: np.ndarray,
) -> None:
    x = np.linspace(0.0, 1.0, 100)
    ax.plot(x, _eval_poly(ref_coeffs, x), "-",
            color=(0.85, 0.0, 0.0),  # standard red
            linewidth=1.6, label="r-test polynomial")
    ax.plot(x, _eval_poly(pyb_coeffs, x), "-",
            color=(0.0, 0.0, 0.85),  # standard blue
            linewidth=1.6, label="pyBmodes polynomial")
    ax.scatter(x_local, phi_local, s=22, facecolors="none",
               edgecolors=(0.20, 0.20, 0.20), linewidths=0.9,
               label="pyBmodes FEM (raw, tower segment)")
    ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=0.5, zorder=0)
    ax.axhline(1.0, color=(0.6, 0.6, 0.6), linewidth=0.5,
               linestyle=":", zorder=0)
    ax.set_xlabel("Normalised tower height  z / H  (0 = TP, 1 = top)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main", type=pathlib.Path, default=_OC3_MAIN,
        help="OC3 ElastoDyn main .dat (default: r-test 5MW_OC3Mnpl_DLL_WTurb_WavesIrr)",
    )
    parser.add_argument(
        "--subdyn", type=pathlib.Path, default=_OC3_SUBDYN,
        help="OC3 SubDyn .dat (default: r-test SubDyn for the same case)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs"
                          / "polynomial_comparison_5mw_monopile_TwFA2_TwSS2.png",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="NREL 5MW OC3 Monopile",
        help="turbine label for the figure suptitle and stdout summary",
    )
    args = parser.parse_args(argv)

    for label, path in (("main", args.main), ("subdyn", args.subdyn)):
        if not path.is_file():
            print(f"error: {label} deck not found: {path}", file=sys.stderr)
            print(
                "  r-test data is gitignored under external/OpenFAST_files/;\n"
                "  see VALIDATION.md for how to clone it.",
                file=sys.stderr,
            )
            return 2

    print(f"Reading ElastoDyn main: {args.main}")
    print(f"Reading SubDyn         : {args.subdyn}")
    main_ed = read_elastodyn_main(args.main)
    tower_ed = read_elastodyn_tower(args.main.parent / main_ed.twr_file)
    subdyn = read_subdyn(args.subdyn)

    z_seabed = float(min(j.z for j in subdyn.joints
                         if j.joint_id == subdyn.reaction_joint_id))
    z_tp = float(next(j.z for j in subdyn.joints
                      if j.joint_id == subdyn.interface_joint_id))
    z_top = float(main_ed.tower_ht)
    combined_length = z_top - z_seabed
    tp_frac = (z_tp - z_seabed) / combined_length
    print(f"  combined cantilever length = {combined_length:.2f} m "
          f"(seabed z={z_seabed:+.1f}, TP z={z_tp:+.1f}, top z={z_top:+.1f})")
    print(f"  tower segment fraction in combined coords: "
          f"[{tp_frac:.4f}, 1.0]  (tower length {z_top - z_tp:.2f} m)")

    print("  building Tower.from_elastodyn_with_subdyn + solving FEM ...")
    model = Tower.from_elastodyn_with_subdyn(args.main, args.subdyn)
    modal = model.run(n_modes=12)

    fa_idxs = _select_first_two(modal.shapes, want_fa=True)
    ss_idxs = _select_first_two(modal.shapes, want_fa=False)
    fa1, fa2 = modal.shapes[fa_idxs[0]], modal.shapes[fa_idxs[1]]
    ss1, ss2 = modal.shapes[ss_idxs[0]], modal.shapes[ss_idxs[1]]
    print(f"  FA1 = mode {fa1.mode_number} ({fa1.freq_hz:.4f} Hz)")
    print(f"  FA2 = mode {fa2.mode_number} ({fa2.freq_hz:.4f} Hz)")
    print(f"  SS1 = mode {ss1.mode_number} ({ss1.freq_hz:.4f} Hz)")
    print(f"  SS2 = mode {ss2.mode_number} ({ss2.freq_hz:.4f} Hz)")

    # Extract tower-segment bending shape + fit polynomial for each mode.
    fits: dict[str, tuple[np.ndarray, np.ndarray, PolyFitResult, np.ndarray]] = {}
    for name, shape, is_fa, ref_coeffs in (
        ("TwFAM1Sh", fa1, True, np.asarray(tower_ed.tw_fa_m1_sh, dtype=float)),
        ("TwSSM1Sh", ss1, False, np.asarray(tower_ed.tw_ss_m1_sh, dtype=float)),
        ("TwFAM2Sh", fa2, True, np.asarray(tower_ed.tw_fa_m2_sh, dtype=float)),
        ("TwSSM2Sh", ss2, False, np.asarray(tower_ed.tw_ss_m2_sh, dtype=float)),
    ):
        x_local, phi_local = _extract_tower_segment_bending(
            shape, tp_frac=tp_frac, is_fa=is_fa,
        )
        fit = fit_mode_shape(x_local, phi_local)
        fits[name] = (x_local, phi_local, fit, ref_coeffs)

    # Diagnostics — same format as the 5MW land and IEA-3.4 scripts.
    for name, (_, _, fit, ref_coeffs) in fits.items():
        _print_diagnostic_table(name, _coeffs_from_fit(fit), ref_coeffs)

    print()
    print("=== Amplitude ratios (max|phi_rtest| / max|phi_pyB| on [0, 1]) ===")
    for name, (_, _, fit, ref_coeffs) in fits.items():
        ratio = _amplitude_ratio(_coeffs_from_fit(fit), ref_coeffs)
        print(f"  {name}: {ratio:>5.2f} x")

    print()
    print("=== RMS residuals at FEM stations (polynomial - tip-normalised FEM) ===")
    print(f"  {'block':<8}  {'file_rms':>10}  {'pyB_rms':>10}  {'ratio':>7}")
    for name, (x_local, phi_local, fit, ref_coeffs) in fits.items():
        file_rms, pyb_rms = _rms_residuals(
            ref_coeffs, x_local, phi_local, _coeffs_from_fit(fit),
        )
        ratio = file_rms / pyb_rms if pyb_rms > 0 else float("inf")
        print(f"  {name:<8}  {file_rms:>10.5f}  {pyb_rms:>10.5f}  {ratio:>6.2f} x")

    # Plot — 2nd modes only, matching the 5MW land script's layout.
    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")
    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    fa2_x, fa2_y, fa2_fit, fa2_ref = fits["TwFAM2Sh"]
    ss2_x, ss2_y, ss2_fit, ss2_ref = fits["TwSSM2Sh"]
    _plot_one(
        axes[0], title="TwFAM2Sh — fore-aft 2nd bending (tower segment)",
        x_local=fa2_x, phi_local=fa2_y,
        pyb_coeffs=_coeffs_from_fit(fa2_fit), ref_coeffs=fa2_ref,
    )
    _plot_one(
        axes[1], title="TwSSM2Sh — side-side 2nd bending (tower segment)",
        x_local=ss2_x, phi_local=ss2_y,
        pyb_coeffs=_coeffs_from_fit(ss2_fit), ref_coeffs=ss2_ref,
    )
    axes[0].set_ylabel("Modal displacement  phi(z)  (TP-rigid-motion subtracted)")
    fig.suptitle(
        f"{args.label} tower: 2nd-mode polynomial vs FEM tower segment — "
        f"{args.main.name}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out)
    print()
    print(f"Wrote {args.out}")

    # Interpretation against the land-based and IEA-3.4 baselines.
    fa2_ratio = _amplitude_ratio(_coeffs_from_fit(fa2_fit), fa2_ref)
    ss2_ratio = _amplitude_ratio(_coeffs_from_fit(ss2_fit), ss2_ref)
    avg_ratio = 0.5 * (fa2_ratio + ss2_ratio)
    print()
    print("=== Comparison vs land-based / IEA-3.4 2nd-mode amplitude ratios ===")
    print("  NREL 5MW land-based : avg 2.37 x")
    print("  IEA-3.4-130-RWT     : avg 1.85 x")
    print(f"  {args.label:<20}: avg {avg_ratio:.2f} x")
    if avg_ratio < 1.3:
        print("  Outcome: monopile polynomial is internally consistent with the FEM")
        print("  (ratio ~ 1 x). The land-based ~ 2.4 x mismatch is BC-specific —")
        print("  the OC3-monopile polynomial was regenerated more carefully than")
        print("  the land-based one.")
    elif avg_ratio < 2.0:
        print("  Outcome: monopile polynomial is somewhat better than land-based")
        print("  (ratio ~ 1.5-2 x). Some BC-specific improvement, but the same")
        print("  systematic pattern is partially present.")
    else:
        print("  Outcome: monopile polynomial shows the same ~ 2.3 x amplitude")
        print("  mismatch as the land-based deck. The pattern is BC-INDEPENDENT —")
        print("  same polynomial-derivation approach was used for both, and it")
        print("  produces a 2nd-mode amplitude that disagrees with the FEM by")
        print("  the same factor regardless of substructure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
