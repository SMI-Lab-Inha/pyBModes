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

"""Three-substructure polynomial comparison for the IEA-22-280-RWT tower.

The IEA-22 OpenFAST repo ships three different tower polynomial blocks
under the same RWT label — one per substructure configuration — with
TwFAM2Sh(2) leading coefficients of +70.39 (land-based), -4.11
(monopile), and -177.11 (UMaine semi-submersible). That's three
independent polynomial sets for a tower that has the same structural
properties between TowerBsHt and TowerHt; the only thing that changes
is the boundary condition the polynomial was fit under (rigid clamp
at ground for land, pile-mounted for monopile, platform-mounted for
semi). This script plots all three on the same axes so the divergence
is visible at a glance, alongside the pyBmodes FEM tower-segment
shape obtained by running the validated SubDyn-splice path on the
monopile sub-case (the only one that ships a complete SubDyn deck).

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\visualise_polynomial_comparison_iea22.py

Outputs ``scripts/outputs/polynomial_comparison_iea22_TwFA2_TwSS2.png``.
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

from visualise_polynomial_comparison import _eval_poly  # noqa: E402
from visualise_polynomial_comparison_5mw_monopile import (  # noqa: E402
    _amplitude_ratio,
    _coeffs_from_fit,
    _extract_tower_segment_bending,
    _rms_residuals,
    _select_first_two,
)

from pybmodes.fitting.poly_fit import fit_mode_shape  # noqa: E402
from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.io.subdyn_reader import read_subdyn  # noqa: E402
from pybmodes.models import Tower  # noqa: E402

_IEA22_OPENFAST = REPO_ROOT / "external" / "OpenFAST_files" / "IEA-22-280-RWT" / "OpenFAST"
_LAND_TOWER = (
    _IEA22_OPENFAST / "IEA-22-280-RWT" / "IEA-22-280-RWT_ElastoDyn_tower_land_based.dat"
)
_MONOPILE_DIR = _IEA22_OPENFAST / "IEA-22-280-RWT-Monopile"
_MONOPILE_MAIN = _MONOPILE_DIR / "IEA-22-280-RWT_ElastoDyn.dat"
_MONOPILE_TOWER = _MONOPILE_DIR / "IEA-22-280-RWT_ElastoDyn_tower.dat"
_MONOPILE_SUBDYN = _MONOPILE_DIR / "IEA-22-280-RWT_SubDyn.dat"
_SEMI_TOWER = (
    _IEA22_OPENFAST / "IEA-22-280-RWT-Semi"
    / "IEA-22-280-RWT-Semi_ElastoDyn_tower.dat"
)


def _print_three_way_table(
    name: str,
    land_coeffs: np.ndarray,
    monop_coeffs: np.ndarray,
    semi_coeffs: np.ndarray,
    pyb_coeffs: np.ndarray | None = None,
) -> None:
    sample_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    fine_x = np.linspace(0.0, 1.0, 1001)
    cols = [
        ("land     ", land_coeffs),
        ("monopile ", monop_coeffs),
        ("semi     ", semi_coeffs),
    ]
    if pyb_coeffs is not None:
        cols.append(("pyBmodes ", pyb_coeffs))

    print()
    print(f"=== {name} ===")
    for label, coeffs in cols:
        print(f"  {label} coeffs: {np.round(coeffs, 3).tolist()}")
    print(f"  {'x':>5}  " + "  ".join(f"{label}phi(x)" for label, _ in cols))
    for x in sample_x:
        cells = "  ".join(f"{_eval_poly(c, np.array([x]))[0]:14.4f}"
                          for _, c in cols)
        print(f"  {x:5.2f}  {cells}")
    print("  max |phi(x)| on [0, 1]:")
    for label, coeffs in cols:
        print(f"    {label}: {np.max(np.abs(_eval_poly(coeffs, fine_x))):.4f}")


def _plot_panel(
    ax, *, title: str,
    fem_x: np.ndarray, fem_phi: np.ndarray,
    pyb_coeffs: np.ndarray,
    land_coeffs: np.ndarray, monop_coeffs: np.ndarray, semi_coeffs: np.ndarray,
) -> None:
    x = np.linspace(0.0, 1.0, 100)
    ax.plot(x, _eval_poly(land_coeffs, x), "--",
            color=(0.85, 0.0, 0.0),  # standard red
            linewidth=1.5, label="IEA-22 land polynomial")
    ax.plot(x, _eval_poly(monop_coeffs, x), "--",
            color=(0.0, 0.6, 0.0),  # standard green
            linewidth=1.5, label="IEA-22 monopile polynomial")
    ax.plot(x, _eval_poly(semi_coeffs, x), "--",
            color=(0.8, 0.0, 0.8),  # standard magenta
            linewidth=1.5, label="IEA-22 semi polynomial")
    ax.plot(x, _eval_poly(pyb_coeffs, x), "-",
            color=(0.0, 0.0, 0.85),  # standard blue
            linewidth=1.7, label="pyBmodes polynomial (monopile FEM)")
    ax.scatter(fem_x, fem_phi, s=22, facecolors="none",
               edgecolors=(0.10, 0.10, 0.10), linewidths=0.9,
               label="pyBmodes FEM (monopile, tower segment)")
    ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=0.5, zorder=0)
    ax.axhline(1.0, color=(0.6, 0.6, 0.6), linewidth=0.5,
               linestyle=":", zorder=0)
    ax.set_xlabel("Normalised tower height  z / H  (0 = TP/base, 1 = top)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs"
                          / "polynomial_comparison_iea22_TwFA2_TwSS2.png",
    )
    args = parser.parse_args(argv)

    for label, path in (
        ("land tower", _LAND_TOWER),
        ("monopile main", _MONOPILE_MAIN),
        ("monopile tower", _MONOPILE_TOWER),
        ("monopile SubDyn", _MONOPILE_SUBDYN),
        ("semi tower", _SEMI_TOWER),
    ):
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            print(
                "  IEA-22-280-RWT data is gitignored under external/OpenFAST_files/;\n"
                "  see VALIDATION.md for how to clone it.",
                file=sys.stderr,
            )
            return 2

    # --- Read the three tower polynomials -------------------------------
    land = read_elastodyn_tower(_LAND_TOWER)
    monop = read_elastodyn_tower(_MONOPILE_TOWER)
    semi = read_elastodyn_tower(_SEMI_TOWER)

    land_fa2 = np.asarray(land.tw_fa_m2_sh, dtype=float)
    land_ss2 = np.asarray(land.tw_ss_m2_sh, dtype=float)
    monop_fa2 = np.asarray(monop.tw_fa_m2_sh, dtype=float)
    monop_ss2 = np.asarray(monop.tw_ss_m2_sh, dtype=float)
    semi_fa2 = np.asarray(semi.tw_fa_m2_sh, dtype=float)
    semi_ss2 = np.asarray(semi.tw_ss_m2_sh, dtype=float)

    # --- pyBmodes FEM on the monopile sub-case --------------------------
    print(f"Reading IEA-22 monopile main : {_MONOPILE_MAIN}")
    print(f"Reading IEA-22 monopile SubDyn: {_MONOPILE_SUBDYN}")
    main_ed = read_elastodyn_main(_MONOPILE_MAIN)
    subdyn = read_subdyn(_MONOPILE_SUBDYN)

    z_seabed = float(min(j.z for j in subdyn.joints
                         if j.joint_id == subdyn.reaction_joint_id))
    z_tp = float(next(j.z for j in subdyn.joints
                      if j.joint_id == subdyn.interface_joint_id))
    z_top = float(main_ed.tower_ht)
    combined_length = z_top - z_seabed
    tp_frac = (z_tp - z_seabed) / combined_length
    print(f"  combined cantilever {combined_length:.2f} m "
          f"(seabed z={z_seabed:+.1f}, TP z={z_tp:+.1f}, top z={z_top:+.1f})")
    print(f"  tower segment in combined coords: [{tp_frac:.4f}, 1.0]")

    print("  Tower.from_elastodyn_with_subdyn + solving FEM ...")
    model = Tower.from_elastodyn_with_subdyn(_MONOPILE_MAIN, _MONOPILE_SUBDYN)
    modal = model.run(n_modes=12)

    fa_idxs = _select_first_two(modal.shapes, want_fa=True)
    ss_idxs = _select_first_two(modal.shapes, want_fa=False)
    fa2 = modal.shapes[fa_idxs[1]]
    ss2 = modal.shapes[ss_idxs[1]]
    print(f"  FA2 = mode {fa2.mode_number} ({fa2.freq_hz:.4f} Hz)")
    print(f"  SS2 = mode {ss2.mode_number} ({ss2.freq_hz:.4f} Hz)")

    fa2_x, fa2_phi = _extract_tower_segment_bending(
        fa2, tp_frac=tp_frac, is_fa=True,
    )
    ss2_x, ss2_phi = _extract_tower_segment_bending(
        ss2, tp_frac=tp_frac, is_fa=False,
    )
    fa2_fit = fit_mode_shape(fa2_x, fa2_phi)
    ss2_fit = fit_mode_shape(ss2_x, ss2_phi)
    pyb_fa2 = _coeffs_from_fit(fa2_fit)
    pyb_ss2 = _coeffs_from_fit(ss2_fit)

    # --- Diagnostics -----------------------------------------------------
    _print_three_way_table(
        "TwFAM2Sh", land_fa2, monop_fa2, semi_fa2, pyb_fa2,
    )
    _print_three_way_table(
        "TwSSM2Sh", land_ss2, monop_ss2, semi_ss2, pyb_ss2,
    )

    print()
    print("=== Amplitude ratios vs pyBmodes (max|phi_file| / max|phi_pyB|) ===")
    print(f"  {'block':<10}  {'land':>7}  {'monopile':>9}  {'semi':>7}")
    for name, refs, pyb in (
        ("TwFAM2Sh", (land_fa2, monop_fa2, semi_fa2), pyb_fa2),
        ("TwSSM2Sh", (land_ss2, monop_ss2, semi_ss2), pyb_ss2),
    ):
        ratios = [_amplitude_ratio(pyb, r) for r in refs]
        print(f"  {name:<10}  {ratios[0]:>6.2f}x  {ratios[1]:>8.2f}x  "
              f"{ratios[2]:>6.2f}x")

    print()
    print("=== RMS residuals at FEM stations (file polynomial - tip-norm FEM) ===")
    print(f"  {'block':<10}  {'land_rms':>9}  {'monop_rms':>10}  "
          f"{'semi_rms':>9}  {'pyB_rms':>9}")
    for name, refs, fit_x, fit_phi, pyb in (
        ("TwFAM2Sh", (land_fa2, monop_fa2, semi_fa2),
         fa2_x, fa2_phi, pyb_fa2),
        ("TwSSM2Sh", (land_ss2, monop_ss2, semi_ss2),
         ss2_x, ss2_phi, pyb_ss2),
    ):
        rms_each = []
        for r in refs:
            file_rms, _ = _rms_residuals(r, fit_x, fit_phi, pyb)
            rms_each.append(file_rms)
        _, pyb_rms = _rms_residuals(pyb, fit_x, fit_phi, pyb)
        print(f"  {name:<10}  {rms_each[0]:>8.4f}   {rms_each[1]:>8.4f}    "
              f"{rms_each[2]:>7.4f}   {pyb_rms:>8.4f}")

    # --- Plot ------------------------------------------------------------
    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")
    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    _plot_panel(
        axes[0], title="TwFAM2Sh — fore-aft 2nd bending",
        fem_x=fa2_x, fem_phi=fa2_phi, pyb_coeffs=pyb_fa2,
        land_coeffs=land_fa2, monop_coeffs=monop_fa2, semi_coeffs=semi_fa2,
    )
    _plot_panel(
        axes[1], title="TwSSM2Sh — side-side 2nd bending",
        fem_x=ss2_x, fem_phi=ss2_phi, pyb_coeffs=pyb_ss2,
        land_coeffs=land_ss2, monop_coeffs=monop_ss2, semi_coeffs=semi_ss2,
    )
    axes[0].set_ylabel("Modal displacement  phi(z)  (TP-rigid-motion subtracted)")
    fig.suptitle(
        "IEA-22-280-RWT tower: three substructure polynomials vs FEM "
        "tower segment (monopile)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out)
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
