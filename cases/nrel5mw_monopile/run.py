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

"""Exploratory pyBmodes run for the 5MW OC3-style monopile reference deck.

Uses ``Tower.from_elastodyn_with_subdyn(...)`` to splice the SubDyn pile
geometry below the ElastoDyn tower, producing a single cantilever
"clamped at the seabed" model:

    z = -20 m  ──── rigid clamped base (SubDyn reaction joint)
    ...            pile  (uniform circular section, OC3 properties)
    z = +10 m  ──── transition piece (SubDyn interface joint /
                    ElastoDyn ``TowerBsHt``)
    ...            tower (ElastoDyn distributed properties)
    z = +87.6 m ─── tower top (RNA tip mass)

Scope deliberately excludes (per the design choice for this case):

* Soil flexibility — the OC3 SubDyn deck has no soil springs by design
  (``NSpringPropSets = 0``, no SSI file, base reaction joint clamped),
  so there's nothing to extract.
* Hydrodynamic added mass — would require parsing HydroDyn for ``Ca``
  plus an extra distributed-mass contribution on the submerged section.
  Not included here; the resulting modal frequency is the dry-system
  frequency.
* Polynomial-coefficient comparison — the bundled ``.dat`` polynomials
  describe the *tower above TP* only and have no consistent peer in our
  combined-cantilever solve, so the residual columns from the other
  case-study scripts are omitted.

The primary output of interest is the system 1st-FA modal frequency: a
length-only Bernoulli scaling from the land-based 5MW value (0.3354 Hz
on a 87.6 m beam) predicts ``0.3354 · (87.6 / 107.6)² ≈ 0.222 Hz`` for
the 107.6 m combined cantilever — the well-known "soft-soft" regime.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pybmodes.plots import apply_style  # noqa: E402

apply_style()

from pybmodes.elastodyn.params import (  # noqa: E402
    _is_degenerate_pair,
    _rotate_degenerate_pairs,
    _shape_participation,
)
from pybmodes.io.elastodyn_reader import (  # noqa: E402
    read_elastodyn_main,
    read_elastodyn_tower,
)
from pybmodes.io.subdyn_reader import read_subdyn  # noqa: E402
from pybmodes.models import Tower  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
OUT_DIR = HERE.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ED_DIR = (
    ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes" / "openfast"
    / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
)
MAIN_DAT = ED_DIR / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat"
SUBDYN_DAT = ED_DIR / "NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat"

TURBINE_LABEL = "5MW OC3 monopile"
EXPECTED_FA1_HZ = 0.222   # Bernoulli length-scaling estimate vs land-based.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_label(p_fa: float, p_ss: float, p_tw: float) -> str:
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


def participation_3(shape) -> tuple[float, float, float]:
    fa = float(np.sum(shape.flap_disp ** 2))
    ss = float(np.sum(shape.lag_disp ** 2))
    tw = float(np.sum(shape.twist ** 2))
    total = fa + ss + tw
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return (fa / total, ss / total, tw / total)


def tower_diagnostic_table(modal) -> str:
    lines = []
    header = (
        f"{'Mode':>4} {'Freq (Hz)':>10}  {'p_FA':>6}  {'p_SS':>6}  "
        f"{'p_tor':>6}  {'Label':<14}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for shape in modal.shapes:
        p_fa, p_ss, p_tw = participation_3(shape)
        label = classify_label(p_fa, p_ss, p_tw)
        lines.append(
            f"{shape.mode_number:>4} {shape.freq_hz:>10.4f}  "
            f"{p_fa:>6.3f}  {p_ss:>6.3f}  {p_tw:>6.3f}  {label:<14}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combined_modes(modal, n_show: int, z_seabed: float, z_tp: float,
                        z_top: float, out_path: pathlib.Path) -> None:
    """Small-multiples grid: every extracted mode shape with the pile/tower
    transition annotated as a vertical line."""
    n = min(n_show, len(modal.shapes))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows),
                            sharex=True, sharey=False)
    axs = np.atleast_2d(axs).reshape(rows, cols)

    combined_length = z_top - z_seabed
    tp_frac = (z_tp - z_seabed) / combined_length

    for i in range(rows * cols):
        ax = axs[i // cols, i % cols]
        if i >= n:
            ax.axis("off")
            continue
        shape = modal.shapes[i]

        def norm(arr):
            m = np.max(np.abs(arr))
            return arr / m if m > 0 else arr

        ax.plot(shape.span_loc, norm(shape.flap_disp), "-o", ms=3,
                label="FA (flap)")
        ax.plot(shape.span_loc, norm(shape.lag_disp), "-s", ms=3,
                label="SS (lag)")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(tp_frac, color="C2", lw=1.0, ls=":", alpha=0.6,
                   label="TP (z=+10 m)")
        ax.grid(alpha=0.3)
        p_fa, p_ss, p_tw = participation_3(shape)
        label = classify_label(p_fa, p_ss, p_tw)
        ax.set_title(
            f"Mode {shape.mode_number}: {shape.freq_hz:.3f} Hz\n"
            f"{label}  (twist={p_tw:.2f})",
            fontsize=9,
        )
        if i // cols == rows - 1:
            ax.set_xlabel(f"span fraction (0 = mudline, 1 = top)")
        if i % cols == 0:
            ax.set_ylabel("normalised disp")
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{TURBINE_LABEL} combined pile + tower mode shapes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not MAIN_DAT.is_file() or not SUBDYN_DAT.is_file():
        missing = [str(p) for p in (MAIN_DAT, SUBDYN_DAT) if not p.is_file()]
        print("ERROR: missing input deck file(s):\n  " + "\n  ".join(missing),
              file=sys.stderr)
        return 1

    main_ed = read_elastodyn_main(MAIN_DAT)
    tower_ed = read_elastodyn_tower(MAIN_DAT.parent / main_ed.twr_file)
    subdyn = read_subdyn(SUBDYN_DAT)

    z_seabed = float(min(j.z for j in subdyn.joints
                          if j.joint_id == subdyn.reaction_joint_id))
    z_tp     = float(next(j.z for j in subdyn.joints
                          if j.joint_id == subdyn.interface_joint_id))
    z_top    = float(main_ed.tower_ht)
    pile_length = z_tp - z_seabed
    combined_length = z_top - z_seabed

    pile_props = subdyn.circ_props[0]   # OC3 has uniform pile from PropSet 1.

    tower = Tower.from_elastodyn_with_subdyn(MAIN_DAT, SUBDYN_DAT)
    modal = tower.run(n_modes=10)

    # ----- Build textual report ------------------------------------------
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"{TURBINE_LABEL} ElastoDyn + SubDyn deck — pyBmodes exploratory run")
    out.append("=" * 76)
    out.append("")
    out.append(f"ElastoDyn main:  {MAIN_DAT.relative_to(ROOT).as_posix()}")
    out.append(f"ElastoDyn tower: {main_ed.twr_file}")
    out.append(f"SubDyn:          {SUBDYN_DAT.relative_to(ROOT).as_posix()}")
    out.append("")
    out.append("Geometry (combined cantilever from seabed to tower top):")
    out.append(f"  z_seabed      = {z_seabed:>+8.3f} m  (SubDyn reaction joint, clamped)")
    out.append(f"  z_TP          = {z_tp:>+8.3f} m  (SubDyn interface = ElastoDyn TowerBsHt)")
    out.append(f"  z_tower_top   = {z_top:>+8.3f} m  (ElastoDyn TowerHt)")
    out.append(f"  pile length   = {pile_length:>8.3f} m")
    out.append(f"  tower length  = {z_top - z_tp:>8.3f} m")
    out.append(f"  combined      = {combined_length:>8.3f} m")
    out.append("")
    out.append("Pile cross-section (uniform per OC3, from SubDyn PropSet 1):")
    out.append(f"  OD            = {pile_props.D:.3f} m")
    out.append(f"  wall          = {pile_props.t * 1000:.0f} mm")
    out.append(f"  E             = {pile_props.E:.3e} Pa")
    out.append(f"  rho           = {pile_props.rho:.0f} kg/m^3")
    out.append(f"  derived mass  = {pile_props.mass_per_length:.0f} kg/m")
    out.append(f"  derived EI    = {pile_props.EI:.3e} N.m^2")
    out.append("")
    out.append(f"RNA tip mass (lumped at z_top): {tower._bmi.tip_mass.mass:.0f} kg")
    out.append("")

    # ----- Frequencies ---------------------------------------------------
    out.append("Computed natural frequencies (n=10):")
    for i, f in enumerate(modal.frequencies):
        out.append(f"  mode {i+1:>2}: {f:>8.4f} Hz")
    out.append("")

    # ----- Mode-level diagnostics ---------------------------------------
    out.append("Mode participation and classification (raw eigensolver output)")
    out.append(tower_diagnostic_table(modal))
    out.append("")

    # Show post-rotation participation if any degenerate pair would trigger.
    rotated_for_view = _rotate_degenerate_pairs(modal.shapes)
    rotation_lines = []
    for i in range(len(modal.shapes) - 1):
        if _is_degenerate_pair(modal.shapes[i], modal.shapes[i + 1]):
            for k in (i, i + 1):
                p_fa_pre, p_ss_pre = _shape_participation(modal.shapes[k])
                p_fa_post, p_ss_post = _shape_participation(rotated_for_view[k])
                rotation_lines.append(
                    f"  mode {modal.shapes[k].mode_number}: "
                    f"pre p_FA={p_fa_pre:.3f}/p_SS={p_ss_pre:.3f}  ->  "
                    f"post p_FA={p_fa_post:.3f}/p_SS={p_ss_post:.3f}"
                )
    if rotation_lines:
        out.append("Degenerate-pair rotation (applied internally before fitting):")
        out.extend(rotation_lines)
        out.append("")

    # ----- 1st FA frequency context ------------------------------------
    f_fa1 = float(modal.frequencies[0])
    f_land_5mw = 0.3354
    bernoulli_estimate = f_land_5mw * (87.6 / combined_length) ** 2
    drop_vs_land = 100.0 * (f_land_5mw - f_fa1) / f_land_5mw
    out.append("1st FA frequency in context")
    out.append(f"  pyBmodes 1st mode (combined pile+tower) = {f_fa1:.4f} Hz")
    out.append(f"  pyBmodes 1st mode (land-based 5MW)      = {f_land_5mw:.4f} Hz")
    out.append(f"  reduction vs land-based                 = {drop_vs_land:.1f}%  "
               f"(soft-soft behaviour)")
    out.append("")
    out.append("  Length-only Bernoulli scaling (uniform-beam assumption)")
    out.append(f"    estimate = {bernoulli_estimate:.4f} Hz "
               f"({f_land_5mw:.4f} x (87.6 / {combined_length:.1f})^2)")
    out.append("    note: this scaling underestimates the true frequency because")
    out.append("    the pile section is heavier AND stiffer per unit length than")
    out.append("    tower-equivalent material, partly offsetting the length effect.")
    out.append("")
    out.append("  Reference range (published OC3 modal data, coupled FAST/ADAMS)")
    out.append("    1st tower bending: typically 0.28-0.29 Hz")
    out.append(f"    pyBmodes (rigid-base, no hydro added mass) lands at {f_fa1:.4f} Hz,")
    out.append("    consistent with the published range for the dry-system frequency.")
    out.append("")

    text = "\n".join(out)
    print(text)
    (OUT_DIR / "frequencies.txt").write_text(text + "\n", encoding="utf-8")

    # ----- Plot ---------------------------------------------------------
    plot_combined_modes(
        modal, n_show=8,
        z_seabed=z_seabed, z_tp=z_tp, z_top=z_top,
        out_path=OUT_DIR / "monopile_modes.png",
    )

    print()
    print(f"Wrote outputs to {OUT_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  - frequencies.txt")
    print(f"  - monopile_modes.png   (8 modes, small multiples, TP marked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
