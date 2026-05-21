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

"""Reproduce Bir 2010 Figures 6a / 6b / 6c (floating-platform-supported
tower) in the same plot convention.

Bir 2010 Figure 6 (NREL/CP-500-47953) shows three panels for the
floating-barge-supported turbine, each plotted in the convention:

    x-axis = Modal displacement (mass-normalised)
    y-axis = Tower section height / h   (0 at base, 1 at tip)

Bir's panels:

    Fig 6a — Longitudinal-plane modes (F-A): Surge + Pitch + 1st-3rd F-A
              All decoupled because the longitudinal plane of symmetry
              keeps F-A and S-S separated.
    Fig 6b — Lateral-plane modes (S-S + twist): Sway + Roll +
              1st-3rd S-S, each with its small twist-coupled part
              shown dashed. The barge c.m. / mooring asymmetry
              introduces side-to-side ↔ torsion coupling.
    Fig 6c — Out-of-plane / torsion: Yaw + 1st twist + 2nd twist.

This script uses the **OC3Hywind.bmi** deck from
``external/BModes/docs/examples/`` — the *NREL 5MW Reference Turbine*
(Jonkman et al. 2009) on the *OC3 Hywind* floating spar (Jonkman 2010).
OC3Hywind is a spar-buoy, not a barge, so the absolute frequencies
differ from Bir's Fig 6 reference; the *plot convention* and the
qualitative mode-shape topology transfer.

pyBmodes matches BModes JJ on this deck to ≤ 0.0003 % across the first
9 modes (see ``test_certtest_oc3hywind`` in the validation suite).

Output: ``outputs/bir_fig6a_fa.png``,
        ``outputs/bir_fig6b_ss_twist.png``,
        ``outputs/bir_fig6c_yaw_twist.png``

Run from the repo root with::

    set PYTHONPATH=%CD%\\src
    python cases/bir_2010_floating/run.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from pybmodes.models import Tower  # noqa: E402
from pybmodes.plots import apply_style, bir_mode_shape_plot  # noqa: E402

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OC3_BMI = (
    REPO_ROOT / "external" / "BModes" / "docs" / "examples" / "OC3Hywind.bmi"
)
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _classify(result) -> dict[str, list[int]]:
    """Group OC3Hywind modes into platform-rigid, F-A bending, S-S
    bending, and twist-dominant categories.

    Platform rigid-body modes (surge / sway / heave / roll / pitch /
    yaw) appear at very low frequency (< 0.2 Hz on OC3Hywind) and have
    near-linear or near-constant mode shapes. We label them by their
    primary translation/rotation component and lump them as 'rigid'.
    """
    rigid: list[tuple[int, str]] = []   # (mode, label)
    fa: list[int] = []
    ss: list[int] = []
    twist: list[int] = []

    for shape in result.shapes:
        flap = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag = float(np.dot(shape.lag_disp, shape.lag_disp))
        tw = float(np.dot(shape.twist, shape.twist))
        bending = flap + lag

        # Platform rigid modes: very low freq AND mode shape near-linear
        # in z (i.e. the curvature is tiny relative to the average
        # amplitude). Use a frequency threshold for the first cut.
        if shape.freq_hz < 0.20:
            # Pick whichever component dominates.
            if flap > lag and flap > tw:
                rigid.append((shape.mode_number, f"surge/pitch (mode {shape.mode_number})"))
            elif lag > flap and lag > tw:
                rigid.append((shape.mode_number, f"sway/roll (mode {shape.mode_number})"))
            elif tw > 0:
                rigid.append((shape.mode_number, f"yaw (mode {shape.mode_number})"))
            continue

        # Higher frequencies: classify as bending or twist by energy.
        if tw > 3.0 * bending:
            twist.append(shape.mode_number)
        elif flap >= lag:
            fa.append(shape.mode_number)
        else:
            ss.append(shape.mode_number)

    return {"rigid": rigid, "fa": fa, "ss": ss, "twist": twist}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_6a_fa(result, groups, out_path: pathlib.Path) -> None:
    """Bir 2010 Fig 6a — F-A bending + all longitudinal-plane platform modes."""
    # Bir Fig 6a shows BOTH platform surge AND pitch alongside the
    # tower-bending F-A modes. Plot every flap-dominated rigid-body mode
    # the classifier has identified.
    rigid_fa_modes = [m for m, lbl in groups["rigid"] if "surge" in lbl]

    mode_specs: list[tuple[int, str, str]] = [
        (m, "flap", f"platform surge/pitch (mode {m})") for m in rigid_fa_modes
    ]
    for m, lbl in zip(groups["fa"][:3], ("1st F-A", "2nd F-A", "3rd F-A")):
        mode_specs.append((m, "flap", lbl))

    fig = bir_mode_shape_plot(
        result, mode_specs,
        title="Bir 2010 Figure 6a-style — floating tower, F-A modes",
        height_label="Tower section height / h",
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


def figure_6b_ss_twist(result, groups, out_path: pathlib.Path) -> None:
    """Bir 2010 Fig 6b — S-S bending + all lateral-plane platform modes,
    with twist overlays.

    Bir Fig 6b shows BOTH platform sway AND roll, since both lateral-plane
    rigid-body modes contribute to the side-to-side / torsion coupling.
    """
    rigid_ss_modes = [m for m, lbl in groups["rigid"] if "sway" in lbl]

    mode_specs: list[tuple[int, str, str]] = [
        (m, "lag", f"platform sway/roll (mode {m})") for m in rigid_ss_modes
    ]
    twist_overlay: list[tuple[int, str, str]] = []
    for m, lbl in zip(groups["ss"][:3], ("1st S-S", "2nd S-S", "3rd S-S")):
        mode_specs.append((m, "lag", lbl))
        twist_overlay.append((m, "twist", f"{lbl} (twist part)"))

    fig = bir_mode_shape_plot(
        result, mode_specs,
        title="Bir 2010 Figure 6b-style — floating tower, S-S + twist",
        height_label="Tower section height / h",
        coupling_overlay=twist_overlay if twist_overlay else None,
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


def figure_6c_yaw_twist(result, groups, out_path: pathlib.Path) -> None:
    """Bir 2010 Fig 6c — all platform yaw modes + 1st twist + 2nd twist."""
    rigid_yaw_modes = [m for m, lbl in groups["rigid"] if "yaw" in lbl]

    mode_specs: list[tuple[int, str, str]] = [
        (m, "twist", f"platform yaw (mode {m})") for m in rigid_yaw_modes
    ]
    for m, lbl in zip(groups["twist"][:2], ("1st twist", "2nd twist")):
        mode_specs.append((m, "twist", lbl))

    fig = bir_mode_shape_plot(
        result, mode_specs,
        title="Bir 2010 Figure 6c-style — floating tower, yaw + twist",
        height_label="Tower section height / h",
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not OC3_BMI.is_file():
        print(f"ERROR: OC3Hywind deck not found at {OC3_BMI}")
        print("       (these BModes example decks are local-only — "
              "see the README *Independence stance*)")
        return 1

    print("Bir 2010 Figure 6 reproduction — floating-platform tower")
    print(f"  source deck: {OC3_BMI.name}  (OC3Hywind floating spar)")
    print()

    result = Tower(OC3_BMI).run(n_modes=20)
    groups = _classify(result)

    print("Mode classification:")
    print(f"  rigid (platform): {len(groups['rigid'])} -> "
          f"{[(m, lbl) for m, lbl in groups['rigid']]}")
    print(f"  F-A bending:      {len(groups['fa'])} -> {groups['fa']}")
    print(f"  S-S bending:      {len(groups['ss'])} -> {groups['ss']}")
    print(f"  twist-dominant:   {len(groups['twist'])} -> {groups['twist']}")
    print()

    figure_6a_fa(result, groups, OUTPUT_DIR / "bir_fig6a_fa.png")
    figure_6b_ss_twist(
        result, groups, OUTPUT_DIR / "bir_fig6b_ss_twist.png"
    )
    figure_6c_yaw_twist(
        result, groups, OUTPUT_DIR / "bir_fig6c_yaw_twist.png"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
