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

"""Reproduce Bir 2010 Figure 8 (monopile-supported tower) in the same
plot convention.

Bir 2010 Figure 8 (NREL/CP-500-47953) shows three side-by-side panels
for the monopile-supported turbine — S-S bending, F-A bending, and
torsion — with:

    x-axis = Modal displacement (mass-normalised)
    y-axis = Tower section height / H   (0 at base, 1 at tip)

and horizontal dotted markers labelling Mean Sea Level (MSL) and Mud
Line. Bir's monopile config: H = 143.6 m total flexible length, MSL at
y ≈ 0.40, Mud Line at y ≈ 0.25 (i.e. 36 m of buried-pile + 20 m of
water column + 87.6 m of tower above MSL).

This script uses the **CS_Monopile.bmi** deck from
``external/BModes/docs/examples/`` — the *NREL 5MW Reference Turbine*
(Jonkman et al. 2009) on an OC3-style soft monopile (Jonkman & Musial
2010), with pyBmodes matching the BModes JJ reference at < 0.005 % per
mode (see ``test_certtest_cs_monopile`` in the validation suite).

Geometry of CS_Monopile differs slightly from Bir's reference setup:

    H_total  = radius + draft = 87.6 + 20 = 107.6 m
    MSL      at y = draft / H_total = 20 / 107.6 ≈ 0.186
    Mud Line at the FEM base (y = 0)

The CS_Monopile deck represents the buried-pile soil interaction as a
distributed lateral stiffness over the lower 36 m of the flexible
tower (``hub_conn = 3``: axial + torsion locked, lateral + rocking
free), so there is no separate buried-pile region in the FEM
coordinates — the FEM base is effectively the mud line for plotting
purposes.

Output: ``outputs/bir_fig8_monopile_modes.png``

Run from the repo root with::

    set PYTHONPATH=%CD%\\src
    python cases/bir_2010_monopile/run.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from pybmodes.models import Tower  # noqa: E402
from pybmodes.plots import apply_style, bir_mode_shape_subplot  # noqa: E402

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CS_MONOPILE_BMI = (
    REPO_ROOT / "external" / "BModes" / "docs" / "examples" / "CS_Monopile.bmi"
)
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# CS_Monopile geometric annotations (computed from the deck inputs).
RADIUS = 87.6
DRAFT = 20.0
H_TOTAL = RADIUS + DRAFT          # 107.6 m
MSL_FRAC = DRAFT / H_TOTAL        # 0.186
# Mud-line marker — for CS_Monopile the base IS effectively the mud line,
# so we draw the line very close to the axis bottom for visual parity
# with Bir Fig 8 rather than at exactly y = 0 (which the axis already
# marks).
MUD_LINE_FRAC = 0.001


def _identify_modes(result) -> dict[str, list[tuple[int, str]]]:
    """Group modes by dominant component, with hybrid-mode awareness.

    For axisymmetric or near-axisymmetric towers (CS_Monopile + the
    head-mass c.m. offset along x) the eigensolver returns hybrid F-A
    + twist modes inside near-degenerate eigenspaces. A mode whose
    twist energy dominates (twist > 3·bending) is classified as
    'twist', but if it also carries non-trivial bending content
    (flap or lag energy > 0.005 in mass-normalised units) we tag it
    as a hybrid in the label so the figure tells the reader what
    happened.

    Returns dict with keys 'fa', 'ss', 'twist', each a list of
    ``(mode_number, label)`` tuples in ascending-frequency order.
    Mode order within each bucket maps to "1st …", "2nd …", etc.
    """
    fa: list[tuple[int, str]] = []
    ss: list[tuple[int, str]] = []
    twist: list[tuple[int, str]] = []

    fa_count = ss_count = twist_count = 0
    ordinals = ("1st", "2nd", "3rd", "4th", "5th", "6th")
    HYBRID_BENDING_THRESHOLD = 5.0e-3   # mass-normalised energy

    for shape in result.shapes:
        flap_n = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_n = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_n = float(np.dot(shape.twist, shape.twist))
        bending_n = flap_n + lag_n
        if (bending_n + twist_n) < 1e-12:
            continue

        if twist_n > 3.0 * bending_n:
            label = f"{ordinals[twist_count % len(ordinals)]} twist"
            # Hybrid if there's still meaningful bending content.
            if flap_n > HYBRID_BENDING_THRESHOLD:
                label += " (+ F-A part)"
            elif lag_n > HYBRID_BENDING_THRESHOLD:
                label += " (+ S-S part)"
            twist.append((shape.mode_number, label))
            twist_count += 1
        elif flap_n >= lag_n:
            label = f"{ordinals[fa_count % len(ordinals)]} F-A"
            fa.append((shape.mode_number, label))
            fa_count += 1
        else:
            label = f"{ordinals[ss_count % len(ordinals)]} S-S"
            ss.append((shape.mode_number, label))
            ss_count += 1

    return {"fa": fa, "ss": ss, "twist": twist}


def main() -> int:
    if not CS_MONOPILE_BMI.is_file():
        print(f"ERROR: CS_Monopile deck not found at {CS_MONOPILE_BMI}")
        print("       (these BModes example decks are local-only — "
              "see the README *Independence stance*)")
        return 1

    print("Bir 2010 Figure 8 reproduction — monopile-supported tower")
    print(f"  source deck: {CS_MONOPILE_BMI.name}")
    print(f"  H_total = {H_TOTAL} m")
    print(f"  MSL at y = {MSL_FRAC:.3f}, Mud Line at y = {MUD_LINE_FRAC:.3f}")
    print()

    result = Tower(CS_MONOPILE_BMI).run(n_modes=20)
    groups = _identify_modes(result)

    # Pick the first three S-S, three F-A, and two twist modes for the
    # three subplots. This mirrors Bir Fig 8's layout (1st-3rd S-S,
    # 1st-3rd F-A, 1st-2nd twist). Labels come from the classifier and
    # may include hybrid annotations like "1st twist (+ F-A part)".
    ss_specs = [(m, "lag",   lbl) for m, lbl in groups["ss"][:3]]
    fa_specs = [(m, "flap",  lbl) for m, lbl in groups["fa"][:3]]
    tw_specs = [(m, "twist", lbl) for m, lbl in groups["twist"][:2]]

    panels = [
        ("S-S bending", ss_specs),
        ("F-A bending", fa_specs),
        ("Torsion", tw_specs),
    ]

    fig = bir_mode_shape_subplot(
        result, panels,
        suptitle="Bir 2010 Figure 8 — monopile-supported tower",
        height_label="Tower section height / H",
        annotations={
            "Mean Sea Level": MSL_FRAC,
            "Mud Line": MUD_LINE_FRAC,
        },
    )
    out_path = OUTPUT_DIR / "bir_fig8_monopile_modes.png"
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")

    print()
    print("Identified mode counts:")
    for k, v in groups.items():
        print(f"  {k:>5}: {len(v)} modes -> {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
