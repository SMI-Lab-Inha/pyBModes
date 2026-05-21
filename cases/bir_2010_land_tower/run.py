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

"""Reproduce Bir 2010 Figures 4, 5a, 5b in the same plot convention.

Bir, G. S. (2010), NREL/CP-500-47953, "Verification of BModes: Rotary
Beam and Tower Modal Analysis Code," presents tower mode-shape plots
using the convention:

    x-axis = Modal displacement (mass-normalised, NOT unit-tip)
    y-axis = Tower section height / h   (0 at base, 1 at tip)

Figure 4   — Land-based tower, no head mass.  All modes show the
              uncoupled cantilever bending shapes; the 1st F-A mode and
              1st S-S mode are coincident in an axisymmetric tower, etc.
Figure 5a  — Same tower with a head mass at the top: F-A bending modes.
Figure 5b  — Same tower with a head mass: S-S bending modes (which
              couple weakly to the first twist mode through the offset
              c.m. of the head mass).

This script generates three PNGs:

    outputs/bir_fig4_no_head_mass.png
    outputs/bir_fig5a_fa_with_head_mass.png
    outputs/bir_fig5b_ss_twist_with_head_mass.png

For Figure 4 we synthesise a uniform-cantilever tower in a temp
directory (no third-party data). For Figures 5a / 5b we use the
``Test03_tower`` deck from ``external/BModes/CertTest/`` — pyBmodes already
matches this BModes reference at six-digit precision (``test_certtest_03``
in the validation suite).

Run from the repo root with::

    set PYTHONPATH=%CD%\\src
    python cases/bir_2010_land_tower/run.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402

from pybmodes.models import Tower  # noqa: E402
from pybmodes.plots import apply_style, bir_mode_shape_plot  # noqa: E402

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CERT_DIR = REPO_ROOT / "external" / "BModes" / "CertTest"
TEST03_BMI = CERT_DIR / "Test03_tower.bmi"

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic uniform cantilever (no head mass) — Bir 2010 Figure 4
# ---------------------------------------------------------------------------

# Bir Fig 4 caption: "h = 87.6 m". We use uniform stiffness / mass density
# representative of a land-based steel tower; the absolute numbers are
# the script author's choice, the figure shape is what matters.
_FIG4_HEIGHT = 87.6        # [m]
_FIG4_MASS_DEN = 5000.0    # [kg/m]
_FIG4_EI_FA = 5.0e11       # [N·m²]  (axisymmetric: EI_FA = EI_SS)
_FIG4_EI_SS = 5.0e11
_FIG4_GJ = 1.0e11
_FIG4_EAC = 1.0e12
_FIG4_NELT = 12


def _write_uniform_tower_files(workdir: pathlib.Path) -> pathlib.Path:
    """Write a synthetic uniform-cantilever tower .bmi + .dat pair in
    *workdir*; return the .bmi path."""
    bmi_path = workdir / "uniform_tower.bmi"
    dat_path = workdir / "uniform_tower.dat"

    # --- section properties (uniform along span) -------------------------
    n_secs = 5
    span = np.linspace(0.0, 1.0, n_secs)
    rows = "\n".join(
        f"{s:.4f}  0.0  0.0  {_FIG4_MASS_DEN}  "
        f"{_FIG4_MASS_DEN * 0.1}  {_FIG4_MASS_DEN * 0.1}  "
        f"{_FIG4_EI_FA}  {_FIG4_EI_SS}  {_FIG4_GJ}  {_FIG4_EAC}  "
        "0.0  0.0  0.0"
        for s in span
    )
    dat_path.write_text(
        "synthetic uniform tower section properties\n"
        f"{n_secs}  n_secs\n"
        "\n"
        "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
        "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
        "  -      deg    deg     kg/m     kg.m     kg.m     "
        "N.m^2   N.m^2     N.m^2   N        m        m        m\n"
        + rows + "\n",
        encoding="utf-8",
    )

    # --- bmi (matches the section/separator pattern the parser expects) ---
    el_loc = list(np.linspace(0.0, 1.0, _FIG4_NELT + 1))
    lines = [
        "================= bir 2010 figure 4 — uniform cantilever ==========",
        "'uniform_tower'",
        "--------------- general parameters ---------------",
        "--------------- (echo through mid_node_tw) -------",
        "f    ! echo",
        "2    ! beam_type",
        "0.0  ! rot_rpm",
        "1.0  ! rpm_mult",
        f"{_FIG4_HEIGHT}  ! radius",
        "0.0  ! hub_rad",
        "0.0  ! precone",
        "0.0  ! bl_thp",
        "1    ! hub_conn (cantilever)",
        "10   ! n_modes_print",
        "f    ! tab_delim",
        "f    ! mid_node_tw",
        "--------------- tip mass -------------------------",
        "--------------- (mass + 9 inertia/offsets) -------",
        "0.0  ! tip_mass",
        "0.0  ! cm_offset",
        "0.0  ! cm_axial",
        "0.0  ! ixx",
        "0.0  ! iyy",
        "0.0  ! izz",
        "0.0  ! ixy",
        "0.0  ! izx",
        "0.0  ! iyz",
        "--------------- distributed properties -----------",
        "--------------- (id_mat + filename) --------------",
        "1    ! id_mat",
        f"'{dat_path.name}'    ! sec_props_file",
        "--------------- scaling factors ------------------",
        "--------------- (10 unity multipliers) -----------",
        "1.0  ! sec_mass",
        "1.0  ! flp_iner",
        "1.0  ! lag_iner",
        "1.0  ! flp_stff",
        "1.0  ! edge_stff",
        "1.0  ! tor_stff",
        "1.0  ! axial_stff",
        "1.0  ! cg_offst",
        "1.0  ! sc_offst",
        "1.0  ! tc_offst",
        "--------------- fe discretisation ----------------",
        "--------------- (nselt + el_loc) -----------------",
        f"{_FIG4_NELT}    ! n_elements",
        "--- el_loc ---",
        "  ".join(f"{v:.6f}" for v in el_loc),
        "================= tower support =================",
        "------- tow_support -------",
        "0    ! tow_support (cantilever; no extra support)",
    ]
    bmi_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bmi_path


def figure_4_no_head_mass(out_path: pathlib.Path) -> None:
    """Reproduce Bir 2010 Fig 4: uniform cantilever, no head mass.

    The first three F-A and S-S modes are coincident (axisymmetric) so
    we plot only one set, alongside the first twist mode.
    """
    with tempfile.TemporaryDirectory(prefix="bir_fig4_") as td:
        bmi = _write_uniform_tower_files(pathlib.Path(td))
        result = Tower(bmi).run(n_modes=12)

    # For an axisymmetric cantilever, modes pair up: 1st F-A with 1st S-S
    # at the same frequency, etc. The eigensolver returns either ordering
    # depending on numerical accident; pick the FA-dominated and twist
    # modes by inspecting flap_disp / lag_disp / twist content.
    fa_modes: list[int] = []
    twist_modes: list[int] = []
    for shape in result.shapes:
        flap_n = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_n = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_n = float(np.dot(shape.twist, shape.twist))
        if twist_n > 5.0 * (flap_n + lag_n):
            twist_modes.append(shape.mode_number)
        elif flap_n >= lag_n:
            fa_modes.append(shape.mode_number)

    mode_specs = [
        (fa_modes[0], "flap", "1st F-A (= 1st S-S)"),
        (fa_modes[1], "flap", "2nd F-A (= 2nd S-S)"),
        (fa_modes[2], "flap", "3rd F-A (= 3rd S-S)"),
    ]
    if twist_modes:
        mode_specs.append((twist_modes[0], "twist", "1st twist"))

    fig = bir_mode_shape_plot(
        result, mode_specs,
        title="Bir 2010 Figure 4 — land-based tower, no head mass",
        height_label="Tower section height / h",
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# CertTest Test03 — Bir 2010 Figures 5a, 5b (with head mass)
# ---------------------------------------------------------------------------

def _identify_tower_modes(result) -> dict[str, int]:
    """Sort modes into 1st/2nd/3rd F-A, 1st/2nd/3rd S-S, 1st twist by
    inspecting the dominant component of each mode shape.

    Returns a dict mapping label -> mode_number_1based.
    """
    fa_sorted: list[int] = []
    ss_sorted: list[int] = []
    twist_sorted: list[int] = []
    for shape in result.shapes:
        flap_n = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_n = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_n = float(np.dot(shape.twist, shape.twist))
        total = flap_n + lag_n + twist_n
        if total < 1e-12:
            continue
        if twist_n / total > 0.5:
            twist_sorted.append(shape.mode_number)
        elif flap_n >= lag_n:
            fa_sorted.append(shape.mode_number)
        else:
            ss_sorted.append(shape.mode_number)

    out: dict[str, int] = {}
    for k, label in enumerate(("1st F-A", "2nd F-A", "3rd F-A")):
        if k < len(fa_sorted):
            out[label] = fa_sorted[k]
    for k, label in enumerate(("1st S-S", "2nd S-S", "3rd S-S")):
        if k < len(ss_sorted):
            out[label] = ss_sorted[k]
    if twist_sorted:
        out["1st twist"] = twist_sorted[0]
    return out


def figure_5a_fa_with_head_mass(out_path: pathlib.Path) -> None:
    """Reproduce Bir 2010 Fig 5a: F-A bending modes with head mass.

    Uses the CertTest Test03 deck (cantilever tower + head mass + c.m.
    offsets), which pyBmodes matches BModes on at six-digit precision.
    """
    if not TEST03_BMI.is_file():
        print(f"  skipping fig 5a — Test03 deck not present at {TEST03_BMI}")
        return

    result = Tower(TEST03_BMI).run(n_modes=12)
    labels = _identify_tower_modes(result)

    mode_specs = []
    for lbl in ("1st F-A", "2nd F-A", "3rd F-A"):
        if lbl in labels:
            mode_specs.append((labels[lbl], "flap", lbl))

    fig = bir_mode_shape_plot(
        result, mode_specs,
        title="Bir 2010 Figure 5a — land tower, head mass, F-A modes",
        height_label="Tower section height / h",
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


def figure_5b_ss_twist_with_head_mass(out_path: pathlib.Path) -> None:
    """Reproduce Bir 2010 Fig 5b: S-S bending modes coupled with twist.

    The head-mass c.m. offset couples lateral and torsional motion, so
    each S-S mode also has a small twist component. We plot the S-S
    component as the solid line and the twist part of the same modes as
    dashed overlays.
    """
    if not TEST03_BMI.is_file():
        print(f"  skipping fig 5b — Test03 deck not present at {TEST03_BMI}")
        return

    result = Tower(TEST03_BMI).run(n_modes=12)
    labels = _identify_tower_modes(result)

    ss_specs = []
    twist_overlay = []
    for lbl in ("1st S-S", "2nd S-S", "3rd S-S"):
        if lbl in labels:
            ss_specs.append((labels[lbl], "lag", lbl))
            twist_overlay.append(
                (labels[lbl], "twist", f"{lbl} (twist part)")
            )

    if "1st twist" in labels:
        ss_specs.append((labels["1st twist"], "twist", "1st twist"))

    fig = bir_mode_shape_plot(
        result, ss_specs,
        title="Bir 2010 Figure 5b — land tower, head mass, S-S + twist",
        height_label="Tower section height / h",
        coupling_overlay=twist_overlay if twist_overlay else None,
    )
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Bir 2010 land-tower mode-shape reproductions")
    print(f"  output dir: {OUTPUT_DIR}")
    print()

    figure_4_no_head_mass(OUTPUT_DIR / "bir_fig4_no_head_mass.png")
    figure_5a_fa_with_head_mass(
        OUTPUT_DIR / "bir_fig5a_fa_with_head_mass.png"
    )
    figure_5b_ss_twist_with_head_mass(
        OUTPUT_DIR / "bir_fig5b_ss_twist_with_head_mass.png"
    )

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
