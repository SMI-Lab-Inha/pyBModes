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

"""Canonical re-emitters for the three ElastoDyn ``.dat`` flavours.

Each ``write_*`` emits text that parses back to a dataclass equal to
the original. Whitespace, label column position, and comment text are
normalised — the output is **not** byte-identical to any bundled RWT
file, but the test suite checks the parse → emit → re-parse fixed
point under ``np.allclose`` (rtol 1e-12).
"""

from __future__ import annotations

import io
import pathlib

import numpy as np

from pybmodes.io._elastodyn.types import (
    ElastoDynBlade,
    ElastoDynMain,
    ElastoDynTower,
)

# Width constants for the canonical scalar-line format. Values left-
# aligned in a 22-char column followed by a 12-char label column.
_VAL_W = 22
_LBL_W = 12


def _fmt_scalar_line(value: object, label: str, comment: str = "") -> str:
    if isinstance(value, bool):
        sval = "True" if value else "False"
    elif isinstance(value, int):
        sval = str(value)
    elif isinstance(value, float):
        sval = f"{value:.17g}"
    elif isinstance(value, str):
        sval = value
    else:
        sval = str(value)
    line = f"{sval:<{_VAL_W}} {label:<{_LBL_W}}"
    if comment:
        line = f"{line} - {comment}"
    return line


def _scalar_or_default(obj: ElastoDynMain, label: str, fallback: str) -> str:
    """Return the original value string for ``label`` from
    ``obj.scalars`` if captured, otherwise the fallback string."""
    return obj.scalars.get(label, fallback)


def write_elastodyn_main(
    obj: ElastoDynMain, path: str | pathlib.Path | None = None
) -> str:
    """Re-emit a main ElastoDyn file. Returns the text and optionally
    writes it to ``path``."""
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    # Re-emit every captured scalar with the typed-field value when known,
    # falling back to the original token when we did not specialise it.
    typed_overrides = {
        "NumBl":     str(obj.num_bl),
        "TipRad":    f"{obj.tip_rad:.17g}",
        "HubRad":    f"{obj.hub_rad:.17g}",
        "HubCM":     f"{obj.hub_cm:.17g}",
        "OverHang":  f"{obj.overhang:.17g}",
        "ShftTilt":  f"{obj.shft_tilt:.17g}",
        "Twr2Shft":  f"{obj.twr2shft:.17g}",
        "TowerHt":   f"{obj.tower_ht:.17g}",
        "TowerBsHt": f"{obj.tower_bs_ht:.17g}",
        "NacCMxn":   f"{obj.nac_cm_xn:.17g}",
        "NacCMyn":   f"{obj.nac_cm_yn:.17g}",
        "NacCMzn":   f"{obj.nac_cm_zn:.17g}",
        "RotSpeed":  f"{obj.rot_speed_rpm:.17g}",
        "HubMass":   f"{obj.hub_mass:.17g}",
        "HubIner":   f"{obj.hub_iner:.17g}",
        "GenIner":   f"{obj.gen_iner:.17g}",
        "NacMass":   f"{obj.nac_mass:.17g}",
        "NacYIner":  f"{obj.nac_y_iner:.17g}",
        "YawBrMass": f"{obj.yaw_br_mass:.17g}",
        "TwrFile":   f'"{obj.twr_file}"',
    }
    # Indexed overrides: PreCone(i), TipMass(i), BldFile(i)
    for i, vf in enumerate(obj.pre_cone):
        typed_overrides[f"PreCone({i + 1})"] = f"{vf:.17g}"
    for i, vf in enumerate(obj.tip_mass):
        typed_overrides[f"TipMass({i + 1})"] = f"{vf:.17g}"
    for i, vs in enumerate(obj.bld_file):
        # Match whichever indexing form the original used.
        if f"BldFile({i + 1})" in obj.scalars:
            typed_overrides[f"BldFile({i + 1})"] = f'"{vs}"'
        elif f"BldFile{i + 1}" in obj.scalars:
            typed_overrides[f"BldFile{i + 1}"] = f'"{vs}"'

    for label, raw_val in obj.scalars.items():
        val = typed_overrides.get(label, raw_val)
        out.write(f"{val:<{_VAL_W}} {label:<{_LBL_W}}\n")

    # OutList sections, verbatim.
    if obj.out_list:
        out.write("---------------------- OUTPUT --------------------------------------------------\n")  # noqa: E501
        for ln in obj.out_list:
            out.write(ln + "\n")
    if obj.nodal_out_list:
        out.write("====== Outputs for all blade stations ===========================\n")
        for ln in obj.nodal_out_list:
            out.write(ln + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text


def write_elastodyn_tower(
    obj: ElastoDynTower, path: str | pathlib.Path | None = None
) -> str:
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    out.write("---------------------- TOWER PARAMETERS ----------------------------------------\n")
    out.write(_fmt_scalar_line(obj.n_tw_inp_st, "NTwInpSt") + "\n")
    out.write(_fmt_scalar_line(obj.twr_fa_dmp[0], "TwrFADmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_fa_dmp[1], "TwrFADmp(2)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_ss_dmp[0], "TwrSSDmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.twr_ss_dmp[1], "TwrSSDmp(2)") + "\n")
    out.write("---------------------- TOWER ADJUSTMUNT FACTORS --------------------------------\n")
    out.write(_fmt_scalar_line(obj.fa_st_tunr[0], "FAStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.fa_st_tunr[1], "FAStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.ss_st_tunr[0], "SSStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.ss_st_tunr[1], "SSStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.adj_tw_ma, "AdjTwMa") + "\n")
    out.write(_fmt_scalar_line(obj.adj_fa_st, "AdjFASt") + "\n")
    out.write(_fmt_scalar_line(obj.adj_ss_st, "AdjSSSt") + "\n")

    out.write("---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------------\n")
    if obj.distr_header_lines:
        for ln in obj.distr_header_lines:
            out.write(ln + "\n")
    else:
        out.write("  HtFract       TMassDen         TwFAStif       TwSSStif\n")
        out.write("   (-)           (kg/m)           (Nm^2)         (Nm^2)\n")
    cols = [obj.ht_fract, obj.t_mass_den, obj.tw_fa_stif, obj.tw_ss_stif]
    for extra in (obj.tw_fa_iner, obj.tw_ss_iner, obj.tw_fa_cg_of, obj.tw_ss_cg_of):
        if extra.size:
            cols.append(extra)
    for r in range(len(obj.ht_fract)):
        out.write("  ".join(f"{c[r]:.17e}" for c in cols) + "\n")

    out.write("---------------------- TOWER FORE-AFT MODE SHAPES ------------------------------\n")
    for k, v in enumerate(obj.tw_fa_m1_sh):
        out.write(_fmt_scalar_line(v, f"TwFAM1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.tw_fa_m2_sh):
        out.write(_fmt_scalar_line(v, f"TwFAM2Sh({k + 2})") + "\n")
    out.write("---------------------- TOWER SIDE-TO-SIDE MODE SHAPES --------------------------\n")
    for k, v in enumerate(obj.tw_ss_m1_sh):
        out.write(_fmt_scalar_line(v, f"TwSSM1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.tw_ss_m2_sh):
        out.write(_fmt_scalar_line(v, f"TwSSM2Sh({k + 2})") + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text


def write_elastodyn_blade(
    obj: ElastoDynBlade, path: str | pathlib.Path | None = None
) -> str:
    out = io.StringIO()
    out.write(obj.header + "\n")
    out.write(obj.title + "\n")

    out.write("---------------------- BLADE PARAMETERS ----------------------------------------\n")
    out.write(_fmt_scalar_line(obj.n_bl_inp_st, "NBlInpSt") + "\n")
    out.write(_fmt_scalar_line(obj.bld_fl_dmp[0], "BldFlDmp(1)") + "\n")
    out.write(_fmt_scalar_line(obj.bld_fl_dmp[1], "BldFlDmp(2)") + "\n")
    out.write(_fmt_scalar_line(obj.bld_ed_dmp[0], "BldEdDmp(1)") + "\n")
    out.write("---------------------- BLADE ADJUSTMENT FACTORS --------------------------------\n")
    out.write(_fmt_scalar_line(obj.fl_st_tunr[0], "FlStTunr(1)") + "\n")
    out.write(_fmt_scalar_line(obj.fl_st_tunr[1], "FlStTunr(2)") + "\n")
    out.write(_fmt_scalar_line(obj.adj_bl_ms, "AdjBlMs") + "\n")
    out.write(_fmt_scalar_line(obj.adj_fl_st, "AdjFlSt") + "\n")
    out.write(_fmt_scalar_line(obj.adj_ed_st, "AdjEdSt") + "\n")

    out.write("---------------------- DISTRIBUTED BLADE PROPERTIES ----------------------------\n")
    if obj.distr_header_lines:
        for ln in obj.distr_header_lines:
            out.write(ln + "\n")
    else:
        if obj.pitch_axis is not None:
            out.write("    BlFract      PitchAxis      StrcTwst       BMassDen        FlpStff        EdgStff\n")  # noqa: E501
            out.write("      (-)           (-)          (deg)          (kg/m)         (Nm^2)         (Nm^2)\n")  # noqa: E501
        else:
            out.write("    BlFract               StrcTwst               BMassDen               FlpStff                 EdgStff\n")  # noqa: E501
            out.write("      (-)                   (deg)                 (kg/m)                 (Nm^2)                  (Nm^2)\n")  # noqa: E501
    cols: list[np.ndarray] = [obj.bl_fract]
    if obj.pitch_axis is not None:
        cols.append(obj.pitch_axis)
    cols.extend([obj.strc_twst, obj.b_mass_den, obj.flp_stff, obj.edg_stff])
    for r in range(len(obj.bl_fract)):
        out.write("  ".join(f"{c[r]:.17e}" for c in cols) + "\n")

    out.write("---------------------- BLADE MODE SHAPES ---------------------------------------\n")
    for k, v in enumerate(obj.bld_fl1_sh):
        out.write(_fmt_scalar_line(v, f"BldFl1Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.bld_fl2_sh):
        out.write(_fmt_scalar_line(v, f"BldFl2Sh({k + 2})") + "\n")
    for k, v in enumerate(obj.bld_edg_sh):
        out.write(_fmt_scalar_line(v, f"BldEdgSh({k + 2})") + "\n")

    text = out.getvalue()
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="latin-1")
    return text
