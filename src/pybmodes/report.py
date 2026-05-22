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

"""Generate a human-readable report on a pyBmodes modal analysis.

The entry point is :func:`generate_report`. It builds a structured
representation of the report (sections + tables) once, then renders
that representation in one of three output formats:

* **Markdown** (``format="md"``) — default. Human-readable plain text
  with section headers, tables, and inline code blocks. Useful for
  paste-into-issue or check-in-to-repo workflows.
* **HTML** (``format="html"``) — a self-contained HTML5 document with
  inline CSS. Same content as the markdown version; emitted directly
  rather than via an md→html converter so the report module has no
  runtime dependency on the third-party ``markdown`` package.
* **CSV** (``format="csv"``) — frequencies + polynomial-coefficient
  rows only, suitable for spreadsheet ingestion. The narrative
  sections (assumptions, validation verdict, warnings) are dropped
  because CSV can't represent them faithfully.

The report's content is driven by what the caller supplies. The
minimum is a :class:`ModalResult`; everything else
(``model``, ``campbell``, ``validation``, ``check_warnings``,
``tower_params``, ``blade_params``) is optional and unlocks the
corresponding section when supplied.
"""

from __future__ import annotations

import csv
import datetime
import html as _html
import io
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pybmodes.campbell import CampbellResult
    from pybmodes.checks import ModelWarning
    from pybmodes.elastodyn.params import (
        BladeElastoDynParams,
        TowerElastoDynParams,
    )
    from pybmodes.elastodyn.validate import ValidationResult
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.result import ModalResult
    from pybmodes.models.tower import Tower

ReportFormat = Literal["md", "html", "csv"]


# ---------------------------------------------------------------------------
# Internal structured representation
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """One top-level section of the report. Header text is the section
    title; ``body`` is a list of paragraph strings or table dicts.

    A table dict has shape::

        {"kind": "table", "header": [str, ...], "rows": [[str, ...], ...]}
    """

    title: str
    body: list[Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    result: ModalResult,
    output_path: str | pathlib.Path,
    *,
    format: ReportFormat = "md",
    model: Tower | RotatingBlade | None = None,
    campbell: CampbellResult | None = None,
    validation: ValidationResult | None = None,
    check_warnings: list[ModelWarning] | None = None,
    tower_params: TowerElastoDynParams | None = None,
    blade_params: BladeElastoDynParams | None = None,
    elastodyn_compatible: bool | None = None,
    source_file: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Render a modal-analysis report to ``output_path``.

    ``source_file`` is recorded in the *Model summary* section and the
    metadata block. When omitted, the function falls back to
    ``result.metadata["source_file"]`` and then to
    ``model._bmi.source_file`` if those are populated.

    Returns the resolved output path so the caller can chain.
    """
    # Materialise metadata at report time so the Model summary section
    # always has pybmodes_version + timestamp + git_hash. The supplied
    # ``source_file`` (typically the deck path the CLI invoked us on)
    # wins over any pre-existing metadata.source_file.
    from pybmodes.io._serialize import _capture_metadata

    base_meta = dict(result.metadata) if result.metadata else {}
    fresh = _capture_metadata(source_file=source_file)
    # Existing metadata wins for everything except source_file when an
    # explicit source_file was passed.
    merged: dict[str, Any] = {**fresh, **base_meta}
    if source_file is not None:
        merged["source_file"] = str(source_file)
    elif merged.get("source_file") is None and model is not None:
        bmi_src = getattr(model._bmi, "source_file", None)
        if bmi_src is not None:
            merged["source_file"] = str(bmi_src)
    result_for_render = result
    if result_for_render.metadata != merged:
        # Don't mutate the caller's ModalResult — shallow-copy via the
        # dataclass.replace pattern would be ideal, but ModalResult
        # carries large arrays; sharing references is fine.
        from dataclasses import replace
        result_for_render = replace(result, metadata=merged)

    sections = _build_sections(
        result=result_for_render,
        model=model,
        campbell=campbell,
        validation=validation,
        check_warnings=check_warnings,
        tower_params=tower_params,
        blade_params=blade_params,
        elastodyn_compatible=elastodyn_compatible,
    )
    path = pathlib.Path(output_path)
    if format == "md":
        text = _render_markdown(sections)
    elif format == "html":
        text = _render_html(sections)
    elif format == "csv":
        text = _render_csv(result, tower_params, blade_params)
    else:
        raise ValueError(
            f"format must be 'md', 'html', or 'csv'; got {format!r}"
        )
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_sections(
    *,
    result: ModalResult,
    model: Tower | RotatingBlade | None,
    campbell: CampbellResult | None,
    validation: ValidationResult | None,
    check_warnings: list[ModelWarning] | None,
    tower_params: TowerElastoDynParams | None,
    blade_params: BladeElastoDynParams | None,
    elastodyn_compatible: bool | None,
) -> list[_Section]:
    sections: list[_Section] = []
    sections.append(_section_model_summary(result, model))
    sections.append(_section_assumptions(model, elastodyn_compatible))
    sections.append(_section_frequencies(result))
    sections.append(_section_mode_classification(result))
    if tower_params is not None or blade_params is not None:
        sections.append(_section_polynomial_coefficients(
            tower_params, blade_params,
        ))
    if validation is not None:
        sections.append(_section_validation(validation))
    if check_warnings is not None:
        sections.append(_section_check_warnings(check_warnings))
    if campbell is not None:
        sections.append(_section_campbell(campbell))
    return sections


def _section_model_summary(
    result: ModalResult, model: Tower | RotatingBlade | None,
) -> _Section:
    meta = result.metadata or {}
    body: list[Any] = []
    table_rows: list[list[str]] = []
    title = "unknown"
    beam_type = "unknown"
    source = meta.get("source_file") or "—"
    if model is not None:
        bmi = model._bmi
        title = (bmi.title or "—").strip()
        beam_type = "Tower" if bmi.beam_type == 2 else "Blade"
        if bmi.source_file is not None:
            source = str(bmi.source_file)
    table_rows.extend([
        ["Turbine / title", title],
        ["Beam type", beam_type],
        ["Source file", source],
        ["pyBmodes version", str(meta.get("pybmodes_version", "—"))],
        ["Generated at (UTC)",
         meta.get("timestamp")
         or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")],
        ["Git hash", str(meta.get("git_hash") or "—")],
    ])
    body.append({"kind": "table",
                 "header": ["Field", "Value"], "rows": table_rows})
    return _Section("1. Model summary", body)


def _section_assumptions(
    model: Tower | RotatingBlade | None,
    elastodyn_compatible: bool | None,
) -> _Section:
    body: list[Any] = []
    if model is None:
        body.append(
            "Model object not supplied; assumptions section is limited "
            "to whatever metadata was attached to the ModalResult."
        )
        return _Section("2. Assumptions", body)

    bmi = model._bmi
    bc_map = {
        1: "cantilever (`hub_conn = 1`) — axial + lag + flap + slopes + twist locked at root",
        2: "free-free (`hub_conn = 2`) — all 6 base DOFs released; reactions via PlatformSupport",
        3: "soft monopile (`hub_conn = 3`) — axial + torsion locked; lateral + rocking free",
        4: "pinned-free (`hub_conn = 4`) — axial + lag/flap + twist locked; bending slopes free",
    }
    bc_text = bc_map.get(bmi.hub_conn, f"hub_conn = {bmi.hub_conn}")

    rows: list[list[str]] = [
        ["Boundary condition", bc_text],
        ["Beam length", f"{bmi.radius:.3f} m"],
        ["Hub radius / TowerBsHt", f"{bmi.hub_rad:.3f} m"],
        ["Rotor speed", f"{bmi.rot_rpm:.4f} rpm"],
        ["Number of elements", str(bmi.n_elements)],
    ]
    # RNA / tip-mass assembly
    if bmi.tip_mass is not None and bmi.tip_mass.mass > 0.0:
        tm = bmi.tip_mass
        rows.extend([
            ["Tip mass (RNA, lumped)", f"{tm.mass:,.1f} kg"],
            ["Tip-mass CM offset (transverse)", f"{tm.cm_offset:.4f} m"],
            ["Tip-mass CM offset (axial)", f"{tm.cm_axial:.4f} m"],
            ["Tip-mass inertia (Ixx, Iyy, Izz)",
             f"{tm.ixx:.3e}, {tm.iyy:.3e}, {tm.izz:.3e} kg·m²"],
        ])
    else:
        rows.append(["Tip mass", "0 kg (no concentrated tip mass)"])
    # ElastoDyn-compatibility flag (blade adapter only)
    if elastodyn_compatible is not None:
        rows.append([
            "ElastoDyn-compatibility (blade adapter)",
            "ON — str_tw / tw_iner / offsets zeroed per Jonkman 2015"
            if elastodyn_compatible else
            "OFF — full structural model preserved",
        ])
    body.append({"kind": "table",
                 "header": ["Assumption", "Value"], "rows": rows})
    return _Section("2. Assumptions", body)


def _section_frequencies(result: ModalResult) -> _Section:
    body: list[Any] = []
    rows = []
    for i, f in enumerate(result.frequencies, start=1):
        period = 1.0 / float(f) if float(f) > 0 else float("inf")
        rows.append([
            str(i),
            f"{float(f):.4f}",
            f"{period:.4f}" if period != float("inf") else "—",
        ])
    body.append({
        "kind": "table",
        "header": ["Mode #", "Frequency (Hz)", "Period (s)"],
        "rows": rows,
    })
    return _Section("3. Natural frequencies", body)


def _section_mode_classification(result: ModalResult) -> _Section:
    body: list[Any] = []
    if not result.shapes:
        body.append("No mode shapes available.")
        return _Section("4. Mode classification", body)

    rows = []
    have_participation = result.participation is not None
    # Floating models carry per-mode platform rigid-body names
    # (surge / sway / heave / roll / pitch / yaw); show them as an
    # extra column. ``None`` for a mode the classifier couldn't
    # attribute to one DOF (flexible / coupled), and the whole list
    # is ``None`` for a non-floating model — in which case the column
    # is omitted entirely so existing reports are unchanged.
    labels = result.mode_labels
    have_labels = labels is not None
    for i, shape in enumerate(result.shapes):
        flap_n = float(np.dot(shape.flap_disp, shape.flap_disp))
        lag_n = float(np.dot(shape.lag_disp, shape.lag_disp))
        twist_n = float(np.dot(shape.twist, shape.twist))
        total = flap_n + lag_n + twist_n
        if total <= 0.0:
            axis = "—"
            shares = "—"
        else:
            f_pct = 100.0 * flap_n / total
            e_pct = 100.0 * lag_n / total
            t_pct = 100.0 * twist_n / total
            shares = f"flap={f_pct:.1f}%, edge/SS={e_pct:.1f}%, twist={t_pct:.1f}%"
            biggest = max((f_pct, "flap/FA"), (e_pct, "edge/SS"),
                          (t_pct, "twist"))
            axis = biggest[1]
        row = [
            str(shape.mode_number),
            f"{float(shape.freq_hz):.4f}",
            axis,
            shares,
        ]
        if labels is not None:
            lbl = labels[i] if i < len(labels) else None
            row.insert(2, lbl if lbl is not None else "—")
        rows.append(row)
    header = ["Mode #", "Freq (Hz)", "Dominant axis", "Participation"]
    if have_labels:
        header.insert(2, "Platform DOF")
    body.append({
        "kind": "table",
        "header": header,
        "rows": rows,
    })
    if have_labels:
        body.append(
            "The *Platform DOF* column names the floating-platform "
            "rigid-body modes (surge / sway / heave / roll / pitch / "
            "yaw); ``—`` marks a mode no single platform DOF dominates "
            "(a flexible tower mode, or a strongly coupled pair). The "
            "same labels are on ``result.mode_labels``."
        )
    if have_participation:
        body.append(
            "Per-mode participation array attached to ``result.participation`` "
            "(N × 3). The table above derives shares from the FEM eigenvector "
            "energies for self-containedness; the attached array is the "
            "preferred metric for downstream code."
        )
    return _Section("4. Mode classification", body)


def _section_polynomial_coefficients(
    tower_params: TowerElastoDynParams | None,
    blade_params: BladeElastoDynParams | None,
) -> _Section:
    body: list[Any] = []

    if tower_params is not None:
        body.append("**Tower coefficients (C2 .. C6)**")
        rows = []
        for name in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh"):
            fit = getattr(tower_params, name)
            rows.append([
                name,
                f"{fit.c2:+.4e}", f"{fit.c3:+.4e}", f"{fit.c4:+.4e}",
                f"{fit.c5:+.4e}", f"{fit.c6:+.4e}",
                f"{fit.rms_residual:.4f}",
                f"{fit.cond_number:.2e}",
            ])
        body.append({
            "kind": "table",
            "header": ["Block", "C2", "C3", "C4", "C5", "C6",
                       "RMS residual", "Cond number"],
            "rows": rows,
        })

    if blade_params is not None:
        body.append("**Blade coefficients (C2 .. C6)**")
        rows = []
        for name in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh"):
            fit = getattr(blade_params, name)
            rows.append([
                name,
                f"{fit.c2:+.4e}", f"{fit.c3:+.4e}", f"{fit.c4:+.4e}",
                f"{fit.c5:+.4e}", f"{fit.c6:+.4e}",
                f"{fit.rms_residual:.4f}",
                f"{fit.cond_number:.2e}",
            ])
        body.append({
            "kind": "table",
            "header": ["Block", "C2", "C3", "C4", "C5", "C6",
                       "RMS residual", "Cond number"],
            "rows": rows,
        })

    if not body:
        body.append("No polynomial coefficients supplied.")
    return _Section("5. Polynomial coefficients with fit residuals", body)


def _section_validation(validation: ValidationResult) -> _Section:
    body: list[Any] = []
    body.append(f"**Overall verdict**: {validation.overall}")
    rows = []
    for block in validation.all_blocks().values():
        rows.append([
            block.name,
            f"{block.file_rms:.4f}",
            f"{block.pybmodes_rms:.4f}",
            f"{block.ratio:.2f}",
            block.verdict,
        ])
    if rows:
        body.append({
            "kind": "table",
            "header": ["Block", "file RMS", "pyBmodes RMS",
                       "ratio (file/pyB)", "verdict"],
            "rows": rows,
        })
    return _Section("6. Validation verdict", body)


def _section_check_warnings(
    check_warnings: list[ModelWarning],
) -> _Section:
    body: list[Any] = []
    if not check_warnings:
        body.append("No check_model warnings.")
        return _Section("7. check_model warnings", body)
    rows = [
        [w.severity, w.location, w.message]
        for w in check_warnings
    ]
    body.append({
        "kind": "table",
        "header": ["Severity", "Location", "Message"],
        "rows": rows,
    })
    return _Section("7. check_model warnings", body)


def _section_campbell(campbell: CampbellResult) -> _Section:
    body: list[Any] = []
    body.append(
        f"Sweep over {campbell.omega_rpm.size} rotor speed(s) "
        f"({float(campbell.omega_rpm.min()):.2f} – "
        f"{float(campbell.omega_rpm.max()):.2f} rpm), "
        f"{campbell.n_blade_modes} blade mode(s) + "
        f"{campbell.n_tower_modes} tower mode(s)."
    )
    body.append(
        "Frequencies at the first and last rotor speed for every mode:"
    )
    rows = []
    for k, lbl in enumerate(campbell.labels):
        rows.append([
            lbl,
            f"{float(campbell.frequencies[0, k]):.4f}",
            f"{float(campbell.frequencies[-1, k]):.4f}",
        ])
    body.append({
        "kind": "table",
        "header": ["Mode label",
                   f"f @ {float(campbell.omega_rpm[0]):.2f} rpm (Hz)",
                   f"f @ {float(campbell.omega_rpm[-1]):.2f} rpm (Hz)"],
        "rows": rows,
    })
    return _Section("8. Campbell sweep", body)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_markdown(sections: list[_Section]) -> str:
    out = io.StringIO()
    out.write("# pyBmodes Modal Analysis Report\n\n")
    for section in sections:
        out.write(f"## {section.title}\n\n")
        for item in section.body:
            if isinstance(item, str):
                out.write(item)
                out.write("\n\n")
            elif isinstance(item, dict) and item.get("kind") == "table":
                _render_md_table(out, item["header"], item["rows"])
                out.write("\n")
    return out.getvalue()


def _render_md_table(
    out: io.StringIO, header: list[str], rows: list[list[str]],
) -> None:
    out.write("| " + " | ".join(header) + " |\n")
    out.write("| " + " | ".join("---" for _ in header) + " |\n")
    for row in rows:
        out.write("| " + " | ".join(row) + " |\n")


def _render_html(sections: list[_Section]) -> str:
    out = io.StringIO()
    out.write(
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        '  <title>pyBmodes Modal Analysis Report</title>\n'
        "  <style>\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Helvetica, Arial, sans-serif; "
        "max-width: 960px; margin: 2em auto; padding: 0 1em; }\n"
        "    h1 { border-bottom: 2px solid #333; padding-bottom: 0.2em; }\n"
        "    h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.1em; "
        "margin-top: 2em; }\n"
        "    table { border-collapse: collapse; margin: 1em 0; }\n"
        "    th, td { border: 1px solid #ccc; padding: 4px 10px; "
        "text-align: left; }\n"
        "    th { background: #f0f0f0; }\n"
        "    code { background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>pyBmodes Modal Analysis Report</h1>\n"
    )
    for section in sections:
        out.write(f"<h2>{_html.escape(section.title)}</h2>\n")
        for item in section.body:
            if isinstance(item, str):
                out.write(f"<p>{_md_inline_to_html(item)}</p>\n")
            elif isinstance(item, dict) and item.get("kind") == "table":
                _render_html_table(out, item["header"], item["rows"])
    out.write("</body>\n</html>\n")
    return out.getvalue()


def _md_inline_to_html(text: str) -> str:
    """Minimal inline-markdown to HTML conversion: ``**bold**``,
    ``*italic*``, and ```code```. The structured tables are rendered
    separately so this only needs to handle paragraph text."""
    import re
    s = _html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return s


def _render_html_table(
    out: io.StringIO, header: list[str], rows: list[list[str]],
) -> None:
    out.write("<table>\n  <thead><tr>")
    for h in header:
        out.write(f"<th>{_html.escape(h)}</th>")
    out.write("</tr></thead>\n  <tbody>\n")
    for row in rows:
        out.write("    <tr>")
        for cell in row:
            out.write(f"<td>{_html.escape(cell)}</td>")
        out.write("</tr>\n")
    out.write("  </tbody>\n</table>\n")


def _render_csv(
    result: ModalResult,
    tower_params: TowerElastoDynParams | None,
    blade_params: BladeElastoDynParams | None,
) -> str:
    """CSV emission is intentionally narrower than markdown / HTML —
    we drop the narrative sections (assumptions, validation, warnings)
    that CSV cannot represent without faking columns, and emit:

    1. A frequencies block (one row per mode).
    2. A blank-row separator.
    3. A polynomial-coefficient block (one row per coefficient block)
       with explicit C2..C6 + RMS + cond-number columns — exactly the
       data spreadsheet workflows need.
    """
    out = io.StringIO()
    writer = csv.writer(out)

    # Frequencies block
    writer.writerow(["section", "mode", "frequency_hz", "period_s"])
    for i, f in enumerate(result.frequencies, start=1):
        period = 1.0 / float(f) if float(f) > 0 else float("nan")
        writer.writerow(["frequencies", i, float(f), period])

    # Blank separator row
    writer.writerow([])

    # Coefficients block
    writer.writerow([
        "section", "block", "C2", "C3", "C4", "C5", "C6",
        "rms_residual", "cond_number",
    ])
    if tower_params is not None:
        for name in ("TwFAM1Sh", "TwFAM2Sh", "TwSSM1Sh", "TwSSM2Sh"):
            fit = getattr(tower_params, name)
            writer.writerow([
                "tower", name,
                fit.c2, fit.c3, fit.c4, fit.c5, fit.c6,
                fit.rms_residual, fit.cond_number,
            ])
    if blade_params is not None:
        for name in ("BldFl1Sh", "BldFl2Sh", "BldEdgSh"):
            fit = getattr(blade_params, name)
            writer.writerow([
                "blade", name,
                fit.c2, fit.c3, fit.c4, fit.c5, fit.c6,
                fit.rms_residual, fit.cond_number,
            ])
    return out.getvalue()
