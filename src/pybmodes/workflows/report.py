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

"""``pybmodes report`` workflow as a typed library function.

Runs the full modal-analysis pipeline on one ElastoDyn deck — tower +
blade FEM solves, polynomial fits, optional coefficient validation,
optional Campbell sweep — and writes a single combined Markdown / HTML
/ CSV report. The report module itself
(:func:`pybmodes.report.generate_report`) writes the file; this workflow
is responsible for orchestrating the inputs.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pybmodes.workflows._base import WorkflowResult

if TYPE_CHECKING:
    from pybmodes.campbell import CampbellResult
    from pybmodes.checks import ModelWarning
    from pybmodes.elastodyn.params import (
        BladeElastoDynParams,
        TowerElastoDynParams,
    )
    from pybmodes.elastodyn.validate import ValidationResult
    from pybmodes.models.result import ModalResult

ReportFormat = Literal["md", "html", "csv"]


@dataclass
class ReportResult(WorkflowResult):
    """Result of :func:`run_report`.

    Attributes
    ----------
    out_path : pathlib.Path | None
        Absolute path of the written report file.
    tower_modal, blade_modal
        Modal-analysis results for the tower and blade sides. The
        report itself is keyed off ``tower_modal`` (one combined
        report per deck); both are exposed here so callers can inspect
        the underlying frequencies / shapes without re-solving.
    tower_params, blade_params
        Fitted polynomial coefficient blocks.
    validation : ValidationResult | None
        Coefficient validation result; ``None`` when ``validate=False``.
    campbell : CampbellResult | None
        Campbell sweep result; ``None`` when ``campbell=False``.
    check_warnings : list[ModelWarning]
        Pre-solve check findings (tower + blade), captured for the
        report's ``check_model`` section. Surfaces both sides — the
        blade-side findings were missing in 0.x and are restored here.
    """

    out_path: pathlib.Path | None = None
    tower_modal: ModalResult | None = None
    blade_modal: ModalResult | None = None
    tower_params: TowerElastoDynParams | None = None
    blade_params: BladeElastoDynParams | None = None
    validation: ValidationResult | None = None
    campbell: CampbellResult | None = None
    check_warnings: list[ModelWarning] = field(default_factory=list)


def run_report(
    dat_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    *,
    n_modes: int = 10,
    format: ReportFormat = "md",
    validate: bool = True,
    campbell: bool = False,
    max_rpm: float = 15.0,
    n_steps: int = 16,
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
) -> ReportResult:
    """Run modal analysis + (optional) validation + (optional) Campbell
    on one ElastoDyn deck and write a combined report.

    Library entry point for :command:`pybmodes report`.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        ElastoDyn main ``.dat`` file.
    out_path : str or pathlib.Path
        Destination report file. Parent directory is created if missing.
    n_modes : int, default 10
        Number of FEM modes to solve per side.
    format : {"md", "html", "csv"}, default "md"
        Report format.
    validate : bool, default True
        Run :func:`~pybmodes.elastodyn.validate_dat_coefficients` and
        attach the result to the report's validation section.
    campbell : bool, default False
        Run a rotor-speed sweep from 0 to ``max_rpm`` in ``n_steps``
        points and attach the result to the report's Campbell section.
    max_rpm, n_steps, n_blade_modes, n_tower_modes
        Campbell-sweep parameters. Ignored when ``campbell=False``.

    Returns
    -------
    ReportResult
        Carries the rendered file path plus every intermediate
        artefact (modal results, fitted params, validation, Campbell
        sweep, pre-solve warnings) so callers can introspect without
        re-running the workflow.

    Raises
    ------
    FileNotFoundError
        When ``dat_path`` does not exist.
    """
    import numpy as np

    from pybmodes.checks import check_model as _check_model
    from pybmodes.elastodyn import (
        compute_blade_params,
        compute_tower_params,
        validate_dat_coefficients,
    )
    from pybmodes.models import RotatingBlade, Tower
    from pybmodes.report import generate_report

    main_dat = pathlib.Path(dat_path).resolve()
    if not main_dat.is_file():
        raise FileNotFoundError(f"file not found: {main_dat}")

    out = pathlib.Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    messages.append(
        f"report: building tower + blade models from {main_dat.name}"
    )

    tower_model = Tower.from_elastodyn(main_dat)
    blade_model = RotatingBlade.from_elastodyn(main_dat)
    tower_modal = tower_model.run(n_modes=n_modes, check_model=False)
    blade_modal = blade_model.run(n_modes=n_modes, check_model=False)
    tower_params = compute_tower_params(tower_modal)
    blade_params = compute_blade_params(blade_modal)

    # Pre-solve warnings (captured, not raised — surfaced via the
    # report's check_model section). Includes BOTH tower-side and
    # blade-side findings.
    check_warnings = list(_check_model(tower_model, n_modes=n_modes))
    check_warnings.extend(_check_model(blade_model, n_modes=n_modes))

    validation = (
        validate_dat_coefficients(main_dat) if validate else None
    )

    campbell_result = None
    if campbell:
        from pybmodes.campbell import campbell_sweep
        rpm_grid = np.linspace(0.0, max_rpm, n_steps)
        campbell_result = campbell_sweep(
            main_dat, rpm_grid,
            n_blade_modes=n_blade_modes,
            n_tower_modes=n_tower_modes,
        )

    generate_report(
        tower_modal,
        out,
        format=format,
        model=tower_model,
        validation=validation,
        check_warnings=check_warnings,
        tower_params=tower_params,
        blade_params=blade_params,
        campbell=campbell_result,
        source_file=main_dat,
    )
    messages.append(f"wrote {out}")

    return ReportResult(
        exit_code=0,
        messages=messages,
        out_path=out,
        tower_modal=tower_modal,
        blade_modal=blade_modal,
        tower_params=tower_params,
        blade_params=blade_params,
        validation=validation,
        campbell=campbell_result,
        check_warnings=check_warnings,
    )
