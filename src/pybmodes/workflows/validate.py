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

"""``pybmodes validate`` workflow as a typed library function.

Wraps :func:`pybmodes.elastodyn.validate_dat_coefficients` with a
:class:`ValidateResult` carrying the validation report plus
CLI-mapping fields (exit code, messages, errors).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pybmodes.workflows._base import WorkflowResult

if TYPE_CHECKING:
    from pybmodes.elastodyn.validate import (
        CoeffBlockResult,
        ValidationResult,
    )


def _fmt_ratio(ratio: float) -> str:
    """Right-align a coefficient-block ratio for the printed report."""
    if ratio != ratio:  # NaN
        return "  nan"
    if ratio == float("inf"):
        return "   inf"
    if ratio >= 100.0:
        return f"{ratio:>5.0f}"
    if ratio >= 10.0:
        return f"{ratio:>5.1f}"
    return f"{ratio:>5.2f}"


def _format_block_row(block: "CoeffBlockResult") -> str:
    """One per-block line for the printed report."""
    flag = ""
    if block.verdict == "FAIL":
        flag = "  FAIL <-"
    elif block.verdict == "WARN":
        flag = "  WARN"
    elif block.verdict == "PASS":
        flag = "  PASS"
    return (
        f"  {block.name:<8}  file RMS={block.file_rms:7.4f}  "
        f"pyB RMS={block.pybmodes_rms:7.4f}  "
        f"ratio={_fmt_ratio(block.ratio)}  {flag}"
    )


def _render_validation_report(result: "ValidationResult") -> list[str]:
    """Render a :class:`ValidationResult` as the line-by-line CLI
    report. Returned as a list of lines (no trailing newlines);
    the workflow caller / CLI joins with ``\\n`` and prints."""
    lines: list[str] = []
    lines.append("pyBmodes coefficient validator")
    lines.append("==============================")
    lines.append(f"File: {result.dat_path.name}")
    lines.append("")
    lines.append("Tower modes:")
    for block in result.tower_results.values():
        lines.append(_format_block_row(block))
    lines.append("")
    lines.append("Blade modes:")
    for block in result.blade_results.values():
        lines.append(_format_block_row(block))
    lines.append("")

    n_fail = len(result.failing_blocks())
    n_warn = len(result.warning_blocks())
    if result.overall == "FAIL":
        lines.append(f"Overall: FAIL ({n_fail} block(s))")
        lines.append(
            f"Recommendation: run `pybmodes patch {result.dat_path}` to "
            "update coefficients from structural inputs."
        )
    elif result.overall == "WARN":
        lines.append(f"Overall: WARN ({n_warn} block(s))")
    else:
        lines.append("Overall: PASS")
    return lines


@dataclass
class ValidateResult(WorkflowResult):
    """Result of :func:`run_validate`.

    Attributes
    ----------
    validation : pybmodes.elastodyn.validate.ValidationResult | None
        The full validation report — per-block PASS / WARN / FAIL
        verdicts, file-RMS and pyBmodes-RMS values, and the
        overall verdict. ``None`` only if the workflow short-
        circuited before validation could run (currently no such
        path; reserved for future strict-mode failures).
    """

    validation: "ValidationResult | None" = None


def run_validate(dat_path: "str | pathlib.Path") -> ValidateResult:
    """Validate the coefficient blocks in an ElastoDyn ``.dat`` deck.

    Library entry point for the :command:`pybmodes validate`
    subcommand. Parses the deck's polynomial blocks, re-fits each
    block from the structural inputs in the same file, scores
    file-RMS against pyBmodes-RMS, and returns a structured
    verdict.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to an ElastoDyn main ``.dat`` file. Resolved to an
        absolute path before validation.

    Returns
    -------
    ValidateResult
        Carries the full :class:`~pybmodes.elastodyn.ValidationResult`
        plus ``exit_code`` mapping (``0`` for PASS / WARN, ``1``
        for FAIL) and the per-block printable lines in
        ``messages``.

    Raises
    ------
    FileNotFoundError
        When ``dat_path`` does not exist or is not a regular file.
        Callers (CLI or library) should treat this as a usage /
        IO error rather than a verdict failure.

    Examples
    --------

    From a notebook::

        from pybmodes.workflows import run_validate

        res = run_validate("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat")
        if res.validation.overall == "FAIL":
            print("polynomial blocks are stale — run patch")
        for line in res.messages:
            print(line)
    """
    from pybmodes.elastodyn import validate_dat_coefficients

    p = pathlib.Path(dat_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {p}")

    validation = validate_dat_coefficients(p)
    messages = _render_validation_report(validation)

    if validation.overall == "FAIL":
        exit_code = 1
    else:
        # WARN counts as 0 — warnings are informational, not a hard
        # failure. Same convention as the legacy CLI.
        exit_code = 0

    return ValidateResult(
        exit_code=exit_code,
        messages=messages,
        validation=validation,
    )
