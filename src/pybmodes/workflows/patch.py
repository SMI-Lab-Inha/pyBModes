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

"""``pybmodes patch`` workflow as a typed library function.

Regenerates the tower + blade polynomial coefficient blocks in an
ElastoDyn ``.dat`` deck from the structural-property inputs.

Five mutually-supportive output modes, exposed both via the CLI and as
keyword arguments of :func:`run_patch`:

* **default in-place** — overwrites the user's ``.dat`` files.
* ``backup=True`` — same as default plus ``.bak`` copies first.
* ``output_dir=DIR`` — writes ``DIR/<filename>.dat`` instead of in-place,
  leaving the originals untouched.
* ``dry_run=True`` — computes the patched text but writes nothing.
* ``diff=True`` — implies dry-run, additionally produces a PR-ready
  coefficient-only diff with per-block RMS-improvement ratios.

The compute / write split is deliberate: each side's patched text is
generated into a temporary file regardless of mode, so dry-run and diff
share a code path with the real writes and can't drift.
"""
from __future__ import annotations

import difflib
import math
import pathlib
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from pybmodes.workflows._base import WorkflowResult

if TYPE_CHECKING:
    from pybmodes.elastodyn.params import (
        BladeElastoDynParams,
        TowerElastoDynParams,
    )
    from pybmodes.elastodyn.validate import ValidationResult


@dataclass
class PatchResult(WorkflowResult):
    """Result of :func:`run_patch`.

    Attributes
    ----------
    main_dat : pathlib.Path | None
        Resolved absolute path of the ElastoDyn main ``.dat`` file the
        workflow operated on. ``None`` only when the workflow short-
        circuited before resolving the main file (currently no such
        path; reserved).
    tower_dat : pathlib.Path | None
        Resolved absolute path of the tower side-deck (``TwrFile``
        referenced from the main).
    blade_dat : pathlib.Path | None
        Resolved absolute path of the blade-1 side-deck (``BldFile(1)``).
    tower_params, blade_params
        Fitted polynomial coefficient blocks for the tower and blade
        sides. ``None`` only if the workflow failed before the fit.
    validation : pybmodes.elastodyn.validate.ValidationResult | None
        Populated only in ``diff`` mode (the validator is needed for
        per-block RMS-improvement annotations); ``None`` otherwise.
    tower_patched_text, blade_patched_text : str | None
        The full post-patch text of each side-deck, computed without
        modifying the user's files. Always populated on success
        regardless of mode (so callers can compare / diff / persist
        elsewhere without re-running the workflow).
    wrote : list[pathlib.Path]
        Absolute paths of files actually written. Empty in dry-run /
        diff mode; one entry per side in ``output_dir`` mode; two
        entries (tower + blade) in in-place mode.
    n_tower_changed, n_blade_changed : int
        Number of changed lines that would (or did) result from the
        patch, useful for summary print-outs.
    """

    main_dat: "pathlib.Path | None" = None
    tower_dat: "pathlib.Path | None" = None
    blade_dat: "pathlib.Path | None" = None
    tower_params: "TowerElastoDynParams | None" = None
    blade_params: "BladeElastoDynParams | None" = None
    validation: "ValidationResult | None" = None
    tower_patched_text: "str | None" = None
    blade_patched_text: "str | None" = None
    wrote: list[pathlib.Path] = field(default_factory=list)
    n_tower_changed: int = 0
    n_blade_changed: int = 0


def _patched_text(
    source: pathlib.Path,
    params: Union["BladeElastoDynParams", "TowerElastoDynParams"],
) -> str:
    """Apply ``patch_dat`` to a temp copy and return the resulting text.

    Decouples the "compute" step from the "write output" step so dry-
    run / diff / output-dir / in-place all reuse the same patched-text
    bytes — they only differ in what they do with the result.
    """
    from pybmodes.elastodyn import patch_dat

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=source.suffix, delete=False, encoding="utf-8",
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        shutil.copy2(source, tmp_path)
        patch_dat(tmp_path, params)
        return tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp_path.unlink(missing_ok=True)


def _count_changed_lines(original: pathlib.Path, new_text: str) -> int:
    """How many added-or-removed lines a unified diff between
    ``original`` and ``new_text`` would carry."""
    original_text = original.read_text(encoding="utf-8", errors="replace")
    return sum(
        1 for line in difflib.unified_diff(
            original_text.splitlines(),
            new_text.splitlines(),
            lineterm="",
        )
        if line and line[0] in "+-" and not line.startswith(("+++", "---"))
    )


def _format_diff_block(
    name: str,
    file_label: str,
    block: "object",
) -> list[str]:
    """Render one coefficient block in the PR-ready ``--diff`` format.

    ``block`` is a :class:`~pybmodes.elastodyn.validate.CoeffBlockResult`
    (typed as object here to keep the heavy import lazy).
    """
    lines: list[str] = [f"@@ {name}  ({file_label}) @@"]
    for k, c in enumerate(block.file_coeffs):  # type: ignore[attr-defined]
        lines.append(f"-   {name}({k + 2}) = {float(c):+.4e}")
    for k, c in enumerate(block.pybmodes_coeffs):  # type: ignore[attr-defined]
        lines.append(f"+   {name}({k + 2}) = {float(c):+.4e}")
    file_rms = block.file_rms  # type: ignore[attr-defined]
    pyb_rms = block.pybmodes_rms  # type: ignore[attr-defined]
    if pyb_rms > 0.0 and math.isfinite(pyb_rms):
        ratio = file_rms / pyb_rms
        ratio_str = (
            f"{ratio:>5.0f}×" if ratio >= 100.0
            else f"{ratio:>5.1f}×" if ratio >= 10.0
            else f"{ratio:>5.2f}×"
        )
        lines.append(
            f"  RMS improvement: {file_rms:.4f} -> {pyb_rms:.4f} "
            f"({ratio_str} better)"
        )
    else:
        lines.append(
            f"  RMS improvement: {file_rms:.4f} -> {pyb_rms:.4f} "
            "(already at numerical precision)"
        )
    lines.append("")
    return lines


def run_patch(
    dat_path: "str | pathlib.Path",
    *,
    n_modes: int = 10,
    backup: bool = False,
    output_dir: "str | pathlib.Path | None" = None,
    dry_run: bool = False,
    diff: bool = False,
) -> PatchResult:
    """Regenerate the tower + blade polynomial blocks of an ElastoDyn deck.

    Library entry point for :command:`pybmodes patch`. Builds the
    cantilever-basis tower model and the rotating-blade model from the
    deck's structural inputs, fits 6th-order polynomial mode-shape
    coefficients, and either writes them back to the deck or returns
    them for inspection.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        ElastoDyn main ``.dat`` file. The tower side-deck
        (``TwrFile``) and blade-1 side-deck (``BldFile(1)``) are
        resolved relative to it.
    n_modes : int, default 10
        Number of FEM modes to solve before extracting the polynomial
        blocks. The default matches the CLI default.
    backup : bool, default False
        In-place mode only: copy each side-deck to a ``.bak`` sibling
        before overwriting.
    output_dir : str, pathlib.Path, or None, default None
        Write patched copies into this directory instead of overwriting
        the originals. Mutually exclusive with ``dry_run`` / ``diff``.
    dry_run : bool, default False
        Compute the patched text without writing anything. Mutually
        exclusive with ``output_dir``.
    diff : bool, default False
        Compute the patched text without writing anything, AND emit a
        PR-ready coefficient-only diff into :attr:`PatchResult.messages`
        with per-block RMS-improvement annotations. Implies dry-run.

    Returns
    -------
    PatchResult
        Carries the resolved side-deck paths, the fitted parameters,
        the patched text for both sides, and (in ``diff`` mode) the
        :class:`~pybmodes.elastodyn.validate.ValidationResult` used to
        derive the RMS-improvement ratios. ``exit_code`` is ``0`` on
        success.

    Raises
    ------
    FileNotFoundError
        When ``dat_path``, the tower side-deck, or the blade side-deck
        does not exist.
    ValueError
        When ``output_dir`` is combined with ``dry_run`` / ``diff``
        (those modes write nothing, so a destination is meaningless).
    """
    from pybmodes.elastodyn import (
        compute_blade_params,
        compute_tower_params,
        validate_dat_coefficients,
    )
    from pybmodes.io.elastodyn_reader import read_elastodyn_main
    from pybmodes.models import RotatingBlade, Tower

    if output_dir is not None and (dry_run or diff):
        raise ValueError(
            "output_dir is incompatible with dry_run / diff "
            "(those modes write nothing, so an output destination is "
            "meaningless)"
        )

    main_dat = pathlib.Path(dat_path).resolve()
    if not main_dat.is_file():
        raise FileNotFoundError(f"file not found: {main_dat}")

    main = read_elastodyn_main(main_dat)
    tower_dat = main_dat.parent / main.twr_file
    blade_dat = main_dat.parent / main.bld_file[0]

    if not tower_dat.is_file():
        raise FileNotFoundError(f"tower file not found: {tower_dat}")
    if not blade_dat.is_file():
        raise FileNotFoundError(f"blade file not found: {blade_dat}")

    out_dir = pathlib.Path(output_dir).resolve() if output_dir else None
    write_mode = "skip" if (dry_run or diff) else (
        "output_dir" if out_dir is not None else "in_place"
    )

    messages: list[str] = []
    messages.append("pyBmodes coefficient patch")
    messages.append("==========================")
    messages.append(f"Main:  {main_dat}")
    messages.append(f"Tower: {tower_dat}")
    messages.append(f"Blade: {blade_dat}")
    if write_mode == "skip":
        messages.append("Mode:  dry-run (no files will be modified)")
    elif write_mode == "output_dir":
        messages.append(f"Mode:  write to {out_dir}/")
    else:
        messages.append(
            "Mode:  in-place" + (" (with .bak backup)" if backup else "")
        )
        if not backup:
            # First-time-run hint promised by README's 1.0 milestone.
            messages.append(
                "       (recommend `--dry-run --diff` for a first-time "
                "review; add `--backup` or use `--output-dir` to keep "
                "the originals)"
            )
    messages.append("")

    messages.append("  building tower model + fitting polynomials ...")
    tower = Tower.from_elastodyn(main_dat)
    tower_modal = tower.run(n_modes=n_modes, check_model=False)
    tower_params = compute_tower_params(tower_modal)

    messages.append("  building blade model + fitting polynomials ...")
    blade = RotatingBlade.from_elastodyn(main_dat)
    blade_modal = blade.run(n_modes=n_modes, check_model=False)
    blade_params = compute_blade_params(blade_modal)

    tower_patched_text = _patched_text(tower_dat, tower_params)
    blade_patched_text = _patched_text(blade_dat, blade_params)

    n_tower_changed = _count_changed_lines(tower_dat, tower_patched_text)
    n_blade_changed = _count_changed_lines(blade_dat, blade_patched_text)
    messages.append("")
    messages.append("  summary of proposed changes:")
    messages.append(
        f"    {tower_dat.name}: {n_tower_changed} line(s) would change"
    )
    messages.append(
        f"    {blade_dat.name}: {n_blade_changed} line(s) would change"
    )

    validation = None
    if diff:
        validation = validate_dat_coefficients(main_dat)
        tower_block_names = set(validation.tower_results.keys())
        messages.append("")
        messages.append("--- original")
        messages.append("+++ patched")
        for name, block in validation.all_blocks().items():
            file_label = (
                tower_dat.name if name in tower_block_names else blade_dat.name
            )
            messages.extend(_format_diff_block(name, file_label, block))

    wrote: list[pathlib.Path] = []
    if write_mode == "output_dir":
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        for source, new_text in (
            (tower_dat, tower_patched_text),
            (blade_dat, blade_patched_text),
        ):
            target = out_dir / source.name
            target.write_text(new_text, encoding="utf-8")
            wrote.append(target)
            messages.append(f"  wrote {target}")
        messages.append("")
        messages.append(
            f"Done. Patched files in {out_dir}/; run "
            "`pybmodes validate` against a corresponding ElastoDyn main "
            "file referring to them to confirm consistency."
        )
    elif write_mode == "in_place":
        from pybmodes.elastodyn import patch_dat

        if backup:
            messages.append("")
            for target in (tower_dat, blade_dat):
                bak = target.with_suffix(target.suffix + ".bak")
                shutil.copy2(target, bak)
                messages.append(f"  backed up {target.name} -> {bak.name}")
        messages.append("")
        messages.append("  patching tower .dat in place ...")
        patch_dat(tower_dat, tower_params)
        wrote.append(tower_dat)
        messages.append("  patching blade .dat in place ...")
        patch_dat(blade_dat, blade_params)
        wrote.append(blade_dat)
        messages.append("")
        messages.append(
            "Done. Re-run `pybmodes validate` to confirm consistency."
        )
    else:  # skip
        messages.append("")
        messages.append("Dry-run complete; no files modified.")

    return PatchResult(
        exit_code=0,
        messages=messages,
        main_dat=main_dat,
        tower_dat=tower_dat,
        blade_dat=blade_dat,
        tower_params=tower_params,
        blade_params=blade_params,
        validation=validation,
        tower_patched_text=tower_patched_text,
        blade_patched_text=blade_patched_text,
        wrote=wrote,
        n_tower_changed=n_tower_changed,
        n_blade_changed=n_blade_changed,
    )
