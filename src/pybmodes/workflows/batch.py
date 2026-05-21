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

"""``pybmodes batch`` workflow as a typed library function.

Walks a directory tree for ElastoDyn main ``.dat`` files, runs
validate + optional patch on each deck, and writes a summary CSV plus
optional per-deck validation reports.

Exit-code policy mirrors the original CLI:

* ``0`` — every deck reaches a non-FAIL overall verdict (PASS or WARN).
* ``1`` — at least one deck remained at FAIL after patching, or
  errored during parse / fit. The summary CSV still lists every deck
  the workflow attempted.
"""
from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass, field
from typing import Literal

from pybmodes.workflows._base import WorkflowResult
from pybmodes.workflows.validate import _render_validation_report

_ELASTODYN_EXCLUDE_TOKENS = (
    "_tower",
    "_blade",
    "_subdyn",
    "_hydrodyn",
    "_moordyn",
    "_servodyn",
    "_aerodyn",
    "_inflowwind",
    "_beamdyn",
    "_icefloe",
)

BatchKind = Literal["elastodyn"]


@dataclass
class BatchResult(WorkflowResult):
    """Result of :func:`run_batch`.

    Attributes
    ----------
    root : pathlib.Path | None
        Resolved absolute path of the directory walked.
    out_dir : pathlib.Path | None
        Resolved absolute path of the output directory (parent of
        ``summary.csv``).
    summary_path : pathlib.Path | None
        Path of the written ``summary.csv``.
    decks_found : int
        Number of ElastoDyn main decks discovered under ``root``.
    decks_failed : int
        Number of decks that ended at ``FAIL`` or ``ERROR`` (drives the
        non-zero exit code).
    summary_rows : list[dict]
        Per-deck summary rows. Each row is the dict written as a CSV
        line (``filename`` relative to ``root``, ``overall_verdict``,
        ``TwFAM2Sh_ratio``, ``TwSSM2Sh_ratio``, ``n_fail``, ``n_warn``).
    """

    root: "pathlib.Path | None" = None
    out_dir: "pathlib.Path | None" = None
    summary_path: "pathlib.Path | None" = None
    decks_found: int = 0
    decks_failed: int = 0
    summary_rows: list[dict[str, object]] = field(default_factory=list)


def find_elastodyn_main_dats(root: pathlib.Path) -> list[pathlib.Path]:
    """Walk ``root`` recursively and return every file that looks like
    an ElastoDyn **main** input.

    Two-stage filter:

    1. Name heuristic: must contain ``ElastoDyn`` (case-insensitive)
       and must NOT contain any auxiliary-file token
       (``_Tower``, ``_Blade``, ``_SubDyn``, etc.).
    2. Parse confirmation: must round-trip through
       :func:`pybmodes.io.elastodyn_reader.read_elastodyn_main` and
       carry a non-empty ``TwrFile`` reference. Files that fail to
       parse are silently skipped — the batch workflow can't act on
       them anyway.
    """
    from pybmodes.io.elastodyn_reader import read_elastodyn_main

    out: list[pathlib.Path] = []
    for p in sorted(root.rglob("*.dat")):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if "elastodyn" not in name_lower:
            continue
        if any(tok in name_lower for tok in _ELASTODYN_EXCLUDE_TOKENS):
            continue
        try:
            main = read_elastodyn_main(p)
        except (OSError, ValueError, IndexError, AttributeError):
            continue
        if not main.twr_file:
            continue
        out.append(p)
    return out


def _ratio(name: str, result: "object") -> float:
    block = result.tower_results.get(name)  # type: ignore[attr-defined]
    return float(block.ratio) if block is not None else float("nan")


def run_batch(
    root: "str | pathlib.Path",
    out_dir: "str | pathlib.Path",
    *,
    kind: BatchKind = "elastodyn",
    validate: bool = False,
    patch: bool = False,
    n_modes: int = 10,
    dry_run: bool = False,
    backup: bool = True,
    output_dir: "str | pathlib.Path | None" = None,
) -> BatchResult:
    """Walk a directory tree of ElastoDyn decks, validate + optionally
    patch each one, and write a summary CSV.

    Library entry point for :command:`pybmodes batch`.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to walk recursively for ElastoDyn main decks.
    out_dir : str or pathlib.Path
        Output directory. Created if missing. Receives per-deck
        validation reports (when ``validate=True``) and the
        ``summary.csv``. **Distinct** from ``output_dir`` below —
        ``out_dir`` carries the batch *reports*, ``output_dir``
        carries the patched *decks*.
    kind : {"elastodyn"}, default "elastodyn"
        Which deck flavour to look for. Only ElastoDyn is supported
        today; passing anything else raises ``ValueError``.
    validate : bool, default False
        Write a per-deck ``<deckname>_validate.txt`` containing the
        validation report. The validator itself ALWAYS runs (its
        ``overall_verdict`` populates the summary CSV); this flag
        only controls the per-deck text file.
    patch : bool, default False
        Regenerate the polynomial blocks for each deck and re-validate.
        Under the **default** ``backup=True`` (new in 1.8.0 — see
        :func:`run_patch`), each deck's tower / blade side-decks are
        copied to ``.bak`` siblings before the in-place rewrite so a
        botched run is recoverable. Pass ``backup=False`` for the
        legacy "modify in place without safety net" semantics. When
        combined with ``validate=True``, a second per-deck text file
        named ``<deckname>_validate_after.txt`` captures the post-patch
        state.
    n_modes : int, default 10
        Number of FEM modes to solve when patching.
    dry_run : bool, default False
        Patch-mode safety lever — compute the patched coefficients for
        each deck without writing anything. The summary CSV still
        carries the BEFORE-patch verdict (source files unchanged); the
        per-deck messages report what *would* have changed. Mutually
        exclusive with ``output_dir``.
    backup : bool, default ``True``
        Patch-mode safety lever — copy each tower / blade side-deck to
        a ``.bak`` sibling before overwriting in place. Default changed
        from ``False`` to ``True`` in 1.8.0: ``pybmodes batch --patch``
        sweeps a directory tree, so a single bad run can mutate decks
        the user didn't realise discovery picked up. Local-only safety
        artefact; ``*.bak`` is gitignored. Ignored in ``dry_run`` or
        ``output_dir`` mode (those write nothing or write elsewhere).
    output_dir : str, pathlib.Path, or None, default None
        Patch-mode safety lever — write the patched copies of each
        deck's tower / blade side-decks into
        ``output_dir / <deck stem>/`` instead of overwriting the
        originals. The source tree is untouched. Mutually exclusive
        with ``dry_run``.

    Returns
    -------
    BatchResult
        Carries the discovered-deck count, per-deck summary rows, the
        summary CSV path, and the failed-deck count. ``exit_code`` is
        ``0`` on all-good and ``1`` when any deck failed or errored.

    Raises
    ------
    ValueError
        When ``kind`` is anything other than ``"elastodyn"``, or when
        ``dry_run`` is combined with ``output_dir`` (those modes are
        mutually exclusive — same convention as :func:`run_patch`).
    FileNotFoundError
        When ``root`` does not exist or is not a directory.
    """
    if kind != "elastodyn":
        raise ValueError(
            f"kind {kind!r} not supported (only 'elastodyn' for now)"
        )
    # Patch-mode safety levers are only consulted when ``patch=True``;
    # a validation-only batch call carrying leftover ``--dry-run`` /
    # ``--output-dir`` flags must NOT raise (the CLI help text
    # explicitly documents the levers as ignored unless ``--patch`` is
    # set). Static-review follow-up on PR #77.
    if patch and dry_run and output_dir is not None:
        raise ValueError(
            "dry_run is mutually exclusive with output_dir "
            "(dry_run writes nothing, output_dir writes elsewhere — "
            "pass one or the other, not both)"
        )

    from pybmodes.elastodyn import validate_dat_coefficients
    from pybmodes.workflows.patch import run_patch

    root_p = pathlib.Path(root).resolve()
    if not root_p.is_dir():
        raise FileNotFoundError(f"root directory not found: {root_p}")

    out_p = pathlib.Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    output_root = (
        pathlib.Path(output_dir).resolve() if output_dir is not None else None
    )

    decks = find_elastodyn_main_dats(root_p)
    messages: list[str] = []
    messages.append(
        f"batch: found {len(decks)} ElastoDyn main deck(s) under {root_p}"
    )
    if patch:
        if dry_run:
            mode = "dry-run (no files modified)"
        elif output_root is not None:
            mode = f"write to {output_root}/<deck>/"
        else:
            mode = (
                "in-place with .bak backup" if backup
                else "in-place (no backup — pass backup=True for safety)"
            )
        messages.append(f"  patch mode: {mode}")

    summary_rows: list[dict[str, object]] = []
    for deck in decks:
        try:
            rel = deck.relative_to(root_p)
        except ValueError:
            rel = deck
        messages.append("")
        messages.append(f"[{rel}]")

        # --- 1. Initial validate. Always runs; cheap, and we need it
        # for the summary row regardless of validate / patch flags.
        try:
            result = validate_dat_coefficients(deck)
        except Exception as exc:
            messages.append(f"  parse / validate ERROR: {exc!r}")
            summary_rows.append({
                "filename": str(rel),
                "overall_verdict": "ERROR",
                "TwFAM2Sh_ratio": float("nan"),
                "TwSSM2Sh_ratio": float("nan"),
                "n_fail": 0,
                "n_warn": 0,
            })
            continue

        if validate:
            report_path = out_p / f"{deck.stem}_validate.txt"
            report_text = "\n".join(_render_validation_report(result)) + "\n"
            report_path.write_text(report_text, encoding="utf-8")
            messages.append(f"  wrote {report_path.name}")

        # --- 2. Optional patch — delegate to run_patch per deck so the
        # full safety machinery (dry_run / backup / output_dir / compute
        # -before-write split) is reused unchanged.
        if patch:
            try:
                # Use the deck's RELATIVE path under ``root`` (not just
                # the stem) as the per-deck destination, so two decks
                # with the same filename in different sub-directories
                # land in distinct sub-trees instead of silently
                # overwriting each other. Static-review follow-up on
                # PR #77.
                per_deck_out: "pathlib.Path | None" = None
                if output_root is not None:
                    try:
                        rel_parent = deck.relative_to(root_p).parent
                    except ValueError:
                        rel_parent = pathlib.Path()
                    per_deck_out = output_root / rel_parent / deck.stem
                patch_result = run_patch(
                    deck,
                    n_modes=n_modes,
                    dry_run=dry_run,
                    backup=backup,
                    output_dir=per_deck_out,
                )
                for line in patch_result.messages:
                    messages.append(f"  {line}")
                # Only re-validate when the patch actually wrote new
                # coefficients back into the deck's source tree (in-
                # place mode). Dry-run and output-dir mode leave the
                # source unchanged, so the summary row keeps the
                # BEFORE-patch verdict — which IS the source-on-disk
                # state — without misleading callers into thinking a
                # re-validation happened.
                if (not dry_run) and output_root is None:
                    result = validate_dat_coefficients(deck)
                    if validate:
                        after_path = (
                            out_p / f"{deck.stem}_validate_after.txt"
                        )
                        after_text = (
                            "\n".join(_render_validation_report(result))
                            + "\n"
                        )
                        after_path.write_text(after_text, encoding="utf-8")
                        messages.append(f"  wrote {after_path.name}")
            except Exception as exc:
                messages.append(f"  patch ERROR: {exc!r}")
                summary_rows.append({
                    "filename": str(rel),
                    "overall_verdict": "ERROR",
                    "TwFAM2Sh_ratio": float("nan"),
                    "TwSSM2Sh_ratio": float("nan"),
                    "n_fail": 0,
                    "n_warn": 0,
                })
                continue

        # --- 3. Summary row from the (possibly post-patch) result.
        summary_rows.append({
            "filename": str(rel),
            "overall_verdict": result.overall,
            "TwFAM2Sh_ratio": _ratio("TwFAM2Sh", result),
            "TwSSM2Sh_ratio": _ratio("TwSSM2Sh", result),
            "n_fail": len(result.failing_blocks()),
            "n_warn": len(result.warning_blocks()),
        })

    # --- 4. Write summary CSV.
    summary_path = out_p / "summary.csv"
    fieldnames = [
        "filename", "overall_verdict",
        "TwFAM2Sh_ratio", "TwSSM2Sh_ratio",
        "n_fail", "n_warn",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            # csv.DictWriter writes math.nan as "nan", which downstream
            # readers handle. Normalise explicitly so the output is
            # stable across platforms.
            r = dict(row)
            for k in ("TwFAM2Sh_ratio", "TwSSM2Sh_ratio"):
                v = r.get(k)
                if isinstance(v, float) and math.isnan(v):
                    r[k] = "nan"
            writer.writerow(r)
    messages.append("")
    messages.append(f"wrote summary: {summary_path}")

    n_bad = sum(
        1 for r in summary_rows
        if r["overall_verdict"] in ("FAIL", "ERROR")
    )
    if n_bad:
        messages.append("")
        messages.append(
            f"{n_bad}/{len(summary_rows)} deck(s) at FAIL / ERROR; "
            "exit code 1"
        )

    return BatchResult(
        exit_code=1 if n_bad else 0,
        messages=messages,
        root=root_p,
        out_dir=out_p,
        summary_path=summary_path,
        decks_found=len(decks),
        decks_failed=n_bad,
        summary_rows=summary_rows,
    )
