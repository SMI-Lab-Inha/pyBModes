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

"""``pybmodes examples --copy`` workflow as a typed library function.

Vendors the bundled ``sample_inputs/`` and / or ``reference_decks/``
trees out of the installed :mod:`pybmodes._examples` package into a
user-supplied destination directory. Works for wheel installs (where
the bundles are package-data) and editable source installs (where
they're a literal ``src/pybmodes/_examples/`` directory) — the same
resolver locates ``pybmodes.__file__`` and reads from the sibling
``_examples`` folder.
"""
from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass, field
from typing import Literal

from pybmodes.workflows._base import WorkflowResult

# bundle-name -> (sub-directory under pybmodes/_examples/, human description)
_EXAMPLE_BUNDLES: dict[str, tuple[str, str]] = {
    "samples": (
        "sample_inputs",
        "analytical-reference cases + 7 RWT samples",
    ),
    "decks": (
        "reference_decks",
        "6 patched ElastoDyn decks (land + monopile + floating)",
    ),
}

Kind = Literal["all", "samples", "decks"]


def _resolve_examples_root() -> pathlib.Path:
    """Locate ``pybmodes/_examples/`` on the installed package.

    Both wheel-installed and source-installed callers find the
    bundle tree alongside the imported :mod:`pybmodes` package.
    """
    import pybmodes

    pkg_dir = pathlib.Path(pybmodes.__file__).resolve().parent
    return pkg_dir / "_examples"


@dataclass
class ExamplesResult(WorkflowResult):
    """Result of :func:`run_examples_copy`.

    Attributes
    ----------
    dest : pathlib.Path | None
        The destination directory the bundles were copied into.
        ``None`` if the workflow short-circuited before copying
        anything (e.g. no requested bundles found in the installed
        package — exit code 2).
    copied : list[pathlib.Path]
        Absolute paths to each bundle successfully copied
        (typically ``dest / "sample_inputs"`` and / or ``dest /
        "reference_decks"``).
    skipped : list[str]
        Bundle names (``"samples"`` / ``"decks"``) that were
        requested but not found on disk inside the installed
        package. Each entry produces a warning line in
        ``messages`` / a ``WARN:`` prefix.
    """

    dest: "pathlib.Path | None" = None
    copied: list[pathlib.Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_examples_copy(
    dest: "str | pathlib.Path",
    *,
    kind: Kind = "all",
    force: bool = False,
) -> ExamplesResult:
    """Copy bundled example trees out of the installed pyBmodes package.

    Library entry point for :command:`pybmodes examples --copy DIR`.

    Parameters
    ----------
    dest : str or pathlib.Path
        Destination directory. Created if missing. Existing
        sub-bundles in this directory are an error unless
        ``force=True``.
    kind : {"all", "samples", "decks"}, default "all"
        Which bundle(s) to copy. ``"all"`` copies both
        ``sample_inputs/`` and ``reference_decks/``; ``"samples"``
        or ``"decks"`` selects only that one.
    force : bool, default False
        Overwrite existing destination sub-directories rather
        than erroring. Useful in CI scripts that re-vendor on
        every run.

    Returns
    -------
    ExamplesResult
        Carries the destination, the list of copied bundle paths,
        and any skipped bundle names. Exit code is ``0`` on
        success, ``2`` if no requested bundles were found in the
        installed package or if a destination existed without
        ``force=True``.
    """
    examples_root = _resolve_examples_root()

    requested: list[str] = (
        ["samples", "decks"] if kind == "all" else [kind]
    )
    selected: list[tuple[str, pathlib.Path]] = []
    skipped: list[str] = []
    for name in requested:
        subdir = _EXAMPLE_BUNDLES[name][0]
        src = examples_root / subdir
        if src.is_dir():
            selected.append((name, src))
        else:
            skipped.append(name)

    if not selected:
        return ExamplesResult(
            exit_code=2,
            errors=[
                "error: example bundles not found inside the "
                "installed pybmodes package.",
                f"       looked under: {examples_root}",
                "       expected the wheel to ship "
                "`pybmodes/_examples/sample_inputs/` and "
                "`pybmodes/_examples/reference_decks/` as package "
                "data; if you installed from a wheel and the "
                "directories are absent, the wheel is malformed. "
                "From a source checkout run `pip install -e .` "
                "from the repo root and retry.",
            ],
            skipped=skipped,
        )

    messages: list[str] = []
    errors: list[str] = []
    if skipped:
        errors.append(
            f"warning: skipping bundle(s) not found on disk: "
            f"{', '.join(skipped)}"
        )

    dest_root = pathlib.Path(dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    copied: list[pathlib.Path] = []
    for name, src in selected:
        target = dest_root / src.name
        if target.exists():
            if not force:
                return ExamplesResult(
                    exit_code=2,
                    dest=dest_root,
                    copied=copied,
                    skipped=skipped,
                    errors=[
                        f"error: destination already exists: {target}",
                        "       pass force=True (CLI: --force) to "
                        "overwrite, or pick an empty directory.",
                    ],
                )
            shutil.rmtree(target)
        shutil.copytree(src, target)
        copied.append(target)
        messages.append(f"copied {name}: {src} -> {target}")

    return ExamplesResult(
        exit_code=0,
        messages=messages,
        errors=errors,
        dest=dest_root,
        copied=copied,
        skipped=skipped,
    )
