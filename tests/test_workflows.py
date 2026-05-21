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

"""Tests for the :mod:`pybmodes.workflows` library entry points.

Phase 2 PR B1 of the v1.x architecture refactor introduced two
workflow functions: :func:`run_validate` and
:func:`run_examples_copy`. These tests exercise each function
**directly** as a library call (no ``subprocess`` of the CLI, no
argparse), confirming that the typed result dataclass carries the
fields downstream callers (notebooks, external scripts) expect.

CLI-level smoke tests (testing the same workflows via the
``pybmodes`` subprocess invocation) live in ``test_validate.py`` /
``test_examples_cli.py`` and continue to pass against the slimmed-
down delegation wrappers in ``cli.py``.
"""
from __future__ import annotations

import pathlib

import pytest

from pybmodes.cli import _resolve_examples_root
from pybmodes.workflows import (
    ExamplesResult,
    ValidateResult,
    WorkflowResult,
    run_examples_copy,
    run_validate,
)

_SAMPLES = _resolve_examples_root() / "sample_inputs"
_REFERENCE_DECKS = _resolve_examples_root() / "reference_decks"
_NREL5MW_LAND_DAT = (
    _REFERENCE_DECKS
    / "nrel5mw_land"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)


# ---------------------------------------------------------------------------
# WorkflowResult / inheritance contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [ValidateResult, ExamplesResult])
def test_result_classes_inherit_workflow_result(cls) -> None:
    """Every per-workflow result dataclass inherits from the
    shared :class:`WorkflowResult` base. Lets callers branch on
    ``isinstance(res, WorkflowResult)`` and read the common
    ``exit_code`` / ``messages`` / ``errors`` fields regardless
    of which workflow ran."""
    assert issubclass(cls, WorkflowResult)


# ---------------------------------------------------------------------------
# run_validate
# ---------------------------------------------------------------------------

def test_validate_returns_typed_result_on_bundled_reference_deck() -> None:
    """The NREL 5MW land reference deck ships pre-patched
    (validation matrix track B). ``run_validate`` should return
    exit_code 0 with a populated validation report and printable
    messages."""
    if not _NREL5MW_LAND_DAT.is_file():
        pytest.skip(
            f"bundled reference deck not present at {_NREL5MW_LAND_DAT}; "
            "run `python scripts/build_reference_decks.py` first."
        )
    res = run_validate(_NREL5MW_LAND_DAT)
    assert isinstance(res, ValidateResult)
    assert res.exit_code == 0           # PASS or WARN both map to 0
    assert res.validation is not None
    assert res.messages                 # non-empty printable report
    # Heading line + at least one block row + an Overall verdict.
    joined = "\n".join(res.messages)
    assert "pyBmodes coefficient validator" in joined
    assert "Overall:" in joined


def test_validate_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    """A missing file is a usage / IO error — workflow raises
    :class:`FileNotFoundError` rather than returning a result
    with exit_code 2. The CLI wrapper catches and translates."""
    with pytest.raises(FileNotFoundError):
        run_validate(tmp_path / "does_not_exist.dat")


def test_validate_accepts_string_or_path(tmp_path: pathlib.Path) -> None:
    """Both string and pathlib.Path inputs work — the workflow
    resolves to an absolute Path internally."""
    if not _NREL5MW_LAND_DAT.is_file():
        pytest.skip("bundled reference deck not present")
    res_path = run_validate(_NREL5MW_LAND_DAT)
    res_str = run_validate(str(_NREL5MW_LAND_DAT))
    assert res_path.exit_code == res_str.exit_code
    assert res_path.validation.overall == res_str.validation.overall


# ---------------------------------------------------------------------------
# run_examples_copy
# ---------------------------------------------------------------------------

def test_examples_copy_default_kind_all(tmp_path: pathlib.Path) -> None:
    """``kind="all"`` copies both bundles."""
    dest = tmp_path / "out"
    res = run_examples_copy(dest)
    assert isinstance(res, ExamplesResult)
    assert res.exit_code == 0
    assert res.dest == dest.resolve()
    # Two bundles successfully copied (sample_inputs + reference_decks).
    assert len(res.copied) == 2
    assert (dest / "sample_inputs").is_dir()
    assert (dest / "reference_decks").is_dir()
    # No skipped bundles when both are present in the installed package.
    assert res.skipped == []
    # Per-bundle messages.
    joined = "\n".join(res.messages)
    assert "copied samples:" in joined
    assert "copied decks:" in joined


def test_examples_copy_kind_samples_only(tmp_path: pathlib.Path) -> None:
    """``kind="samples"`` copies only ``sample_inputs/``."""
    dest = tmp_path / "out"
    res = run_examples_copy(dest, kind="samples")
    assert res.exit_code == 0
    assert len(res.copied) == 1
    assert (dest / "sample_inputs").is_dir()
    assert not (dest / "reference_decks").exists()


def test_examples_copy_kind_decks_only(tmp_path: pathlib.Path) -> None:
    """``kind="decks"`` copies only ``reference_decks/``."""
    dest = tmp_path / "out"
    res = run_examples_copy(dest, kind="decks")
    assert res.exit_code == 0
    assert len(res.copied) == 1
    assert (dest / "reference_decks").is_dir()
    assert not (dest / "sample_inputs").exists()


def test_examples_copy_errors_on_existing_target_without_force(
    tmp_path: pathlib.Path,
) -> None:
    """Existing destination directory is a usage error unless
    ``force=True``. Returns exit_code 2 with an explanatory
    ``errors`` entry — does not raise (matches the original CLI
    semantics)."""
    dest = tmp_path / "out"
    # First copy: succeeds.
    run_examples_copy(dest, kind="samples")
    # Second copy without force: errors.
    res = run_examples_copy(dest, kind="samples")
    assert res.exit_code == 2
    assert any(
        "destination already exists" in line for line in res.errors
    )


def test_examples_copy_force_overwrites(tmp_path: pathlib.Path) -> None:
    """``force=True`` overwrites an existing destination."""
    dest = tmp_path / "out"
    run_examples_copy(dest, kind="samples")
    # Drop a marker file inside the bundle so we can verify the
    # rewrite cleared the prior tree.
    marker = dest / "sample_inputs" / ".marker"
    marker.write_text("pre-force", encoding="utf-8")
    assert marker.is_file()

    res = run_examples_copy(dest, kind="samples", force=True)
    assert res.exit_code == 0
    # The whole sub-tree was rewritten — marker is gone.
    assert not marker.is_file()
