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
    BatchResult,
    ExamplesResult,
    PatchResult,
    ReportResult,
    ValidateResult,
    WorkflowResult,
    run_batch,
    run_examples_copy,
    run_patch,
    run_report,
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

@pytest.mark.parametrize(
    "cls",
    [ValidateResult, ExamplesResult, PatchResult, ReportResult, BatchResult],
)
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


def test_examples_copy_preserves_skipped_warning_on_destination_conflict(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When one requested bundle is missing on disk AND another
    bundle hits an existing destination without ``force=True``, the
    result must surface BOTH diagnostics — the skipped-bundle
    warning and the destination-conflict error. (Regression for the
    static-review P3 finding on PR #69: the earlier implementation
    rebuilt ``errors`` in the early-return path and dropped the
    accumulated skipped-bundle warning.)"""
    from pybmodes.workflows import examples as examples_mod

    # Build a fake examples root that ships only ``sample_inputs/``
    # (no ``reference_decks/``) so ``kind="all"`` selects samples and
    # skips decks.
    fake_root = tmp_path / "_examples"
    samples_src = fake_root / "sample_inputs"
    samples_src.mkdir(parents=True)
    (samples_src / "marker.txt").write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        examples_mod, "_resolve_examples_root", lambda: fake_root
    )

    dest = tmp_path / "out"
    # First copy: samples succeeds, decks skipped (warning only,
    # exit_code 0 because at least one bundle copied).
    first = run_examples_copy(dest, kind="all")
    assert first.exit_code == 0
    assert first.skipped == ["decks"]
    assert any(
        "skipping bundle(s) not found" in line for line in first.errors
    )

    # Second copy without force: samples hits destination conflict
    # AND decks is still skipped. Both diagnostics must appear.
    res = run_examples_copy(dest, kind="all")
    assert res.exit_code == 2
    assert res.skipped == ["decks"]
    # Skipped-bundle warning preserved.
    assert any(
        "skipping bundle(s) not found" in line for line in res.errors
    )
    # Destination-conflict error also present.
    assert any(
        "destination already exists" in line for line in res.errors
    )


# ---------------------------------------------------------------------------
# run_patch
# ---------------------------------------------------------------------------

def _stage_deck_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy the bundled NREL 5MW land reference deck into ``tmp_path``
    so a workflow that mutates the deck files in place does not touch
    the repo's checked-in copy. Returns the main-``.dat`` path inside
    the staged tree, or skips if the source bundle is absent."""
    src_dir = _NREL5MW_LAND_DAT.parent
    if not _NREL5MW_LAND_DAT.is_file():
        pytest.skip(f"bundled reference deck not present at {src_dir}")
    dest_dir = tmp_path / src_dir.name
    import shutil as _shutil
    _shutil.copytree(src_dir, dest_dir)
    return dest_dir / _NREL5MW_LAND_DAT.name


def test_patch_dry_run_writes_nothing(tmp_path: pathlib.Path) -> None:
    """``dry_run=True`` produces a typed result with the patched text
    populated, but doesn't touch the source files."""
    main_dat = _stage_deck_tree(tmp_path)
    before = main_dat.read_bytes()

    res = run_patch(main_dat, dry_run=True)
    assert isinstance(res, PatchResult)
    assert res.exit_code == 0
    assert res.wrote == []
    assert res.tower_patched_text is not None
    assert res.blade_patched_text is not None
    # Source main .dat untouched (patch only mutates tower/blade
    # side-decks, but the same invariant should hold for the whole
    # tree on a no-op mode).
    assert main_dat.read_bytes() == before
    # Tower / blade .dat resolved.
    assert res.tower_dat is not None and res.tower_dat.is_file()
    assert res.blade_dat is not None and res.blade_dat.is_file()


def test_patch_diff_emits_pr_ready_diff(tmp_path: pathlib.Path) -> None:
    """``diff=True`` implies dry-run AND emits a coefficient-only diff
    in messages with per-block RMS-improvement annotations."""
    main_dat = _stage_deck_tree(tmp_path)
    res = run_patch(main_dat, diff=True)
    assert res.exit_code == 0
    assert res.wrote == []
    assert res.validation is not None
    joined = "\n".join(res.messages)
    assert "--- original" in joined
    assert "+++ patched" in joined
    assert "RMS improvement" in joined


def test_patch_output_dir_writes_copies(tmp_path: pathlib.Path) -> None:
    """``output_dir=DIR`` writes patched copies into DIR and leaves
    the source files untouched."""
    main_dat = _stage_deck_tree(tmp_path)
    # First, learn the tower/blade side-deck paths via a dry-run so
    # the assertion below doesn't have to know the deck's filename
    # convention.
    probe = run_patch(main_dat, dry_run=True)
    assert probe.tower_dat is not None
    tower_before = probe.tower_dat.read_bytes()
    out_dir = tmp_path / "patched_out"

    res = run_patch(main_dat, output_dir=out_dir)
    assert res.exit_code == 0
    assert len(res.wrote) == 2
    for p in res.wrote:
        assert p.parent == out_dir.resolve()
        assert p.is_file()
    # Source tower .dat untouched by the output-dir write.
    assert probe.tower_dat.read_bytes() == tower_before


def test_patch_rejects_output_dir_with_dry_run(tmp_path: pathlib.Path) -> None:
    """``output_dir`` and ``dry_run`` / ``diff`` are mutually exclusive
    (the dry-run modes write nothing). Raises ``ValueError`` — the CLI
    catches and translates to exit code 2."""
    main_dat = _stage_deck_tree(tmp_path)
    with pytest.raises(ValueError, match="output_dir is incompatible"):
        run_patch(main_dat, output_dir=tmp_path / "out", dry_run=True)


def test_patch_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    """Missing main file → ``FileNotFoundError`` (usage / IO error)."""
    with pytest.raises(FileNotFoundError):
        run_patch(tmp_path / "does_not_exist.dat", dry_run=True)


# ---------------------------------------------------------------------------
# run_report
# ---------------------------------------------------------------------------

def test_report_writes_markdown(tmp_path: pathlib.Path) -> None:
    """``run_report`` writes a Markdown report to the requested path
    and returns a populated typed result."""
    if not _NREL5MW_LAND_DAT.is_file():
        pytest.skip("bundled reference deck not present")
    out_path = tmp_path / "report.md"
    res = run_report(_NREL5MW_LAND_DAT, out_path, format="md")
    assert isinstance(res, ReportResult)
    assert res.exit_code == 0
    assert res.out_path == out_path.resolve()
    assert out_path.is_file()
    # Validation ran (validate=True is the default).
    assert res.validation is not None
    # Modal results populated for both sides.
    assert res.tower_modal is not None
    assert res.blade_modal is not None
    # Markdown content sanity.
    md = out_path.read_text(encoding="utf-8")
    assert "# " in md  # at least one heading


def test_report_no_validate_skips_validation(tmp_path: pathlib.Path) -> None:
    """``validate=False`` skips the validator and leaves
    ``result.validation`` as ``None``."""
    if not _NREL5MW_LAND_DAT.is_file():
        pytest.skip("bundled reference deck not present")
    out_path = tmp_path / "report.md"
    res = run_report(_NREL5MW_LAND_DAT, out_path, validate=False)
    assert res.exit_code == 0
    assert res.validation is None


def test_report_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_report(tmp_path / "missing.dat", tmp_path / "report.md")


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

def test_batch_discovers_reference_decks(tmp_path: pathlib.Path) -> None:
    """``run_batch`` over ``reference_decks/`` finds the six bundled
    main decks, writes the summary CSV, and returns exit_code 0."""
    if not _REFERENCE_DECKS.is_dir():
        pytest.skip("reference_decks/ not present")
    res = run_batch(_REFERENCE_DECKS, tmp_path)
    assert isinstance(res, BatchResult)
    assert res.exit_code == 0
    assert res.decks_found == 6
    assert res.decks_failed == 0
    assert res.summary_path is not None and res.summary_path.is_file()
    assert len(res.summary_rows) == 6
    # Every row carries the expected column set.
    for row in res.summary_rows:
        assert set(row.keys()) >= {
            "filename", "overall_verdict",
            "TwFAM2Sh_ratio", "TwSSM2Sh_ratio",
            "n_fail", "n_warn",
        }


def test_batch_rejects_unknown_kind(tmp_path: pathlib.Path) -> None:
    """Unsupported ``kind`` raises ``ValueError`` (usage error → CLI
    translates to exit code 2)."""
    with pytest.raises(ValueError, match="not supported"):
        run_batch(tmp_path, tmp_path / "out", kind="bmi")  # type: ignore[arg-type]


def test_batch_raises_on_missing_root(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_batch(tmp_path / "does_not_exist", tmp_path / "out")


def test_batch_empty_tree_writes_empty_summary(tmp_path: pathlib.Path) -> None:
    """A directory with no ElastoDyn main decks produces a summary
    CSV with just the header row and exit_code 0."""
    root = tmp_path / "empty"
    root.mkdir()
    out = tmp_path / "out"
    res = run_batch(root, out)
    assert res.exit_code == 0
    assert res.decks_found == 0
    assert res.summary_rows == []
    assert res.summary_path is not None
    # Header-only CSV.
    text = res.summary_path.read_text(encoding="utf-8")
    assert "filename" in text
    assert text.count("\n") == 1
