"""Tests for ``pybmodes batch``.

Three spec-named tests:

* ``test_batch_finds_dat_files`` — discovery filter picks main decks
  but rejects ``_Tower`` / ``_Blade`` / ``_SubDyn`` siblings.
* ``test_batch_summary_csv_columns`` — summary CSV has the exact
  column set the spec requires.
* ``test_batch_patch_all_pass_after`` — integration: running
  ``pybmodes batch --patch`` over the upstream r-test 5MW deck
  flips every block to PASS / WARN.

Supporting tests check the ``--kind elastodyn``-only enforcement and
the exit-code-1 behaviour when at least one deck remains at FAIL.
"""

from __future__ import annotations

import csv
import pathlib
import shutil

import pytest

from pybmodes.cli import _find_elastodyn_main_dats, _resolve_examples_root
from pybmodes.cli import main as cli_main

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_DECKS = _resolve_examples_root() / "reference_decks"
NREL5MW_RTEST_DECK = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_Land_DLL_WTurb"
    / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
)


# ---------------------------------------------------------------------------
# Discovery filter
# ---------------------------------------------------------------------------

def test_batch_finds_dat_files() -> None:
    """``_find_elastodyn_main_dats`` returns the six main decks under
    ``reference_decks/`` and rejects every ``_Tower.dat`` / ``_Blade.dat``
    sibling."""
    if not REFERENCE_DECKS.is_dir():
        pytest.skip("reference_decks/ not present")
    found = _find_elastodyn_main_dats(REFERENCE_DECKS)
    names = [p.name for p in found]

    # Every entry must look like a main deck (no aux tokens) and parse.
    assert len(found) == 6, (
        f"expected 6 main decks under reference_decks/, found {len(found)}: "
        f"{names}"
    )
    forbidden_tokens = ("_Tower", "_Blade", "_SubDyn", "_HydroDyn", "_MoorDyn")
    for name in names:
        for tok in forbidden_tokens:
            assert tok not in name, (
                f"discovery returned {name!r} which contains aux-file "
                f"token {tok!r}; the filter should have excluded it"
            )
    # Spot-check that every reference-deck directory contributed
    # exactly one main file.
    parents = {p.parent.name for p in found}
    assert parents == {
        "nrel5mw_land",
        "nrel5mw_oc3monopile",
        "iea34_land",
        "nrel5mw_oc3spar",
        "nrel5mw_oc4semi",
        "iea15mw_umainesemi",
    }


def test_batch_skips_non_main_dat_files(tmp_path: pathlib.Path) -> None:
    """A tree with only ``_Tower.dat`` / ``_Blade.dat`` files returns
    an empty list — the discovery filter rejects them by name."""
    (tmp_path / "MyTurbine_ElastoDyn_Tower.dat").write_text("not a main")
    (tmp_path / "MyTurbine_ElastoDyn_Blade.dat").write_text("not a main")
    (tmp_path / "MyTurbine_HydroDyn.dat").write_text("hydro")
    found = _find_elastodyn_main_dats(tmp_path)
    assert found == []


# ---------------------------------------------------------------------------
# Summary CSV format
# ---------------------------------------------------------------------------

def test_batch_summary_csv_columns(tmp_path: pathlib.Path) -> None:
    """Running batch on the reference-decks tree writes a summary CSV
    with exactly the six column names the spec calls for, in order."""
    if not REFERENCE_DECKS.is_dir():
        pytest.skip("reference_decks/ not present")
    out_dir = tmp_path / "reports"
    rc = cli_main([
        "batch", str(REFERENCE_DECKS),
        "--validate",
        "--out", str(out_dir),
    ])
    # Reference decks have one known WARN (IEA-15 TwSSM2Sh); the rest
    # PASS. The batch's exit-code logic counts only FAIL/ERROR as bad,
    # so we should get 0.
    assert rc == 0, f"batch over patched reference decks exited {rc}"
    summary_csv = out_dir / "summary.csv"
    assert summary_csv.is_file()
    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    expected_header = [
        "filename",
        "overall_verdict",
        "TwFAM2Sh_ratio",
        "TwSSM2Sh_ratio",
        "n_fail",
        "n_warn",
    ]
    assert header == expected_header
    # Every row has six columns.
    for row in rows:
        assert len(row) == len(expected_header), (
            f"row {row!r} has wrong column count"
        )


def test_batch_rejects_unsupported_kind(
    tmp_path: pathlib.Path, capsys
) -> None:
    """``--kind beamdyn`` (or anything other than ``elastodyn``) exits
    2 with a clear error message. argparse rejects it at parse time
    because ``choices=['elastodyn']``."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["batch", str(tmp_path), "--kind", "beamdyn"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "beamdyn" in err or "choices" in err.lower()


def test_batch_rejects_missing_root(
    tmp_path: pathlib.Path, capsys
) -> None:
    """A non-existent root directory exits 2."""
    bogus = tmp_path / "does_not_exist"
    rc = cli_main(["batch", str(bogus)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


# ---------------------------------------------------------------------------
# Patch flow — integration (needs the upstream r-test deck which ships
# the broken polynomial blocks)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_batch_patch_all_pass_after(tmp_path: pathlib.Path) -> None:
    """Running batch with ``--patch`` over the upstream NREL 5MW r-test
    deck (which ships the wild TwFAM2Sh / TwSSM2Sh coefficients) must
    turn every block to PASS or WARN — no FAIL after patching."""
    if not NREL5MW_RTEST_DECK.is_file():
        pytest.skip(
            f"r-test 5MW deck not present at {NREL5MW_RTEST_DECK}"
        )
    src_dir = NREL5MW_RTEST_DECK.parent
    blade_src = (
        REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
        / "openfast" / "5MW_Baseline"
    )
    # Stage the deck + its blade-baseline sibling so the relative
    # BldFile path inside the main resolves against tmp_path.
    deck_dst = tmp_path / src_dir.name
    shutil.copytree(src_dir, deck_dst)
    shutil.copytree(blade_src, tmp_path / blade_src.name)

    out_dir = tmp_path / "reports"
    rc = cli_main([
        "batch", str(tmp_path),
        "--validate",
        "--patch",
        "--out", str(out_dir),
        "--n-modes", "10",
    ])
    # All blocks PASS or WARN (no FAIL) means exit 0.
    assert rc == 0, f"batch --patch left at least one deck at FAIL/ERROR (rc={rc})"

    # The summary CSV should report no FAIL rows.
    with (out_dir / "summary.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "summary CSV is empty; batch found no decks"
    for row in rows:
        assert row["overall_verdict"] in ("PASS", "WARN"), (
            f"deck {row['filename']!r} remained at "
            f"{row['overall_verdict']!r} after --patch"
        )
        assert int(row["n_fail"]) == 0, (
            f"deck {row['filename']!r} has {row['n_fail']} FAIL block(s) "
            "after --patch"
        )
