"""Tests for ``pybmodes examples --copy <dir>``.

The subcommand vendors ``pybmodes._examples.sample_inputs`` and
``pybmodes._examples.reference_decks`` from the package install into a
user-supplied directory. Tests exercise the three ``--kind`` modes
(``all`` / ``samples`` / ``decks``), the destination-conflict guard,
and the ``--force`` override.

All tests skip if both source bundles are missing from the installed
package (which only happens on a malformed install where setuptools
didn't pick up the ``_examples`` package-data tree).
"""

from __future__ import annotations

import pathlib

import pytest

from pybmodes.cli import _resolve_examples_root
from pybmodes.cli import main as cli_main

EXAMPLES_ROOT = _resolve_examples_root()
SAMPLES_SRC = EXAMPLES_ROOT / "sample_inputs"
DECKS_SRC = EXAMPLES_ROOT / "reference_decks"


def _skip_if_no_bundles() -> None:
    if not SAMPLES_SRC.is_dir() and not DECKS_SRC.is_dir():
        pytest.skip(
            "neither sample_inputs/ nor reference_decks/ is "
            "present alongside the installed package"
        )


def test_examples_copy_samples_only(tmp_path: pathlib.Path) -> None:
    """``--kind samples`` copies sample_inputs/ into the destination
    directory; reference_decks/ is left untouched."""
    if not SAMPLES_SRC.is_dir():
        pytest.skip("sample_inputs/ not present")
    dest = tmp_path / "out"
    rc = cli_main(["examples", "--copy", str(dest), "--kind", "samples"])
    assert rc == 0
    assert (dest / "sample_inputs").is_dir()
    # At least the four analytical-reference subdirs should be present
    for name in (
        "01_uniform_blade",
        "02_tower_topmass",
        "03_rotating_uniform_blade",
        "04_pinned_free_cable",
    ):
        assert (dest / "sample_inputs" / name).is_dir()
    # The reference_turbines/ subdirectory should be copied too
    assert (dest / "sample_inputs" / "reference_turbines").is_dir()
    # And reference_decks/ should NOT have been vendored
    assert not (dest / "reference_decks").exists()


def test_examples_copy_decks_only(tmp_path: pathlib.Path) -> None:
    """``--kind decks`` copies reference_decks/ but leaves
    sample_inputs/ untouched."""
    if not DECKS_SRC.is_dir():
        pytest.skip("reference_decks/ not present")
    dest = tmp_path / "out"
    rc = cli_main(["examples", "--copy", str(dest), "--kind", "decks"])
    assert rc == 0
    assert (dest / "reference_decks").is_dir()
    # Spot-check at least one of the six reference decks
    assert (dest / "reference_decks" / "nrel5mw_land").is_dir()
    assert not (dest / "sample_inputs").exists()


def test_examples_copy_all_by_default(tmp_path: pathlib.Path) -> None:
    """Without ``--kind``, the command vendors both bundles."""
    _skip_if_no_bundles()
    dest = tmp_path / "out"
    rc = cli_main(["examples", "--copy", str(dest)])
    assert rc == 0
    if SAMPLES_SRC.is_dir():
        assert (dest / "sample_inputs").is_dir()
    if DECKS_SRC.is_dir():
        assert (dest / "reference_decks").is_dir()


def test_examples_destination_conflict_without_force(
    tmp_path: pathlib.Path,
) -> None:
    """A destination directory that already contains the target
    subdirectory raises exit code 2 unless ``--force`` is set."""
    if not SAMPLES_SRC.is_dir():
        pytest.skip("sample_inputs/ not present")
    dest = tmp_path / "out"
    (dest / "sample_inputs").mkdir(parents=True)
    (dest / "sample_inputs" / "preexisting.txt").write_text("hi")
    rc = cli_main(["examples", "--copy", str(dest), "--kind", "samples"])
    assert rc == 2
    # The preexisting file is still there
    assert (dest / "sample_inputs" / "preexisting.txt").read_text(encoding="utf-8") == "hi"


def test_examples_force_overwrites(tmp_path: pathlib.Path) -> None:
    """``--force`` removes the existing target subdir before copying."""
    if not SAMPLES_SRC.is_dir():
        pytest.skip("sample_inputs/ not present")
    dest = tmp_path / "out"
    (dest / "sample_inputs").mkdir(parents=True)
    (dest / "sample_inputs" / "stale.txt").write_text("stale")
    rc = cli_main([
        "examples", "--copy", str(dest), "--kind", "samples", "--force",
    ])
    assert rc == 0
    # The stale file is gone, replaced by the real bundle
    assert not (dest / "sample_inputs" / "stale.txt").exists()
    assert (dest / "sample_inputs" / "01_uniform_blade").is_dir()


def test_examples_requires_copy_flag(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting ``--copy`` is a parse error (argparse exits 2)."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["examples"])
    assert excinfo.value.code == 2
