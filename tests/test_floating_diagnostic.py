"""Tests for ``pybmodes.elastodyn.report_floating_frequency_gap``.

The dataclass behaviour and the ``format_report`` text are exercised
purely with synthetic frequencies so they run in the default
no-external-data pytest pass. The end-to-end function path that runs
two FEM solves on a real floating deck is gated by the
``integration`` marker because it needs the upstream OpenFAST r-test
clone under ``external/``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.elastodyn import (
    FloatingFrequencyGap,
    report_floating_frequency_gap,
)
from pybmodes.elastodyn.diagnostics import _drop_rigid_body_shapes

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

OC3_DECK_DIR = (
    REPO_ROOT / "external" / "OpenFAST_files" / "r-test" / "glue-codes"
    / "openfast" / "5MW_OC3Spar_DLL_WTurb_WavesIrr"
)
OC3_ELASTODYN = OC3_DECK_DIR / "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat"
OC3_MOORDYN = OC3_DECK_DIR / "NRELOffshrBsline5MW_OC3Hywind_MoorDyn.dat"
OC3_HYDRODYN = OC3_DECK_DIR / "NRELOffshrBsline5MW_OC3Hywind_HydroDyn.dat"


def test_gap_signs_for_stiffening_platform() -> None:
    """Coupled frequency above cantilever should yield a positive gap."""
    gap = FloatingFrequencyGap(
        cantilever_fa_1=0.385,
        cantilever_ss_1=0.385,
        coupled_fa_1=0.493,
        coupled_ss_1=0.493,
    )
    assert gap.gap_fa_1_pct == pytest.approx(28.05, rel=1e-3)
    assert gap.gap_ss_1_pct == pytest.approx(28.05, rel=1e-3)


def test_gap_signs_for_softening_platform() -> None:
    """Coupled frequency below cantilever should yield a negative gap."""
    gap = FloatingFrequencyGap(
        cantilever_fa_1=0.5,
        cantilever_ss_1=0.5,
        coupled_fa_1=0.4,
        coupled_ss_1=0.45,
    )
    assert gap.gap_fa_1_pct == pytest.approx(-20.0, abs=1e-9)
    assert gap.gap_ss_1_pct == pytest.approx(-10.0, abs=1e-9)


def test_format_report_contains_expected_labels() -> None:
    """The formatted report names both FA and SS, with units and the gap row."""
    gap = FloatingFrequencyGap(
        cantilever_fa_1=0.385,
        cantilever_ss_1=0.390,
        coupled_fa_1=0.493,
        coupled_ss_1=0.495,
    )
    report = gap.format_report()
    assert "Cantilever 1st FA: 0.385 Hz" in report
    assert "Coupled 1st FA:    0.493 Hz" in report
    assert "Cantilever 1st SS: 0.390 Hz" in report
    assert "Coupled 1st SS:    0.495 Hz" in report
    assert "Gap:" in report
    assert "%" in report


def test_format_report_signed_gap_prefix() -> None:
    """A positive gap should carry an explicit ``+`` sign."""
    gap_up = FloatingFrequencyGap(
        cantilever_fa_1=0.4,
        cantilever_ss_1=0.4,
        coupled_fa_1=0.5,
        coupled_ss_1=0.5,
    )
    assert "+25.0%" in gap_up.format_report()

    gap_down = FloatingFrequencyGap(
        cantilever_fa_1=0.5,
        cantilever_ss_1=0.5,
        coupled_fa_1=0.4,
        coupled_ss_1=0.4,
    )
    assert "-20.0%" in gap_down.format_report()


def test_drop_rigid_body_drops_unlabelled_platform_modes() -> None:
    """Regression for the Codex P2 finding on PR #114.

    ``classify_platform_modes`` can leave a strongly-coupled or
    rotated rigid-body pair tagged ``None``. The label-based filter
    forwarded those candidates into the tower-family classifier and
    the diagnostic could have landed on a low-frequency platform mode
    as the coupled 1st FA / SS.

    Synthesises a ten-mode ``ModalResult`` where two of the six
    rigid-body slots carry ``None`` (the bug condition Codex flagged),
    then asserts the filter still drops them via the frequency-floor
    cut at 0.2 Hz.
    """
    from pybmodes.models.result import ModalResult

    frequencies = np.array([
        0.008,
        0.008,
        0.032,
        0.039,
        0.039,
        0.120,
        0.482,
        0.491,
        1.570,
        1.598,
    ])
    shapes = [f"shape_{i}" for i in range(10)]
    mode_labels = [
        "surge",
        "sway",
        "yaw",
        "pitch",
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    modal = ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert len(filtered.shapes) == 4
    assert list(filtered.frequencies) == [0.482, 0.491, 1.570, 1.598]
    assert filtered.shapes == ["shape_6", "shape_7", "shape_8", "shape_9"]


def test_drop_rigid_body_keeps_unlabelled_mode_above_floor() -> None:
    """Unlabelled tower modes above the 0.2 Hz floor are preserved.

    Mirrors the actual OC3 Hywind shape (asymmetric platform support
    leaves the eigensolver returning four labelled rigid-body modes
    rather than six; the 1st tower FA / SS at 0.49 Hz must not be
    dropped just because their label is ``None``).
    """
    from pybmodes.models.result import ModalResult

    frequencies = np.array([
        0.0081,
        0.0081,
        0.0324,
        0.0412,
        0.4887,
        0.4903,
        2.9639,
        3.4724,
    ])
    shapes = [f"shape_{i}" for i in range(8)]
    mode_labels = [
        "surge",
        "sway",
        "heave",
        "yaw",
        None,
        None,
        None,
        None,
    ]
    modal = ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert filtered.shapes == [
        "shape_4", "shape_5", "shape_6", "shape_7",
    ]


def test_drop_rigid_body_drops_labelled_mode_above_floor() -> None:
    """Labelled rigid-body modes are dropped even above the 0.2 Hz floor.

    Defensive guard for a stiff platform (such as a TLP) where one of
    the six rigid-body frequencies could overlap with the 1st tower
    bending pair. As long as ``classify_platform_modes`` tags the
    rigid-body mode with a label, the filter excludes it regardless
    of frequency.
    """
    from pybmodes.models.result import ModalResult

    frequencies = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.40, 0.42, 0.48])
    shapes = [f"shape_{i}" for i in range(8)]
    mode_labels = [
        "surge",
        "sway",
        "yaw",
        "roll",
        "pitch",
        "heave",   # the labelled rigid-body mode that sits in the tower-bending band
        None,
        None,
    ]
    modal = ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert filtered.shapes == ["shape_6", "shape_7"]


def test_drop_rigid_body_passes_through_cantilever_solve() -> None:
    """A cantilever modal result has ``mode_labels = None`` and is unchanged."""
    from pybmodes.models.result import ModalResult

    frequencies = np.array([0.4, 0.4, 1.5, 1.5])
    shapes = [f"shape_{i}" for i in range(4)]
    modal = ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        mode_labels=None,
    )
    assert _drop_rigid_body_shapes(modal) is modal


@pytest.mark.integration
@pytest.mark.skipif(
    not OC3_ELASTODYN.is_file() or not OC3_MOORDYN.is_file(),
    reason=(
        "OpenFAST r-test 5MW_OC3Spar deck not present under "
        f"{OC3_DECK_DIR}; clone the upstream r-test repo to run this test."
    ),
)
def test_report_floating_frequency_gap_on_oc3_spar() -> None:
    """End-to-end run on the OC3 Hywind spar deck.

    Asserts the gap is non-trivial (>= 5 percent on at least one axis)
    so the diagnostic actually surfaces what it is meant to surface,
    and that both cantilever and coupled frequencies land in a
    physically plausible band for a 5 MW floating tower bending mode.
    """
    hydrodyn = OC3_HYDRODYN if OC3_HYDRODYN.is_file() else None
    gap = report_floating_frequency_gap(
        OC3_ELASTODYN,
        OC3_MOORDYN,
        hydrodyn,
    )
    for f in (
        gap.cantilever_fa_1,
        gap.cantilever_ss_1,
        gap.coupled_fa_1,
        gap.coupled_ss_1,
    ):
        assert 0.1 < f < 1.5

    nontrivial = max(abs(gap.gap_fa_1_pct), abs(gap.gap_ss_1_pct))
    assert nontrivial >= 5.0
