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

import pytest

from pybmodes.elastodyn import (
    FloatingFrequencyGap,
    report_floating_frequency_gap,
)

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
