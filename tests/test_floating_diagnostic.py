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
from pybmodes.fem.normalize import NodeModeShape

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


def _rigid_body_shape(mode_number: int, freq_hz: float, axis: str) -> NodeModeShape:
    """Build a synthetic rigid-body mode shape (uniform translation along ``axis``)."""
    span = np.linspace(0.0, 1.0, 11)
    zeros = np.zeros_like(span)
    ones = np.ones_like(span)
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=ones if axis == "flap" else zeros,
        flap_slope=zeros,
        lag_disp=ones if axis == "lag" else zeros,
        lag_slope=zeros,
        twist=zeros,
    )


def _tower_bending_shape(mode_number: int, freq_hz: float, axis: str) -> NodeModeShape:
    """Build a synthetic 1st cantilever bending shape on ``axis``.

    Uses phi(s) = s**2 (3 - s) / 2 (a smooth cantilever-class mode with
    phi(0) = phi'(0) = 0 and phi(1) = 1), so the affine root-tangent
    subtraction is a no-op and the elastic tip ratio is exactly 1.
    """
    span = np.linspace(0.0, 1.0, 11)
    phi = span**2 * (3.0 - span) / 2.0
    phi_slope = span * (3.0 - 1.5 * span)
    zeros = np.zeros_like(span)
    return NodeModeShape(
        mode_number=mode_number,
        freq_hz=freq_hz,
        span_loc=span,
        flap_disp=phi if axis == "flap" else zeros,
        flap_slope=phi_slope if axis == "flap" else zeros,
        lag_disp=phi if axis == "lag" else zeros,
        lag_slope=phi_slope if axis == "lag" else zeros,
        twist=zeros,
    )


def test_drop_rigid_body_drops_unlabelled_low_frequency_platform_mode() -> None:
    """Regression for the first Codex P2 finding on PR #114.

    ``classify_platform_modes`` can leave a strongly-coupled or
    rotated rigid-body pair tagged ``None``. A pure label-based filter
    forwarded those candidates into the tower-family classifier and
    the diagnostic could have landed on a low-frequency platform mode
    as the coupled 1st FA / SS. Asserts the shape-content filter drops
    them via the elastic-tip ratio cut.
    """
    from pybmodes.models.result import ModalResult

    shapes = [
        _rigid_body_shape(1, 0.008, "flap"),
        _rigid_body_shape(2, 0.008, "lag"),
        _rigid_body_shape(3, 0.032, "flap"),
        _rigid_body_shape(4, 0.039, "lag"),
        _rigid_body_shape(5, 0.039, "flap"),
        _rigid_body_shape(6, 0.120, "lag"),
        _tower_bending_shape(7, 0.482, "flap"),
        _tower_bending_shape(8, 0.491, "lag"),
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    mode_labels = ["surge", "sway", "yaw", "pitch", None, None, None, None]
    modal = ModalResult(
        frequencies=frequencies, shapes=shapes, mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert len(filtered.shapes) == 2
    assert [s.mode_number for s in filtered.shapes] == [7, 8]


def test_drop_rigid_body_drops_unlabelled_high_frequency_platform_mode() -> None:
    """Regression for the second Codex P2 finding on PR #114.

    On a stiff floating support such as a TLP a rigid-body mode can
    sit above the spar-class 0.2 Hz band. A frequency-floor filter
    would let that unlabelled high-frequency platform mode through.
    The shape-content filter drops it because the elastic tip ratio
    is still negligible regardless of frequency.
    """
    from pybmodes.models.result import ModalResult

    shapes = [
        _rigid_body_shape(1, 0.5, "flap"),     # the offending unlabelled mode
        _tower_bending_shape(2, 0.6, "flap"),
        _tower_bending_shape(3, 0.62, "lag"),
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    mode_labels = [None, None, None]
    modal = ModalResult(
        frequencies=frequencies, shapes=shapes, mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert [s.mode_number for s in filtered.shapes] == [2, 3]


def test_drop_rigid_body_keeps_unlabelled_tower_modes() -> None:
    """Mirrors the actual OC3 Hywind shape.

    The asymmetric platform-support eigensolve leaves the 1st tower
    FA / SS at indices 4 and 5 with ``label = None``. The filter must
    keep them via the elastic-tip ratio gate.
    """
    from pybmodes.models.result import ModalResult

    shapes = [
        _rigid_body_shape(1, 0.0081, "flap"),
        _rigid_body_shape(2, 0.0081, "lag"),
        _rigid_body_shape(3, 0.0324, "flap"),
        _rigid_body_shape(4, 0.0412, "lag"),
        _tower_bending_shape(5, 0.4887, "lag"),
        _tower_bending_shape(6, 0.4903, "flap"),
        _tower_bending_shape(7, 2.9639, "flap"),
        _tower_bending_shape(8, 3.4724, "lag"),
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    mode_labels = ["surge", "sway", "heave", "yaw", None, None, None, None]
    modal = ModalResult(
        frequencies=frequencies, shapes=shapes, mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert [s.mode_number for s in filtered.shapes] == [5, 6, 7, 8]


def test_drop_rigid_body_drops_labelled_rigid_modes_at_tower_band_frequency() -> None:
    """Defensive guard: a labelled rigid-body mode at tower-band frequency drops.

    Handles the stiff-platform corner case where ``classify_platform_modes``
    successfully tags one of the rigid-body modes but its frequency
    happens to overlap the 1st tower bending band. The label-based
    half of the filter excludes it regardless of frequency.
    """
    from pybmodes.models.result import ModalResult

    shapes = [
        _rigid_body_shape(1, 0.40, "flap"),     # labelled heave above 0.2 Hz
        _tower_bending_shape(2, 0.42, "flap"),
        _tower_bending_shape(3, 0.48, "lag"),
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    mode_labels = ["heave", None, None]
    modal = ModalResult(
        frequencies=frequencies, shapes=shapes, mode_labels=mode_labels,
    )

    filtered = _drop_rigid_body_shapes(modal)

    assert [s.mode_number for s in filtered.shapes] == [2, 3]


def test_drop_rigid_body_passes_through_cantilever_solve() -> None:
    """A cantilever modal result has ``mode_labels = None`` and is unchanged."""
    from pybmodes.models.result import ModalResult

    shapes = [_tower_bending_shape(i + 1, 0.4 + i, "flap") for i in range(4)]
    frequencies = np.array([s.freq_hz for s in shapes])
    modal = ModalResult(
        frequencies=frequencies, shapes=shapes, mode_labels=None,
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
