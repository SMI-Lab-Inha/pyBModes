"""Small regression tests for Campbell helper functions.

These stay independent of the bundled reference decks, so they run fast
and keep low-level labelling / assignment contracts pinned down.
"""

from __future__ import annotations

import numpy as np

from pybmodes.campbell import (
    _greedy_assignment,
    _hungarian_assignment,
    _label_blade_modes,
    _label_tower_modes,
    _label_tower_modes_with_overrides,
    _ordinal,
    _participation,
)
from pybmodes.fem.normalize import NodeModeShape


def _shape(flap, lag, twist) -> NodeModeShape:
    flap = np.asarray(flap, dtype=float)
    lag = np.asarray(lag, dtype=float)
    twist = np.asarray(twist, dtype=float)
    n = flap.size
    return NodeModeShape(
        mode_number=1,
        freq_hz=1.0,
        span_loc=np.linspace(0.0, 1.0, n),
        flap_disp=flap,
        flap_slope=np.zeros(n),
        lag_disp=lag,
        lag_slope=np.zeros(n),
        twist=twist,
    )


def test_ordinal_handles_teens_as_th() -> None:
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21)] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
    ]


def test_label_blade_modes_counts_each_axis_independently() -> None:
    participation = np.array([
        [0.9, 0.1, 0.0],
        [0.2, 0.7, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.8, 0.1],
    ])
    assert _label_blade_modes(participation) == [
        "1st flap",
        "1st edge",
        "2nd flap",
        "1st torsion",
        "2nd edge",
    ]


def test_label_tower_modes_uses_fa_ss_torsion_names() -> None:
    participation = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    assert _label_tower_modes(participation) == [
        "1st tower FA",
        "1st tower SS",
        "2nd tower SS",
        "1st tower torsion",
    ]


def test_tower_overrides_none_matches_plain_labeller() -> None:
    """mode_labels=None (every cantilever / monopile tower) reproduces
    _label_tower_modes exactly — the pre-#47 contract is unchanged."""
    participation = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    assert _label_tower_modes_with_overrides(
        participation, None
    ) == _label_tower_modes(participation)


def test_tower_overrides_use_classified_platform_dofs() -> None:
    """For a floating tower the FEM-classified platform DOFs are used
    verbatim and the flexible bending modes get a *bending-only*
    ordinal (issue #47): the first real bending mode is '1st tower FA'
    even though six rigid modes precede it."""
    # 6 rigid (named by classify_platform_modes) + 2 flexible bending.
    participation = np.array([
        [0.5, 0.3, 0.2],   # surge   (override wins; participation moot)
        [0.3, 0.5, 0.2],   # sway
        [0.4, 0.4, 0.2],   # heave
        [0.4, 0.3, 0.3],   # roll
        [0.3, 0.4, 0.3],   # pitch
        [0.2, 0.3, 0.5],   # yaw
        [0.8, 0.1, 0.1],   # 1st tower FA  (flexible)
        [0.1, 0.8, 0.1],   # 1st tower SS  (flexible)
    ])
    mode_labels = ["surge", "sway", "heave", "roll", "pitch", "yaw",
                   None, None]
    assert _label_tower_modes_with_overrides(participation, mode_labels) == [
        "surge", "sway", "heave", "roll", "pitch", "yaw",
        "1st tower FA", "1st tower SS",
    ]


def test_tower_overrides_partial_none_falls_back() -> None:
    """A None entry (classifier stayed conservative on a rotated pair)
    falls back to the participation label, counted over flexible modes
    only."""
    participation = np.array([
        [0.5, 0.3, 0.2],   # surge
        [0.9, 0.05, 0.05],  # None -> participation -> 1st tower FA
        [0.05, 0.9, 0.05],  # None -> participation -> 1st tower SS
    ])
    assert _label_tower_modes_with_overrides(
        participation, ["surge", None, None]
    ) == ["surge", "1st tower FA", "1st tower SS"]


def test_participation_returns_axis_energy_fractions() -> None:
    shape = _shape([3.0, 4.0], [0.0, 5.0], [0.0, 0.0])
    np.testing.assert_allclose(_participation(shape), [0.5, 0.5, 0.0])


def test_participation_zero_shape_returns_zero_vector() -> None:
    shape = _shape([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    np.testing.assert_array_equal(_participation(shape), np.zeros(3))


def test_hungarian_assignment_beats_greedy_local_choice() -> None:
    mac = np.array([
        [0.95, 0.90],
        [0.94, 0.10],
    ])
    # A local first-row argmax would choose [0, 1] for a total of 1.05;
    # the global optimum chooses [1, 0] for a total of 1.84.
    np.testing.assert_array_equal(_hungarian_assignment(mac), [1, 0])
    np.testing.assert_array_equal(_greedy_assignment(mac), [1, 0])


def test_hungarian_assignment_marks_unmatched_rows_for_rectangular_input() -> None:
    mac = np.array([
        [0.1, 0.8],
        [0.9, 0.2],
        [0.3, 0.4],
    ])
    order = _hungarian_assignment(mac)
    assert order.shape == (3,)
    assert sorted(order[order >= 0].tolist()) == [0, 1]
    assert np.count_nonzero(order < 0) == 1
