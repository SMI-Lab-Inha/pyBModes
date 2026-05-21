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

"""Modal Assurance Criterion (MAC) utilities for comparing mode shapes.

Two public entry points:

* :func:`mac_matrix` — pairwise MAC between two lists of mode shapes.
* :func:`compare_modes` — full mode-by-mode comparison between two
  :class:`~pybmodes.models.result.ModalResult` records: MAC matrix,
  Hungarian-optimal mode pairing, per-pair frequency shift in
  percent, and the source labels for display.

The MAC formula::

    MAC_ij = |φ_i · φ_j|² / ((φ_i · φ_i) (φ_j · φ_j))

where each ``φ`` is the concatenated FEM displacement vector across
flap / lag / twist axes (3 × n_nodes entries). The metric is
sign- and amplitude-invariant; the only quantity that survives is
the directional alignment of the two shapes.

Typical use cases:

* ``compare_modes(baseline_result, patched_result)`` — confirm the
  polynomial-coefficient patch didn't actually change the underlying
  mode shapes (MAC diagonal stays at ~ 1.0; frequency shifts are
  the only visible delta).
* ``compare_modes(land_result, monopile_result)`` — quantify the
  boundary-condition effect on a tower's mode shapes; the MAC
  diagonal drops as the lower modes pick up rigid-body contributions
  from the soft monopile base.

:func:`plot_mac` is a lightweight matplotlib helper that renders the
MAC matrix as a heatmap with the Hungarian-paired cells highlighted.
matplotlib is imported lazily so the rest of this module works
without the ``[plots]`` extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pybmodes.fem.normalize import NodeModeShape

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pybmodes.models.result import ModalResult


# ---------------------------------------------------------------------------
# MAC core
# ---------------------------------------------------------------------------

def shape_to_vector(shape: NodeModeShape) -> np.ndarray:
    """Flatten a :class:`NodeModeShape` into one (3·n_nodes,) vector.

    The concatenation order is ``[flap_disp, lag_disp, twist]`` —
    matches the convention used internally by the Campbell tracker.
    Slope arrays are not included; the MAC contract operates on
    displacement directions only.
    """
    return np.concatenate([
        np.asarray(shape.flap_disp, dtype=float),
        np.asarray(shape.lag_disp, dtype=float),
        np.asarray(shape.twist, dtype=float),
    ])


def mac_matrix(
    shapes_A: list[NodeModeShape],
    shapes_B: list[NodeModeShape],
) -> np.ndarray:
    """Compute the pairwise MAC matrix between two mode-shape lists.

    Returns an ``(n, m)`` ndarray where ``out[i, j]`` is the MAC
    value between ``shapes_A[i]`` and ``shapes_B[j]``. Values are
    in ``[0, 1]``; 0 means perfectly orthogonal, 1 means perfectly
    aligned (or anti-aligned — MAC squares the inner product).

    Empty inputs are accepted and return a correctly-shaped zero-
    dimensional ndarray. Zero-norm shapes (i.e. all-zero
    displacements) get MAC = 0 against every counterpart, since the
    denominator is undefined and the closest reasonable answer is
    "no correlation".
    """
    n = len(shapes_A)
    m = len(shapes_B)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    V_A = np.stack([shape_to_vector(s) for s in shapes_A])
    V_B = np.stack([shape_to_vector(s) for s in shapes_B])
    if V_A.shape[1] != V_B.shape[1]:
        raise ValueError(
            f"shape vectors must have the same length to compute MAC; "
            f"shapes_A has {V_A.shape[1]} DOFs per mode, shapes_B has "
            f"{V_B.shape[1]}"
        )
    inner = V_A @ V_B.T
    norms_A = np.einsum("ij,ij->i", V_A, V_A)
    norms_B = np.einsum("ij,ij->i", V_B, V_B)
    denom = np.outer(norms_A, norms_B)
    out: np.ndarray = np.zeros_like(inner)
    safe = denom > 0.0
    out[safe] = (inner[safe] ** 2) / denom[safe]
    return out


# ---------------------------------------------------------------------------
# Mode-by-mode comparison
# ---------------------------------------------------------------------------

@dataclass
class ModeComparison:
    """Result of a :func:`compare_modes` call.

    Attributes
    ----------
    mac : (n, m) MAC matrix between ``result_A.shapes`` and
        ``result_B.shapes``.
    frequency_shift : (n_pairs,) percent change in frequency from
        ``result_A`` to ``result_B`` for each Hungarian-paired mode.
        Positive values mean ``result_B`` is *higher* in frequency.
        ``NaN`` for pairs where either side's frequency is
        non-positive (e.g. rigid-body modes).
    paired_modes : list of ``(i, j)`` tuples mapping
        ``result_A.shapes[i]`` → ``result_B.shapes[j]`` under the
        Hungarian-optimal MAC assignment. Length is
        ``min(n, m)``.
    label_A / label_B : free-text labels for the two sources, used
        by plot_mac as axis titles.
    freqs_A / freqs_B : the two raw frequency arrays, copied here so
        the comparison object is self-contained for downstream
        reporting.
    """

    mac: np.ndarray
    frequency_shift: np.ndarray
    paired_modes: list[tuple[int, int]]
    label_A: str = "A"
    label_B: str = "B"
    freqs_A: np.ndarray = field(default_factory=lambda: np.empty(0))
    freqs_B: np.ndarray = field(default_factory=lambda: np.empty(0))


def compare_modes(
    result_A: "ModalResult",
    result_B: "ModalResult",
    *,
    label_A: str = "baseline",
    label_B: str = "modified",
) -> ModeComparison:
    """Compare two :class:`ModalResult` records mode-by-mode.

    Builds the full MAC matrix, finds the Hungarian-optimal pairing
    (each ``result_A`` mode mapped to one ``result_B`` mode), and
    computes the per-pair frequency shift ``(f_B - f_A) / f_A`` in
    percent.

    Pairs are returned in ``result_A`` mode-index order
    (i.e. ``paired_modes[0]`` is ``(0, j₀)`` for whatever ``j₀``
    matched the first ``result_A`` mode).
    """
    from scipy.optimize import linear_sum_assignment

    mac = mac_matrix(result_A.shapes, result_B.shapes)
    if mac.size == 0:
        return ModeComparison(
            mac=mac,
            frequency_shift=np.empty(0),
            paired_modes=[],
            label_A=label_A,
            label_B=label_B,
            freqs_A=np.asarray(result_A.frequencies, dtype=float),
            freqs_B=np.asarray(result_B.frequencies, dtype=float),
        )

    row_ind, col_ind = linear_sum_assignment(mac, maximize=True)
    paired = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

    freqs_A = np.asarray(result_A.frequencies, dtype=float)
    freqs_B = np.asarray(result_B.frequencies, dtype=float)
    shift = np.full(len(paired), np.nan, dtype=float)
    for k, (i, j) in enumerate(paired):
        if 0 <= i < freqs_A.size and 0 <= j < freqs_B.size:
            fa = float(freqs_A[i])
            fb = float(freqs_B[j])
            if fa > 0.0 and np.isfinite(fa) and np.isfinite(fb):
                shift[k] = 100.0 * (fb - fa) / fa

    return ModeComparison(
        mac=mac,
        frequency_shift=shift,
        paired_modes=paired,
        label_A=label_A,
        label_B=label_B,
        freqs_A=freqs_A,
        freqs_B=freqs_B,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_mac(
    comparison: ModeComparison,
    ax: "Axes | None" = None,
    *,
    annotate: bool = True,
    cmap: str = "viridis",
) -> "Figure":
    """Render the MAC matrix as an annotated heatmap.

    Hungarian-paired cells get a red outline; other cells are plain.
    Cell colour ramps from 0 (dark) to 1 (light) via the supplied
    ``cmap``. Set ``annotate=False`` to drop the numerical
    overlay (useful for large matrices where labels collide).

    Returns the parent :class:`Figure` so the caller can save / show.
    Raises :class:`ImportError` if matplotlib isn't installed (the
    optional ``[plots]`` extra is required).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot_mac; install with "
            '`pip install "pybmodes[plots]"`'
        ) from exc

    mac = comparison.mac
    n, m = mac.shape

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.6 * m + 2, 0.6 * n + 2),
        )
    else:
        # ``Axes.figure`` is typed as ``Figure | SubFigure`` upstream
        # (matplotlib >= 3.7); plot_mac's contract returns a real
        # ``Figure`` so callers can ``fig.savefig(...)``. The cast is
        # safe because pyBmodes ax arguments only ever come from
        # plt.subplots / fig.add_subplot, which always produce an
        # ``Axes`` whose ``.figure`` is the top-level ``Figure``.
        from typing import cast as _cast

        from matplotlib.figure import Figure as _FigureCls
        fig = _cast(_FigureCls, ax.figure)

    im = ax.imshow(mac, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(m))
    ax.set_xticklabels([str(k + 1) for k in range(m)])
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(k + 1) for k in range(n)])
    ax.set_xlabel(f"{comparison.label_B} mode index")
    ax.set_ylabel(f"{comparison.label_A} mode index")
    ax.set_title(f"MAC: {comparison.label_A} vs {comparison.label_B}")

    if annotate:
        # Threshold for switching annotation colour from black to white,
        # mirroring the convention used by matplotlib's own imshow
        # examples — light cells get dark text, dark cells get light.
        cmap_lookup = plt.get_cmap(cmap)
        for i in range(n):
            for j in range(m):
                val = float(mac[i, j])
                rgba = cmap_lookup(val)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                colour = "black" if luminance > 0.5 else "white"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=colour, fontsize=8,
                )

    # Highlight Hungarian-paired cells with a red rectangle outline.
    from matplotlib.patches import Rectangle
    for i, j in comparison.paired_modes:
        rect = Rectangle(
            (j - 0.5, i - 0.5), 1.0, 1.0,
            fill=False, edgecolor="red", linewidth=1.8,
        )
        ax.add_patch(rect)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAC")
    fig.tight_layout()
    return fig
