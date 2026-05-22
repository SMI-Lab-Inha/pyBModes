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

"""``pybmodes campbell`` workflow as a typed library function.

Sweeps a rotor-speed grid, writes the resulting :class:`CampbellResult`
to CSV alongside a Campbell-diagram PNG (with per-rev excitation
orders overlaid), and returns a typed result carrying both paths.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pybmodes.workflows._base import WorkflowResult

if TYPE_CHECKING:
    from pybmodes.campbell import CampbellResult as _CampbellSweepResult


@dataclass
class CampbellWorkflowResult(WorkflowResult):
    """Result of :func:`run_campbell`.

    Attributes
    ----------
    sweep : pybmodes.campbell.CampbellResult | None
        The underlying :class:`~pybmodes.campbell.CampbellResult`
        (frequencies × rpm grid + per-mode MAC tracking).
    png_path : pathlib.Path | None
        Path of the Campbell-diagram PNG. ``None`` only when figure
        rendering was skipped (e.g. matplotlib unavailable).
    csv_path : pathlib.Path | None
        Path of the CSV summary (frequencies + per-step MAC tracking
        confidence — produced via :meth:`CampbellResult.to_csv`).
    orders : list[int]
        Per-rev excitation orders overlaid on the diagram.
    """

    sweep: _CampbellSweepResult | None = None
    png_path: pathlib.Path | None = None
    csv_path: pathlib.Path | None = None
    orders: list[int] = field(default_factory=list)


def run_campbell(
    input_path: str | pathlib.Path,
    *,
    max_rpm: float,
    n_steps: int = 16,
    orders: str | list[int] = "1,2,3,6,9",
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
    tower_input: str | pathlib.Path | None = None,
    rated_rpm: float | None = None,
    out_path: str | pathlib.Path | None = None,
) -> CampbellWorkflowResult:
    """Run a rotor-speed sweep and write the Campbell diagram + CSV.

    Library entry point for :command:`pybmodes campbell`. Delegates
    the sweep itself to :func:`pybmodes.campbell.campbell_sweep` and
    renders the diagram via :func:`pybmodes.campbell.plot_campbell`.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Source model: a ``.bmi`` deck or an ElastoDyn main ``.dat``.
    max_rpm : float
        Upper end of the rotor-speed sweep, in rpm. Must be > 0.
    n_steps : int, default 16
        Number of rotor-speed points in the sweep (including 0 and
        ``max_rpm``). Must be >= 2.
    orders : str or list[int], default ``"1,2,3,6,9"``
        Per-rev excitation orders to overlay. Strings are parsed as a
        comma-separated list of integers (this matches the CLI
        ``--orders`` flag); lists are used as-is.
    n_blade_modes, n_tower_modes : int, defaults 4, 4
        Modes to track per side across the sweep.
    tower_input : str, pathlib.Path, or None
        Optional tower override. Mirrors the CLI ``--tower`` flag.
    rated_rpm : float or None
        Operating rotor speed (rpm) drawn as a vertical reference line.
    out_path : str, pathlib.Path, or None
        Output PNG path. ``None`` →
        ``<input_stem>_campbell.png`` alongside ``input_path``. The
        CSV is written next to the PNG with a ``.csv`` suffix.

    Returns
    -------
    CampbellWorkflowResult
        Carries the resolved PNG and CSV paths, the underlying
        :class:`~pybmodes.campbell.CampbellResult`, and the parsed
        excitation orders. ``exit_code`` is ``0`` on success.

    Raises
    ------
    FileNotFoundError
        When ``input_path`` does not exist.
    ValueError
        When ``orders`` cannot be parsed, ``max_rpm <= 0``, or
        ``n_steps < 2``.
    """
    import numpy as np

    from pybmodes.campbell import campbell_sweep, plot_campbell

    src = pathlib.Path(input_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"file not found: {src}")

    if isinstance(orders, str):
        try:
            orders_list = [int(x) for x in orders.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(
                f"orders must be a comma-separated list of integers; "
                f"got {orders!r}"
            ) from exc
    else:
        orders_list = list(orders)
    if not orders_list:
        raise ValueError("orders must list at least one integer")

    if max_rpm <= 0.0:
        raise ValueError(f"max_rpm must be > 0; got {max_rpm}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")

    rpm = np.linspace(0.0, max_rpm, n_steps)
    tower_path = (
        pathlib.Path(tower_input).resolve() if tower_input else None
    )

    messages: list[str] = []
    messages.append(f"Campbell sweep: {src.name}")
    messages.append(f"  rpm grid       : 0..{max_rpm} ({n_steps} points)")
    messages.append(f"  blade modes    : {n_blade_modes}")
    messages.append(f"  tower modes    : {n_tower_modes}")
    if tower_path is not None:
        messages.append(f"  tower override : {tower_path}")

    sweep = campbell_sweep(
        src,
        rpm,
        n_blade_modes=n_blade_modes,
        n_tower_modes=n_tower_modes,
        tower_input=tower_path,
    )

    png_path = (
        pathlib.Path(out_path).resolve() if out_path is not None
        else src.with_name(src.stem + "_campbell.png")
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Use CampbellResult.to_csv() rather than a hand-rolled np.savetxt
    # so the MAC tracking-confidence columns travel alongside the
    # frequencies — the canonical schema.
    csv_path = png_path.with_suffix(".csv")
    sweep.to_csv(csv_path)
    messages.append(f"  wrote {csv_path}")

    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        pass

    fig = plot_campbell(
        sweep, excitation_orders=orders_list, rated_rpm=rated_rpm
    )
    fig.savefig(png_path)
    messages.append(f"  wrote {png_path}")

    return CampbellWorkflowResult(
        exit_code=0,
        messages=messages,
        sweep=sweep,
        png_path=png_path,
        csv_path=csv_path,
        orders=orders_list,
    )
