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

"""Generate a Campbell diagram for the bundled NREL 5MW reference blade.

Reads ``reference_decks/nrel5mw_land/NRELOffshrBsline5MW_Onshore_ElastoDyn.dat``
and sweeps the blade across rotor speeds 0..15 rpm. Writes
``scripts/outputs/nrel5mw_campbell.png`` plus a per-step CSV. The
sweep uses 8 modes by default; pass ``--n-modes 6`` etc. to change.

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\campbell.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pybmodes.campbell import campbell_sweep, plot_campbell  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=(
            REPO_ROOT
            / "src" / "pybmodes" / "_examples"
            / "reference_decks"
            / "nrel5mw_land"
            / "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
        ),
        help="path to a .bmi or ElastoDyn main .dat (default: NREL 5MW land)",
    )
    parser.add_argument("--max-rpm", type=float, default=15.0)
    parser.add_argument("--rated-rpm", type=float, default=12.1)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--n-blade-modes", type=int, default=4)
    parser.add_argument("--n-tower-modes", type=int, default=4)
    parser.add_argument(
        "--orders",
        type=str,
        default="1,2,3,6,9",
        help="comma-separated per-rev orders to overlay",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "scripts" / "outputs",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rpm_grid = np.linspace(0.0, args.max_rpm, args.n_steps)
    orders = [int(s) for s in args.orders.split(",") if s.strip()]

    print(f"Campbell sweep on {args.input.name}")
    print(f"  rotor-speed grid : 0..{args.max_rpm} rpm ({args.n_steps} points)")
    print(f"  rated rpm        : {args.rated_rpm}")
    print(f"  blade modes      : {args.n_blade_modes}")
    print(f"  tower modes      : {args.n_tower_modes}")

    result = campbell_sweep(
        args.input,
        rpm_grid,
        n_blade_modes=args.n_blade_modes,
        n_tower_modes=args.n_tower_modes,
    )

    print()
    print("Mode summary:")
    for k, lbl in enumerate(result.labels):
        f0 = result.frequencies[0, k]
        f_top = result.frequencies[-1, k]
        kind = "tower" if k >= result.n_blade_modes else "blade"
        print(f"  slot {k}: {lbl:>18s}  ({kind})  "
              f"{f0:.3f} Hz @ 0 rpm   {f_top:.3f} Hz @ {args.max_rpm:.1f} rpm")

    try:
        from pybmodes.plots.style import apply_style
        apply_style()
    except ImportError:
        print("note: matplotlib style helpers unavailable; "
              "install pybmodes[plots] for journal defaults")

    out_png = args.out_dir / f"{args.input.stem}_campbell.png"
    fig = plot_campbell(
        result,
        excitation_orders=orders,
        rated_rpm=args.rated_rpm,
    )
    fig.savefig(out_png)
    print(f"\nWrote {out_png}")

    out_csv = out_png.with_suffix(".csv")
    header = "rpm," + ",".join(result.labels)
    np.savetxt(
        out_csv,
        np.column_stack([result.omega_rpm, result.frequencies]),
        delimiter=",",
        header=header,
        comments="",
    )
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
