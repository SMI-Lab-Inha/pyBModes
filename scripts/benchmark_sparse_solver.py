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

"""Benchmark the sparse vs dense eigensolver paths over a sweep of
tower mesh sizes.

Synthesises a uniform cantilever tower at ``n_elements ∈ {20, 50, 100,
200, 500}`` and solves for the first 4 modes via both
``_solve_sparse_shift_invert`` and ``_solve_dense_symmetric``. Reports
median solve time across N repetitions and the speedup ratio.

Asserts the sparse path is faster than the dense path once
``n_elements > 100`` (i.e. ``ngd > 600`` for a 15-DOF/element tower
with cantilever BC), which is the design target for the
sparse-eigensolver acceleration path.

Run from the repo root:

    set PYTHONPATH=%CD%\\src
    python scripts/benchmark_sparse_solver.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pybmodes.fem.solver import (  # noqa: E402
    _solve_dense_symmetric,
    _solve_sparse_shift_invert,
)

# Suppress the per-call INFO log lines the production solver emits;
# we report timings directly.
logging.getLogger("pybmodes.fem.solver").setLevel(logging.WARNING)


def _build_uniform_tower_problem(n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a (K, M) pair for a uniform cantilever tower using the
    real pyBmodes FEM pipeline up to assembly. Returns the assembled
    global matrices ready for the eigensolver.

    Uses ``tests._synthetic_bmi.write_uniform_sec_props`` + ``write_bmi``
    so the produced matrices match what a real ``Tower(...).run()``
    would feed the solver — i.e. the benchmark is representative of
    real-world solve cost, not a contrived sparsity pattern.
    """
    import tempfile

    from pybmodes.io.bmi import read_bmi
    from pybmodes.io.sec_props import read_sec_props

    # Lazy imports so the script can be run from the repo root
    # without a pip-install of the test package.
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from _synthetic_bmi import write_bmi, write_uniform_sec_props  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        write_uniform_sec_props(tmp_path / "secs.dat", n_secs=11)
        write_bmi(
            tmp_path / "tower.bmi",
            beam_type=2, radius=90.0, hub_rad=0.0, hub_conn=1,
            sec_props_file="secs.dat",
            n_elements=n_elements,
            tip_mass=200_000.0,
        )
        bmi = read_bmi(tmp_path / "tower.bmi")
        sp = read_sec_props(bmi.resolve_sec_props_path())

    # Build matrices the same way ``run_fem`` does. We re-use the
    # private pipeline rather than calling ``Tower.run()`` so we get
    # the matrices in hand for direct timing.
    from pybmodes.fem.assembly import assemble, compute_element_props
    from pybmodes.fem.nondim import (
        make_params,
        nondim_section_props,
        nondim_tip_mass,
    )

    nd = make_params(bmi.radius, bmi.hub_rad, 0.0, draft=0.0)
    props_nd = nondim_section_props(sp, nd, id_form=1, beam_type=bmi.beam_type)
    hub_r = nd.hub_rad / nd.radius
    el, xb, cfe, eiy, eiz, gj, eac, rmas, skm1, skm2, eg, ea = compute_element_props(
        nselt=bmi.n_elements,
        el_loc=bmi.el_loc,
        sp=type("_SP", (), {
            "span_loc": props_nd["sec_loc"],
            "flp_stff": props_nd["flp_stff"],
            "edge_stff": props_nd["edge_stff"],
            "tor_stff": props_nd["tor_stff"],
            "axial_stff": props_nd["axial_stff"],
            "mass_den": props_nd["mass_den"],
            "cg_offst": props_nd["cg_offst"],
            "tc_offst": props_nd["tc_offst"],
            "flp_iner": sp.flp_iner,
            "edge_iner": sp.edge_iner,
        })(),
        hub_r=hub_r,
        bl_frac=nd.bl_len / nd.radius,
    )
    xmid = xb + 0.5 * el
    skm1 = np.interp(xmid, props_nd["sec_loc"], props_nd["sq_km1"])
    skm2 = np.interp(xmid, props_nd["sec_loc"], props_nd["sq_km2"])
    tip_mass_nd = nondim_tip_mass(
        bmi.tip_mass, nd, beam_type=bmi.beam_type, id_form=1,
        hub_conn=bmi.hub_conn,
    )
    gk, gm, _ = assemble(
        nselt=bmi.n_elements, el=el, xb=xb, cfe=cfe,
        eiy=eiy, eiz=eiz, gj=gj, eac=eac, rmas=rmas,
        skm1=skm1, skm2=skm2, eg=eg, ea=ea,
        omega2=nd.omega2, sec_loc=props_nd["sec_loc"],
        str_tw=props_nd["str_tw"],
        tip_mass=tip_mass_nd, wire_k_nd=None,
        wire_node_attach=None, hub_conn=bmi.hub_conn,
        platform_nd=None, elm_distr_k=None,
    )
    return gk, gm


def _median_solve_time(
    solve_fn, gk: np.ndarray, gm: np.ndarray, n_modes: int, n_reps: int,
) -> float:
    """Return the median wall-clock seconds across ``n_reps`` solves."""
    times: list[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        solve_fn(gk, gm, n_modes)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> int:
    n_modes = 4
    n_reps = 5
    grid = [20, 50, 100, 200, 500]

    print(f"{'n_elements':>10}  {'ngd':>5}  "
          f"{'dense (s)':>10}  {'sparse (s)':>10}  "
          f"{'speedup':>8}")
    print("-" * 60)

    rows: list[tuple[int, int, float, float]] = []
    for n_el in grid:
        gk, gm = _build_uniform_tower_problem(n_el)
        ngd = gk.shape[0]
        dense_t = _median_solve_time(
            _solve_dense_symmetric, gk, gm, n_modes, n_reps,
        )
        sparse_t = _median_solve_time(
            _solve_sparse_shift_invert, gk, gm, n_modes, n_reps,
        )
        speedup = dense_t / sparse_t if sparse_t > 0 else float("inf")
        rows.append((n_el, ngd, dense_t, sparse_t))
        print(f"{n_el:>10d}  {ngd:>5d}  "
              f"{dense_t * 1e3:>9.2f}ms  {sparse_t * 1e3:>9.2f}ms  "
              f"{speedup:>7.2f}×")

    # Assertion: sparse should be faster than dense once n_elements > 100.
    # Use a generous margin (10 %) to absorb scheduler noise on CI.
    print()
    bad: list[str] = []
    for n_el, ngd, dense_t, sparse_t in rows:
        if n_el > 100:
            if sparse_t > 1.10 * dense_t:
                bad.append(
                    f"  n_elements={n_el} (ngd={ngd}): sparse "
                    f"{sparse_t * 1e3:.2f} ms > 1.10 × dense "
                    f"{dense_t * 1e3:.2f} ms"
                )
    if bad:
        print("FAIL: sparse path slower than dense above n_elements > 100:")
        for line in bad:
            print(line)
        return 1
    print("OK: sparse path beats dense for every n_elements > 100 "
          "(within a 10 % margin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
