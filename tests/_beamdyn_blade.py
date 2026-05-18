"""Minimal BeamDyn blade ``.dat`` reader — **test harness only**
(issue #35; test-harness validation oracle).

Not imported by ``src/pybmodes`` and never part of the runtime: it
exists so the WindIO composite-blade reduction can be cross-checked
against the companion ``*_BeamDyn_blade.dat`` 6×6 sectional matrices
that ship with the IEA RWT decks (those tables were themselves
WISDEM-PreComp-generated, so they are the natural diagonal-property
oracle). Integration-only; the upstream decks are gitignored.

BeamDyn distributed-properties block: ``station_total`` stations, each
a non-dimensional ``eta`` followed by a 6×6 stiffness matrix then a
6×6 mass matrix, in the BeamDyn sectional frame
``[F1, F2, F3(axial), M1, M2, M3(torsion)]`` (axis 3 = blade span).
So per station: ``EA = K[2,2]``, the two bending stiffnesses are
``K[3,3]`` / ``K[4,4]`` (flap/edge depending on the local twist
frame — compared as an unordered pair), ``GJ = K[5,5]``, and mass
per length ``= M[0,0]``.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import numpy as np

_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")


@dataclass
class BeamDynBlade:
    eta: np.ndarray      # (n,) non-dimensional station
    EA: np.ndarray       # (n,) axial stiffness, N
    EI_a: np.ndarray     # (n,) bending stiffness K[3,3], N·m² (raw)
    EI_b: np.ndarray     # (n,) bending stiffness K[4,4], N·m² (raw)
    GJ: np.ndarray       # (n,) torsion stiffness K[5,5], N·m² (raw)
    mpl: np.ndarray      # (n,) mass per unit length M[0,0], kg/m
    # Decoupled (elastic/shear centre + principal-axis) properties —
    # the apples-to-apples oracle for the issue #50 published path
    # (the BeamDyn 6×6 is also referenced at the blade axis, so it
    # must be decoupled the same way before comparison).
    EI_flap_dec: np.ndarray = None  # type: ignore[assignment]
    EI_edge_dec: np.ndarray = None  # type: ignore[assignment]
    GJ_dec: np.ndarray = None       # type: ignore[assignment]


def read_beamdyn_blade(path: str | pathlib.Path) -> BeamDynBlade:
    text = pathlib.Path(path).read_text(encoding="latin-1")

    m = re.search(r"(\d+)\s+station_total", text)
    if not m:
        raise ValueError(f"{path}: no 'station_total' found")
    n = int(m.group(1))

    hdr = re.search(r"DISTRIBUTED PROPERTIES[^\n]*\n", text)
    if not hdr:
        raise ValueError(f"{path}: no DISTRIBUTED PROPERTIES block")
    body = text[hdr.end():]

    # 1 (eta) + 36 (K) + 36 (M) numbers per station, in order.
    toks = [t.replace("D", "e").replace("d", "e")
            for t in _FLOAT.findall(body)]
    need = n * 73
    if len(toks) < need:
        raise ValueError(
            f"{path}: expected {need} numbers for {n} stations, "
            f"found {len(toks)}"
        )
    vals = np.array([float(t) for t in toks[:need]], dtype=float)
    vals = vals.reshape(n, 73)

    eta = vals[:, 0]
    K = vals[:, 1:37].reshape(n, 6, 6)
    M = vals[:, 37:73].reshape(n, 6, 6)

    from pybmodes.io._precomp.decouple import decouple_stiffness

    dec = [decouple_stiffness(K[i]) for i in range(n)]
    return BeamDynBlade(
        eta=eta,
        EA=K[:, 2, 2],
        EI_a=K[:, 3, 3],
        EI_b=K[:, 4, 4],
        GJ=K[:, 5, 5],
        mpl=M[:, 0, 0],
        EI_flap_dec=np.array([d.EI_flap for d in dec]),
        EI_edge_dec=np.array([d.EI_edge for d in dec]),
        GJ_dec=np.array([d.GJ for d in dec]),
    )
