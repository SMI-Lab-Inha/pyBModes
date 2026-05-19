"""Campbell-diagram support: rotor-speed sweep with MAC-tracked blade
modes and constant-frequency tower modes overlaid on the same plot.

A Campbell diagram plots a turbine's natural frequencies against rotor
speed and overlays the per-revolution excitation lines (1P, 2P, 3P,
…); crossings between excitation lines and structural-mode lines flag
resonance risks. For a wind-turbine blade the centrifugal-stiffening
contribution to the FEM stiffness matrix raises flap-dominated
frequencies markedly with rotor speed while edgewise (lag-dominated)
modes barely move. The tower lives in an Earth-fixed frame, so its
fore-aft / side-to-side bending frequencies don't depend on rotor
speed at all and show up as horizontal lines on the diagram. The
NREL 5MW turbine's canonical resonance call-out — 3P crossing the
1st tower fore-aft mode near ~6.4 rpm — sits right where the cut-in
operating envelope begins, which is exactly the kind of constraint
this diagram is designed to surface.

Public API
----------

- :func:`campbell_sweep` — given an OpenFAST ElastoDyn main ``.dat``,
  loads the blade and tower from the same deck, sweeps the blade
  across ``omega_rpm`` (with MAC-based mode tracking), solves the
  tower once, and packs both into a single :class:`CampbellResult`.
  ``.bmi`` inputs are also accepted and route to blade-only or
  tower-only sweeps based on ``beam_type``; an explicit
  ``tower_input=...`` keyword adds a tower file alongside a blade
  ``.bmi``.
- :func:`plot_campbell` — renders the result with blade modes as
  solid coloured lines, tower modes as horizontal dashed dark-grey
  lines, and the per-rev excitation family as light grey rays from
  the origin. Optional vertical marker at the rated rotor speed.

Defaults are deliberately spare (``n_blade_modes=4``, ``n_tower_modes=4``)
so the diagram shows the modes that actually drive resonance design —
1st/2nd flap, 1st/2nd edge, 1st/2nd tower FA, 1st/2nd tower SS —
without crowding the plot with high-order modes that the per-rev
family doesn't reach inside any realistic operating envelope.
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.io.bmi import BMIFile, read_bmi
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models._pipeline import run_fem

if TYPE_CHECKING:
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.tower import Tower


@dataclass
class CampbellResult:
    """Frequencies and labels from a Campbell sweep — blade + tower combined.

    Attributes
    ----------
    omega_rpm : (N,) array of rotor speeds in rpm.
    frequencies : (N, n_total_modes) array of natural frequencies in Hz.
        Columns are ordered *blade modes first, then tower modes*. With
        MAC tracking enabled, blade columns hold the same physical mode
        across all rotor speeds. Tower columns are constant across rows
        (tower frequencies don't depend on rotor speed).
    labels : list of length ``n_total_modes`` with human-readable mode
        names — blade modes look like ``"1st flap"`` / ``"2nd edge"``,
        tower modes are prefixed with ``"tower"`` (e.g.
        ``"1st tower FA"``, ``"1st tower SS"``) so callers can split
        the two by string match if needed.
    participation : (N, n_total_modes, 3) array of energy fractions in
        the FEM's per-mode (flap or FA, edge or SS, torsion) axes.
        Each row sums to 1. Note the axis interpretation is
        beam-type-specific: blade columns use flap/edge/torsion, tower
        columns use FA/SS/torsion.
    mac_to_previous : (N, n_total_modes) array of per-step MAC values
        between each output slot's mode shape at step ``k`` and the
        same slot at step ``k - 1`` (i.e. the tracking confidence).
        Row 0 is filled with NaN (no previous step). Tower columns are
        also NaN (tower modes don't change with rotor speed, so a MAC
        confidence is not meaningful for them).
    n_blade_modes : how many of the leading columns are blade modes.
    n_tower_modes : how many of the trailing columns are tower modes.
    """

    omega_rpm: np.ndarray
    frequencies: np.ndarray
    labels: list[str]
    participation: np.ndarray
    n_blade_modes: int
    n_tower_modes: int
    mac_to_previous: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # ------------------------------------------------------------------
    # Schema validation (shared by save / to_csv)
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Assert the documented array-shape contract before any
        export, so ``save`` / ``to_csv`` can't silently emit a
        malformed archive or CSV that loads back inconsistent.

        One uniform contract — no special-cased "empty sweep"
        exemption (that ad-hoc branch repeatedly leaked edge cases:
        missing arrays, ``.size`` vs ``.shape``). ``frequencies`` is
        ``(n_steps, n_modes)`` (2-D, always); ``omega_rpm`` is
        ``(n_steps,)``; ``labels`` has ``n_modes`` entries;
        ``participation`` is ``(n_steps, n_modes, 3)``;
        ``mac_to_previous`` is **either** the empty ``(0, 0)`` default
        (unset) **or** exactly ``(n_steps, n_modes)``;
        ``n_blade_modes, n_tower_modes`` are non-negative and sum to
        ``n_modes``. A genuinely empty sweep is simply the
        ``n_steps == n_modes == 0`` instance of this contract — every
        check below holds vacuously for the canonical empty shapes
        (``frequencies (0,0)``, ``omega_rpm (0,)``, ``participation
        (0,0,3)``, ``mac (0,0)``) and fails for any malformed
        zero-size variant such as ``(0,3)`` / ``(2,0)`` / ``(0,2)``.
        """
        freqs = np.asarray(self.frequencies)
        if freqs.ndim != 2:
            raise ValueError(
                f"frequencies must be 2-D (n_steps, n_modes); got "
                f"shape {freqs.shape}"
            )
        n_steps, n_modes = freqs.shape
        omega = np.asarray(self.omega_rpm)
        if omega.shape != (n_steps,):
            raise ValueError(
                f"omega_rpm shape {omega.shape} != (n_steps,) = "
                f"({n_steps},)"
            )
        if len(self.labels) != n_modes:
            raise ValueError(
                f"len(labels)={len(self.labels)} != n_modes={n_modes}"
            )
        part = np.asarray(self.participation)
        if part.shape != (n_steps, n_modes, 3):
            raise ValueError(
                f"participation shape {part.shape} != "
                f"(n_steps, n_modes, 3) = ({n_steps}, {n_modes}, 3)"
            )
        mac = np.asarray(self.mac_to_previous)
        # Unset iff the canonical empty default ``(0, 0)`` — *not*
        # merely ``size == 0`` (a ``(2, 0)`` / ``(0, 2)`` array is
        # size-0 but malformed). Otherwise exactly ``(n_steps,
        # n_modes)``. For the empty sweep both collapse to ``(0, 0)``.
        if mac.shape != (0, 0) and mac.shape != (n_steps, n_modes):
            raise ValueError(
                f"mac_to_previous shape {mac.shape} is neither the "
                f"empty (0, 0) default nor (n_steps, n_modes) = "
                f"({n_steps}, {n_modes})"
            )
        if self.n_blade_modes < 0 or self.n_tower_modes < 0:
            raise ValueError(
                f"mode counts must be non-negative; got "
                f"n_blade_modes={self.n_blade_modes}, "
                f"n_tower_modes={self.n_tower_modes}"
            )
        if self.n_blade_modes + self.n_tower_modes != n_modes:
            raise ValueError(
                f"n_blade_modes ({self.n_blade_modes}) + n_tower_modes "
                f"({self.n_tower_modes}) != n_modes ({n_modes})"
            )
        # Physical arrays must be finite. ``mac_to_previous`` is
        # exempt from the finite check — NaN there is the documented
        # "not meaningful" sentinel (row 0 / tower columns) — but
        # ``inf`` is *not* a valid sentinel and is rejected.
        for nm, a in (("frequencies", freqs),
                      ("omega_rpm", omega),
                      ("participation", part)):
            if not np.all(np.isfinite(np.asarray(a, dtype=float))):
                raise ValueError(
                    f"{nm} contains non-finite (NaN / inf) values"
                )
        if mac.size and np.isinf(np.asarray(mac, dtype=float)).any():
            raise ValueError(
                "mac_to_previous contains inf — NaN is the only "
                "permitted non-finite sentinel"
            )
        # participation: documented energy fractions — every row sums
        # to 1, or to 0 for a null mode shape (the documented
        # zero-shape sentinel, mirroring the mac NaN one). Negative
        # entries or any other row sum is corruption.
        if np.any(part < 0.0):
            raise ValueError(
                "participation contains negative values (energy "
                "fractions must be >= 0)"
            )
        rs = part.sum(axis=-1)
        ok = np.isclose(rs, 1.0, atol=1e-6) | np.isclose(
            rs, 0.0, atol=1e-9
        )
        if not np.all(ok):
            raise ValueError(
                "participation rows must each sum to 1 (or 0 for a "
                "null mode); got sums outside that set"
            )

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def save(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the sweep result to a ``.npz`` archive.

        Arrays go in as named keys; labels and the two integer scalars
        ride in via the embedded JSON ``__meta__`` blob alongside the
        standard pyBmodes-version / timestamp / source-file / git-hash
        metadata captured by :func:`pybmodes.io._serialize._capture_metadata`.
        """
        from pybmodes.io._serialize import _capture_metadata, _metadata_to_npz_value

        self._validate()
        meta = _capture_metadata(source_file=source_file)
        meta["labels"] = list(self.labels)
        meta["n_blade_modes"] = int(self.n_blade_modes)
        meta["n_tower_modes"] = int(self.n_tower_modes)

        np.savez_compressed(
            pathlib.Path(path),
            omega_rpm=np.asarray(self.omega_rpm, dtype=float),
            frequencies=np.asarray(self.frequencies, dtype=float),
            participation=np.asarray(self.participation, dtype=float),
            mac_to_previous=np.asarray(self.mac_to_previous, dtype=float),
            __meta__=_metadata_to_npz_value(meta),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CampbellResult":
        """Read a sweep result back from a ``.npz`` archive saved by
        :meth:`save`."""
        from pybmodes.io._serialize import _read_npz_meta

        path = pathlib.Path(path)
        with np.load(path, allow_pickle=False) as npz:
            meta = _read_npz_meta(npz, path)
            inst = cls(
                omega_rpm=np.asarray(npz["omega_rpm"], dtype=float),
                frequencies=np.asarray(npz["frequencies"], dtype=float),
                labels=list(meta["labels"]),
                participation=np.asarray(npz["participation"], dtype=float),
                n_blade_modes=int(meta["n_blade_modes"]),
                n_tower_modes=int(meta["n_tower_modes"]),
                mac_to_previous=np.asarray(npz["mac_to_previous"], dtype=float),
            )
        # Validate on ingest, not only on export — a corrupt /
        # hand-edited archive must fail loudly at load(), not later
        # in plotting / CSV export.
        inst._validate()
        return inst

    # ------------------------------------------------------------------
    # CSV emission
    # ------------------------------------------------------------------

    def to_csv(self, path: str | pathlib.Path) -> None:
        """Write a spreadsheet-friendly CSV with one row per rotor-speed
        step.

        Columns: ``rpm``, then one frequency column per mode (named by
        the mode's label), then one MAC-confidence column per mode
        suffixed with ``_mac``. Tower-mode MAC columns are NaN
        throughout because tower modes don't change with rotor speed —
        kept as columns for shape-stability across blade-only / tower-
        only / mixed sweeps.
        """
        import csv

        self._validate()
        n_steps, n_modes = self.frequencies.shape
        freq_cols = list(self.labels)
        mac_cols = [f"{lbl}_mac" for lbl in self.labels]
        header = ["rpm", *freq_cols, *mac_cols]

        with pathlib.Path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for step in range(n_steps):
                row: list[object] = [float(self.omega_rpm[step])]
                row.extend(float(self.frequencies[step, k]) for k in range(n_modes))
                # Per-mode MAC confidence (NaN where unset / not meaningful).
                if self.mac_to_previous.shape == self.frequencies.shape:
                    row.extend(
                        float(self.mac_to_previous[step, k])
                        for k in range(n_modes)
                    )
                else:
                    row.extend([float("nan")] * n_modes)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# A "model" is a (BMIFile, SectionProperties|None) tuple. ``None`` for the
# section-properties slot signals that ``run_fem`` should re-read them
# from disk via ``BMIFile.resolve_sec_props_path``; ElastoDyn-derived
# models supply them directly.
_Model = tuple[BMIFile, SectionProperties | None]


def _model_pair(obj: object) -> "tuple[str, _Model] | None":
    """``(role, (_bmi, _sp))`` for an already-loaded ``RotatingBlade``
    / ``Tower`` (issue #51 — single point of load-in; also the only
    way to Campbell a ``from_windio`` / ``from_elastodyn`` model,
    which carry pre-built section properties no path can re-read).
    ``None`` if ``obj`` is not a loaded model (treat as a path).

    Deferred import keeps ``campbell`` free of a ``models`` import
    cycle. ``role`` is taken from ``beam_type`` so a blade or tower
    can be passed in either positional slot."""
    from pybmodes.models.blade import RotatingBlade
    from pybmodes.models.tower import Tower

    if not isinstance(obj, (RotatingBlade, Tower)):
        return None
    bmi = obj._bmi
    sp = obj._sp
    if bmi.beam_type == 1:
        return "blade", (bmi, sp)
    if bmi.beam_type == 2:
        return "tower", (bmi, sp)
    raise ValueError(
        f"loaded {type(obj).__name__} has unsupported beam_type "
        f"{bmi.beam_type} (expected 1 = blade or 2 = tower)"
    )


def _load_models(
    input_path: "str | pathlib.Path | object",
    tower_input: "str | pathlib.Path | object | None",
) -> tuple[_Model | None, _Model | None]:
    """Resolve the input(s) to (blade, tower) model pairs.

    ``input_path`` may be a path **or an already-loaded
    ``RotatingBlade`` / ``Tower``** (issue #51): a loaded model is
    used directly — no disk re-read — which is the only way to sweep a
    ``from_windio`` / ``from_elastodyn`` model and gives a single
    point of load-in. For an ElastoDyn ``.dat`` file the deck carries
    both, so we load both unless the corresponding files can't be
    resolved. ``.bmi`` inputs are routed to blade or tower by their
    ``beam_type``. The ``tower_input`` keyword (a tower ``.bmi`` path
    *or* a loaded ``Tower``) pairs an explicit tower with a blade
    input; if the primary input was an ElastoDyn deck and
    ``tower_input`` is also given, ``tower_input`` overrides the
    deck-supplied tower (useful when the deck's tower file points
    somewhere unhelpful).
    """
    blade: _Model | None = None
    tower: _Model | None = None

    # Already-loaded RotatingBlade / Tower — use verbatim.
    mp = _model_pair(input_path)
    if mp is not None:
        role, pair = mp
        if role == "blade":
            blade = pair
        else:
            tower = pair
        if tower_input is not None:
            tmp = _model_pair(tower_input)
            if tmp is not None:
                if tmp[0] != "tower":
                    raise ValueError(
                        "tower_input must be a Tower (beam_type=2) or a "
                        "tower .bmi"
                    )
                tower = tmp[1]
            else:
                tp = pathlib.Path(tower_input)  # type: ignore[arg-type]
                if tp.suffix.lower() != ".bmi":
                    raise ValueError(
                        f"tower_input must be a .bmi file; got "
                        f"{tp.suffix!r}"
                    )
                tbmi = read_bmi(tp)
                if tbmi.beam_type != 2:
                    raise ValueError(
                        f"tower_input {tp} has beam_type "
                        f"{tbmi.beam_type}, expected 2 (tower)"
                    )
                tower = (tbmi, None)
        return blade, tower

    input_path = pathlib.Path(input_path)  # type: ignore[arg-type]
    suffix = input_path.suffix.lower()

    if suffix == ".dat":
        from pybmodes.io.elastodyn_reader import (
            read_elastodyn_blade,
            read_elastodyn_main,
            read_elastodyn_tower,
            to_pybmodes_blade,
            to_pybmodes_tower,
        )
        main = read_elastodyn_main(input_path)
        bld_path = input_path.parent / main.bld_file[0]
        blade_data = read_elastodyn_blade(bld_path)
        blade = to_pybmodes_blade(main, blade_data)

        twr_path = input_path.parent / main.twr_file
        if twr_path.is_file():
            tower_data = read_elastodyn_tower(twr_path)
            tower = to_pybmodes_tower(main, tower_data, blade=blade_data)
        else:
            # ElastoDyn main references a TwrFile that we couldn't
            # locate on disk. Continuing blade-only is a useful
            # degraded-mode for blade-focused Campbell sweeps, but
            # silently dropping the tower modes from the result has
            # surprised users. Warn explicitly so the absence is
            # visible — caller can still opt in to blade-only by
            # ignoring the warning.
            warnings.warn(
                f"campbell_sweep: TwrFile referenced by {input_path} "
                f"as {main.twr_file!r} not found at {twr_path}. "
                f"Continuing blade-only — the resulting CampbellResult "
                f"will carry zero tower modes. To suppress this "
                f"warning explicitly, pass a .bmi blade file directly "
                f"instead of the ElastoDyn main.",
                UserWarning,
                stacklevel=2,
            )
    elif suffix == ".bmi":
        bmi = read_bmi(input_path)
        if bmi.beam_type == 1:
            blade = (bmi, None)
        elif bmi.beam_type == 2:
            tower = (bmi, None)
        else:
            raise ValueError(
                f"unsupported beam_type {bmi.beam_type} in {input_path}"
            )
    else:
        raise ValueError(
            f"campbell_sweep input must be .bmi or ElastoDyn .dat; "
            f"got {input_path.suffix!r}"
        )

    if tower_input is not None:
        tmp = _model_pair(tower_input)
        if tmp is not None:                      # a loaded Tower
            if tmp[0] != "tower":
                raise ValueError(
                    "tower_input must be a Tower (beam_type=2) or a "
                    "tower .bmi"
                )
            tower = tmp[1]
        else:
            tpath = pathlib.Path(tower_input)    # type: ignore[arg-type]
            if tpath.suffix.lower() != ".bmi":
                raise ValueError(
                    f"tower_input must be a .bmi file; got "
                    f"{tpath.suffix!r}"
                )
            tower_bmi = read_bmi(tpath)
            if tower_bmi.beam_type != 2:
                raise ValueError(
                    f"tower_input {tpath} has beam_type "
                    f"{tower_bmi.beam_type}, expected 2 (tower)"
                )
            tower = (tower_bmi, None)

    return blade, tower


def _shape_vector(shape: NodeModeShape) -> np.ndarray:
    """Thin wrapper kept for backwards-compatibility with the internal
    Campbell tracker. New code should call
    ``pybmodes.mac.shape_to_vector`` directly."""
    from pybmodes.mac import shape_to_vector
    return shape_to_vector(shape)


def _participation(shape: NodeModeShape) -> np.ndarray:
    """Energy fractions in axes 0 / 1 / 2 (sum to 1; zeros if shape is null).

    For a blade these read flap / edge / torsion; for a tower they read
    FA / SS / torsion (same FEM DOF layout, different physical naming).
    """
    flap = float(np.dot(shape.flap_disp, shape.flap_disp))
    edge = float(np.dot(shape.lag_disp, shape.lag_disp))
    twist = float(np.dot(shape.twist, shape.twist))
    total = flap + edge + twist
    if total <= 0.0:
        return np.zeros(3)
    return np.array([flap, edge, twist]) / total


def _mac_matrix(
    curr: list[NodeModeShape],
    prev: list[NodeModeShape],
) -> np.ndarray:
    """Thin wrapper around :func:`pybmodes.mac.mac_matrix` kept for
    backwards-compatibility inside the Campbell tracker. New code
    should call ``pybmodes.mac.mac_matrix`` directly."""
    from pybmodes.mac import mac_matrix
    return mac_matrix(curr, prev)


def _hungarian_assignment(mac: np.ndarray) -> np.ndarray:
    """Global MAC-maximising assignment via the Hungarian (Munkres)
    algorithm.

    Returns ``order[i] = j`` mapping current-step mode ``i`` to the
    previous-step slot ``j`` that maximises the sum of MAC values
    across all matched pairs. This is the standard industry approach
    for mode tracking — it avoids the failure mode of the older
    greedy ``argmax(mac)`` scheme, which can lock in a slightly-
    better first match and force later modes into worse pairings.

    Non-square inputs are handled natively by
    ``scipy.optimize.linear_sum_assignment``: it returns
    ``min(n_curr, n_prev)`` matched pairs, and any current-step row
    that did not receive a previous-step pairing stays at the
    sentinel ``-1`` in the output. The caller (``_solve_blade_sweep``)
    fills those slots from any free previous-step indices, so a
    non-square call still produces a well-defined ordering for every
    current-step mode. In practice the Campbell sweep always supplies
    square ``(n_modes, n_modes)`` inputs; the non-square fallback is
    defensive.
    """
    from scipy.optimize import linear_sum_assignment

    n_curr, _ = mac.shape
    row_ind, col_ind = linear_sum_assignment(mac, maximize=True)
    order = -np.ones(n_curr, dtype=int)
    order[row_ind] = col_ind
    return order


# Kept as a thin wrapper for backwards compatibility — older callers
# (and tests) may import ``_greedy_assignment`` by name. Delegates to
# the Hungarian-based implementation.
def _greedy_assignment(mac: np.ndarray) -> np.ndarray:
    """Deprecated alias for :func:`_hungarian_assignment` — kept for
    backwards compatibility; new code should call the Hungarian
    version directly."""
    return _hungarian_assignment(mac)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _label_blade_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st flap"`` / ``"2nd edge"`` / … from participation at one rotor speed."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("flap", "edge", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} {names[axis]}")
    return out


def _label_tower_modes(participation_row: np.ndarray) -> list[str]:
    """``"1st tower FA"`` / ``"1st tower SS"`` / …."""
    n = participation_row.shape[0]
    counts = [0, 0, 0]
    names = ("FA", "SS", "torsion")
    out: list[str] = []
    for i in range(n):
        axis = int(np.argmax(participation_row[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} tower {names[axis]}")
    return out


def _label_tower_modes_with_overrides(
    participation: np.ndarray,
    mode_labels: "list[str | None] | None",
) -> list[str]:
    """Tower-column labels, preferring the FEM's own classification.

    For a free-free floating tower the leading modes are the platform
    rigid-body modes (surge / sway / heave / roll / pitch / yaw), which
    :func:`pybmodes.fem.platform_modes.classify_platform_modes`
    already names on the :class:`~pybmodes.models.result.ModalResult`
    (``mode_labels``) and which BModes-cross-validates. Participation
    argmax (flap/edge/torsion energy) is meaningless for those rigid
    modes — it produced spurious ``"1st tower FA"`` … names for the
    platform DOFs (issue #47). So: where ``mode_labels[i]`` is a
    classified platform DOF, use it verbatim; everywhere else fall
    back to the participation-derived ``"Nth tower FA/SS/torsion"``
    label, with the ordinal counted over the *flexible* tower modes
    only so the first real bending mode is ``"1st tower FA"`` even
    when six rigid modes precede it. ``mode_labels=None`` (every
    cantilever / monopile tower) reproduces :func:`_label_tower_modes`
    exactly.
    """
    n = participation.shape[0]
    counts = [0, 0, 0]
    names = ("FA", "SS", "torsion")
    out: list[str] = []
    for i in range(n):
        plat = (
            mode_labels[i]
            if mode_labels is not None and i < len(mode_labels)
            else None
        )
        if plat is not None:
            out.append(str(plat))
            continue
        axis = int(np.argmax(participation[i]))
        counts[axis] += 1
        out.append(f"{_ordinal(counts[axis])} tower {names[axis]}")
    return out


def _solve_blade_sweep(
    blade: _Model,
    omega_rpm: np.ndarray,
    n_modes: int,
    track_by_mac: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Run the rotor-speed sweep on the blade model.

    Returns ``(frequencies, participation, labels, mac_to_previous)``
    with shapes ``(n_steps, n_modes)``, ``(n_steps, n_modes, 3)``, a
    list of ``n_modes`` labels, and ``(n_steps, n_modes)`` per-step
    MAC values vs the immediately-preceding step (row 0 is NaN).
    Restores the original ``bbmi.rot_rpm`` after the sweep so the
    caller's BMI object is unmutated.
    """
    bbmi, bsp = blade
    original_rpm = float(getattr(bbmi, "rot_rpm", 0.0))
    n_steps = omega_rpm.size
    freqs = np.zeros((n_steps, n_modes))
    parts = np.zeros((n_steps, n_modes, 3))
    mac_to_prev = np.full((n_steps, n_modes), np.nan, dtype=float)
    slot_shapes: list[NodeModeShape] | None = None

    try:
        for step, rpm in enumerate(omega_rpm):
            bbmi.rot_rpm = float(rpm)
            modal = run_fem(bbmi, n_modes=n_modes, sp=bsp)
            # Defensive: the symmetric ``eigh`` solver always returns
            # exactly ``n_modes`` rows, but the rare general-eig
            # fallback (floating platforms with non-symmetric K,
            # ``solve_modes`` → ``scipy.linalg.eig``) can drop NaN
            # rows from a degenerate eigenproblem. Detect and fail
            # loudly rather than silently NaN-padding downstream.
            if len(modal.frequencies) < n_modes:
                raise RuntimeError(
                    f"campbell_sweep: at rot_rpm = {rpm:.3f}, the FEM "
                    f"solver returned only {len(modal.frequencies)} of "
                    f"the requested {n_modes} modes — typically a sign "
                    f"of a near-degenerate eigenproblem (rotating "
                    f"floating platforms with a non-symmetric "
                    f"PlatformSupport block at certain rotor speeds). "
                    f"Reduce ``n_blade_modes`` or use a finer RPM grid "
                    f"that avoids the degeneracy."
                )
            shapes = list(modal.shapes[:n_modes])
            f_step = np.asarray(modal.frequencies[:n_modes], dtype=float)
            p_step = np.array([_participation(s) for s in shapes])

            if step == 0 or not track_by_mac or slot_shapes is None:
                order = np.arange(n_modes, dtype=int)
                mac_row = np.full(n_modes, np.nan, dtype=float)
            else:
                mac = _mac_matrix(shapes, slot_shapes)
                order = _hungarian_assignment(mac)
                free = [s for s in range(n_modes) if s not in order]
                for k in range(n_modes):
                    if order[k] < 0 and free:
                        order[k] = free.pop(0)
                # MAC confidence of the chosen pairing per output slot.
                mac_row = np.empty(n_modes, dtype=float)
                for k in range(n_modes):
                    slot = int(order[k])
                    mac_row[slot] = float(mac[k, slot]) if slot >= 0 else np.nan

            for k in range(n_modes):
                slot = int(order[k])
                freqs[step, slot] = f_step[k]
                parts[step, slot, :] = p_step[k]
            mac_to_prev[step, :] = mac_row

            new_slot_shapes: list[NodeModeShape | None] = [None] * n_modes
            for k in range(n_modes):
                new_slot_shapes[int(order[k])] = shapes[k]
            slot_shapes = [s for s in new_slot_shapes if s is not None]
    finally:
        bbmi.rot_rpm = original_rpm

    labels = _label_blade_modes(parts[0])
    return freqs, parts, labels, mac_to_prev


def _solve_tower_once(
    tower: _Model,
    n_modes: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Solve the tower once at ``rot_rpm = 0`` and broadcast across the sweep.

    Tower modes are rotor-speed-independent (the tower lives in an
    Earth-fixed frame), so a single eigensolve is enough; we tile the
    result across ``n_steps`` rows for shape compatibility with the
    blade-sweep output.
    """
    tbmi, tsp = tower
    # Save and restore the caller's ``rot_rpm`` — tower modes are
    # rotor-speed-independent so we force ``0.0`` for the solve, but
    # we mustn't leave the caller's BMI mutated on the way out
    # (mirrors the try / finally pattern in ``_solve_blade_sweep``;
    # Pre-1.0 review caught this).
    original_rpm = tbmi.rot_rpm
    try:
        tbmi.rot_rpm = 0.0
        modal = run_fem(tbmi, n_modes=n_modes, sp=tsp)
    finally:
        tbmi.rot_rpm = original_rpm
    # Defensive: mirror the blade-sweep "too few modes" guard. The
    # symmetric ``eigh`` path always returns exactly ``n_modes`` rows,
    # but the rare general-eig fallback (floating ``PlatformSupport``
    # with non-symmetric ``hydro_K`` / ``mooring_K``) can drop NaN rows
    # from a degenerate eigenproblem and return fewer. Without this
    # guard the downstream ``np.broadcast_to`` call would raise a
    # cryptic shape error; we want the same friendly diagnostic the
    # blade path emits.
    if len(modal.frequencies) < n_modes:
        raise RuntimeError(
            f"campbell_sweep: tower solve returned only "
            f"{len(modal.frequencies)} of the requested {n_modes} "
            f"modes — typically a sign of a near-degenerate "
            f"eigenproblem (floating tower with a non-symmetric "
            f"PlatformSupport block). Reduce ``n_tower_modes``."
        )
    tshapes = list(modal.shapes[:n_modes])
    tfreqs = np.asarray(modal.frequencies[:n_modes], dtype=float)
    tparts = np.array([_participation(s) for s in tshapes])

    freqs = np.broadcast_to(tfreqs, (n_steps, n_modes)).copy()
    parts = np.broadcast_to(tparts, (n_steps, n_modes, 3)).copy()
    # Prefer the FEM's own platform-mode classification for a floating
    # tower (``ModalResult.mode_labels`` — populated only for
    # ``hub_conn == 2``); fall back to participation argmax for
    # cantilever / monopile towers and for flexible bending modes.
    labels = _label_tower_modes_with_overrides(tparts, modal.mode_labels)
    return freqs, parts, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def campbell_sweep(
    input_path: "str | pathlib.Path | RotatingBlade | Tower",
    omega_rpm: np.ndarray,
    n_blade_modes: int = 4,
    n_tower_modes: int = 4,
    *,
    tower_input: "str | pathlib.Path | Tower | None" = None,
    track_by_mac: bool = True,
) -> CampbellResult:
    """Build a Campbell-diagram dataset for the given turbine.

    Parameters
    ----------
    input_path :
        Either a path **or an already-loaded model** (issue #51):

        - an OpenFAST ElastoDyn main ``.dat`` file — the function
          loads the blade *and* the tower from the deck and runs both;
        - a blade ``.bmi`` (``beam_type = 1``) — blade-only sweep
          unless ``tower_input`` is also supplied;
        - a tower ``.bmi`` (``beam_type = 2``) — tower-only result
          (frequencies are constant across ``omega_rpm``; the result
          is mostly useful for overlay against the per-rev family);
        - an already-constructed :class:`~pybmodes.models.RotatingBlade`
          or :class:`~pybmodes.models.Tower` (from *any* constructor —
          ``__init__``, ``from_elastodyn``, ``from_windio``,
          ``from_windio_floating``, …). The model is used **verbatim,
          with no disk re-read**, so a single load point feeds both
          ``.run()`` and the sweep, and a ``from_windio`` /
          ``from_elastodyn`` model (whose section properties no path
          can re-read) can finally be swept. Routed to blade/tower by
          its ``beam_type`` so either may be passed here.
    omega_rpm :
        1-D array of rotor speeds in rpm. ``Ω = 0`` is fine and
        produces the parked-rotor frequencies.
    n_blade_modes :
        Number of blade modes to extract per speed and report in
        ``frequencies[:, :n_blade_modes]``. Default 4 covers
        1st/2nd flap and 1st/2nd edge — the modes that actually drive
        resonance design. Pushing this much higher just adds
        high-order flap modes that no realistic per-rev family
        crosses inside the operating envelope; raise it deliberately
        when you need them.
    n_tower_modes :
        Number of tower modes (default 4 — 1st/2nd FA + 1st/2nd SS).
        Drop to 2 to overlay only the 1st FA + 1st SS pair, or push
        higher for offshore decks where 3rd-mode crossings matter.
        Ignored when no tower model is available.
    tower_input :
        Optional explicit tower — a tower ``.bmi`` path **or a loaded
        :class:`~pybmodes.models.Tower`** (keyword-only). Useful when
        ``input_path`` is a blade-only deck or a loaded
        ``RotatingBlade``. Overrides the deck-supplied tower if
        ``input_path`` was an ElastoDyn ``.dat``. So the
        single-load-point form is
        ``campbell_sweep(blade, omega, tower_input=tower)``.
    track_by_mac :
        Whether to use MAC across consecutive rotor speeds to keep
        each blade output column corresponding to the same physical
        mode. ``False`` returns the eigensolver's native order (useful
        for debugging mode re-ordering issues). Tower modes don't
        change with rotor speed and are unaffected by this flag.

    Returns
    -------
    :class:`CampbellResult`.
    """
    # ``_load_models`` accepts a path *or* a loaded RotatingBlade /
    # Tower for either argument (issue #51) — do not coerce to Path
    # here (that would break a model object).
    blade, tower = _load_models(input_path, tower_input)

    omega_rpm = np.asarray(omega_rpm, dtype=float).ravel()
    if omega_rpm.size == 0:
        raise ValueError("omega_rpm must contain at least one rotor speed")
    if not np.all(np.isfinite(omega_rpm)):
        raise ValueError(
            "omega_rpm must be finite; found NaN or inf in "
            f"{omega_rpm.tolist()!r}"
        )
    if np.any(omega_rpm < 0.0):
        raise ValueError(
            "omega_rpm must be non-negative (rotor speeds in rpm); found "
            f"min = {float(omega_rpm.min())!r}"
        )
    if omega_rpm.size >= 2 and np.any(np.diff(omega_rpm) < 0.0):
        raise ValueError(
            "omega_rpm must be sorted ascending so MAC tracking can pair "
            "consecutive steps; got "
            f"{omega_rpm.tolist()!r}"
        )
    if not isinstance(n_blade_modes, int) or n_blade_modes < 0:
        raise ValueError(
            f"n_blade_modes must be a non-negative integer; got {n_blade_modes!r}"
        )
    if not isinstance(n_tower_modes, int) or n_tower_modes < 0:
        raise ValueError(
            f"n_tower_modes must be a non-negative integer; got {n_tower_modes!r}"
        )

    # Silently zero-out mode counts for components that aren't present —
    # easier on the caller than raising for the common "no tower" case.
    if blade is None:
        n_blade_modes = 0
    if tower is None:
        n_tower_modes = 0
    if n_blade_modes + n_tower_modes < 1:
        raise ValueError(
            "no modes to compute: input had neither a blade nor a tower "
            "component, or both n_blade_modes and n_tower_modes were 0"
        )

    n_steps = omega_rpm.size
    blade_freqs = blade_parts = None
    blade_labels: list[str] = []
    blade_mac: np.ndarray | None = None
    if blade is not None and n_blade_modes > 0:
        blade_freqs, blade_parts, blade_labels, blade_mac = _solve_blade_sweep(
            blade, omega_rpm, n_blade_modes, track_by_mac,
        )

    tower_freqs = tower_parts = None
    tower_labels: list[str] = []
    if tower is not None and n_tower_modes > 0:
        tower_freqs, tower_parts, tower_labels = _solve_tower_once(
            tower, n_tower_modes, n_steps,
        )

    parts_pieces = [a for a in (blade_parts, tower_parts) if a is not None]
    freqs_pieces = [a for a in (blade_freqs, tower_freqs) if a is not None]
    frequencies = np.concatenate(freqs_pieces, axis=1)
    participation = np.concatenate(parts_pieces, axis=1)
    labels = blade_labels + tower_labels

    # Build the per-step MAC table: blade columns get the tracked
    # MACs from the sweep; tower columns are NaN (no rotor-speed
    # dependence, so a MAC confidence is not meaningful).
    mac_pieces: list[np.ndarray] = []
    if blade_mac is not None:
        mac_pieces.append(blade_mac)
    if tower_freqs is not None:
        mac_pieces.append(np.full((n_steps, n_tower_modes), np.nan))
    if mac_pieces:
        mac_to_previous = np.concatenate(mac_pieces, axis=1)
    else:
        mac_to_previous = np.empty((n_steps, 0))

    return CampbellResult(
        omega_rpm=omega_rpm,
        frequencies=frequencies,
        labels=labels,
        participation=participation,
        n_blade_modes=n_blade_modes,
        n_tower_modes=n_tower_modes,
        mac_to_previous=mac_to_previous,
    )


def plot_campbell(
    result: CampbellResult,
    excitation_orders: list[int] | None = None,
    rated_rpm: float | None = None,
    ax=None,
    platform_modes: "list[tuple[str, float]] | None" = None,
    log_freq: bool = False,
    *,
    operating_rpm: "tuple[float, float] | None" = None,
    freq_max: float | None = None,
):
    """Render a Campbell diagram from a :class:`CampbellResult`.

    Engineering-report style (issue #54): structural modes are
    coloured by **family**, the legend carries only those four family
    keys, mode names are written inline next to their lines, and the
    per-rev family is the only thing the legend enumerates as a group:

    * **Blades** — green; per-blade curves, name written at the line.
    * **Tower** — black; horizontal lines, name at the line.
    * **Platform** — red; floating-platform rigid-body modes (surge /
      sway / heave / roll / pitch / yaw), near-degenerate symmetric
      pairs merged (``surge/sway``) to keep the figure clean.
    * **Blade Passing** — blue; the per-rev rays (default 1P / 3P /
      6P / 9P), each tagged ``↑ nP`` inline (no legend clutter).

    ``operating_rpm=(lo, hi)`` shades the operating rotor-speed window
    grey (outside it stays white) and draws a ``↔ Operating Speed
    Range`` marker.

    Parameters
    ----------
    result :
        Output of :func:`campbell_sweep`.
    excitation_orders :
        Per-rev orders. Default ``[1, 3, 6, 9]``.
    rated_rpm :
        If given, a thin reference line at the operating rotor speed.
    ax :
        Existing Axes to draw into; a fresh figure if ``None``.
    platform_modes :
        Optional ``[(dof, freq_hz), ...]`` floating rigid-body modes
        for the *screening* path (when the result has no platform
        columns). Drawn in the red **Platform** family. The
        coupled-tower path classifies these natively — see
        :func:`campbell_sweep`.
    log_freq :
        Log-scaled frequency axis (the per-rev rays are densely
        sampled so they render correctly). Default ``False``.
    operating_rpm :
        ``(lo, hi)`` rotor-speed operating window (rpm) — shaded grey
        with an ``Operating Speed Range`` marker. ``None`` (default)
        draws no band — backward compatible.
    freq_max :
        Upper frequency-axis limit (Hz). ``None`` (default) auto-caps
        the axis just above the highest *structural* mode so the
        modes of interest fill the figure (the steep per-rev rays run
        off the top, as in a standard Campbell report) instead of the
        axis stretching to the highest ray. Ignored when
        ``log_freq=True``.

    Note on blade-line jitter
    -------------------------
    For ElastoDyn-derived blade FEMs the 1st-flap line typically shows
    ~5 % step-to-step scatter — *not* real Southwell dynamics. The
    BMI adapter floors rotary inertia and forces near-rigid axial
    behaviour (``EA / EI ≈ 1e6``), leaving the dense FEM matrices
    ill-conditioned (κ(M) ≈ 1e11), which makes LAPACK's subset
    eigenvalue routines wobble on the lowest mode even when the
    underlying eigenvector is identical step to step. The MAC tracker
    catches this — the participation array stays > 98 % flap-dominant
    in the 1st-flap slot — so the mode *identity* is correct, only
    the eigenvalue precision suffers. Centrifugal stiffening is
    monotonic in physics (Wright 1982); endpoint-to-endpoint
    comparisons (parked vs rated) are reliable, individual-step
    monotonicity is not.

    Returns
    -------
    :class:`matplotlib.figure.Figure` for the rendered axes.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_campbell; install with "
            'pip install "pybmodes[plots]"'
        ) from exc

    if excitation_orders is None:
        excitation_orders = [1, 3, 6, 9]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
    else:
        fig = ax.figure

    # Family colours (issue #54 — engineering-report convention):
    # Blades green, Tower black, Platform red, Blade Passing blue.
    C_BLADE = (0.0, 0.62, 0.0)
    C_TOWER = (0.0, 0.0, 0.0)
    C_PLAT = (0.85, 0.0, 0.0)
    C_BP = (0.0, 0.0, 0.62)
    C_OPR = (0.85, 0.85, 0.85)        # operating-speed-range shade

    rpm = result.omega_rpm
    rpm_max = float(rpm.max()) if rpm.size > 0 else 0.0
    xmax = rpm_max if rpm_max > 0.0 else 1.0
    n_blade = result.n_blade_modes
    n_tower = result.n_tower_modes

    from pybmodes.fem.platform_modes import _PLATFORM_DOF_NAMES

    plat_name_set = set(_PLATFORM_DOF_NAMES)
    # Split tower columns into flexible-tower vs rigid-body platform.
    flex_modes: list[tuple[str, float]] = []
    plat_pairs: list[tuple[str, float]] = []
    for k in range(n_blade, n_blade + n_tower):
        f = float(result.frequencies[0, k])
        lbl = result.labels[k]
        if lbl in plat_name_set:
            plat_pairs.append((lbl, f))
        else:
            flex_modes.append((lbl.replace("tower ", ""), f))
    if platform_modes:
        plat_pairs += [(str(nm), float(fv)) for nm, fv in platform_modes]
    seen: set[str] = set()
    plat_clean: list[tuple[str, float]] = []
    for nm, f in plat_pairs:
        if nm in seen or not np.isfinite(f) or f <= 0.0:
            continue
        seen.add(nm)
        plat_clean.append((nm, f))
    # Merge near-degenerate symmetric pairs (surge≈sway, roll≈pitch)
    # within 2 % into one "a/b" label — matches the reference and
    # keeps a symmetric floater's labels from stacking.
    plat_clean.sort(key=lambda t: t[1])
    plat_merged: list[tuple[str, float]] = []
    for nm, f in plat_clean:
        if (plat_merged
                and abs(plat_merged[-1][1] - f) <= 0.02 * max(f, 1e-9)):
            pn, pf = plat_merged[-1]
            plat_merged[-1] = (f"{pn}/{nm}", 0.5 * (pf + f))
        else:
            plat_merged.append((nm, f))

    # Operating-speed-range shading: the *window itself* is grey,
    # outside stays white (behind everything).
    op: tuple[float, float] | None = None
    if operating_rpm is not None:
        lo, hi = sorted(float(v) for v in operating_rpm)
        lo, hi = max(0.0, lo), min(xmax, hi)
        if hi > lo:
            op = (lo, hi)
            ax.axvspan(lo, hi, color=C_OPR, lw=0, zorder=0)

    # Per-rev "Blade Passing" rays — uniform blue. Dense grid so the
    # rays render correctly on a log axis too.
    n_ray = 256
    if log_freq and rpm_max > 0.0:
        ray = np.linspace(rpm_max * 1.0e-3, rpm_max, n_ray)
    else:
        ray = np.linspace(0.0, xmax, n_ray)
    for order in excitation_orders:
        ax.plot(ray, order * ray / 60.0, "-", color=C_BP,
                linewidth=1.8, zorder=2)

    # Blade modes — uniform green curves ("Blades" family).
    for k in range(n_blade):
        ax.plot(rpm, result.frequencies[:, k], "-", color=C_BLADE,
                linewidth=1.8, zorder=3)

    # Flexible tower modes — black horizontal lines; platform modes —
    # red horizontal lines.
    for _nm, f in flex_modes:
        ax.axhline(f, color=C_TOWER, linewidth=1.6, zorder=2)
    # Platform modes cluster within a narrow low-frequency band, so
    # one red colour alone makes them indistinguishable — give each a
    # distinct line *style* (still the red family) so they can be told
    # apart and traced to their labels.
    _PLAT_LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, (_nm, f) in enumerate(plat_merged):
        ax.axhline(f, color=C_PLAT, linewidth=1.6,
                   linestyle=_PLAT_LS[i % len(_PLAT_LS)], zorder=2)

    # Rated-speed reference (thin; the operating-range band is the
    # primary speed annotation, so this stays out of the legend).
    if rated_rpm is not None:
        ax.axvline(float(rated_rpm), color=(0.4, 0.4, 0.4),
                   linestyle=":", linewidth=0.9, zorder=1)

    ax.set_xlabel("Rotor Speed [RPM]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Campbell Diagram")
    ax.set_xlim(0.0, xmax)
    if log_freq:
        cand = [float(v) for v in np.asarray(result.frequencies).ravel()
                if np.isfinite(v) and v > 0.0]
        cand += [f for _, f in plat_merged + flex_modes]
        floor = max(1.0e-4, 0.5 * min(cand)) if cand else 1.0e-3
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
    else:
        # Cap the axis just above the highest *structural* mode so the
        # modes of interest fill the figure; the steep per-rev rays
        # simply run off the top (standard Campbell-report framing).
        struct_max = 0.0
        if n_blade > 0:
            struct_max = float(np.nanmax(
                result.frequencies[:, :n_blade]))
        for _nm, f in flex_modes + plat_merged:
            struct_max = max(struct_max, f)
        if freq_max is not None:
            top = float(freq_max)
        elif struct_max > 0.0:
            top = 1.30 * struct_max     # headroom for the op-range bar
        else:
            top = None        # nothing structural — let matplotlib pick
        ax.set_ylim(0.0, top)
    ymin, ymax = ax.get_ylim()
    log_scale = log_freq and ymin > 0.0 and ymax > ymin

    def _to_frac(yv: float) -> float:
        if log_scale:
            return (np.log10(max(yv, ymin)) - np.log10(ymin)) / (
                np.log10(ymax) - np.log10(ymin))
        return (yv - ymin) / (ymax - ymin) if ymax > ymin else 0.0

    def _from_frac(fr: float) -> float:
        if log_scale:
            return float(10.0 ** (np.log10(ymin) + fr * (
                np.log10(ymax) - np.log10(ymin))))
        return ymin + fr * (ymax - ymin)

    # Operating-speed-range marker — set just below the top so it
    # clears the legend / frame.
    if op is not None:
        tr = ax.get_xaxis_transform()        # x = data, y = axes frac
        ax.annotate("", xy=(op[1], 0.92), xytext=(op[0], 0.92),
                    xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="<->", color="0.30",
                                    lw=1.3), zorder=5)
        ax.text(0.5 * (op[0] + op[1]), 0.93, "Operating Speed Range",
                transform=tr, ha="center", va="bottom", fontsize=9,
                color="0.20", zorder=5)

    # Inline nP tags along the blue rays, heights staggered so
    # successive orders don't collide. A white backing box keeps the
    # tag from sitting on top of (overlapping) its ray.
    no = len(excitation_orders)
    for i, order in enumerate(excitation_orders):
        ty = _from_frac(0.28 + 0.52 * (i / max(no - 1, 1)))
        tx = ty * 60.0 / order
        if tx > 0.90 * xmax:
            tx = 0.55 * xmax
            ty = order * tx / 60.0
        ax.text(tx, ty, f"{order}P", color=C_BP, fontsize=9,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.80))

    # Structural mode names written inline *along* their lines at
    # staggered x positions (the reference technique) rather than
    # stacked at the margin — a FOWT clusters several modes within a
    # narrow frequency band, so vertical-only de-overlap piles them
    # into one corner. Spreading the labels horizontally separates
    # freq-adjacent modes; a white backing keeps them readable where
    # they cross the per-rev rays. Only modes too close to the axis
    # floor are nudged up (with a thin leader to the true line).
    # Spell the engineering terms out for the figure (the terse
    # tokens in CampbellResult.labels stay as-is for CSV / API).
    _pretty_tok = {"flap": "flapwise", "edge": "edgewise",
                   "FA": "Fore-Aft", "SS": "Side-to-Side"}

    def _pretty(name: str) -> str:
        return " ".join(_pretty_tok.get(t, t) for t in name.split(" "))

    structural: list[tuple[str, float, tuple]] = []
    for k in range(n_blade):
        structural.append((_pretty(result.labels[k]),
                           float(result.frequencies[0, k]), C_BLADE))
    for _nm, f in flex_modes:
        structural.append((_pretty(_nm), f, C_TOWER))
    for _nm, f in plat_merged:
        structural.append((_pretty(_nm), f, C_PLAT))
    structural.sort(key=lambda e: e[1])
    # Interleaved x comb (well inside the axes, never at the y-axis
    # edge) so frequency-adjacent labels land far apart horizontally
    # and clear of the top-right legend.
    _comb = [0.07, 0.45, 0.24, 0.63, 0.36, 0.78, 0.15, 0.55]
    floor_fr = 0.05
    lifted = 0          # sub-floor modes stack up a low band, not pile
    for i, (nm, f, col) in enumerate(structural):
        xl = _comb[i % len(_comb)] * xmax
        tf = _to_frac(f)
        if tf < floor_fr:                       # too near the x-axis
            ty = _from_frac(floor_fr + 0.075 * lifted)
            lifted += 1
            ax.plot([xl, xl], [f, ty], color=col, linewidth=0.6,
                    alpha=0.45, zorder=2)        # thin leader, no head
        else:
            ty = f
        ax.text(
            xl, ty, f"{nm} ({f:.3g} Hz)", color=col, fontsize=8.5,
            ha="left", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.70),
        )

    # Four family keys only (those that are present).
    handles = []
    if n_blade > 0:
        handles.append(Line2D([], [], color=C_BLADE, lw=2.0,
                              label="Blades"))
    if flex_modes:
        handles.append(Line2D([], [], color=C_TOWER, lw=2.0,
                              label="Tower"))
    if plat_merged:
        handles.append(Line2D([], [], color=C_PLAT, lw=2.0,
                              label="Platform"))
    handles.append(Line2D([], [], color=C_BP, lw=2.0,
                          label="Blade Passing"))
    ax.legend(handles=handles, loc="upper left", frameon=True,
              fontsize=9)
    return fig


__all__ = ["CampbellResult", "campbell_sweep", "plot_campbell"]
