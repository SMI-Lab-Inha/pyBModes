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

"""ModalResult dataclass returned by RotatingBlade and Tower."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pybmodes.fem.normalize import NodeModeShape


@dataclass
class ModalResult:
    """Frequencies and mode shapes from a single FEM solve.

    Attributes
    ----------
    frequencies : (n_modes,) array of natural frequencies in Hz.
    shapes : list of :class:`NodeModeShape`, one per mode, ordered root
        to tip.
    participation : (n_modes, 3) array of energy fractions in the
        per-mode (flap or FA, edge or SS, torsion) axes — populated by
        downstream classification code such as
        :func:`pybmodes.campbell._participation`. Each row sums to 1.
        ``None`` when not yet computed; included in the saved archive
        only when set.
    fit_residuals : optional ``{block_name: rms_value}`` dict of
        polynomial-fit RMS residuals — populated by
        :func:`pybmodes.elastodyn.compute_tower_params` /
        ``compute_blade_params`` callers that want to embed the fit
        quality in the serialised result. ``None`` when not set.
    metadata : optional metadata dict (pyBmodes version, source file,
        save timestamp, git hash) attached automatically by
        :meth:`save` / :meth:`to_json` if not already populated.
    mode_labels : optional per-mode classification labels, one entry
        per mode (parallel to ``shapes`` / ``frequencies``). For a
        **floating** model (``hub_conn = 2`` with a
        :class:`~pybmodes.io.bmi.PlatformSupport`) the platform
        rigid-body modes are named ``"surge"`` / ``"sway"`` /
        ``"heave"`` / ``"roll"`` / ``"pitch"`` / ``"yaw"``; a mode the
        classifier can't confidently attribute to a single platform
        DOF (a flexible tower mode, or a strongly coupled / rotated
        pair) is left as ``None``. The whole list is ``None`` for a
        non-floating model (cantilever / monopile have no rigid-body
        modes to name). Added 1.3.0 — the **last** dataclass field, so
        the pre-1.3.0 positional constructor ABI is preserved (see the
        field-order note below); included in the saved archive only
        when set, like ``participation`` / ``fit_residuals``.
    """

    # NOTE: field order is the positional constructor ABI for this
    # semver-frozen 1.x public class. ``mode_labels`` (added 1.3.0)
    # MUST stay LAST — appended after ``metadata`` — so the historical
    # positional signature
    # ``ModalResult(frequencies, shapes, participation, fit_residuals,
    # metadata)`` is preserved. Inserting it before ``metadata`` would
    # silently bind a positional metadata dict to ``mode_labels`` for
    # existing callers (a backward-compat break in a minor release).
    # Any future optional field goes at the end too.
    frequencies: np.ndarray
    shapes: list[NodeModeShape]
    participation: np.ndarray | None = None
    fit_residuals: dict[str, float] | None = None
    metadata: dict[str, Any] | None = field(default=None)
    mode_labels: list[str | None] | None = None

    # ------------------------------------------------------------------
    # Shared integrity check
    # ------------------------------------------------------------------

    def _validate_lengths(self) -> None:
        """Raise if the parallel per-mode arrays disagree in length.

        Enforced by **both** serialisers (:meth:`save` and
        :meth:`to_json`) so neither can silently write a result whose
        per-mode arrays are inconsistent. The fully-empty case
        (``frequencies`` and ``shapes`` both empty — a failed-solve
        round-trip) is the only exemption.
        """
        farr = np.asarray(self.frequencies)
        if farr.ndim != 1:
            raise ValueError(
                f"frequencies must be a 1-D array; got ndim="
                f"{farr.ndim}, shape {farr.shape}"
            )
        n = int(farr.size)
        n_shapes = len(self.shapes)
        if n != n_shapes:
            raise ValueError(
                f"len(frequencies)={n} != len(shapes)={n_shapes}"
            )
        if self.mode_labels is not None and len(self.mode_labels) != n:
            raise ValueError(
                f"len(mode_labels)={len(self.mode_labels)} != "
                f"len(frequencies)={n}"
            )
        if self.participation is not None:
            part = np.asarray(self.participation)
            if part.ndim != 2 or part.shape[1] != 3:
                raise ValueError(
                    f"participation must be a 2-D (n_modes, 3) array "
                    f"(flap / lag / twist fractions); got shape "
                    f"{part.shape}"
                )
            n_part = int(part.shape[0])
            if n_part != n:
                raise ValueError(
                    f"len(participation)={n_part} != len(frequencies)={n}"
                )
            if not np.all(np.isfinite(part)):
                raise ValueError(
                    "participation contains non-finite (NaN / inf) "
                    "values"
                )
            # Documented energy fractions: each row sums to 1, or to 0
            # for a null mode shape (the documented zero-shape
            # sentinel). Negative entries / other row sums = corruption.
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
                    "participation rows must each sum to 1 (or 0 for "
                    "a null mode); got sums outside that set"
                )

        # Physical arrays must be finite — a scientific result with
        # NaN / inf frequencies or mode shapes must not be persisted.
        if n and not np.all(np.isfinite(farr)):
            raise ValueError("frequencies contains non-finite "
                             "(NaN / inf) values")
        for s in self.shapes:
            for nm in ("span_loc", "flap_disp", "flap_slope",
                       "lag_disp", "lag_slope", "twist"):
                if not np.all(np.isfinite(np.asarray(getattr(s, nm)))):
                    raise ValueError(
                        f"mode {s.mode_number}: {nm} contains "
                        f"non-finite (NaN / inf) values"
                    )

    def _validate_shared_span_for_npz(self) -> None:
        """Raise if the shapes cannot be represented by the NPZ layout.

        The NPZ archive stores one shared ``span_loc`` array and stacks
        all per-mode displacement arrays against it. JSON stores
        ``span_loc`` inside each shape, so this check intentionally
        applies only to the NPZ serializer / loader path.
        """
        if self.shapes:
            ref = np.asarray(self.shapes[0].span_loc, dtype=float)
            for s in self.shapes[1:]:
                sl = np.asarray(s.span_loc, dtype=float)
                if sl.shape != ref.shape or not np.array_equal(sl, ref):
                    raise ValueError(
                        f"mode {s.mode_number}: span_loc differs from "
                        f"mode {self.shapes[0].mode_number}'s grid — "
                        f"the NPZ format stores one shared span grid"
                    )

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def save(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the result to a ``.npz`` archive.

        ``source_file`` is recorded in the metadata when supplied
        (typically the BMI / ElastoDyn deck the solve came from).
        """
        from pybmodes.io._serialize import _capture_metadata, _metadata_to_npz_value

        self._validate_lengths()
        self._validate_shared_span_for_npz()
        path = pathlib.Path(path)
        if self.metadata is None:
            meta = _capture_metadata(source_file=source_file)
        else:
            meta = dict(self.metadata)

        if not self.shapes:
            # Allow an empty result (e.g. failed solve) to round-trip.
            shared_span = np.empty(0, dtype=float)
            stacked: dict[str, np.ndarray] = {
                name: np.empty((0, 0), dtype=float)
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            mode_numbers = np.empty(0, dtype=int)
        else:
            shared_span = np.asarray(self.shapes[0].span_loc, dtype=float)
            stacked = {
                name: np.stack([np.asarray(getattr(s, name), dtype=float)
                                for s in self.shapes])
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            mode_numbers = np.array([int(s.mode_number) for s in self.shapes])

        kwargs: dict[str, np.ndarray] = {
            "frequencies": np.asarray(self.frequencies, dtype=float),
            "mode_numbers": mode_numbers,
            "span_loc": shared_span,
            **stacked,
            "__meta__": _metadata_to_npz_value(meta),
        }
        if self.participation is not None:
            kwargs["participation"] = np.asarray(self.participation, dtype=float)
        if self.fit_residuals is not None:
            keys = list(self.fit_residuals.keys())
            # Unicode array (NOT object dtype) so the archive — like
            # ``__meta__`` and ``mode_labels`` — stays loadable with
            # ``allow_pickle=False``. Block names are always non-empty
            # ASCII identifiers, so a fixed-width string array is exact.
            kwargs["fit_residual_keys"] = np.array(keys, dtype=np.str_)
            kwargs["fit_residual_values"] = np.array(
                [float(self.fit_residuals[k]) for k in keys], dtype=float,
            )
        if self.mode_labels is not None:
            # Store as a fixed-width Unicode array (NOT object dtype) so
            # the archive stays loadable with ``allow_pickle=False``.
            # An unclassified mode (``None``) is written as the empty
            # string — a safe sentinel because a real label is always a
            # non-empty DOF name — and mapped back to ``None`` on load.
            kwargs["mode_labels"] = np.array(
                ["" if x is None else str(x) for x in self.mode_labels],
                dtype=np.str_,
            )
        # Length consistency was verified by ``_validate_lengths()`` at
        # the top of this method (shared with ``to_json``).
        # The numpy stub for savez_compressed mis-types **kwargs as a
        # positional ``bool`` first arg; the call is correct at runtime.
        np.savez_compressed(path, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ModalResult":
        """Read a result back from a ``.npz`` archive saved by
        :meth:`save`. The reconstructed instance is value-equal to the
        original modulo numpy dtype promotion."""
        from pybmodes.io._serialize import _read_npz_meta

        path = pathlib.Path(path)
        with np.load(path, allow_pickle=False) as npz:
            frequencies = np.asarray(npz["frequencies"], dtype=float)
            mode_numbers = np.asarray(npz["mode_numbers"], dtype=int)
            span_loc = np.asarray(npz["span_loc"], dtype=float)
            arrays = {
                name: np.asarray(npz[name], dtype=float)
                for name in ("flap_disp", "flap_slope", "lag_disp",
                             "lag_slope", "twist")
            }
            metadata = _read_npz_meta(npz, path)
            participation: np.ndarray | None = None
            if "participation" in npz.files:
                participation = np.asarray(npz["participation"], dtype=float)
            fit_residuals: dict[str, float] | None = None
            if "fit_residual_keys" in npz.files:
                keys = [str(k) for k in npz["fit_residual_keys"]]
                vals = [float(v) for v in npz["fit_residual_values"]]
                fit_residuals = dict(zip(keys, vals))
            mode_labels: list[str | None] | None = None
            if "mode_labels" in npz.files:
                # Empty-string sentinel → None (see ``save``); a
                # genuine label is never empty.
                mode_labels = [
                    None if not v else str(v)
                    for v in npz["mode_labels"].tolist()
                ]

        # Validate on ingest: a corrupt / hand-edited archive with
        # ragged per-mode arrays must fail with a clear message, not
        # an opaque IndexError mid-reconstruction.
        n_modes = int(mode_numbers.size)
        for nm, a in (("frequencies", frequencies),
                      *(arr for arr in arrays.items())):
            if int(np.asarray(a).shape[0]) != n_modes:
                raise ValueError(
                    f"corrupt archive: '{nm}' has "
                    f"{np.asarray(a).shape[0]} rows but mode_numbers "
                    f"has {n_modes}"
                )
        shapes = [
            NodeModeShape(
                mode_number=int(mode_numbers[i]),
                freq_hz=float(frequencies[i]),
                span_loc=span_loc,
                flap_disp=arrays["flap_disp"][i],
                flap_slope=arrays["flap_slope"][i],
                lag_disp=arrays["lag_disp"][i],
                lag_slope=arrays["lag_slope"][i],
                twist=arrays["twist"][i],
            )
            for i in range(n_modes)
        ]
        inst = cls(
            frequencies=frequencies,
            shapes=shapes,
            participation=participation,
            fit_residuals=fit_residuals,
            mode_labels=mode_labels,
            metadata=metadata,
        )
        # Validate on ingest, not only on export: a corrupt / hand-
        # edited archive must fail loudly at load(), not silently
        # downstream.
        inst._validate_lengths()
        return inst

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_json(
        self, path: str | pathlib.Path, *,
        source_file: str | pathlib.Path | None = None,
    ) -> None:
        """Write the result to a JSON file. Arrays are emitted as
        nested lists; metadata is embedded under ``"metadata"``."""
        from pybmodes.io._serialize import _capture_metadata

        self._validate_lengths()
        path = pathlib.Path(path)
        meta = self.metadata if self.metadata is not None else _capture_metadata(
            source_file=source_file,
        )
        payload: dict[str, Any] = {
            "schema_version": "1",
            "metadata": meta,
            "frequencies": [float(f) for f in self.frequencies],
            "shapes": [
                {
                    "mode_number": int(s.mode_number),
                    "freq_hz": float(s.freq_hz),
                    "span_loc": [float(x) for x in s.span_loc],
                    "flap_disp": [float(x) for x in s.flap_disp],
                    "flap_slope": [float(x) for x in s.flap_slope],
                    "lag_disp": [float(x) for x in s.lag_disp],
                    "lag_slope": [float(x) for x in s.lag_slope],
                    "twist": [float(x) for x in s.twist],
                }
                for s in self.shapes
            ],
            # NOTE: the inner ``[float(c) for c in row]`` and outer
            # ``[... for row in ...]`` both use square brackets, so
            # this is a *list of lists*, not a generator — it
            # serialises through ``json.dumps`` as a nested array of
            # numbers and round-trips cleanly through ``from_json``.
            "participation": (
                [[float(c) for c in row] for row in self.participation]
                if self.participation is not None else None
            ),
            "fit_residuals": (
                {k: float(v) for k, v in self.fit_residuals.items()}
                if self.fit_residuals is not None else None
            ),
            "mode_labels": (
                [None if x is None else str(x) for x in self.mode_labels]
                if self.mode_labels is not None else None
            ),
        }
        # No ``default=str``: every value above is constructed as a
        # JSON-native type (float / int / str / list / dict / None) and
        # the metadata dict from ``_capture_metadata`` is likewise all
        # str / None. A non-native object reaching here is a regression
        # — let ``json.dumps`` raise ``TypeError`` loudly rather than
        # silently stringifying it into an un-round-trippable blob.
        # ``allow_nan=False``: emit standards-compliant JSON or raise —
        # never write the non-standard ``NaN`` / ``Infinity`` literals
        # strict parsers reject (``_validate_lengths`` above already
        # rejects non-finite physical arrays; this is the last guard).
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "ModalResult":
        """Read a result back from a JSON file saved by
        :meth:`to_json`."""
        path = pathlib.Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        shapes = [
            NodeModeShape(
                mode_number=int(s["mode_number"]),
                freq_hz=float(s["freq_hz"]),
                span_loc=np.asarray(s["span_loc"], dtype=float),
                flap_disp=np.asarray(s["flap_disp"], dtype=float),
                flap_slope=np.asarray(s["flap_slope"], dtype=float),
                lag_disp=np.asarray(s["lag_disp"], dtype=float),
                lag_slope=np.asarray(s["lag_slope"], dtype=float),
                twist=np.asarray(s["twist"], dtype=float),
            )
            for s in payload["shapes"]
        ]
        participation = (
            np.asarray(payload["participation"], dtype=float)
            if payload.get("participation") is not None else None
        )
        fit_residuals = (
            {k: float(v) for k, v in payload["fit_residuals"].items()}
            if payload.get("fit_residuals") is not None else None
        )
        mode_labels = (
            [None if x is None else str(x) for x in payload["mode_labels"]]
            if payload.get("mode_labels") is not None else None
        )
        inst = cls(
            frequencies=np.asarray(payload["frequencies"], dtype=float),
            shapes=shapes,
            participation=participation,
            fit_residuals=fit_residuals,
            mode_labels=mode_labels,
            metadata=payload.get("metadata"),
        )
        inst._validate_lengths()       # validate on ingest, not only export
        return inst
