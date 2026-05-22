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

"""Internal serialisation helpers shared by ``ModalResult`` and
``CampbellResult``.

Two output formats are supported:

* ``.npz`` — NumPy's compressed multi-array format. Arrays are stored
  as ``ndarray`` keys; scalar / dict metadata is stored as a single
  ``__meta__`` key holding a JSON-serialised dict (kept as a 0-d
  string array so ``np.load`` returns it without `allow_pickle`).
* ``.json`` — UTF-8 JSON with arrays serialised as nested lists. Used
  for ``ModalResult.to_json`` / ``from_json``.

A small ``_capture_metadata`` helper grabs the pyBmodes version, the
current UTC timestamp, the source file path (when supplied), and the
git HEAD hash of the working directory (best-effort, silently None
when ``git`` isn't installed or the cwd isn't a repo).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
from typing import Any

import numpy as np


def _capture_metadata(source_file: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Build a metadata dict for embedding in a saved ``ModalResult`` or
    ``CampbellResult``.

    Fields:

    * ``pybmodes_version`` — :data:`pybmodes.__version__` at save time.
    * ``timestamp`` — UTC ISO-8601 timestamp at save time.
    * ``source_file`` — string form of ``source_file`` when supplied,
      otherwise ``None``.
    * ``git_hash`` — short SHA of the current ``git`` HEAD if the cwd
      is a git repo and ``git`` is available; otherwise ``None``.

    The function never raises; missing pieces become ``None``.
    """
    from pybmodes import __version__ as pybmodes_version

    meta: dict[str, Any] = {
        "pybmodes_version": pybmodes_version,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source_file": str(source_file) if source_file is not None else None,
        "git_hash": _try_git_hash(),
    }
    return meta


def _try_git_hash() -> str | None:
    """Return the short git HEAD hash for the cwd, or ``None`` on any
    failure path (no git binary, cwd not a repo, subprocess timeout)."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None
    out = cp.stdout.strip()
    return out or None


def _metadata_to_npz_value(meta: dict[str, Any]) -> np.ndarray:
    """Pack a metadata dict into a 0-d Unicode-string array suitable
    for storing as the ``__meta__`` key of an ``.npz`` archive.

    The dtype is ``np.str_`` (fixed-length unicode) rather than
    ``object`` — that's deliberate. The previous ``dtype=object``
    pickled the array contents, so loading the file back required
    ``allow_pickle=True``, which contradicted the module docstring's
    promise that metadata is "kept loadable without pickle".

    ``ModalResult.load`` / ``CampbellResult.load`` open archives with
    ``allow_pickle=False`` (the safe default). A legacy ``dtype=object``
    ``__meta__`` is **refused by default** — object-array unpickling can
    execute arbitrary code — and is only read when the caller passes
    ``allow_legacy_pickle=True``, and then only with a ``UserWarning``
    (see :func:`_read_npz_meta`). The change is forward-only and the
    common path never enables pickle."""
    return np.array(json.dumps(meta, default=str), dtype=np.str_)


def _metadata_from_npz_value(arr: np.ndarray) -> dict[str, Any]:
    """Unpack the inverse of :func:`_metadata_to_npz_value`. ``arr``
    is typically a 0-d array as returned by ``np.load``. Handles both
    the new ``dtype=np.str_`` form and the legacy ``dtype=object``
    form left by older saves."""
    raw = arr.item() if hasattr(arr, "item") else str(arr)
    return dict(json.loads(raw))


def _read_npz_meta(
    npz: Any, path: pathlib.Path, *, allow_legacy_pickle: bool = False,
) -> dict[str, Any]:
    """Return the parsed ``__meta__`` dict from an ``.npz`` opened with
    ``allow_pickle=False``.

    Every array this codebase writes — including ``__meta__`` (a
    pickle-free ``np.str_`` scalar) — loads fine under
    ``allow_pickle=False``. Only archives written by very old pyBmodes
    versions stored ``__meta__`` as a ``dtype=object`` array, which
    NumPy refuses to materialise without pickle.

    **Loading such a legacy file is refused by default.** Unpickling an
    object array can execute arbitrary code, and ``SECURITY.md`` puts
    NPZ deserialisation in scope, so the safe default is to raise rather
    than silently reach for pickle. Pass ``allow_legacy_pickle=True``
    (surfaced as the same keyword on ``ModalResult.load`` /
    ``CampbellResult.load``) to opt in for a file you trust; that path
    reopens with ``allow_pickle=True`` for the metadata only and emits a
    ``UserWarning`` so it is never silent. The modern path never enables
    pickle at all.
    """
    import warnings

    try:
        raw = npz["__meta__"]
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        if not allow_legacy_pickle:
            raise ValueError(
                f"{path}: refusing to load a legacy pre-1.0 .npz whose "
                f"__meta__ is a pickled object array. Unpickling an object "
                f"array can execute arbitrary code, so pyBmodes will not do "
                f"it implicitly. If you trust this file's origin, re-call "
                f"load(..., allow_legacy_pickle=True); otherwise re-save it "
                f"with a current pyBmodes version for a pickle-free archive."
            ) from exc
        warnings.warn(
            f"{path}: legacy pre-1.0 .npz whose __meta__ is a pickled "
            f"object array — loading metadata with allow_pickle=True "
            f"because allow_legacy_pickle=True was passed. Re-save with "
            f"this pyBmodes version to get a fully pickle-free archive.",
            UserWarning,
            stacklevel=3,
        )
        with np.load(path, allow_pickle=True) as legacy:
            return _metadata_from_npz_value(legacy["__meta__"])
    return _metadata_from_npz_value(raw)
