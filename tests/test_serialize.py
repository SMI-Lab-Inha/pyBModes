"""Round-trip tests for ModalResult / CampbellResult serialisation.

Three named tests come straight from the spec
(``test_modal_result_round_trip_npz``,
``test_modal_result_round_trip_json``, ``test_campbell_csv_columns``)
plus a few support tests that gate metadata capture and the
``CampbellResult.save → load`` round-trip.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest

from pybmodes.campbell import CampbellResult
from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models.result import ModalResult

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_modal_result(n_modes: int = 3, n_nodes: int = 7) -> ModalResult:
    """Build a small in-memory ``ModalResult`` with deterministic values
    so round-trip equality checks are stable."""
    rng = np.random.default_rng(42)
    span = np.linspace(0.0, 1.0, n_nodes)
    shapes = [
        NodeModeShape(
            mode_number=k + 1,
            freq_hz=0.5 * (k + 1) ** 1.5,
            span_loc=span,
            flap_disp=rng.standard_normal(n_nodes),
            flap_slope=rng.standard_normal(n_nodes),
            lag_disp=rng.standard_normal(n_nodes),
            lag_slope=rng.standard_normal(n_nodes),
            twist=rng.standard_normal(n_nodes),
        )
        for k in range(n_modes)
    ]
    frequencies = np.array([s.freq_hz for s in shapes])
    participation = rng.uniform(0.0, 1.0, size=(n_modes, 3))
    participation /= participation.sum(axis=1, keepdims=True)
    fit_residuals = {
        "TwFAM1Sh": 1.0e-5,
        "TwFAM2Sh": 2.3e-3,
    }
    return ModalResult(
        frequencies=frequencies,
        shapes=shapes,
        participation=participation,
        fit_residuals=fit_residuals,
    )


def _make_campbell_result(n_steps: int = 5, n_modes: int = 4) -> CampbellResult:
    rng = np.random.default_rng(7)
    omega = np.linspace(0.0, 12.0, n_steps)
    freqs = rng.uniform(0.3, 5.0, size=(n_steps, n_modes))
    parts = rng.uniform(0.0, 1.0, size=(n_steps, n_modes, 3))
    parts /= parts.sum(axis=-1, keepdims=True)
    mac = np.full((n_steps, n_modes), np.nan)
    mac[1:, :min(2, n_modes)] = rng.uniform(
        0.95, 1.0, size=(n_steps - 1, min(2, n_modes))
    )
    # Keep the fixture *self-consistent* for any n_modes (labels and
    # the blade/tower split derived from n_modes) — an inconsistent
    # CampbellResult is exactly what CampbellResult._validate now
    # rejects, so the fixture must not fabricate one.
    n_blade = n_modes // 2
    n_tower = n_modes - n_blade
    labels = (
        [f"blade {i + 1}" for i in range(n_blade)]
        + [f"tower {i + 1}" for i in range(n_tower)]
    )
    return CampbellResult(
        omega_rpm=omega,
        frequencies=freqs,
        labels=labels,
        participation=parts,
        n_blade_modes=n_blade,
        n_tower_modes=n_tower,
        mac_to_previous=mac,
    )


# ---------------------------------------------------------------------------
# Modal result — npz round trip (spec-named)
# ---------------------------------------------------------------------------

def _shapes_equal(a: list[NodeModeShape], b: list[NodeModeShape]) -> bool:
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa.mode_number != sb.mode_number:
            return False
        if not np.isclose(sa.freq_hz, sb.freq_hz):
            return False
        for attr in ("span_loc", "flap_disp", "flap_slope",
                     "lag_disp", "lag_slope", "twist"):
            if not np.allclose(getattr(sa, attr), getattr(sb, attr)):
                return False
    return True


def test_modal_result_round_trip_npz(tmp_path: pathlib.Path) -> None:
    """ModalResult.save -> ModalResult.load yields a value-equal record
    including frequencies, mode shapes, participation, fit residuals,
    and metadata."""
    result = _make_modal_result()
    out = tmp_path / "modes.npz"
    result.save(out, source_file="dummy.bmi")
    assert out.is_file()

    loaded = ModalResult.load(out)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    assert _shapes_equal(loaded.shapes, result.shapes)
    assert loaded.participation is not None
    np.testing.assert_allclose(loaded.participation, result.participation)
    assert loaded.fit_residuals == result.fit_residuals
    # Metadata block populated automatically and round-trippable.
    assert loaded.metadata is not None
    assert "pybmodes_version" in loaded.metadata
    assert "timestamp" in loaded.metadata
    assert loaded.metadata["source_file"] == "dummy.bmi"


def test_modal_result_round_trip_json(tmp_path: pathlib.Path) -> None:
    """ModalResult.to_json -> ModalResult.from_json yields a value-equal
    record. The JSON file is human-readable and includes the schema
    version key."""
    result = _make_modal_result()
    out = tmp_path / "modes.json"
    result.to_json(out, source_file="another.bmi")
    assert out.is_file()
    # Sanity: file is valid JSON with the expected top-level keys.
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert "metadata" in payload and "frequencies" in payload
    assert "shapes" in payload and len(payload["shapes"]) == len(result.shapes)
    # Round-trip equality.
    loaded = ModalResult.from_json(out)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    assert _shapes_equal(loaded.shapes, result.shapes)
    assert loaded.participation is not None
    np.testing.assert_allclose(loaded.participation, result.participation)
    assert loaded.fit_residuals == result.fit_residuals


def test_modal_result_metadata_capture(tmp_path: pathlib.Path) -> None:
    """Default metadata grab populates pybmodes_version + timestamp +
    (when present) git_hash."""
    result = _make_modal_result()
    out = tmp_path / "modes.npz"
    result.save(out)
    loaded = ModalResult.load(out)
    meta = loaded.metadata
    assert meta is not None
    assert isinstance(meta["pybmodes_version"], str)
    assert meta["pybmodes_version"]  # not empty
    assert "timestamp" in meta
    # git_hash is best-effort; either None or a hex-ish short SHA.
    if meta.get("git_hash") is not None:
        assert isinstance(meta["git_hash"], str)


def test_modal_result_no_participation_or_residuals_omitted(
    tmp_path: pathlib.Path,
) -> None:
    """When participation / fit_residuals are unset, they round-trip as
    None — i.e. saved archives don't have to carry them."""
    result = _make_modal_result()
    result.participation = None
    result.fit_residuals = None
    out = tmp_path / "modes.npz"
    result.save(out)
    loaded = ModalResult.load(out)
    assert loaded.participation is None
    assert loaded.fit_residuals is None


def test_modal_result_empty_result_round_trip_npz(tmp_path: pathlib.Path) -> None:
    """An empty solve result should save/load without inventing shapes."""
    result = ModalResult(frequencies=np.empty(0), shapes=[])
    out = tmp_path / "empty.npz"
    result.save(out)

    loaded = ModalResult.load(out)
    assert loaded.frequencies.shape == (0,)
    assert loaded.shapes == []
    assert loaded.metadata is not None


def test_modal_result_positional_constructor_abi() -> None:
    """ModalResult is semver-frozen public API. The pre-1.3.0
    positional signature
    ``ModalResult(frequencies, shapes, participation, fit_residuals,
    metadata)`` must keep working: ``metadata`` must land in
    ``.metadata`` (not be shifted into the 1.3.0-added
    ``mode_labels``). Pins the field order against an accidental
    reinsertion of a new field before ``metadata``."""
    freqs = np.array([1.0, 2.0])
    shapes: list = []
    meta = {"pybmodes_version": "x", "source_file": None}

    r = ModalResult(freqs, shapes, None, None, meta)  # positional
    assert r.metadata == meta
    assert r.mode_labels is None
    # frequencies/shapes empty/empty so this is a valid round-trip
    r2 = ModalResult(np.empty(0), [], None, None, meta)
    assert r2.metadata == meta and r2.mode_labels is None


def test_modal_result_save_rejects_frequency_shape_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    """Saving catches accidental truncation between frequencies and shapes."""
    result = _make_modal_result(n_modes=2)
    result.frequencies = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=r"len\(frequencies\)=3 != len\(shapes\)=2"):
        result.save(tmp_path / "bad.npz")


def test_modal_result_save_rejects_frequencies_without_shapes(
    tmp_path: pathlib.Path,
) -> None:
    """Frequencies-but-no-shapes is corrupt and must raise — the old
    ``if mode_numbers.size and …`` guard skipped the check entirely
    for this case (only fully-empty/empty is a valid round-trip)."""
    result = ModalResult(frequencies=np.array([1.0]), shapes=[])
    with pytest.raises(ValueError, match=r"len\(frequencies\)=1 != len\(shapes\)=0"):
        result.save(tmp_path / "bad.npz")


def test_modal_result_to_json_enforces_length_checks(
    tmp_path: pathlib.Path,
) -> None:
    """to_json must mirror save's integrity check so a mismatched
    result can't be JSON-serialised and reloaded inconsistent."""
    r1 = _make_modal_result(n_modes=2)
    r1.frequencies = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"len\(frequencies\)=3 != len\(shapes\)=2"):
        r1.to_json(tmp_path / "bad.json")

    r2 = _make_modal_result(n_modes=2)
    r2.mode_labels = ["surge"]            # one short
    with pytest.raises(ValueError, match=r"len\(mode_labels\)=1"):
        r2.to_json(tmp_path / "bad2.json")


def test_modal_result_npz_loads_without_pickle(tmp_path: pathlib.Path) -> None:
    """Every archive member is a Unicode / numeric array (no object
    dtype), so the .npz is loadable with ``allow_pickle=False`` — the
    invariant ``__meta__`` / ``fit_residual_keys`` / ``mode_labels``
    all now hold to."""
    result = _make_modal_result(n_modes=3)
    result.fit_residuals = {"TwFAM1Sh": 0.001, "TwSSM1Sh": 0.002}
    result.mode_labels = ["surge", None, "yaw"]
    out = tmp_path / "nopickle.npz"
    result.save(out)

    with np.load(out, allow_pickle=False) as npz:
        assert set(npz.files) >= {
            "frequencies", "__meta__", "fit_residual_keys", "mode_labels",
        }
    loaded = ModalResult.load(out)
    assert loaded.mode_labels == ["surge", None, "yaw"]
    assert loaded.fit_residuals == {"TwFAM1Sh": 0.001, "TwSSM1Sh": 0.002}


# ---------------------------------------------------------------------------
# Campbell result — CSV columns (spec-named) + npz round-trip
# ---------------------------------------------------------------------------

def test_campbell_csv_columns(tmp_path: pathlib.Path) -> None:
    """CampbellResult.to_csv emits a header with 'rpm', one column per
    mode label, and one '<label>_mac' column per mode label. Data
    rows match the result arrays element-for-element."""
    result = _make_campbell_result()
    out = tmp_path / "campbell.csv"
    result.to_csv(out)
    assert out.is_file()

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, *data = rows

    # Expected column ordering.
    expected_freq_cols = result.labels
    expected_mac_cols = [f"{lbl}_mac" for lbl in result.labels]
    assert header[0] == "rpm"
    assert header[1:1 + len(expected_freq_cols)] == expected_freq_cols
    assert header[1 + len(expected_freq_cols):] == expected_mac_cols

    # Row count matches the sweep length.
    assert len(data) == result.omega_rpm.size

    # Spot-check one row's numerical content.
    row = data[2]
    assert float(row[0]) == pytest.approx(result.omega_rpm[2])
    n_modes = result.frequencies.shape[1]
    for k in range(n_modes):
        assert float(row[1 + k]) == pytest.approx(result.frequencies[2, k])


def test_campbell_save_load_round_trip(tmp_path: pathlib.Path) -> None:
    """CampbellResult.save -> CampbellResult.load yields a value-equal
    record (so the save method is honest about being round-trippable)."""
    result = _make_campbell_result()
    out = tmp_path / "campbell.npz"
    result.save(out, source_file="dummy.dat")
    loaded = CampbellResult.load(out)
    np.testing.assert_allclose(loaded.omega_rpm, result.omega_rpm)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)
    np.testing.assert_allclose(loaded.participation, result.participation)
    # MAC has NaN entries; use equal_nan for the round-trip check.
    np.testing.assert_allclose(
        loaded.mac_to_previous, result.mac_to_previous, equal_nan=True,
    )
    assert loaded.labels == result.labels
    assert loaded.n_blade_modes == result.n_blade_modes
    assert loaded.n_tower_modes == result.n_tower_modes


def test_campbell_csv_uses_nan_mac_when_shape_missing(tmp_path: pathlib.Path) -> None:
    """Older CampbellResult-like objects without MAC data still write stable columns."""
    result = _make_campbell_result(n_steps=2, n_modes=2)
    result.mac_to_previous = np.empty((0, 0))
    out = tmp_path / "campbell_no_mac.csv"
    result.to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["rpm", *result.labels, *[f"{lbl}_mac" for lbl in result.labels]]
    assert rows[1][-2:] == ["nan", "nan"]
    assert rows[2][-2:] == ["nan", "nan"]


# ===========================================================================
# NPZ metadata round-trips without allow_pickle
# ===========================================================================

def test_npz_metadata_loads_without_pickle() -> None:
    """The new ``dtype=np.str_`` form for ``__meta__`` loads cleanly
    under ``np.load(..., allow_pickle=False)`` — closes the
    docstring-vs-implementation drift the previous ``dtype=object``
    introduced."""
    import tempfile

    shapes = [NodeModeShape(
        mode_number=1, freq_hz=0.5,
        span_loc=np.linspace(0, 1, 5),
        flap_disp=np.linspace(0, 1, 5), flap_slope=np.zeros(5),
        lag_disp=np.zeros(5), lag_slope=np.zeros(5), twist=np.zeros(5),
    )]
    r = ModalResult(
        frequencies=np.array([0.5]),
        shapes=shapes,
    )
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "r.npz"
        r.save(p, source_file="dummy.bmi")
        # The load WITHOUT allow_pickle proves the metadata isn't
        # pickle-backed. Just opening the archive used to require
        # allow_pickle=True under the old dtype=object regime.
        with np.load(p, allow_pickle=False) as npz:
            meta_arr = npz["__meta__"]
            # ``kind == "U"`` is unicode fixed-length, NOT object.
            assert meta_arr.dtype.kind == "U", (
                f"__meta__ should be a unicode string array; got "
                f"dtype={meta_arr.dtype}"
            )
            meta = json.loads(str(meta_arr))
            assert "pybmodes_version" in meta
            assert "timestamp" in meta
            assert meta["source_file"] == "dummy.bmi"


# ===========================================================================
# F2 — allow_pickle hardening: modern path is pickle-free + silent;
# only a legacy dtype=object __meta__ takes the warned fallback.
# ===========================================================================

def _forge_legacy_object_meta(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Rewrite ``src`` into ``dst`` with ``__meta__`` downgraded to the
    pre-1.0 ``dtype=object`` form (every other array unchanged)."""
    with np.load(src, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files if k != "__meta__"}
        meta_json = str(npz["__meta__"])
    data["__meta__"] = np.array(meta_json, dtype=object)
    np.savez_compressed(dst, **data)


def test_modal_result_modern_npz_load_is_silent(
    tmp_path: pathlib.Path,
) -> None:
    """A freshly-saved (pickle-free) archive must NOT trip the legacy
    fallback warning — the common path never touches pickle."""
    import warnings

    result = _make_modal_result(n_modes=3)
    out = tmp_path / "modern.npz"
    result.save(out)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any warn → fail
        loaded = ModalResult.load(out)
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)


def test_modal_result_legacy_object_meta_refused_then_opt_in(
    tmp_path: pathlib.Path,
) -> None:
    """A pre-1.0 archive whose ``__meta__`` is a pickled object array is
    **refused by default** (object-array unpickling can execute code). It
    loads only when the caller explicitly opts in via
    ``allow_legacy_pickle=True``, and even then only with a
    ``UserWarning`` — pickle is never enabled silently."""
    result = _make_modal_result(n_modes=2)
    modern = tmp_path / "modern.npz"
    result.save(modern, source_file="legacy.bmi")
    legacy = tmp_path / "legacy.npz"
    _forge_legacy_object_meta(modern, legacy)

    with (
        pytest.raises(ValueError, match="Object arrays cannot be loaded"),
        np.load(legacy, allow_pickle=False) as z,
    ):
        _ = z["__meta__"]                   # confirms the forge worked

    # Default: refuse, never reach for pickle.
    with pytest.raises(ValueError, match="refusing to load a legacy"):
        ModalResult.load(legacy)

    # Explicit opt-in: warns and loads.
    with pytest.warns(UserWarning, match="legacy pre-1.0 .npz"):
        loaded = ModalResult.load(legacy, allow_legacy_pickle=True)
    assert loaded.metadata is not None
    assert loaded.metadata["source_file"] == "legacy.bmi"
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)


def test_campbell_legacy_object_meta_refused_then_opt_in(
    tmp_path: pathlib.Path,
) -> None:
    """Same refuse-by-default + explicit opt-in contract for
    ``CampbellResult``."""
    result = _make_campbell_result(n_steps=4, n_modes=4)
    modern = tmp_path / "c_modern.npz"
    result.save(modern)
    legacy = tmp_path / "c_legacy.npz"
    _forge_legacy_object_meta(modern, legacy)

    with pytest.raises(ValueError, match="refusing to load a legacy"):
        CampbellResult.load(legacy)

    with pytest.warns(UserWarning, match="legacy pre-1.0 .npz"):
        loaded = CampbellResult.load(legacy, allow_legacy_pickle=True)
    assert loaded.labels == result.labels
    np.testing.assert_allclose(loaded.frequencies, result.frequencies)


# ===========================================================================
# F5 — dataclass schema guards before any export
# ===========================================================================

def test_modal_result_rejects_bad_participation_shape(
    tmp_path: pathlib.Path,
) -> None:
    """participation must be (n_modes, 3); a wrong second dim is caught
    before save / to_json can emit an un-round-trippable archive."""
    result = _make_modal_result(n_modes=3)
    result.participation = np.zeros((3, 2))          # 2 cols, not 3
    with pytest.raises(ValueError, match=r"participation must be a 2-D"):
        result.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match=r"participation must be a 2-D"):
        result.to_json(tmp_path / "bad.json")


def test_campbell_validate_rejects_inconsistent_shapes(
    tmp_path: pathlib.Path,
) -> None:
    """CampbellResult._validate fires before save / to_csv so a
    malformed sweep can't be written."""
    result = _make_campbell_result(n_steps=5, n_modes=4)
    result.labels = ["a", "b", "c"]                  # 3 != n_modes=4
    with pytest.raises(ValueError, match=r"len\(labels\)=3 != n_modes=4"):
        result.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match=r"len\(labels\)=3 != n_modes=4"):
        result.to_csv(tmp_path / "bad.csv")

    bad_part = _make_campbell_result(n_steps=5, n_modes=4)
    bad_part.participation = np.zeros((5, 4))         # missing the 3-axis
    with pytest.raises(ValueError, match=r"participation shape"):
        bad_part.save(tmp_path / "bad2.npz")


# ===========================================================================
# Validate-on-ingest: a corrupt archive / JSON must fail loudly at
# load(), not silently downstream (review High #1 / #2)
# ===========================================================================

def test_modal_result_load_validates_corrupt_npz(
    tmp_path: pathlib.Path,
) -> None:
    """A hand-corrupted .npz (frequencies/shape-count mismatch) must
    raise at load(), not return an inconsistent object."""
    result = _make_modal_result(n_modes=3)
    good = tmp_path / "ok.npz"
    result.save(good)
    with np.load(good, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["frequencies"] = np.asarray(data["frequencies"])[:-1]  # drop one
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError,
                       match=r"corrupt archive|len\(frequencies\)"):
        ModalResult.load(bad)


def test_modal_result_from_json_validates_corrupt_payload(
    tmp_path: pathlib.Path,
) -> None:
    result = _make_modal_result(n_modes=3)
    jp = tmp_path / "r.json"
    result.to_json(jp)
    payload = json.loads(jp.read_text(encoding="utf-8"))
    payload["frequencies"] = payload["frequencies"][:-1]   # corrupt
    jp.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"len\(frequencies\)"):
        ModalResult.from_json(jp)


def test_campbell_load_validates_corrupt_npz(
    tmp_path: pathlib.Path,
) -> None:
    result = _make_campbell_result(n_steps=5, n_modes=4)
    good = tmp_path / "c.npz"
    result.save(good)
    with np.load(good, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["frequencies"] = np.asarray(data["frequencies"])[:, :-1]  # drop col
    bad = tmp_path / "cbad.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError):
        CampbellResult.load(bad)


def test_validate_lengths_rejects_2d_frequencies() -> None:
    """A 2-D frequencies array with the same total size as
    len(shapes) used to pass the size-only check (review Medium #5)."""
    result = _make_modal_result(n_modes=4)
    result.frequencies = np.asarray(result.frequencies).reshape(2, 2)
    with pytest.raises(ValueError, match=r"frequencies must be a 1-D"):
        result._validate_lengths()


# ===========================================================================
# Round-2 review: non-finite scientific data + shared-span + strict JSON
# ===========================================================================

def test_modal_result_rejects_non_finite_physical_arrays(
    tmp_path: pathlib.Path,
) -> None:
    """NaN / inf in frequencies or a mode-shape array must not be
    persisted (review High A)."""
    r = _make_modal_result(n_modes=3)
    r.frequencies = np.asarray(r.frequencies, dtype=float).copy()
    r.frequencies[1] = np.inf
    with pytest.raises(ValueError, match="frequencies contains non-finite"):
        r.save(tmp_path / "a.npz")

    r2 = _make_modal_result(n_modes=3)
    r2.shapes[0].flap_disp = np.asarray(
        r2.shapes[0].flap_disp, dtype=float).copy()
    r2.shapes[0].flap_disp[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        r2.to_json(tmp_path / "a.json")


def test_modal_result_to_json_is_standards_compliant(
    tmp_path: pathlib.Path,
) -> None:
    """to_json must never emit the non-standard NaN/Infinity literals
    (review High B). The finite guard fires first; allow_nan=False is
    the belt-and-braces last line."""
    r = _make_modal_result(n_modes=2)
    jp = tmp_path / "ok.json"
    r.to_json(jp)
    txt = jp.read_text(encoding="utf-8")
    assert "NaN" not in txt and "Infinity" not in txt
    json.loads(txt)  # strict parser accepts it


def test_modal_result_rejects_mismatched_span_grids(
    tmp_path: pathlib.Path,
) -> None:
    """Equal-length but different span grids must raise — the NPZ
    format stores one shared grid (review Medium C)."""
    r = _make_modal_result(n_modes=2, n_nodes=7)
    r.shapes[1].span_loc = np.asarray(
        r.shapes[1].span_loc, dtype=float) + 0.01    # same length, shifted
    with pytest.raises(ValueError, match="span_loc differs"):
        r.save(tmp_path / "g.npz")


def test_modal_result_json_allows_per_mode_span_grids(
    tmp_path: pathlib.Path,
) -> None:
    """JSON stores span_loc inside each shape, unlike the NPZ layout
    with one shared grid, so finite per-mode grids must round-trip."""
    r = _make_modal_result(n_modes=2, n_nodes=7)
    shifted = np.asarray(r.shapes[1].span_loc, dtype=float) + 0.01
    r.shapes[1].span_loc = shifted

    jp = tmp_path / "mixed-span.json"
    r.to_json(jp)
    loaded = ModalResult.from_json(jp)

    np.testing.assert_allclose(loaded.shapes[0].span_loc, r.shapes[0].span_loc)
    np.testing.assert_allclose(loaded.shapes[1].span_loc, shifted)


def test_campbell_rejects_non_finite_physical_arrays(
    tmp_path: pathlib.Path,
) -> None:
    """NaN / inf in frequencies / omega_rpm / participation must not
    be persisted; mac_to_previous NaN stays the documented sentinel."""
    c = _make_campbell_result(n_steps=5, n_modes=4)
    c.frequencies = np.asarray(c.frequencies, dtype=float).copy()
    c.frequencies[2, 1] = np.nan
    with pytest.raises(ValueError, match="frequencies contains non-finite"):
        c.save(tmp_path / "c.npz")
    # All-NaN mac_to_previous is still fine (it is the sentinel).
    ok = _make_campbell_result(n_steps=3, n_modes=2)
    ok.mac_to_previous = np.full((3, 2), np.nan)
    ok.save(tmp_path / "ok.npz")


def test_io_reexports_read_out_and_error() -> None:
    """README advertises ``read_out(path, strict=True)`` prominently;
    it (and its error) must be importable from ``pybmodes.io``
    (review Low H)."""
    from pybmodes.io import BModeOutParseError, read_out

    assert callable(read_out)
    assert issubclass(BModeOutParseError, ValueError)


# ===========================================================================
# Round-3 review: tightened CampbellResult / participation invariants
# ===========================================================================

def test_campbell_validate_one_uniform_contract_no_empty_exemption() -> None:
    """The empty-sweep special-case branch was removed: ``_validate``
    is now a single uniform per-array shape contract. The canonical
    empty sweep (``frequencies (0,0)``, ``omega_rpm (0,)``,
    ``participation (0,0,3)``, ``mac (0,0)``, no labels/counts)
    satisfies it *vacuously*; every malformed zero-size variant fails
    a normal per-array check. This closes the whole recurring class of
    finding (missing arrays / ``.size`` vs ``.shape``) by design, not
    by another patch."""

    def _empty(**override):
        base = dict(
            omega_rpm=np.empty(0), frequencies=np.empty((0, 0)),
            labels=[], participation=np.empty((0, 0, 3)),
            n_blade_modes=0, n_tower_modes=0,
            mac_to_previous=np.empty((0, 0)),
        )
        base.update(override)
        return CampbellResult(**base)

    # Canonical empty sweep — passes the uniform contract, no raise.
    _empty()._validate()

    # frequencies must be 2-D (the single shape gate; a 1-D (0,) is
    # no longer special-cased).
    with pytest.raises(ValueError, match="frequencies must be 2-D"):
        _empty(frequencies=np.empty(0))._validate()

    # Every zero-*size* but wrong-*shape* / stray-row variant is now
    # caught by the ordinary per-array check it violates — one
    # message per offending array, not a catch-all exemption.
    with pytest.raises(ValueError, match=r"omega_rpm shape"):
        _empty(omega_rpm=np.array([0.0, 6.0]))._validate()
    with pytest.raises(ValueError, match=r"omega_rpm shape"):
        _empty(omega_rpm=np.empty((0, 2)))._validate()
    with pytest.raises(ValueError, match=r"participation shape"):
        _empty(participation=np.empty((2, 0, 3)))._validate()
    with pytest.raises(ValueError, match=r"mac_to_previous shape"):
        _empty(mac_to_previous=np.full((2, 2), np.nan))._validate()
    with pytest.raises(ValueError, match=r"mac_to_previous shape"):
        _empty(mac_to_previous=np.empty((2, 0)))._validate()
    with pytest.raises(ValueError, match=r"len\(labels\)"):
        _empty(labels=["x"])._validate()

    # A (0,3) frequencies array (size 0, implies 3 modes) is rejected
    # because the *other* arrays no longer match its derived
    # (n_steps=0, n_modes=3) — exactly the desired behaviour.
    c = _make_campbell_result(n_steps=4, n_modes=4)
    c.frequencies = np.empty((0, 3), dtype=float)
    with pytest.raises(ValueError):
        c._validate()


def test_campbell_rejects_negative_mode_counts() -> None:
    c = _make_campbell_result(n_steps=5, n_modes=4)
    c.n_blade_modes, c.n_tower_modes = -1, 5     # sums to 4 but invalid
    with pytest.raises(ValueError, match="mode counts must be non-negative"):
        c._validate()


def test_campbell_mac_allows_nan_rejects_inf() -> None:
    c = _make_campbell_result(n_steps=4, n_modes=4)
    c.mac_to_previous = np.full((4, 4), np.nan)
    c._validate()                                 # NaN sentinel OK
    c.mac_to_previous[1, 1] = np.inf
    with pytest.raises(ValueError, match="mac_to_previous contains inf"):
        c._validate()


def test_participation_must_be_nonneg_and_row_sum_1_or_0() -> None:
    # CampbellResult
    c = _make_campbell_result(n_steps=3, n_modes=2)
    c.participation = c.participation.copy()
    c.participation[0, 0, 0] = -0.1               # negative fraction
    with pytest.raises(ValueError, match="participation contains negative"):
        c._validate()
    c2 = _make_campbell_result(n_steps=3, n_modes=2)
    c2.participation = c2.participation.copy()
    c2.participation[0, 0] = [0.5, 0.5, 0.5]      # row sums to 1.5
    with pytest.raises(ValueError, match="participation rows must each sum"):
        c2._validate()
    # A null-mode all-zero row is the documented sentinel — allowed.
    c3 = _make_campbell_result(n_steps=2, n_modes=2)
    c3.participation = c3.participation.copy()
    c3.participation[0, 0] = [0.0, 0.0, 0.0]
    c3._validate()

    # ModalResult mirrors the rule.
    r = _make_modal_result(n_modes=3)
    r.participation = np.asarray(r.participation, dtype=float).copy()
    r.participation[1] = [0.7, 0.7, 0.7]          # row sums to 2.1
    with pytest.raises(ValueError, match="participation rows must each sum"):
        r._validate_lengths()
    r2 = _make_modal_result(n_modes=3)
    r2.participation = np.asarray(r2.participation, dtype=float).copy()
    r2.participation[0, 0] = -0.2
    with pytest.raises(ValueError, match="participation contains negative"):
        r2._validate_lengths()
