"""Regression tests for the 1.14.0 engineering-hardening pass.

Covers the WindIO structured-discovery parse (item 4) and the report
completeness stamp + ignored-physics surfacing (items 2 / 5). The
fail-closed run path (item 1), solver diagnostics (item 3) and the BMI
ParseError layer (item 6) are pinned in test_checks / test_solver_
diagnostics / test_parser_negative_paths respectively.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pybmodes.fem.normalize import NodeModeShape
from pybmodes.models.result import ModalResult

# ---------------------------------------------------------------------------
# Item 4 — WindIO discovery via structured YAML parse, not text scan
# ---------------------------------------------------------------------------

def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_windio_doc_accepts_real_ontology(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.workflows.windio import _doc_is_floating, _load_windio_doc

    p = _write(tmp_path / "ok.yaml", (
        "name: demo\n"
        "components:\n"
        "  tower:\n"
        "    outer_shape: {}\n"
    ))
    doc = _load_windio_doc(p)
    assert isinstance(doc, dict) and "components" in doc
    assert _doc_is_floating(doc) is False


def test_load_windio_doc_detects_floating(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.workflows.windio import _doc_is_floating, _load_windio_doc

    p = _write(tmp_path / "fowt.yaml", (
        "name: demo\n"
        "components:\n"
        "  tower: {}\n"
        "  floating_platform:\n"
        "    members: []\n"
    ))
    assert _doc_is_floating(_load_windio_doc(p)) is True


def test_load_windio_doc_rejects_non_ontology(tmp_path: pathlib.Path) -> None:
    """A yaml that merely mentions the word 'components' but is not a
    WindIO ontology (no top-level components mapping) is rejected — the
    old substring scan would have wrongly accepted it."""
    pytest.importorskip("yaml")
    from pybmodes.workflows.windio import _load_windio_doc

    # 'components' appears only as free text, not a mapping key.
    p = _write(tmp_path / "notes.yaml", (
        "title: just notes about components and floating_platform ideas\n"
        "items: [a, b, c]\n"
    ))
    assert _load_windio_doc(p) is None


def test_load_windio_doc_rejects_malformed_yaml(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("yaml")
    from pybmodes.workflows.windio import _load_windio_doc

    p = _write(tmp_path / "broken.yaml", "key: [unterminated\n")
    assert _load_windio_doc(p) is None


# ---------------------------------------------------------------------------
# Items 2 / 5 — report completeness stamp + ignored-physics row
# ---------------------------------------------------------------------------

def _one_mode_result(ignored: tuple[str, ...] = ()) -> ModalResult:
    span = np.linspace(0.0, 1.0, 4)
    shape = NodeModeShape(
        mode_number=1, freq_hz=0.5, span_loc=span,
        flap_disp=span, flap_slope=np.zeros(4),
        lag_disp=np.zeros(4), lag_slope=np.zeros(4), twist=np.zeros(4),
    )
    return ModalResult(
        frequencies=np.array([0.5]), shapes=[shape], ignored_physics=ignored,
    )


def test_report_stamps_status(tmp_path: pathlib.Path) -> None:
    from pybmodes.report import generate_report

    out = generate_report(
        _one_mode_result(), tmp_path / "r.md", format="md", status="screening",
    )
    text = out.read_text(encoding="utf-8")
    assert "Report status" in text
    assert "screening" in text


def test_report_omits_status_when_none(tmp_path: pathlib.Path) -> None:
    from pybmodes.report import generate_report

    out = generate_report(_one_mode_result(), tmp_path / "r.md", format="md")
    assert "Report status" not in out.read_text(encoding="utf-8")


def test_report_surfaces_ignored_physics(tmp_path: pathlib.Path) -> None:
    from pybmodes.report import generate_report

    out = generate_report(
        _one_mode_result(ignored=("distributed added mass (distr_m)",)),
        tmp_path / "r.md", format="md",
    )
    text = out.read_text(encoding="utf-8")
    assert "Ignored physics" in text
    assert "distr_m" in text
