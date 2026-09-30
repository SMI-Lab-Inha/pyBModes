"""Shared numerical helpers, and the NumPy-version guard they exist for.

``pyproject.toml`` declares ``numpy>=1.26``. ``np.trapezoid`` only exists
from NumPy 2.0 and ``np.trapz`` is deprecated there, so any direct call
breaks one end of the supported range. CI installs a current NumPy and
would not notice, hence the source scan below.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from pybmodes._numeric import trapezoid

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pybmodes"


def test_linear_integrand_is_exact_on_a_nonuniform_grid():
    x = np.array([0.0, 0.3, 1.1, 2.0, 4.5])
    assert trapezoid(3.0 * x + 2.0, x) == pytest.approx(
        1.5 * 4.5 ** 2 + 2.0 * 4.5, rel=1e-14)


def test_single_point_integrates_to_zero():
    assert trapezoid([5.0], [1.0]) == 0.0


def test_returns_a_python_float():
    assert type(trapezoid([1, 2, 3], [0, 1, 2])) is float


def test_no_version_specific_numpy_integrators_in_the_package():
    pattern = re.compile(r"\bnp\.(trapezoid|trapz)\s*\(")
    offenders = [
        f"{path.relative_to(SRC)}:{n}"
        for path in SRC.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "use pybmodes._numeric.trapezoid instead (numpy>=1.26 support): "
        + ", ".join(offenders)
    )
