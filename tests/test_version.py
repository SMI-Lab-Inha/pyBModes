"""Release-hygiene guard for the package version.

``pybmodes.__version__`` reads the installed package metadata when
available and otherwise falls back to a literal in ``__init__.py`` (for
an uninstalled source tree). That literal silently drifted behind
``pyproject.toml`` once already, so this test pins the two together.

``CITATION.cff`` carries its own ``version:`` field, maintained by hand,
which had drifted several releases behind (1.10.0 while the package was
1.14.x). This test also pins the citation version to ``pyproject`` so a
release that forgets to bump it fails CI rather than shipping a stale
citation.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _init_fallback_version() -> str:
    src = (REPO_ROOT / "src" / "pybmodes" / "__init__.py").read_text(encoding="utf-8")
    # The fallback literal assigned inside the PackageNotFoundError branch.
    matches = re.findall(r'__version__\s*=\s*"([^"]+)"', src)
    assert matches, "no literal __version__ fallback found in __init__.py"
    return matches[-1]


def _citation_version() -> str:
    src = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    # The software ``version:`` field, distinct from the ``cff-version:``
    # schema field. Match a line that starts with ``version:`` exactly.
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', src, re.MULTILINE)
    assert m, "no software version: field found in CITATION.cff"
    return m.group(1)


def test_init_fallback_matches_pyproject() -> None:
    assert _init_fallback_version() == _pyproject_version()


def test_citation_version_matches_pyproject() -> None:
    assert _citation_version() == _pyproject_version()
