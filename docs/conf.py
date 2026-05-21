"""Sphinx configuration for the pyBmodes documentation site.

Built on Read the Docs (see ``.readthedocs.yaml``). MyST is enabled so
the existing top-level Markdown source — README, CHANGELOG, VALIDATION
— can be included directly via ``{include}`` directives and stays in
lockstep with the documentation pages that re-render them.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "pyBmodes"
author = "Jae Hoon Seo"
copyright = "2024-2026, Jae Hoon Seo, Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University"

try:
    release = importlib.metadata.version("pybmodes")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
# The Sphinx tree itself is reStructuredText (flat ``docs/*.rst``); MyST
# stays registered only so the top-level ``README.md`` / ``CHANGELOG.md``
# / ``VALIDATION.md`` / ``CONTRIBUTING.md`` can be pulled in via
# ``.. include::``. Don't add new ``.md`` files to ``docs/``.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Included Markdown files (CHANGELOG, VALIDATION, CONTRIBUTING) carry
# repo-relative paths like ``tests/test_campbell.py`` that GitHub
# renders as links but Sphinx tries to resolve as in-docset cross-
# references. They are intentionally GitHub-rendered links — suppress
# the cross-reference warnings rather than rewrite the source files.
suppress_warnings = ["myst.xref_missing"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "furo"
html_title = f"pyBmodes {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/SMI-Lab-Inha/pyBModes/",
    "source_branch": "master",
    "source_directory": "docs/",
}
