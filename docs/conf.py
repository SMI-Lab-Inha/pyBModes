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
    "sphinx_rtd_theme",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    # ``dollarmath`` enables ``$inline$`` and ``$$block$$`` math in the
    # included root-level Markdown (CHANGELOG / VALIDATION). Without
    # it the dollar signs render as literal text. ``amsmath`` adds
    # the ``\begin{align}...\end{align}`` family for multi-line
    # derivations.
    "dollarmath",
    "amsmath",
    # ``linkify`` was previously enabled but turned bare text like
    # ``README.md`` or ``build.py`` into ``http://README.md`` URLs
    # that 404 from the docs site. We don't need auto-link conversion
    # — every URL in the source uses explicit ``http(s)://`` already.
]
myst_heading_anchors = 3
# The Sphinx tree itself is reStructuredText (flat ``docs/*.rst``); MyST
# stays registered only so the top-level ``README.md`` / ``CHANGELOG.md``
# / ``VALIDATION.md`` / ``CONTRIBUTING.md`` can be pulled in via
# ``.. include::``. Don't add new ``.md`` files to ``docs/``.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# We deliberately do NOT suppress ``myst.xref_missing`` any more — the
# previous policy hid genuinely broken links (relative ``tests/foo.py``
# paths in the included CHANGELOG / VALIDATION that resolve on GitHub
# but 404 from the docs site). Those links are now rewritten to
# absolute GitHub URLs in the source files, so any new occurrence is
# a real defect the build should surface.

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render numpy-style ``Attributes`` sections as ``:ivar:`` directives
# inside the class body instead of as a sibling field-list. The
# default rendering produces a second autodoc object description per
# attribute, colliding with autoclass's own per-attribute description
# and emitting ``WARNING: duplicate object description ...`` — see
# napoleon docs on ``napoleon_use_ivar``.
napoleon_use_ivar = True
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Read the Docs theme (replaces furo, which was failing to provision on
# the docs builder). Declared in the ``docs`` extra; registered via the
# entry point, so ``html_theme = "sphinx_rtd_theme"`` is enough.
html_theme = "sphinx_rtd_theme"
html_title = f"pyBmodes {release}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
