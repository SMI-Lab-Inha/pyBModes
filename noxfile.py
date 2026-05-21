"""Nox sessions — common developer / CI loops.

Run a single session:  ``nox -s lint``
List all sessions:     ``nox -l``
Run the default set:   ``nox`` (lint + type + tests + docs)
"""
from __future__ import annotations

import nox

nox.options.sessions = ["lint", "type", "tests", "docs"]
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the self-contained pytest suite (matches CI default)."""
    session.install("-e", ".[dev,plots,notebook]")
    session.run("pytest", "--tb=short", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def integration(session: nox.Session) -> None:
    """Run integration-marked tests. Needs upstream data under ``external/``."""
    session.install("-e", ".[dev,plots,notebook]")
    session.run("pytest", "--tb=short", "-m", "integration", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """``ruff check`` over the same scope CI lints."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "tests/", "scripts/")


@nox.session
def type(session: nox.Session) -> None:
    """``mypy`` over the public package (matches CI)."""
    session.install("-e", ".[dev]")
    session.install("mypy")
    session.run("mypy", "src/pybmodes")


@nox.session
def docs(session: nox.Session) -> None:
    """Build the Sphinx site (non-strict — matches Read the Docs + CI).

    Strict mode (``-W --keep-going``) is intentionally off while
    there is a backlog of pre-existing docstring-formatting warnings.
    Run ``nox -s docs -- -W --keep-going`` to opt back in locally.
    """
    session.install("-e", ".[docs,plots,windio]")
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "docs",
        "docs/_build/html",
        *session.posargs,
    )


@nox.session
def build(session: nox.Session) -> None:
    """Build the sdist + wheel. Use for release dry-runs."""
    session.install("--upgrade", "pip", "build")
    session.run("python", "-m", "build")


@nox.session
def audit_validation(session: nox.Session) -> None:
    """Mechanical claims-vs-tests audit (matches CI; release-checklist step 4.5)."""
    session.run("python", "scripts/audit_validation_claims.py", external=True)
