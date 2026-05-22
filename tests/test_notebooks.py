"""Headless execution of the bundled walkthrough notebooks.

Three notebooks are exercised:

- ``notebooks/walkthrough.ipynb`` — contractually self-contained tour
  of the public API (synthetic uniform-blade and tower cases with
  closed-form references). The README's "start here" introduction.
  Runs in the **default** suite — every PR exercises it end-to-end.

- ``cases/iea15_umainesemi_walkthrough.ipynb`` — the IEA-15-240-RWT
  UMaineSemi end-to-end walkthrough. Depends on upstream OpenFAST
  decks gitignored under the Independence stance. Two assertions:

  * Default-suite: with the upstream data **absent**, the first code
    cell must raise a *friendly* ``FileNotFoundError`` whose message
    points at the upstream IEA-15-240-RWT repo. This is the
    documented contract; without a test we'd be trusting the design,
    not verifying it.
  * Integration-marked: with the upstream data **present**, every
    cell must execute without error. Skipped on a fresh clone so the
    default suite stays self-contained.

- ``cases/iea15_volturnus_windio_walkthrough.ipynb`` — the WISDEM/
  WindIO one-click FOWT walkthrough. Same friendly-error +
  integration contract as the UMaineSemi notebook, keyed on the
  upstream WindIO ontology ``.yaml`` (and needing the ``[windio]``
  extra, PyYAML).

Requires the optional ``[notebook]`` extra (``nbclient`` /
``nbformat`` / ``ipykernel``); the test skips at collection time if
any are missing so a contributor running with only ``[dev]`` sees a
clean ``SKIPPED`` rather than an import-time error.
"""

from __future__ import annotations

import pathlib

import pytest

nbformat = pytest.importorskip(
    "nbformat", reason="install the [notebook] extra to run this test",
)
nbclient = pytest.importorskip(
    "nbclient", reason="install the [notebook] extra to run this test",
)
# ipykernel is a runtime dep of nbclient's default kernel manager —
# importorskip catches the case where it's missing too.
pytest.importorskip(
    "ipykernel", reason="install the [notebook] extra to run this test",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WALKTHROUGH = REPO_ROOT / "notebooks" / "walkthrough.ipynb"
IEA15_WALKTHROUGH = REPO_ROOT / "cases" / "iea15_umainesemi_walkthrough.ipynb"
IEA15_DECK_DIR = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-15-240-RWT" / "OpenFAST"
    / "IEA-15-240-RWT-UMaineSemi"
)

if not WALKTHROUGH.is_file():
    pytest.skip(
        f"walkthrough notebook not found at {WALKTHROUGH}; "
        "this normally indicates a checkout problem rather than a real skip",
        allow_module_level=True,
    )


def _execute_notebook(path: pathlib.Path, *, timeout: int = 120):
    """Run every cell of ``path`` through nbclient. Returns the
    executed notebook (with outputs attached); re-raises
    :class:`nbclient.exceptions.CellExecutionError` from the first
    failing cell so callers can inspect the message."""
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    return nb


def test_walkthrough_notebook_executes_without_errors() -> None:
    """Execute every code cell in ``notebooks/walkthrough.ipynb`` and
    assert none raise.

    The notebook is synthetic-only (uniform blade + tower with
    closed-form references) so it has no upstream-data dependency
    and runs in a few seconds. ``nbclient`` shells out to an
    ``ipykernel`` Python kernel which is fully isolated from pytest's
    process — a failed cell shows up as a ``CellExecutionError``
    raised from ``client.execute()``.
    """
    from nbclient.exceptions import CellExecutionError

    try:
        _execute_notebook(WALKTHROUGH)
    except CellExecutionError as err:
        # nbclient pretty-prints the kernel traceback inline.
        pytest.fail(f"walkthrough.ipynb failed to execute end-to-end:\n{err}")


# ---------------------------------------------------------------------------
# IEA-15 UMaineSemi walkthrough — friendly-error and integration paths
# ---------------------------------------------------------------------------

if not IEA15_WALKTHROUGH.is_file():
    pytest.skip(
        f"IEA-15 UMaineSemi walkthrough not found at {IEA15_WALKTHROUGH}; "
        "this normally indicates a checkout problem rather than a real skip",
        allow_module_level=False,
    )


def test_iea15_walkthrough_friendly_error_when_data_absent() -> None:
    """When the upstream OpenFAST decks are NOT present under
    ``external/OpenFAST_files/IEA-15-240-RWT/...``, the notebook's first
    code cell must raise a ``FileNotFoundError`` whose message names
    the upstream IEA-15-240-RWT repository. This is the documented
    contract that lets the notebook ship under ``cases/`` (which is
    allowed to depend on gitignored data) without leaving a fresh-
    clone user with a cryptic traceback.

    Skipped on machines where the upstream data IS present — the
    integration-marked counterpart runs there instead.
    """
    if IEA15_DECK_DIR.is_dir():
        pytest.skip(
            f"upstream IEA-15 decks present at {IEA15_DECK_DIR}; "
            "see test_iea15_walkthrough_executes_when_data_present for "
            "the end-to-end path"
        )
    from nbclient.exceptions import CellExecutionError

    with pytest.raises(CellExecutionError) as exc_info:
        _execute_notebook(IEA15_WALKTHROUGH, timeout=60)
    # The kernel traceback is in ``str(exc_info.value)``. Assert two
    # things: the error TYPE inside the kernel was FileNotFoundError
    # (not some random NameError or import failure), and the message
    # carries the documented "Clone the upstream IEA-15-240-RWT
    # GitHub repo" hint.
    err_text = str(exc_info.value)
    assert "FileNotFoundError" in err_text, (
        f"expected a FileNotFoundError on the first code cell; got:\n{err_text}"
    )
    assert "IEA-15-240-RWT" in err_text and "Clone" in err_text, (
        "expected the friendly 'Clone the upstream IEA-15-240-RWT' hint; "
        f"got:\n{err_text}"
    )


@pytest.mark.integration
def test_iea15_walkthrough_executes_when_data_present() -> None:
    """When the upstream OpenFAST decks ARE present, every cell of
    ``cases/iea15_umainesemi_walkthrough.ipynb`` must execute without
    error. Skipped on a fresh clone (no upstream data) so the
    default suite stays self-contained.
    """
    if not IEA15_DECK_DIR.is_dir():
        pytest.skip(
            f"upstream IEA-15 decks not present at {IEA15_DECK_DIR}; "
            "clone the IEA-15-240-RWT repo there to run this test"
        )
    from nbclient.exceptions import CellExecutionError

    try:
        # Higher timeout — the coupled solve + Campbell sweep adds a
        # few seconds per cell and the FEM matrices are larger.
        _execute_notebook(IEA15_WALKTHROUGH, timeout=300)
    except CellExecutionError as err:
        pytest.fail(
            f"iea15_umainesemi_walkthrough.ipynb failed end-to-end:\n{err}"
        )


# ---------------------------------------------------------------------------
# IEA-15 VolturnUS-S WindIO walkthrough — friendly-error and integration paths
# ---------------------------------------------------------------------------

VOLTURNUS_WALKTHROUGH = (
    REPO_ROOT / "cases" / "iea15_volturnus_windio_walkthrough.ipynb"
)
VOLTURNUS_YAML = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-15-240-RWT"
    / "WT_Ontology" / "IEA-15-240-RWT_VolturnUS-S.yaml"
)

if not VOLTURNUS_WALKTHROUGH.is_file():
    pytest.skip(
        f"IEA-15 VolturnUS-S WindIO walkthrough not found at "
        f"{VOLTURNUS_WALKTHROUGH}; this normally indicates a checkout problem",
        allow_module_level=False,
    )


def test_volturnus_windio_walkthrough_friendly_error_when_data_absent() -> None:
    """When the upstream WindIO ontology is NOT present under
    ``external/OpenFAST_files/IEA-15-240-RWT/WT_Ontology/``, the
    notebook's first code cell must raise a ``FileNotFoundError`` whose
    message names the upstream IEA-15-240-RWT repository — the same
    documented contract as the UMaineSemi notebook, so this one can also
    ship under ``cases/`` without leaving a fresh-clone user with a
    cryptic traceback.

    Skipped on machines where the upstream data IS present.
    """
    if VOLTURNUS_YAML.is_file():
        pytest.skip(
            f"upstream VolturnUS-S ontology present at {VOLTURNUS_YAML}; "
            "see test_volturnus_windio_walkthrough_executes_when_data_present"
        )
    from nbclient.exceptions import CellExecutionError

    with pytest.raises(CellExecutionError) as exc_info:
        _execute_notebook(VOLTURNUS_WALKTHROUGH, timeout=60)
    err_text = str(exc_info.value)
    assert "FileNotFoundError" in err_text, (
        f"expected a FileNotFoundError on the first code cell; got:\n{err_text}"
    )
    assert "IEA-15-240-RWT" in err_text and "Clone" in err_text, (
        "expected the friendly 'Clone the upstream IEA-15-240-RWT' hint; "
        f"got:\n{err_text}"
    )


@pytest.mark.integration
def test_volturnus_windio_walkthrough_executes_when_data_present() -> None:
    """When the upstream WindIO ontology + companion decks ARE present,
    every cell of ``cases/iea15_volturnus_windio_walkthrough.ipynb`` must
    execute without error. Needs the ``[windio]`` extra (PyYAML).
    Skipped on a fresh clone (no upstream data) so the default suite
    stays self-contained.
    """
    if not VOLTURNUS_YAML.is_file():
        pytest.skip(
            f"upstream VolturnUS-S ontology not present at {VOLTURNUS_YAML}; "
            "clone the IEA-15-240-RWT repo there to run this test"
        )
    pytest.importorskip(
        "yaml", reason="install the [windio] extra (PyYAML) to run this test",
    )
    from nbclient.exceptions import CellExecutionError

    try:
        _execute_notebook(VOLTURNUS_WALKTHROUGH, timeout=300)
    except CellExecutionError as err:
        pytest.fail(
            f"iea15_volturnus_windio_walkthrough.ipynb failed end-to-end:\n{err}"
        )
