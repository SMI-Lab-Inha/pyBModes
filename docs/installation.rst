Installation
============

``pybmodes`` requires **Python ≥ 3.11**. The runtime depends only
on ``numpy`` and ``scipy``; everything optional — plotting,
notebook execution, WindIO ingestion, this documentation site —
is gated behind an extra so the default install footprint stays
small.

From PyPI
---------

.. code-block:: bash

   pip install pybmodes

This pulls in the runtime dependencies (``numpy>=1.26``,
``scipy>=1.11``) and exposes:

- The Python package: ``import pybmodes``
- The CLI: ``pybmodes`` on ``PATH`` (seven subcommands — see
  :doc:`quickstart`).
- The bundled example library (vendored as package-data,
  reachable via ``pybmodes examples --copy <dir>``).

To pin a specific release in a requirements file or a
``pyproject.toml``:

.. code-block:: text

   pybmodes ==1.9.0       # exact pin
   pybmodes ~=1.9         # latest 1.9.x, blocks 2.x
   pybmodes >=1.9,<2      # 1.9+ but never a major bump

From conda-forge
----------------

``pybmodes`` is also packaged on `conda-forge
<https://anaconda.org/conda-forge/pybmodes>`_:

.. code-block:: bash

   conda install -c conda-forge pybmodes

This pulls the runtime core (``numpy``, ``scipy``) and the
``pybmodes`` CLI, exactly like the PyPI wheel. The pip-style
extras (``[plots]``, ``[windio]``, ...) are a pip concept and have
no conda equivalent, so for the optional features add the matching
conda-forge packages alongside it, ``matplotlib`` for
``pybmodes.plots`` and ``pyyaml`` for the WindIO path.

.. code-block:: bash

   conda install -c conda-forge pybmodes matplotlib pyyaml

From source (editable)
----------------------

For contributors and anyone tracking ``master``:

.. code-block:: bash

   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"

``-e`` installs in editable mode — changes to ``src/pybmodes/*.py``
take effect on the next import without re-installing. End users who
don't need the test/lint extras can install the runtime core alone
with ``pip install .``.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Extra
     - Pulls in
     - When you need it
   * - ``[dev]``
     - ``pytest``, ``pytest-cov``, ``ruff``, ``mypy``, ``pyyaml``
     - Run the test suite, lint, and type-check the package.
       Pulled into editable installs by default; not needed for
       end users.
   * - ``[plots]``
     - ``matplotlib>=3.7``
     - Every plotting helper (``pybmodes.plots``): Campbell, MAC,
       mode-shape, fit-quality, and environmental-loading
       figures. ``pybmodes.plots.apply_style()`` applies the
       project's standard engineering-paper palette.
   * - ``[windio]``
     - ``pyyaml>=6``
     - ``Tower.from_windio(...)``, ``RotatingBlade.from_windio(...)``,
       ``Tower.from_windio_floating(...)``, and the
       ``pybmodes windio`` one-click CLI.
   * - ``[notebook]``
     - ``nbclient``, ``nbformat``, ``ipykernel``, ``matplotlib``
     - Headless execution of bundled walkthrough notebooks under
       :file:`tests/test_notebooks.py`. Test-only — not imported
       by ``pybmodes`` itself.
   * - ``[docs]``
     - ``sphinx<9``, ``sphinx-rtd-theme``, ``myst-parser``,
       ``sphinx-copybutton``
     - Build this documentation site locally.

Combine extras with commas:

.. code-block:: bash

   pip install -e ".[dev,plots,windio]"

Windows + conda quickstart
--------------------------

The lowest-friction path on Windows. The user-facing maintainer
runs this exact sequence:

.. code-block:: bat

   :: 1. install Miniconda or Anaconda first if you don't have it.
   ::    https://docs.conda.io/en/latest/miniconda.html
   ::    Open "Anaconda Prompt" from the Start menu (not regular
   ::    CMD or PowerShell -- the Anaconda Prompt has `conda`
   ::    already on PATH).

   :: 2. create and activate a dedicated env
   conda create -n pybmodes python=3.11 -y
   conda activate pybmodes

   :: 3. clone and install in editable mode with dev + plotting extras
   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"

   :: 4. verify the install
   pytest

.. note::

   Don't try to invoke the conda env's ``python.exe`` directly
   from PowerShell — it errors with
   ``STATUS_DLL_INIT_FAILED`` because the env relies on conda's
   ``PATH`` manipulations. Use Anaconda Prompt, or wrap the call
   in ``cmd /c "call activate.bat pybmodes && python ..."``.

Updating to a new release
-------------------------

New versions are published to PyPI (see the `releases
<https://github.com/SMI-Lab-Inha/pyBModes/releases>`_ and the
:doc:`changelog`). To upgrade an existing PyPI install to the latest
release, run

.. code-block:: bash

   pip install --upgrade pybmodes        # add the same extras you use,
                                         # e.g. -U "pybmodes[plots,windio]"

Check the installed version, and pin one if you need reproducibility.

.. code-block:: bash

   python -c "import pybmodes; print(pybmodes.__version__)"
   pip install "pybmodes==1.14.1"        # install or pin a specific release

For a source checkout, pull and reinstall so new dependencies are
picked up too.

.. code-block:: bash

   git pull
   pip install -e ".[dev,plots]"

Inside a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyBmodes is on conda-forge, so inside a conda env you can install and
upgrade it with conda directly.

.. code-block:: bash

   conda install -c conda-forge pybmodes   # install
   conda update  -c conda-forge pybmodes   # upgrade
   python -c "import pybmodes; print(pybmodes.__version__)"

.. note::

   The conda-forge package ships the runtime core only. For the
   optional features add the matching conda-forge packages
   (``matplotlib`` for ``pybmodes.plots``, ``pyyaml`` for the WindIO
   path), since the pip-style extras have no conda equivalent.

If you installed pyBmodes with ``pip`` inside a conda env instead of
from conda-forge, keep upgrading it with ``pip``, not ``conda update``,
so the two package managers don't both try to manage it. Activate the
env first, then upgrade with pip.

.. code-block:: bash

   conda activate pybmodes               # the env you installed it into
   pip install --upgrade pybmodes        # or -U "pybmodes[plots,windio]"
   python -c "import pybmodes; print(pybmodes.__version__)"

.. note::

   A ``pip``-installed pyBmodes won't respond to ``conda update
   pybmodes``, because conda doesn't track a pip-installed package. If
   you're unsure which env has it, ``conda env list`` shows every env and
   ``pip show pybmodes`` confirms the version installed in the currently
   active one. For a source checkout living in a conda env, ``git pull``
   then re-run the editable ``pip install -e ".[dev,plots]"`` with the env
   active.

Verifying the install
---------------------

After installing, run the **self-contained** test suite — every
test that doesn't need external data:

.. code-block:: bash

   python -c "import pybmodes; print(pybmodes.__version__)"
   pytest

A fresh clone or a fresh PyPI install both pass this with no
external data on the filesystem. Tests that need upstream
OpenFAST / BModes data are gated behind the ``integration``
marker; see :doc:`data_sources` for what to clone and where.

To run the full suite including the integration track once the
upstream data is staged under ``external/``:

.. code-block:: bash

   pytest -m integration

CI runs both steps on every PR. The integration step tolerates
``pytest`` exit code 5 ("no tests collected") so the job stays
green on a runner without the data, but fails on any other
non-zero exit so a custom workflow run that *does* have the data
surfaces real failures immediately.

IDE setup
---------

VS Code
^^^^^^^

Recommended workspace settings (``./.vscode/settings.json``):

.. code-block:: json

   {
     "python.analysis.typeCheckingMode": "basic",
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["tests", "--no-cov"],
     "[python]": {
       "editor.formatOnSave": false,
       "editor.codeActionsOnSave": {"source.organizeImports": "explicit"}
     },
     "ruff.lint.select": ["E", "F", "W", "I"]
   }

The ``--no-cov`` flag in ``pytestArgs`` is a quality-of-life
choice — coverage reports clutter the Test Explorer output.

PyCharm
^^^^^^^

- **Interpreter**: point at the ``pybmodes`` conda env.
- **Test runner**: ``Settings → Tools → Python Integrated Tools →
  Default test runner → pytest``.
- **Ruff plugin**: install ``Ruff`` from the marketplace; the
  project's ``pyproject.toml`` carries the rules.

Common errors
-------------

``ModuleNotFoundError: No module named 'pybmodes'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You ran ``python`` from outside the install's environment, **or**
you cloned the source but didn't ``pip install -e .``. From the
repo root either install editably or set ``PYTHONPATH``:

.. code-block:: bash

   # one-time install
   pip install -e .

   # or one-off invocation
   PYTHONPATH=src python -c "import pybmodes"

``UserWarning: matplotlib is required for plot_campbell``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``[plots]`` extra wasn't pulled in. Add it:

.. code-block:: bash

   pip install "pybmodes[plots]"

``KeyError: 'floating_platform'`` (WindIO yaml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You're calling ``Tower.from_windio_floating(...)`` on a yaml that
lacks a ``components.floating_platform`` block — a land-based or
monopile-only ontology. Use ``Tower.from_windio(...)`` instead,
or supply a ``floating_platform``-bearing yaml.

``FileNotFoundError: ... external/OpenFAST_files/...``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An integration test or a case script needs upstream data that
isn't present. See :doc:`data_sources` for the layout and clone
the required upstream repository under ``external/``.

``MemoryError`` or eigensolver hangs on large towers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a smaller ``n_modes`` (only the lowest few are usually
interesting), or check ``hub_conn`` — a free-free root
(``hub_conn = 2``) without a ``PlatformSupport`` 6×6 matrix is
singular and will hang the solver. The pre-solve sanity checks
(:func:`pybmodes.checks.check_model`) catch this.

Uninstalling
------------

.. code-block:: bash

   pip uninstall pybmodes

The editable install also clears with ``pip uninstall``; ``rm
-rf`` on the cloned repo handles the source.
