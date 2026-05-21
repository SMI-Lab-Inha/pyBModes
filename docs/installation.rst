Installation
============

``pybmodes`` requires Python ≥ 3.11. The runtime depends only on
``numpy`` and ``scipy``; everything optional is gated behind an extra.

From PyPI
---------

.. code-block:: bash

   pip install pybmodes

From source (editable)
----------------------

.. code-block:: bash

   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Extra
     - Pulls in
     - When you need it
   * - ``[dev]``
     - ``pytest``, ``pytest-cov``, ``ruff``, ``mypy``, ``pyyaml``
     - Run the test suite + lint + type-check.
   * - ``[plots]``
     - ``matplotlib``
     - Campbell, MAC, mode-shape, and environmental-loading figures.
   * - ``[windio]``
     - ``pyyaml``
     - ``Tower.from_windio(...)``, ``RotatingBlade.from_windio(...)``,
       and the ``pybmodes windio`` CLI.
   * - ``[notebook]``
     - ``nbclient``, ``nbformat``, ``ipykernel``, ``matplotlib``
     - Headless execution of the bundled walkthrough notebooks under
       :file:`tests/test_notebooks.py`.
   * - ``[docs]``
     - ``sphinx``, ``furo``, ``myst-parser``, ``linkify-it-py``,
       ``sphinx-autodoc-typehints``, ``sphinx-copybutton``
     - Build this documentation site locally.

Combine extras with commas — e.g. ``pip install -e ".[dev,plots,windio]"``.

Windows + conda quickstart
--------------------------

Path of least resistance on Windows:

.. code-block:: bat

   :: 1. install Miniconda or Anaconda first if you don't have it.
   ::    https://docs.conda.io/en/latest/miniconda.html
   ::    Open "Anaconda Prompt" from the Start menu.

   :: 2. create and activate a dedicated env
   conda create -n pybmodes python=3.11 -y
   conda activate pybmodes

   :: 3. clone and install
   git clone https://github.com/SMI-Lab-Inha/pyBModes.git
   cd pyBModes
   pip install -e ".[dev,plots]"

   :: 4. verify
   pytest

Verifying the install
---------------------

.. code-block:: bash

   python -c "import pybmodes; print(pybmodes.__version__)"
   pytest

A fresh clone runs the **self-contained** test suite — synthetic
decks and closed-form-validated FEM cases. The ``integration``
marker gates tests that need upstream OpenFAST / BModes data under
``external/``; see :doc:`data_sources` for what to clone and where.
