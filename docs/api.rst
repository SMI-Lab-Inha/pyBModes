API reference
=============

The full public API of ``pybmodes``. The semver-stable surface is
enumerated in :doc:`api_contract`.

High-level models
-----------------

.. automodule:: pybmodes.models.blade
   :members:
   :show-inheritance:

.. automodule:: pybmodes.models.tower
   :members:
   :show-inheritance:

.. automodule:: pybmodes.models.result
   :members:
   :show-inheritance:

Input / output
--------------

BModes ``.bmi``
^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.bmi
   :members:
   :show-inheritance:

ElastoDyn deck reader
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.elastodyn_reader
   :members:
   :show-inheritance:

SubDyn reader
^^^^^^^^^^^^^

.. automodule:: pybmodes.io.subdyn_reader
   :members:
   :show-inheritance:

WAMIT / HydroDyn reader
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.wamit_reader
   :members:
   :show-inheritance:

BModes ``.out`` output
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.out_parser
   :members:
   :show-inheritance:

WindIO ontology — tubular tower / monopile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.windio
   :members:
   :show-inheritance:

WindIO ontology — composite blade
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.windio_blade
   :members:
   :show-inheritance:

WindIO ontology — floating substructure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.windio_floating
   :members:
   :show-inheritance:

Geometry & section properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pybmodes.io.geometry
   :members:
   :show-inheritance:

.. automodule:: pybmodes.io.sec_props
   :members:
   :show-inheritance:

Polynomial fitting + ElastoDyn helpers
--------------------------------------

.. automodule:: pybmodes.fitting.poly_fit
   :members:
   :show-inheritance:

.. automodule:: pybmodes.elastodyn.params
   :members:
   :show-inheritance:

.. automodule:: pybmodes.elastodyn.validate
   :members:
   :show-inheritance:

.. automodule:: pybmodes.elastodyn.writer
   :members:
   :show-inheritance:

Campbell
--------

.. automodule:: pybmodes.campbell
   :members:
   :show-inheritance:

Modal Assurance Criterion
-------------------------

.. automodule:: pybmodes.mac
   :members:
   :show-inheritance:

Pre-solve sanity checks
-----------------------

.. automodule:: pybmodes.checks
   :members:
   :show-inheritance:

Report generation
-----------------

.. automodule:: pybmodes.report
   :members:
   :show-inheritance:

Mooring
-------

.. automodule:: pybmodes.mooring
   :members:
   :show-inheritance:

Plot helpers (``[plots]`` extra)
--------------------------------

.. automodule:: pybmodes.plots
   :members:
   :show-inheritance:

CLI
---

.. automodule:: pybmodes.cli
   :members:
   :show-inheritance:
