pyBmodes
========

``pybmodes`` is a pure-Python finite-element library for wind-turbine
blade and tower modal analysis. It reads OpenFAST / ElastoDyn /
SubDyn / HydroDyn / MoorDyn decks, BModes ``.bmi`` decks, and WISDEM /
WindIO ontology YAML files; solves the coupled
flap–lag–torsion–axial vibration modes with a 15-DOF Bernoulli-Euler
beam element; and emits ElastoDyn-compatible mode-shape polynomials,
MAC-tracked Campbell diagrams, and bundled Markdown / HTML / CSV
reports.

The full validation matrix — every cross-checked frequency against a
citable reference, with worst-case margin — is at
:doc:`validation`.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   installation
   quickstart
   theory
   data_sources
   units
   limitations
   validation
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   api_contract

.. toctree::
   :maxdepth: 1
   :caption: Developer guide

   contributing
   release_checklist


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
-------

Apache 2.0 — see `LICENSE
<https://github.com/SMI-Lab-Inha/pyBModes/blob/master/LICENSE>`_.

Copyright 2024-2026 Jae Hoon Seo, Marine Structural Mechanics and
Integrity Lab (SMI Lab), Inha University.

If you use pyBmodes in academic work, please cite it via the
``CITATION.cff`` file in the repository root.
