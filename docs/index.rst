pyBmodes
========

A pure-Python finite-element library for wind-turbine **blade and
tower modal analysis** — coupled flap–lag–torsion–axial vibration
modes from a 15-DOF Bernoulli-Euler beam element, with first-class
support for the OpenFAST / BModes / WISDEM-WindIO ecosystem.

.. rubric:: Highlights

- **Reads what the wind-energy field uses.** OpenFAST decks
  (ElastoDyn / SubDyn / HydroDyn / MoorDyn), BModes ``.bmi``, and
  the WISDEM / WindIO ontology ``.yaml`` are all first-class
  inputs. Mooring (Jonkman 2007 catenary), hydrodynamics (WAMIT /
  HydroDyn ``.1`` / ``.hst``), and composite-layup blade reductions
  (PreComp-class, BeamDyn-validated) come built in. No wrapper
  around a Fortran binary — pure Python, ``numpy + scipy`` runtime.

- **Cross-validated to better than 0.01 % on every benchmark case.**
  Six independent verifications against the BModes Fortran reference
  solver (NREL 5MW land / OC3 monopile / OC3 Hywind floating spar +
  IEA-3.4-130-RWT + BModes CertTest Test03 / Test04) plus three
  closed-form analytical references (Wright 1982 rotating blade,
  Bir 2009 pinned-free cable, Blevins tip-mass tower). The
  validation matrix is mechanically audited by
  ``scripts/audit_validation_claims.py`` in CI — claims cannot
  drift ahead of tests.

- **ElastoDyn-compatible polynomial round-trip.** Solve modes,
  fit the constrained 6th-order polynomial ansatz with
  design-matrix condition-number reporting, then write the
  coefficients straight back into the input deck via
  ``pybmodes patch`` (with safety modes: ``--dry-run`` / ``--diff``
  / ``--backup`` / ``--output-dir``). Six pre-patched reference
  decks ship in the wheel — three fixed-base, three floating.

- **Engineering-report-quality Campbell diagrams.** Rotor-speed
  sweeps with Hungarian MAC-based mode tracking; floating
  platforms get six rigid-body modes named natively
  (surge / sway / heave / roll / pitch / yaw). Operating-window
  shading, per-rev excitation rays (1P / 3P / 6P / 9P), inline
  family labels, and explicit log-frequency support.

- **One-click WISDEM / WindIO pipeline.** ``pybmodes windio
  ontology.yaml`` discovers companion HydroDyn / MoorDyn /
  ElastoDyn decks scoped to the turbine root, then solves the
  composite-layup blade, the tubular tower, and (if present) the
  coupled floating platform — with an optional Campbell sweep and
  a bundled Markdown / HTML / CSV report.

.. rubric:: 30-second tour

.. code-block:: python

   from pybmodes.models import Tower
   from pybmodes.elastodyn import compute_tower_params, patch_dat

   # ElastoDyn main + tower files read from one path; rotor mass lumped.
   tower = Tower.from_elastodyn("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat")
   modal = tower.run(n_modes=4)

   # Constrained 6th-order fit; FA/SS family selection with
   # torsion-contamination filter.
   params = compute_tower_params(modal)
   patch_dat("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat", params)

The user guide takes it from there:

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

What this project is *not*
--------------------------

- Not a multi-body dynamics solver. Use OpenFAST + ElastoDyn for
  time-domain simulation.
- Not a CFD code. Hydrodynamics come in as 6 × 6 matrices from
  WAMIT / HydroDyn potential-flow output.
- Not a structural design tool. The supported workflow is
  *analysis of a defined structure*, not *optimisation of one*.
  Use WISDEM for design.

See :doc:`limitations` for the full scope statement.

Status
------

Stable **1.x baseline**. The public API surface enumerated in
:doc:`api_contract` is semver-frozen across 1.x minor releases.
Numerical outputs may still shift between minor releases when
validation tightens or a modelling correction lands; every such
shift is called out in :doc:`changelog` under *Fixed* / *Changed*
with magnitude and affected case.

Citation
--------

If you use pyBmodes in academic work, please cite it via the
``CITATION.cff`` file in the repository root. GitHub's
*"Cite this repository"* widget, Zenodo, and most reference
managers read it automatically.

License
-------

Apache 2.0 — see `LICENSE
<https://github.com/SMI-Lab-Inha/pyBModes/blob/master/LICENSE>`_.

Copyright 2024-2026 Jae Hoon Seo, Marine Structural Mechanics and
Integrity Lab (SMI Lab), Inha University.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
