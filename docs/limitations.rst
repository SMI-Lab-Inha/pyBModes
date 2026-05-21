Limitations
===========

What ``pybmodes`` deliberately does *not* attempt to do, and where
the modelling fidelity stops.

Polynomial representation
-------------------------

ElastoDyn's polynomial ansatz
:math:`\mathrm{SHP}(s) = \sum_{i=2}^{6} c_i s^{\,i}` is a constrained
6th-order form with ``SHP(0) = SHP'(0) = 0`` baked in. Two
consequences:

* Some FEM mode shapes cannot be faithfully represented as a
  polynomial of this form. The most common case is the
  IEA-15-240-RWT UMaine VolturnUS-S **second tower side-side mode**
  (``TwSSM2Sh``): the constrained 6th-order polynomial cannot
  resolve the section-property gradient of that specific tower for
  that mode, so even a fresh patch from the structural inputs ends
  at ``Overall: WARN`` with ratio > 100. This is documented in the
  ``validation_report.txt`` shipped alongside the patched deck.

* Hybrid bending + twist modes are dropped from FA / SS family
  selection by the torsion-contamination filter
  (``T_tor ≥ 0.10``). The polynomial form cannot express twist
  content; including hybrid modes in the fit silently produces
  wrong coefficients. Rejected modes travel through
  ``TowerSelectionReport.rejected_fa_modes`` /
  ``rejected_ss_modes`` so users see what was dropped.

Floating-platform models
------------------------

The ``Tower.from_windio_floating`` path is **two-tier**:

- **Industry-grade** (with companion HydroDyn + MoorDyn +
  ElastoDyn decks supplied or auto-discovered) — byte-identical to
  ``from_elastodyn_with_mooring``, which reproduces OC3 Hywind to
  ≈ 0.0003 % across the first nine modes.
- **Screening preview** (yaml-only) — member-Morison hydrodynamics
  + RAFT end-cap added mass + catenary mooring from the yaml. Useful
  for design-space exploration, **not** for final-design certification
  numbers. Always emits one ``UserWarning`` naming the result as
  ``SCREENING-fidelity (NOT industry-grade)``.

ElastoDyn-compatible polynomial generation for floating decks uses
the **cantilever** path (``Tower.from_elastodyn``), not the coupled
one — see :doc:`theory` for the OpenFAST source-code citation.

Mooring physics
---------------

:class:`pybmodes.mooring.MooringSystem` is a **quasi-static**
catenary model:

- Jonkman 2007 Appendix B B-1 / B-2 for the fully-suspended branch
  and B-7 / B-8 (with ``CB = 0``) for the anchor-on-seabed branch.
- Extensible elastic catenary per line; damped Newton on ``(H, V_F)``
  with an analytical 2×2 Jacobian.
- No bending stiffness, no dynamic effects (drag, added mass, vortex
  shedding), no fluid-structure interaction. For those, use MoorDyn
  in time-domain — pyBmodes consumes the static linearised stiffness.

Numerical scope
---------------

- **Beam element only.** Plates / shells / volume elements are out of
  scope. Section properties come in as a distributed 1-D table.
- **Linear modal analysis only.** Geometric nonlinearity (large
  deflections), material nonlinearity (yield), and contact / impact
  are out of scope.
- **Centrifugal stiffening only on the blade.** Tower centrifugal
  effects (negligible for fixed-base towers) are not modelled.
- **Rotor aerodynamics are not modelled.** A Campbell sweep does
  not include aeroelastic damping or unsteady aerodynamics — those
  belong in OpenFAST. ``pybmodes`` answers the structural-frequency
  question; resonance checks against the per-rev family are read
  off the diagram.
- **Single 1.x public API.** Names listed in
  :doc:`api_contract` are semver-frozen across 1.x minor releases.
  Numerical outputs may shift between minor releases when
  validation tightens or a modelling correction lands; every such
  shift is called out in :doc:`changelog` under *Fixed* / *Changed*
  with magnitude and affected case.

What this is *not*
------------------

- Not a multi-body dynamics solver. Use OpenFAST + ElastoDyn for
  time-domain simulation.
- Not a CFD code. Hydrodynamics come in as 6 × 6 matrices from
  WAMIT / HydroDyn potential-flow output.
- Not a structural design tool. The supported workflow is *analysis
  of a defined structure*, not *optimisation of one*. Use WISDEM for
  design.
- Not a validation-as-a-service product. Numerical accuracy claims
  are documented per case in :doc:`validation`; deltas vs published
  references are surfaced, not hidden.
