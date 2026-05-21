API contract
============

``pybmodes`` is on a **stable 1.x baseline**. The public surface
enumerated below is semver-frozen across 1.x minor releases.
Renaming or removing any name on this list requires a major-version
bump (2.x). Adding new keyword arguments with defaults, new
dataclass fields, and brand-new entry points is non-breaking.

Numerical outputs may still shift between minor releases when
validation tightens or a modelling correction lands. Every such
shift is called out in :doc:`changelog` under *Fixed* / *Changed*
with magnitude and affected case.

Stable public names
-------------------

The authoritative list is the docstring of
:mod:`pybmodes.__init__`; the table below is a categorised summary.

Model constructors
^^^^^^^^^^^^^^^^^^

- :class:`pybmodes.models.RotatingBlade` —
  ``RotatingBlade(bmi_path)``,
  ``RotatingBlade.from_elastodyn(main_dat_path)``,
  ``RotatingBlade.from_windio(yaml_path)``.
- :class:`pybmodes.models.Tower` —
  ``Tower(bmi_path)``,
  ``Tower.from_bmi(bmi_path)``,
  ``Tower.from_elastodyn(main_dat_path)``,
  ``Tower.from_elastodyn_with_subdyn(...)``,
  ``Tower.from_elastodyn_with_mooring(...)``,
  ``Tower.from_geometry(...)``,
  ``Tower.from_windio(yaml_path)``,
  ``Tower.from_windio_floating(yaml_path, ...)``.

Results & serialisation
^^^^^^^^^^^^^^^^^^^^^^^

- :class:`pybmodes.models.ModalResult` — return value of ``.run()``;
  ``save`` / ``load`` (``.npz``), ``to_json`` / ``from_json``
  (``.json``), with embedded pyBmodes version + UTC timestamp +
  source-file + git-hash metadata.
- :class:`pybmodes.campbell.CampbellResult` — ``save`` / ``load``
  (``.npz``), ``to_csv``.

Solvers + sweeps
^^^^^^^^^^^^^^^^

- :func:`pybmodes.campbell.campbell_sweep` —
  rotor-speed sweep with MAC-tracked blade modes + constant tower
  modes.
- :func:`pybmodes.campbell.plot_campbell` — engineering-report-style
  diagram (four family keys: Blades, Tower, Platform, Blade
  Passing).

Polynomial fitting + validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`pybmodes.elastodyn.compute_blade_params`
- :func:`pybmodes.elastodyn.compute_tower_params`
- :func:`pybmodes.elastodyn.compute_tower_params_report`
- :func:`pybmodes.elastodyn.patch_dat`
- :func:`pybmodes.elastodyn.validate_dat_coefficients`

Pre-solve sanity + comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`pybmodes.checks.check_model`
- :func:`pybmodes.mac.mac_matrix`
- :func:`pybmodes.mac.compare_modes`
- :func:`pybmodes.mac.plot_mac`

Reports
^^^^^^^

- :func:`pybmodes.report.generate_report` — Markdown / HTML / CSV
  bundled analysis report (eight sections).

Mooring + hydro
^^^^^^^^^^^^^^^

- :class:`pybmodes.mooring.LineType`
- :class:`pybmodes.mooring.Point`
- :class:`pybmodes.mooring.Line`
- :class:`pybmodes.mooring.MooringSystem` —
  ``from_moordyn``, ``from_windio_mooring``,
  ``stiffness_matrix(body_r6=None)``.
- :class:`pybmodes.io.wamit_reader.HydroDynReader`

CLI subcommands (``pybmodes`` console script)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validate``, ``patch``, ``campbell``, ``batch``, ``report``,
``windio``, ``examples``. Flag names and their semantics are part
of the semver contract.

Optional extras
---------------

The default install pulls in ``numpy`` and ``scipy`` only. Every
other dependency is gated behind an extra (:doc:`installation`).
Adding a runtime dependency to the core requires alignment — the
``numpy + scipy``-only stance is itself a contract.

How to track changes
--------------------

- :doc:`changelog` — every release; numerical shifts called out with
  magnitude.
- :doc:`validation` — the per-case validation matrix, mechanically
  audited by ``scripts/audit_validation_claims.py`` in CI.
- ``CHANGELOG.md`` at the repo root — the source of truth (this
  page is included from it).
