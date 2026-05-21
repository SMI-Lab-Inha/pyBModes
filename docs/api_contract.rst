API contract
============

``pybmodes`` is on a **stable 1.x baseline**. The public surface
enumerated below is semver-frozen across 1.x minor releases.
Renaming or removing any name on this list requires a major-
version bump (2.x). Adding new keyword arguments with defaults,
new dataclass fields, and brand-new entry points is non-
breaking.

Versioning policy
-----------------

We follow `Semantic Versioning 2.0 <https://semver.org>`_ for
the **API surface**. The version string is in
``[project] version`` of ``pyproject.toml`` and is mirrored at
``pybmodes.__version__``.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Bump
     - Example
     - Triggers
   * - **Major** (X.y.z)
     - 1.7.0 → 2.0.0
     - Renaming or removing any name on this page; changing a
       function's *required* parameter; tightening a return-
       type so callers must adapt; dropping a Python version.
   * - **Minor** (x.Y.z)
     - 1.6.0 → 1.7.0
     - New entry points; new keyword arguments with sane
       defaults; new dataclass fields with sane defaults; new
       optional dependency *extra*; new CLI subcommand.
   * - **Patch** (x.y.Z)
     - 1.7.0 → 1.7.1
     - Bug fixes; numerical-accuracy improvements that change
       output values (always called out in the changelog);
       documentation; internal refactors.

**Numerical outputs may shift between minor and patch
releases** when validation tightens or a modelling correction
lands. Every such shift is called out in :doc:`changelog`
under *Fixed* / *Changed* with magnitude and affected case.
For reproducible numerics across runs, **pin to an exact
version** (``pybmodes==1.7.0``) and only upgrade after reading
the changelog.

Deprecation policy
------------------

When a public name is being renamed or removed for a future
major version, we follow a **two-minor-release deprecation
window**:

1. **First minor release** introduces the new name; keeps the
   old name working but emits a ``DeprecationWarning`` with a
   pointer to the replacement.
2. **Second minor release** keeps the old name working but
   prints the warning at module-import time, in addition to
   call-time.
3. **Next major release** removes the old name entirely.

Deprecation warnings name the *minimum version* a fix is
available in, so:

.. code-block:: python

   warnings.warn(
       "RotatingBlade.legacy_method is deprecated since 1.x; "
       "use RotatingBlade.method instead (available from 1.7.0). "
       "The legacy name will be removed in 2.0.",
       DeprecationWarning,
       stacklevel=2,
   )

There are currently no deprecations in flight on the 1.x line.

Stable public names
-------------------

The authoritative list is the docstring of
:mod:`pybmodes.__init__`; the table below is a categorised
summary.

Model constructors
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Name
     - From
   * - :class:`pybmodes.models.RotatingBlade`
     - ``__init__(bmi_path)``,
       ``from_elastodyn(main_dat_path, *, validate_coeffs=False)``,
       ``from_windio(yaml_path, *, component='blade', n_span=30,
       rot_rpm=0.0, n_perim=300, elastic='auto')``.
   * - :class:`pybmodes.models.Tower`
     - ``__init__(bmi_path)``,
       ``from_bmi(bmi_path)``,
       ``from_elastodyn(main_dat_path, *, validate_coeffs=False)``,
       ``from_elastodyn_with_subdyn(main_dat_path, subdyn_dat_path)``,
       ``from_elastodyn_with_mooring(main_dat_path,
       moordyn_dat_path, hydrodyn_dat_path=None)``,
       ``from_geometry(...)``,
       ``from_windio(yaml_path, *, component='tower',
       thickness_interp='linear')``,
       ``from_windio_floating(yaml_path, *, ...)``.

Results + serialisation
^^^^^^^^^^^^^^^^^^^^^^^

- :class:`pybmodes.models.ModalResult` — return value of
  ``.run()``; carries frequencies + per-node mode shapes +
  optional participation + optional fit residuals; ships
  ``save(.npz)`` / ``load(.npz)`` and ``to_json(.json)`` /
  ``from_json(.json)`` with embedded pyBmodes version + UTC
  timestamp + source-file + git-hash metadata.
- :class:`pybmodes.campbell.CampbellResult` — output of
  ``campbell_sweep``; ships ``save(.npz)`` / ``load(.npz)``,
  ``to_csv(.csv)``; carries frequencies + omega_rpm + labels
  + participation + ``mac_to_previous`` per-step tracking
  confidence + integer mode counts.

Solvers + sweeps
^^^^^^^^^^^^^^^^

- :func:`pybmodes.campbell.campbell_sweep` — rotor-speed sweep
  with MAC-tracked blade modes + constant-frequency tower
  modes.
- :func:`pybmodes.campbell.plot_campbell` — engineering-report-
  style diagram (four family keys: Blades, Tower, Platform,
  Blade Passing) with operating-window shading + inline
  labels.

Polynomial fitting + validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`pybmodes.elastodyn.compute_blade_params`
- :func:`pybmodes.elastodyn.compute_tower_params`
- :func:`pybmodes.elastodyn.compute_tower_params_report` —
  same plus a ``TowerSelectionReport`` exposing FA / SS family
  scoring + rejected-mode lists.
- :func:`pybmodes.elastodyn.patch_dat`
- :func:`pybmodes.elastodyn.validate_dat_coefficients`

Pre-solve sanity + comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`pybmodes.checks.check_model`
- :func:`pybmodes.mac.mac_matrix`
- :func:`pybmodes.mac.compare_modes`
- :func:`pybmodes.mac.plot_mac`

Numerical options (1.x architecture refactor — Phase 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frozen dataclasses that centralise numerical thresholds previously
scattered as module-level constants. Adding fields with sensible
defaults is non-breaking; removing or renaming a field is a
semver-major change.

- :class:`pybmodes.SolverOptions` — FEM solver dispatch
  (``sparse_ndof_threshold``, ``symmetry_rtol``).
- :class:`pybmodes.FitOptions` — polynomial-fit + family-selection
  thresholds (``polynomial_rms_threshold``,
  ``torsion_contamination_threshold``, ``fit_cond_warn``,
  ``fit_cond_fail``).
- :class:`pybmodes.CheckOptions` — :func:`check_model` thresholds
  (``stiffness_jump_factor``, ``ei_ratio_min`` / ``_max``,
  ``support_asymmetry_rtol``, ``fit_cond_warn`` / ``_fail``).

Reports
^^^^^^^

- :func:`pybmodes.report.generate_report` — Markdown / HTML /
  CSV bundled analysis report (eight sections: summary,
  assumptions, frequencies, classification, polynomial
  coefficients with fit residuals, validation, check_model
  warnings, Campbell sweep).

Mooring + hydro
^^^^^^^^^^^^^^^

- :class:`pybmodes.mooring.LineType`
- :class:`pybmodes.mooring.Point`
- :class:`pybmodes.mooring.Line`
- :class:`pybmodes.mooring.MooringSystem` —
  ``from_moordyn``, ``from_windio_mooring``,
  ``stiffness_matrix(body_r6=None)``.
- :class:`pybmodes.io.wamit_reader.HydroDynReader`

I/O
^^^

- :func:`pybmodes.io.bmi.read_bmi`
- :func:`pybmodes.io.out_parser.read_out` (with the
  ``strict=True`` option)
- :func:`pybmodes.io.windio.read_windio_tubular`
- :func:`pybmodes.io.windio_blade.read_windio_blade`
- :func:`pybmodes.io.windio_floating.read_windio_floating`
- :class:`pybmodes.io.errors.ParseError` — unified base class for
  every ``pybmodes.io.*`` parser exception; inherits
  :class:`ValueError` so existing ``except ValueError`` callers
  are backward-compatible. Subclasses :class:`BMIParseError`,
  :class:`ElastoDynParseError`, :class:`SubDynParseError`,
  :class:`WAMITParseError`, :class:`MoorDynParseError`,
  :class:`WindIOParseError`, and the existing
  :class:`BModeOutParseError` (now re-rooted under the new base).
  Structured ``file`` / ``line`` / ``column`` / ``context``
  fields + ``format_diagnostic()`` for uniform error messages
  across formats.

Plot helpers (``[plots]`` extra)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`pybmodes.plots.apply_style` — engineering-paper
  defaults (black / red / blue / green).
- :func:`pybmodes.plots.plot_mode_shapes`
- :func:`pybmodes.plots.plot_fit_quality`
- :func:`pybmodes.plots.bir_mode_shape_plot`
- :func:`pybmodes.plots.bir_mode_shape_subplot`
- ``plot_mac`` (re-exported from ``pybmodes.mac``)
- ``plot_campbell`` (re-exported from ``pybmodes.campbell``)

CLI subcommands
^^^^^^^^^^^^^^^

The ``pybmodes`` console script declared in
``[project.scripts]``. Subcommand names and their flag
semantics are part of the semver contract.

.. list-table::
   :header-rows: 1

   * - Subcommand
     - Purpose
   * - ``pybmodes validate <main.dat>``
     - Coefficient-consistency report on one ElastoDyn deck.
   * - ``pybmodes patch <main.dat>``
     - Regenerate polynomial blocks; ``--backup`` / ``--dry-run``
       / ``--diff`` / ``--output-dir`` for safety.
   * - ``pybmodes campbell <input>``
     - Rotor-speed sweep → Campbell diagram PNG + CSV.
   * - ``pybmodes batch ROOT``
     - Walk a directory of decks; per-deck validate + patch +
       summary CSV.
   * - ``pybmodes report <main.dat>``
     - Bundled Markdown / HTML / CSV analysis report.
   * - ``pybmodes windio <yaml | dir>``
     - One-click WISDEM/WindIO → composite blade + tubular
       tower + coupled platform + Campbell.
   * - ``pybmodes examples --copy DIR``
     - Vendor ``sample_inputs/`` and / or ``reference_decks/``
       out of the installed wheel.

Stability tiers
---------------

Not every module is at the same stability level. The public
names above (and only those) are the **stable surface**.

Public + stable
^^^^^^^^^^^^^^^

Listed above. Semver-frozen.

Public but experimental
^^^^^^^^^^^^^^^^^^^^^^^

Currently none. When a name is introduced experimentally
(e.g. a new constructor for a niche WindIO dialect), it's
flagged in its docstring with::

   .. note::
      Experimental — may change without notice until the next
      X.Y.0 release at the earliest.

Internal (underscore-prefixed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modules under ``src/pybmodes/`` whose name starts with an
underscore (``_pipeline.py``, ``_serialize.py``,
``_elastodyn/``, ``_precomp/``) and module-level attributes
starting with an underscore. These can change in any release
including patch versions.

Don't import from them; if you find yourself wanting to,
**open an issue** — that's a signal we have a real-use-case
gap in the public surface, and we'd rather fix that than
have you depend on something we'll break.

Runtime dependency contract
---------------------------

The default install pulls in **``numpy``** and **``scipy``**
only. Every other dependency is gated behind an extra
(:doc:`installation`). Adding a runtime dependency to the
core requires alignment — the ``numpy + scipy``-only stance is
itself part of the contract.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tier
     - Examples
     - Semver impact
   * - Core
     - ``numpy``, ``scipy``
     - Tightening the version pin is a minor bump; adding a
       new core dep is a major bump.
   * - Extras
     - ``matplotlib`` (``[plots]``), ``pyyaml``
       (``[windio]``), ``sphinx`` (``[docs]``)
     - Adding a new extra is a minor bump; renaming or
       removing an extra is a major bump.

How to track changes
--------------------

- :doc:`changelog` — every release, called-out numerical
  shifts.
- :doc:`validation` — the per-case validation matrix,
  mechanically audited by
  ``scripts/audit_validation_claims.py`` in CI.
- ``CHANGELOG.md`` at the repo root — the source of truth
  (the docs page above is included from it).
- GitHub Releases — release-notes copies of the ``[X.Y.Z]``
  changelog block, with merge-PR backlinks.
- `Read the Docs version selector
  <https://pybmodes.readthedocs.io/>`_ — version-pinned docs
  for every published tag.
