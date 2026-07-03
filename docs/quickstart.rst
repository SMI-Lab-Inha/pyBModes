Quickstart
==========

Five minutes from import to a validated modal solve, plus the
patterns that show up in every real workflow. For full API
reference see :doc:`api`; for the *why* behind each step see
:doc:`theory`.

Solving a synthetic tower
-------------------------

Every public constructor accepts a path; the bundled samples
ship with the wheel:

.. code-block:: python

   from pybmodes.models import Tower
   from pybmodes.cli import _resolve_examples_root

   bmi = (
       _resolve_examples_root()
       / "sample_inputs" / "02_tower_topmass" / "tower_topmass.bmi"
   )
   tower = Tower(bmi)
   result = tower.run(n_modes=4)

   for f in result.frequencies:
       print(f"{f:.4f} Hz")

``Tower.run`` returns a :class:`~pybmodes.models.ModalResult`
with:

- ``frequencies`` — shape ``(n_modes,)``, Hz
- ``shapes`` — list of :class:`~pybmodes.fem.normalize.NodeModeShape`,
  one per mode, with per-node flap / lag / twist / axial
  ordinates and slopes
- ``participation`` — *(optional)* per-mode energy fractions
  in the (flap-axis, lag-axis, torsion) basis
- ``fit_residuals`` — *(optional)* RMS residuals from polynomial
  fits

Reading an OpenFAST ElastoDyn deck
----------------------------------

The realistic workflow: industry deck → modes → polynomial
coefficients → patched deck:

.. code-block:: python

   from pybmodes.models import Tower
   from pybmodes.elastodyn import compute_tower_params, patch_dat

   tower = Tower.from_elastodyn(
       "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
   )
   modal = tower.run(n_modes=4)

   # Constrained 6th-order fit + FA/SS family selection + torsion filter.
   params = compute_tower_params(modal)
   patch_dat(
       "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat", params,
   )

The same on the CLI, with built-in safety modes:

.. code-block:: bash

   # Read-only verdict on coefficient consistency
   pybmodes validate NRELOffshrBsline5MW_Onshore_ElastoDyn.dat

   # PR-ready diff preview before writing anything
   pybmodes patch NRELOffshrBsline5MW_Onshore_ElastoDyn.dat \
       --dry-run --diff

   # Write with a ``.bak`` backup
   pybmodes patch NRELOffshrBsline5MW_Onshore_ElastoDyn.dat --backup

   # Write to a separate directory, leaving the source untouched
   pybmodes patch NRELOffshrBsline5MW_Onshore_ElastoDyn.dat \
       --output-dir ./patched/

   # Full Markdown / HTML / CSV bundled report
   pybmodes report NRELOffshrBsline5MW_Onshore_ElastoDyn.dat \
       --format md --out report.md

Monopile decks with SubDyn
--------------------------

For a monopile, splice the SubDyn pile geometry below the
ElastoDyn tower:

.. code-block:: python

   tower = Tower.from_elastodyn_with_subdyn(
       main_dat_path="NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat",
       subdyn_dat_path="NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat",
   )
   modal = tower.run(n_modes=4)

The result is a single combined cantilever (clamped at the
SubDyn reaction joint, no soil flexibility — designed for
OC3-style fixed-base monopiles). For soft-pile soil compliance
on the ``CS_Monopile.bmi`` reference deck pattern, use
``Tower(bmi_path)`` with the corresponding ``hub_conn = 3``
distributed soil-stiffness BMI.

Soft monopile via closed-form mudline springs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have pile geometry and soil properties but no
pre-computed mudline stiffness deck, :class:`pybmodes.MudlineFoundation`
computes the three coupled springs ``K_hh``, ``K_hr``, ``K_rr``
from Shadlou and Bhattacharya (2016) (Yu and Amdahl 2023 Table 1)
or Psaroudakis et al. (2021) (Yu Eq 25) and emits a 6x6 block
that drops into ``PlatformSupport.mooring_K`` of a
``hub_conn = 3`` BMI:

.. code-block:: python

   from pybmodes import MudlineFoundation
   import math

   D_P = 9.0          # pile outer diameter, m
   t_P = 0.110        # pile wall thickness, m
   L_P = 42.0         # embedded pile length, m
   E_steel = 210e9    # steel Young's modulus, Pa
   I_P = math.pi / 64.0 * (D_P**4 - (D_P - 2.0 * t_P) ** 4)

   f = MudlineFoundation.from_soil_properties(
       pile_diameter=D_P,
       pile_length_embedded=L_P,
       pile_EI=E_steel * I_P,
       soil_E=30e6,
       soil_nu=0.3,
       soil_profile="homogeneous",
       pile_behaviour="auto",   # Randolph (1981) classifier
       formula="shadlou",
   )

The ergonomic wiring uses
:meth:`~pybmodes.models.Tower.attach_mudline_foundation` to swap a
clamped monopile model to the ``hub_conn = 3`` soft-monopile path
without hand-building a ``PlatformSupport``:

.. code-block:: python

   from pybmodes.models import Tower

   tower = Tower.from_windio_with_monopile(
       "IEA-15-240-RWT.yaml", tip_mass=991000.0,
       water_depth=30.0,     # clamp at the seabed, not the embedded pile tip
   )
   tower.attach_mudline_foundation(f)        # mutates BMI to hub_conn = 3
   modal = tower.run(n_modes=4)

.. note::

   Pass ``water_depth`` whenever the monopile ``reference_axis.z`` runs
   below the seabed (IEA-15: axis -75 -> +15 m, mudline at -30 m). It
   clamps the cantilever at the mudline and drops the embedded pile; omit
   it and that length is modelled as a free cantilever, dropping the
   frequency well below reference (issue #121). It also auto-reads
   ``environment.water_depth`` from the ontology when present.

   On an IEA-22-class ontology that carries the hub and nacelle
   ``elastic_properties_mb`` blocks, ``lumped_rna_cal=True`` derives the
   tower-top RNA (hub + nacelle + blades) automatically instead of a
   hand-supplied ``tip_mass`` (issue #82; requires ``hub_conn = 1``, the
   clamped-base default) — the same flag works on
   :meth:`~pybmodes.models.Tower.from_windio`.

If you only need the 6 x 6 stiffness block to compose with an
existing ``PlatformSupport`` you have already built (the
``CS_Monopile.bmi`` deck pattern, say), the raw matrix is also
available:

.. code-block:: python

   K6 = f.as_mooring_K()        # 6 x 6 in OpenFAST DOF order

.. note::

   ``MudlineFoundation`` affects the coupled-system frequency only.
   ElastoDyn polynomial generation continues to use the cantilever
   path (``Tower.from_elastodyn``) regardless of soil flexibility,
   for the same architectural reason floating decks use the
   cantilever basis. See :doc:`limitations` for the source-code
   citations.

Floating decks with mooring + hydro
-----------------------------------

The coupled floating tower assembled from upstream OpenFAST
decks:

.. code-block:: python

   tower = Tower.from_elastodyn_with_mooring(
       main_dat_path="IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat",
       moordyn_dat_path="IEA-15-240-RWT-UMaineSemi_MoorDyn.dat",
       hydrodyn_dat_path="IEA-15-240-RWT-UMaineSemi_HydroDyn.dat",
   )
   modal = tower.run(n_modes=12, check_model=False)

The first six modes are platform rigid-body
(surge / sway / heave / roll / pitch / yaw); modes 7–8 are the
first tower bending fore-aft / side-to-side pair; 9–10 are the
second pair. ``ModalResult.mode_labels`` names each rigid-body
mode (the auto-classifier reads the dominant DOF from the
mass-weighted eigenvector).

.. note::

   For **floating-deck polynomial generation** use the
   *cantilever* path (``Tower.from_elastodyn``), not the coupled
   one. ElastoDyn's polynomial ansatz can only express
   clamped-base modes — see :doc:`theory` for the source-code
   audit and :doc:`limitations` for the consequence.

To reconcile the polynomial-basis cantilever frequency against
the coupled-system frequency an OpenFAST linearisation will
report on the same deck, call
:func:`pybmodes.elastodyn.report_floating_frequency_gap`:

.. code-block:: python

   from pybmodes.elastodyn import report_floating_frequency_gap

   gap = report_floating_frequency_gap(
       "NRELOffshrBsline5MW_OC3Hywind_ElastoDyn.dat",
       "NRELOffshrBsline5MW_OC3Hywind_MoorDyn.dat",
       "NRELOffshrBsline5MW_OC3Hywind_HydroDyn.dat",
   )
   print(gap.format_report())

Sample output on the OC3 Hywind spar (numbers depend on the
deck):

.. code-block:: text

   Cantilever 1st FA: 0.396 Hz (ElastoDyn polynomial basis)
   Coupled 1st FA:    0.490 Hz (actual floating system frequency)
   Gap: +23.9% (platform restoring shifts apparent tower bending)

   Cantilever 1st SS: 0.396 Hz
   Coupled 1st SS:    0.489 Hz
   Gap: +23.5%

The 20-30 percent gap on a typical floating platform is expected,
not a bug. The polynomial encodes the cantilever modal basis
ElastoDyn integrates internally; the coupled solve is what
OpenFAST linearisation reports with platform 6-DOF, mooring, and
hydrostatic restoring all engaged.

Campbell diagrams
-----------------

Rotor-speed sweep with Hungarian MAC-based mode tracking,
constant-frequency tower lines, and the per-rev excitation
family overlaid:

.. code-block:: python

   import numpy as np
   from pybmodes.campbell import campbell_sweep, plot_campbell

   res = campbell_sweep(
       "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat",
       omega_rpm=np.linspace(0.0, 14.0, 15),
       n_blade_modes=4,
       n_tower_modes=2,
   )
   fig = plot_campbell(
       res,
       rated_rpm=12.1,
       operating_rpm=(6.9, 12.1),
       excitation_orders=[1, 3, 6, 9],
   )
   fig.savefig("campbell.png", dpi=150)

Engineering-report conventions are baked in:

- Four-family legend (Blades / Tower / Platform / Blade Passing)
- Inline mode labels (``1st Fore-Aft (0.48 Hz)``)
- Operating-window grey shading + double-arrow marker
- Optional ``log_freq=True`` for the per-rev rays
- Per-step MAC confidence in ``res.mac_to_previous``

Floating decks come out natively named — the six platform DOFs
(surge, sway, heave, roll, pitch, yaw) are detected by the FEM
classifier from the eigenvector content and carried through
``CampbellResult.labels`` without callers having to pass
``platform_modes`` by hand.

WISDEM / WindIO one-click
-------------------------

An ontology ``.yaml`` is consumed end-to-end. The library auto-
discovers companion HydroDyn / MoorDyn / ElastoDyn decks if
they're present and degrades to a clearly-labelled screening
preview when they aren't:

.. code-block:: python

   from pybmodes.models import Tower, RotatingBlade

   tower = Tower.from_windio("IEA-15-240-RWT.yaml")
   blade = RotatingBlade.from_windio("IEA-15-240-RWT.yaml", n_span=30)

   # Floating substructure — two-tier:
   # * with companion decks: industry-grade coupled model
   # * without: SCREENING-fidelity (UserWarning emitted)
   tower_f = Tower.from_windio_floating("IEA-15-240-RWT-UMaineSemi.yaml")

CLI wrapper does the discovery + report in one shot:

.. code-block:: bash

   pybmodes windio IEA-15-240-RWT.yaml \
       --campbell --max-rpm 8 \
       --out report.md

.. note::

   The blade reduction defaults to the WindIO **published**
   distributed elastic properties when the ontology carries them
   (``elastic_properties`` / ``elastic_properties_mb``) —
   minimises deltas against the reference model — and falls back
   to the PreComp-class thin-wall multi-cell layup reduction
   when they're absent. Force either with ``elastic="precomp"``
   or ``elastic="file"``.

Mode-by-mode comparison (MAC)
-----------------------------

.. code-block:: python

   from pybmodes.mac import compare_modes, plot_mac

   cmp = compare_modes(
       result_a, result_b,
       label_a="pyBmodes",
       label_b="BModes JJ",
   )
   for i, j, mac, freq_shift_pct in cmp.paired_modes:
       print(
           f"pyB mode {i} <-> ref mode {j}: "
           f"MAC={mac:.3f}, Δf={freq_shift_pct:+.2f}%"
       )
   fig = plot_mac(cmp)
   fig.savefig("mac.png", dpi=150)

The Hungarian-optimal pairing minimises total mismatch across
the full MAC matrix — robust to mode reordering that a naive
"i-th mode = i-th mode" comparison would miss.

Pre-solve sanity checks
-----------------------

Eight cheap, deterministic gates run before every ``.run()``
unless explicitly disabled:

.. code-block:: python

   from pybmodes.checks import check_model

   tower = Tower("my_deck.bmi")
   findings = check_model(tower, n_modes=4)

   for f in findings:
       print(f"[{f.severity}] {f.location}: {f.message}")

WARN and ERROR findings auto-route through ``UserWarning`` in
``.run(check_model=True)`` (the default). Suppress with
``check_model=False`` once a model is known-clean.

Common patterns
---------------

Loop over an entire directory of decks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pybmodes batch path/to/decks/ --validate --patch \
       --out reports/

Discovers every ElastoDyn main file under ``path/to/decks/``,
runs validate + patch on each, writes per-deck reports plus a
summary CSV.

Persist and reload a result
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pybmodes.models import ModalResult

   modal.save("modal.npz")                 # binary, with metadata
   modal.to_json("modal.json")              # text, schema_version "1"

   restored = ModalResult.load("modal.npz")

Each archive embeds pyBmodes version, UTC timestamp, source-file
path, and best-effort git hash — so a result is auditable months
later.

Vendor the bundled examples out of the wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pybmodes examples --copy ./my_examples            # both trees
   pybmodes examples --copy ./decks --kind decks     # reference decks only
   pybmodes examples --copy ./samples --kind samples # sample inputs only

Works whether you installed from PyPI or in editable mode from
source — the resolver finds ``_examples/`` relative to
``pybmodes.__file__``.

What's next
-----------

- :doc:`theory` — the FEM element, the four boundary conditions,
  the polynomial ansatz, and the solver dispatch.
- :doc:`data_sources` — every input format ``pybmodes`` reads
  and the convention for staging upstream data under
  ``external/``.
- :doc:`units` — SI everywhere; conventions on mode-shape
  normalisation, DOF order, and frame.
- :doc:`limitations` — what ``pybmodes`` deliberately does *not*
  attempt to do.
- :doc:`api` — autodoc-generated reference for every public name.
- :doc:`validation` — the per-case validation matrix.
