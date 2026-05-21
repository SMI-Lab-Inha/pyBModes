Quickstart
==========

Five minutes from import to a validated tower modal solve.

A synthetic tower
-----------------

Every public constructor accepts a ``.bmi`` deck path; the bundled
samples ship with the wheel:

.. code-block:: python

   from pybmodes.models import Tower
   from pybmodes.cli import _resolve_examples_root

   bmi = _resolve_examples_root() / "sample_inputs" / "02_tower_topmass" / "tower_topmass.bmi"
   tower = Tower(bmi)
   result = tower.run(n_modes=4)

   for f in result.frequencies:
       print(f"{f:.4f} Hz")

``ModalResult`` carries frequencies, per-node mode shapes (flap /
lag / twist / axial), and — when requested — modal-participation
fractions and polynomial-fit residuals.

From an OpenFAST ElastoDyn deck
-------------------------------

A real-world flow: read an industry deck, solve modes, fit
ElastoDyn-compatible polynomials, write them back.

.. code-block:: python

   from pybmodes.models import Tower
   from pybmodes.elastodyn import compute_tower_params, patch_dat

   # Reads the ElastoDyn main + tower file from one path.
   tower = Tower.from_elastodyn("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat")
   modal = tower.run(n_modes=4)

   # Constrained 6th-order fit, FA/SS family selection, torsion-contamination filter.
   params = compute_tower_params(modal)

   # Rewrite the polynomial blocks in-place (use --dry-run / --diff via the CLI for safety).
   patch_dat("NRELOffshrBsline5MW_Onshore_ElastoDyn.dat", params)

The same workflow on the CLI:

.. code-block:: bash

   pybmodes validate NRELOffshrBsline5MW_Onshore_ElastoDyn.dat
   pybmodes patch    NRELOffshrBsline5MW_Onshore_ElastoDyn.dat --dry-run --diff
   pybmodes patch    NRELOffshrBsline5MW_Onshore_ElastoDyn.dat --backup
   pybmodes report   NRELOffshrBsline5MW_Onshore_ElastoDyn.dat --format md --out report.md

A Campbell diagram
------------------

``campbell_sweep`` solves the blade across a rotor-speed grid with
Hungarian MAC-based mode tracking, then overlays the rotor-speed-
independent tower modes and the per-rev (1P / 3P / 6P / 9P)
excitation rays:

.. code-block:: python

   from pybmodes.campbell import campbell_sweep, plot_campbell
   import numpy as np

   res = campbell_sweep(
       "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat",
       omega_rpm=np.linspace(0.0, 14.0, 15),
       n_blade_modes=4,
       n_tower_modes=2,
   )
   fig = plot_campbell(res, rated_rpm=12.1, operating_rpm=(6.9, 12.1))
   fig.savefig("campbell.png", dpi=150)

For a coupled floating tower the six lowest tower columns come out
natively named ``surge`` / ``sway`` / ``heave`` / ``roll`` / ``pitch``
/ ``yaw``; ``plot_campbell`` automatically labels each one inline
with its frequency in brackets.

From a WindIO ontology
----------------------

A WISDEM ontology ``.yaml`` is consumed end-to-end:

.. code-block:: python

   from pybmodes.models import Tower, RotatingBlade

   tower = Tower.from_windio("IEA-15-240-RWT.yaml")     # tubular tower
   blade = RotatingBlade.from_windio("IEA-15-240-RWT.yaml", n_span=30)

For a floating ``floating_platform`` block, ``Tower.from_windio_floating``
auto-discovers any companion HydroDyn / MoorDyn / ElastoDyn decks and
upgrades to the industry-grade coupled model; without them it degrades
to a screening preview and says so via a ``UserWarning``. The CLI
wraps this:

.. code-block:: bash

   pybmodes windio IEA-15-240-RWT.yaml --campbell --max-rpm 8 --out report.md

Vendor the bundled examples
---------------------------

Every install ships ``sample_inputs/`` (analytical references + RWT
samples) and ``reference_decks/`` (six patched ElastoDyn decks) inside
the wheel. Copy them out with:

.. code-block:: bash

   pybmodes examples --copy ./my_examples

Mode-by-mode comparison
-----------------------

.. code-block:: python

   from pybmodes.mac import compare_modes, plot_mac

   cmp = compare_modes(result_a, result_b, label_a="pyBmodes", label_b="BModes")
   for p in cmp.paired_modes:
       print(p)         # (i, j, MAC, freq_shift_pct)
   plot_mac(cmp).savefig("mac.png")

Next steps
----------

- :doc:`theory` — what FEM element, what boundary conditions, what
  polynomial ansatz, and why.
- :doc:`data_sources` — every input format ``pybmodes`` reads.
- :doc:`limitations` — what ``pybmodes`` deliberately does not
  attempt to do.
- :doc:`api` — full public API reference.
- :doc:`validation` — the per-case validation matrix.
