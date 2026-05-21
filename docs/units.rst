Units & conventions
===================

``pybmodes`` is **strict SI** everywhere on the API boundary.
No imperial units, no degree-vs-radian convention switches at
runtime, no mode-by-mode unit-system flags. The conventions
below are enforced by the public constructors and asserted in
the validation suite.

Quantities & units
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Quantity
     - Unit
     - Notes
   * - Length, displacement, mode-shape ordinates
     - m
     - Section nodes, tower height ``H``, blade tip ``radius``.
   * - Distributed mass density
     - kg / m
     - Section property ``mass_den``; per unit beam length.
   * - Lumped mass
     - kg
     - ``tip_mass``, ``mass_pform``, RNA mass.
   * - Axial stiffness ``EA``
     - N
     - Section property ``axial_stff``.
   * - Bending stiffness ``EI_flap``, ``EI_edge``
     - N · m²
     - Section properties ``flp_stff``, ``edge_stff``.
   * - Torsional stiffness ``GJ``
     - N · m²
     - Section property ``tor_stff``.
   * - Frequency
     - Hz
     - **Ordinary frequency**, not angular frequency
       :math:`\omega`. To convert:
       :math:`f\, [\mathrm{Hz}] = \omega\, [\mathrm{rad/s}] /
       (2\pi)`.
   * - Rotor speed
     - rpm
     - ``omega_rpm`` arrays in
       :func:`pybmodes.campbell.campbell_sweep` and
       ``RotatingBlade.rot_rpm`` are rpm. The FEM nondim path
       converts internally.
   * - Time / period
     - s
     - Mooring quasi-statics have no time dependence; period
       quantities appear only in the environmental-loading
       helpers (Kaimal / JONSWAP spectra).
   * - Angle, pre-twist
     - rad
     - The WindIO reader **auto-detects** degree-convention
       files (where root twist exceeds the radian ceiling) and
       converts to radians — see *Conventions* below.
   * - Gravity
     - m / s²
     - Default ``g = 9.80665`` (CODATA standard gravity).
   * - Water density
     - kg / m³
     - Default ``rho = 1025.0`` (seawater, OpenFAST convention).

Conversion tables for common units
----------------------------------

If you receive a deck in non-SI units, convert at the boundary
before constructing a ``Tower`` / ``RotatingBlade``. The most
common conversions:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - From
     - To
     - Multiplier
   * - inch
     - m
     - 0.0254
   * - ft
     - m
     - 0.3048
   * - lb (mass)
     - kg
     - 0.45359237
   * - lb / ft (mass / length)
     - kg / m
     - 1.48816
   * - psi (stress)
     - Pa
     - 6894.757
   * - ksi (stress)
     - Pa
     - 6.894757 × 10⁶
   * - lbf (force)
     - N
     - 4.4482216
   * - lbf · ft (moment)
     - N · m
     - 1.355818
   * - rad/s (angular freq)
     - Hz (ordinary freq)
     - 1 / (2π) ≈ 0.159155
   * - rad/s (rotational)
     - rpm
     - 60 / (2π) ≈ 9.549297
   * - deg (angle)
     - rad
     - π / 180 ≈ 0.017453
   * - knot (water current)
     - m / s
     - 0.51444
   * - bar (pressure)
     - Pa
     - 1.0 × 10⁵

Mode-shape normalisation
------------------------

Mode shapes are **mass-normalised**:

.. math::

   \boldsymbol{\varphi}^{\mathrm{T}}\, \mathbf{M}\,
   \boldsymbol{\varphi} \;=\; 1

The sign of each mode shape is canonicalised by the rule
"maximum-amplitude DOF is positive", so two solves of the same
problem produce sign-stable shapes that can be MAC-compared
without a sign-flip ambiguity.

To convert to **tip-unit normalisation** (max ordinate = 1)
post-solve:

.. code-block:: python

   for shape in result.shapes:
       peak = max(
           abs(shape.flap_disp).max(),
           abs(shape.lag_disp).max(),
           abs(shape.twist).max(),
       )
       shape.flap_disp /= peak
       shape.lag_disp /= peak
       shape.twist /= peak

The ElastoDyn polynomial coefficients are written in
**tip-unit** nondim (:math:`\mathrm{SHP}(1) = \sum c_i = 1`);
the polynomial-fit routines apply that normalisation
internally before the constrained least-squares fit, so
``compute_blade_params`` / ``compute_tower_params`` operate
correctly on either mass-normalised or tip-unit mode shapes.

DOF order
---------

The canonical **6-DOF order** for every platform-related
matrix (``mooring_K``, ``hydro_K``, ``i_matrix``, MoorDyn
output, WAMIT output) is the **OpenFAST convention**:

.. code-block:: text

   index   DOF
   -----   -----
     0     surge   (translation along X, forward)
     1     sway    (translation along Y, sideways)
     2     heave   (translation along Z, vertical)
     3     roll    (rotation about X)
     4     pitch   (rotation about Y)
     5     yaw     (rotation about Z)

See :mod:`pybmodes.coords` for the documented construction and
``tests/test_mooring.py::test_oc3hywind_bmi_dof_order_matches_jonkman_2010``
for the regression that pins it against Jonkman 2010 Table 5-1.

A common pitfall: BModes JJ uses
``[surge, sway, yaw, roll, pitch, heave]`` in some legacy
contexts. When comparing pyBmodes output against BModes JJ
column-by-column, **re-order via the modal-classifier labels**,
not by positional index. See ``cases/iea15_deep_diagnostic.md``
for the worked example.

Polynomial coefficients
-----------------------

ElastoDyn-compatible polynomials are written in the
dimensionless span coordinate :math:`s = h / H` with the first
two coefficients implicitly zero:

.. math::

   \mathrm{SHP}(s) = c_2 s^2 + c_3 s^3 + c_4 s^4 + c_5 s^5
                  + c_6 s^6

``BladeElastoDynParams.coefficients()`` and the corresponding
tower accessor return :math:`(c_2, \ldots, c_6)` as a length-5
array, with the constraint :math:`\sum c_i = 1` (tip-unit at
the tip). The constrained least-squares fit handles the
normalisation; you don't need to pre-scale the input.

Frame conventions
-----------------

Tower base frame
^^^^^^^^^^^^^^^^

- **z-axis** along the span (tower base at :math:`z = 0`).
- For a fixed-base tower this is also the global frame at the
  tower base elevation (``TowerBsHt``).
- For a **floating** tower this is the *platform-attached
  frame* — the polynomial basis assumes clamped-base modes
  referenced to it, and platform 6-DOF motion is added at
  runtime by ElastoDyn as a separate rigid-body sum. See
  :doc:`theory` and :doc:`limitations`.

Blade root frame
^^^^^^^^^^^^^^^^

- **z-axis** along the span (root at :math:`z = 0`).
- **x-axis** is the chordwise (edgewise) direction of the
  root section before any pre-twist.
- **y-axis** is the flapwise direction of the root section
  before any pre-twist.
- Pre-twist is applied per section as a rotation about z; the
  reported mode-shape ordinates are in the **section** frame
  (i.e. after the pre-twist rotation).

Section centre-of-mass offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cm_loc`` and ``cm_axial`` in the BMI header are the section
centre-of-mass offsets along the local section frame's axes
(``cm_loc`` along axis-1, ``cm_axial`` along axis-3 i.e. the
span).

The BMI's ``inertia`` array carries the section mass moments
of inertia about the centre-of-mass — **not** the elastic
centre — and the parallel-axis transfer is applied during
nondim by :func:`pybmodes.fem.nondim`.

Twist auto-detection (WindIO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The WindIO standard prescribes radians for ``twist`` in the
blade ``outer_shape``, but WISDEM-generated files for IEA-15
ship degrees. Applying ``np.degrees`` unconditionally turns the
already-degree-valued root twist (~15.6°) into ≈894°.

The reader **auto-detects**: if any twist value exceeds the
radian ceiling (~2π), it's assumed degrees and returned as-is;
otherwise it's assumed radians and converted to degrees by
``np.degrees``. The detection threshold lives in
``pybmodes.io.windio_blade._TWIST_RADIAN_CEILING``.

Common pitfalls
---------------

Mixing ω and f
^^^^^^^^^^^^^^^

ElastoDyn / BModes output frequency in **Hz**, but a lot of
hand-written analytical references give **angular frequency
:math:`\omega = 2\pi f` in rad/s** — especially closed forms
for cable / beam modes. When matching against a paper, check
which is reported.

Confusing OpenFAST's two coordinate systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenFAST uses two related but distinct frames:

- **t**: the tower-base frame, origin at ``TowerBsHt``.
- **z**: the platform-reference frame, origin at
  ``PtfmRefzt``.

``PtfmCMzt`` is the platform CM **in the t-frame**;
``PtfmRefzt`` is the platform reference point in the t-frame
(typically the mean water line for a floater). pyBmodes' BMI
field ``ref_msl`` is ``PtfmRefzt`` interpreted that way.

If you transcribe ``cm_pform = PtfmCMzt`` into a BMI by hand,
don't add ``PtfmRefzt`` to it — that's a double-count. See
:func:`pybmodes.models.tower._scan_platform_fields` for the
exact mapping pyBmodes uses.

Forgetting the rotor mass on a tower deck
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Tower.from_elastodyn`` reads the ``BldFile(1)`` referenced
from the main file **only** to lump the rotor mass into the
tower-top assembly. If you build a tower by hand without that
lump (or set ``tip_mass = 0``), the 1st tower fore-aft
frequency comes out 10–30 % too high — the rotor adds a
substantial concentrated mass at the top.

For a hand-authored BMI:

.. code-block:: python

   # IEA-15-240-RWT rotor + nacelle + hub assembly:
   tip_mass_props = TipMassProps(
       mass=1017000.0,   # kg
       cm_loc=0.0,       # m
       cm_axial=4.0,     # m, above TowerBsHt
       ixx=..., iyy=..., izz=..., ixy=..., iyz=..., izx=...,
   )

Treating the polynomial-fit ratio as an absolute error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pybmodes validate`` reports a **ratio** between the file-
shipped polynomial's RMS residual and pyBmodes' own fit's RMS
residual. A ratio of 100 means the file's polynomial is 100×
worse than pyBmodes' would be — but in absolute terms the
file's may still be fine if the underlying mode shape is
nearly linear. The verdict (PASS / WARN / FAIL) is the actual
gate; the ratio is informational.
