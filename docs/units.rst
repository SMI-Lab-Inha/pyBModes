Units
=====

``pybmodes`` is **strict SI** everywhere on the API boundary. No
imperial units, no degree-vs-radian convention switches at runtime,
no mode-by-mode unit-system flags. The conventions below are
enforced by the public constructors and asserted in the validation
suite.

Quantities and units
--------------------

.. list-table::
   :header-rows: 1

   * - Quantity
     - Unit
     - Notes
   * - Length, displacement, mode-shape ordinates
     - m
     -
   * - Mass density (distributed)
     - kg / m
     - per unit beam length
   * - Lumped mass
     - kg
     -
   * - Axial stiffness ``EA``
     - N
     -
   * - Bending stiffness ``EI_flap``, ``EI_edge``
     - N · m²
     -
   * - Torsional stiffness ``GJ``
     - N · m²
     -
   * - Frequency
     - Hz
     - mode frequencies are always ordinary frequency in Hz,
       not angular frequency :math:`\omega` in rad/s.
   * - Rotor speed
     - rpm
     - ``omega_rpm`` arrays in :func:`pybmodes.campbell.campbell_sweep`
       and ``RotatingBlade.rot_rpm`` are rpm. The FEM nondim path
       converts internally.
   * - Time / period
     - s
     -
   * - Angle, pre-twist
     - rad
     - the WindIO reader detects degree-convention files (where
       root twist > the radian ceiling) and converts to radians.
   * - Gravity
     - m / s²
     - default ``g = 9.80665`` (CODATA standard gravity).
   * - Water density
     - kg / m³
     - default ``rho = 1025.0`` (seawater, OpenFAST convention).

Mode-shape normalisation
------------------------

Mode shapes are returned **mass-normalised**: each mode shape ``φ``
satisfies :math:`\varphi^{\mathrm{T}} M \varphi = 1`. The sign of each
mode shape is canonicalised by the rule "maximum-amplitude DOF is
positive", so two solves of the same problem produce sign-stable
shapes that can be MAC-compared without a sign-flip ambiguity.

DOF order
---------

The canonical 6-DOF order (consumed by ``mooring_K``, ``hydro_K``,
``i_matrix``, and every platform-related routine) is
``[surge, sway, heave, roll, pitch, yaw]`` — the OpenFAST
convention. See :mod:`pybmodes.coords` for the documented
construction and ``tests.test_mooring`` for the regression that
pins it against Jonkman 2010 Table 5-1.

Polynomial coefficients
-----------------------

ElastoDyn-compatible polynomials are written in the dimensionless
span coordinate :math:`s = h / H` with the first two coefficients
implicitly zero:

.. math::

   \mathrm{SHP}(s) = c_2 s^2 + c_3 s^3 + c_4 s^4 + c_5 s^5 + c_6 s^6

``BladeElastoDynParams.coefficients()`` and the corresponding tower
accessor return :math:`(c_2, \ldots, c_6)` as a length-5 array, with
the constraint :math:`\sum c_i = 1` (the mode shape is unit at the
tip in the same nondim).

Frame conventions
-----------------

- Tower mode shapes are in the **tower base frame** (z-axis = span,
  origin at ``TowerBsHt``). For a floating tower this is the
  *platform-attached* frame; the polynomial basis assumes
  clamped-base modes referenced to it — see :doc:`theory` and
  :doc:`limitations`.
- Blade mode shapes are in the **blade root frame** (z-axis = span,
  origin at the root, pre-twist applied per section).
- ``cm_loc`` and ``cm_axial`` in the BMI header are the section
  centre-of-mass offsets along the local section frame's axes.
