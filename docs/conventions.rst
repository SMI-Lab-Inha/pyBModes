Coordinate systems, origins & datums
====================================

Modal analysis lives or dies on getting the reference frames right.
This page defines — precisely, and for **every** model type (land,
monopile, floating, blade) — where the origin is, which way the axes
point, what each offset field is measured *from*, and the sign of the
vertical datum. If a result looks wrong, it is almost always a frame
or datum mistake; start here.

Units are strict SI throughout — see :doc:`units`. The DOF ordering is
documented in code in :mod:`pybmodes.coords`; this page is the
human-facing companion.

.. contents::
   :local:
   :depth: 2


The one origin: the tower base
------------------------------

**Everything pyBmodes computes is referenced to a single origin — the
tower base** (the bottom node of the finite-element beam). There is no
second "platform-centre" or "global" frame inside pyBmodes; every
offset you supply is ultimately expressed relative to the tower base.

* **z-axis** runs **along the tower/blade span**, with the **base at**
  :math:`z = 0` and increasing toward the top/tip.
* **x-axis** is fore-aft (downwind), aligned with the *surge* /
  fore-aft-bending direction.
* **y-axis** is lateral (cross-wind), aligned with the *sway* /
  side-side-bending direction.

The finite-element beam is built from the **span length only**
(``radius`` for a blade, the flexible tower length ``TowerHt −
TowerBsHt`` for a tower). The *absolute* elevation of the base above
mean sea level is **not** baked into the beam — for a floating model
you tell pyBmodes where the base sits via ``draft`` (see
:ref:`floating-conventions`). A WindIO ``reference_axis.z`` that starts
at, say, ``+15 m`` contributes its *length* to the beam; the ``+15 m``
itself is conveyed separately and must not be added into the platform
offsets.


Rigid-body DOF order
--------------------

Every 6×6 platform matrix exposed through the API
(``i_matrix`` / ``hydro_M`` / ``hydro_K`` / ``mooring_K``) and every
MoorDyn / WAMIT input uses **OpenFAST DOF order**:

.. list-table::
   :header-rows: 1
   :widths: 8 12 40

   * - idx
     - DOF
     - physical sense
   * - 0
     - surge
     - fore-aft translation (along wind, +x)
   * - 1
     - sway
     - lateral translation (+y)
   * - 2
     - heave
     - vertical translation (+z)
   * - 3
     - roll
     - rotation about the surge (x) axis
   * - 4
     - pitch
     - rotation about the sway (y) axis
   * - 5
     - yaw
     - rotation about the heave (z) axis

This matches Jonkman (2010) NREL/TP-500-47535 Table 5-1 and the
``OC3Hywind.bmi`` blocks. See :mod:`pybmodes.coords` for the
machine-readable constants (``DOF_NAMES`` / ``DOF_INDEX``) and the
regression test that pins it. BModes JJ uses a different legacy column
order in some contexts (``surge, sway, yaw, roll, pitch, heave``) — when
comparing column-by-column, re-order by the modal-classifier *labels*,
never by positional index.


Boundary conditions (``hub_conn``)
----------------------------------

The tower-base / blade-root boundary condition is selected by
``hub_conn``:

.. list-table::
   :header-rows: 1
   :widths: 10 26 64

   * - ``hub_conn``
     - meaning
     - used for
   * - 1
     - Cantilever (all 6 base DOFs clamped)
     - Land towers; fixed-bottom monopiles (clamped at the mudline /
       transition piece).
   * - 2
     - Free-free (all 6 base DOFs released; reactions supplied by a
       ``PlatformSupport``)
     - Floating platforms. The rigid-body modes (surge … yaw) come from
       the platform restoring.
   * - 3
     - Soft monopile (axial + torsion clamped; lateral + rocking free)
     - Monopiles with explicit soil flexibility (Winkler ``distr_k``).
   * - 4
     - Pinned-free (deflections + twist clamped; bending **slopes**
       free)
     - Cable / tension-member modes (Bir 2009).


Tower-top / RNA mass (``tip_mass``)
-----------------------------------

A :class:`~pybmodes.io.bmi.TipMassProps` lumps the rotor-nacelle
assembly (or any concentrated top mass) at the tower top / blade tip:

* ``mass`` — kg.
* ``cm_offset`` — m, CM offset **transverse** to the span (along the
  tip-section local axis aligned with sway/lag).
* ``cm_axial`` — m, CM offset **along the span (z)**, i.e. how far the
  RNA CM sits *above* the tower top.
* ``ixx … iyz`` — kg·m², mass moments of inertia about the CM.

Omitting the RNA (or setting ``mass = 0``) on a tower makes the first
fore-aft frequency come out **10–30 % too high** — the rotor is a large
concentrated top mass. ``Tower.from_elastodyn`` lumps it for you from
the referenced blade file; a hand-built deck must supply it.


Per-case conventions
--------------------

Land-based tower
^^^^^^^^^^^^^^^^

* **Origin / datum:** tower base at :math:`z = 0` (the global
  ``TowerBsHt`` elevation; no sea-level datum is involved).
* **BC:** ``hub_conn = 1`` (clamped).
* **Support:** none, or a ``TensionWireSupport`` (guy wires).
* **Top mass:** RNA via ``tip_mass``.
* No ``draft`` / ``cm_pform`` / hydro / mooring fields apply.

Monopile (fixed-bottom)
^^^^^^^^^^^^^^^^^^^^^^^

* **Origin / datum:** the **mudline** (pile base), :math:`z = 0`.
* **BC:** ``hub_conn = 1`` — the pile + tower are spliced into one
  cantilever clamped at the mudline (``Tower.from_elastodyn_with_subdyn``,
  ``Tower.from_windio_with_monopile``). This is the **rigid fixed-base**
  model: no soil flexibility, so the 1st frequency is a few percent
  stiffer than a soil-included reference.
* **Soil-flexible alternative:** ``hub_conn = 3`` with a distributed
  Winkler foundation (``distr_k``) and/or a populated ``mooring_K`` for
  the soil springs (the ``CS_Monopile.bmi`` pattern).
* The two segments meet at the **transition piece**; each keeps its own
  wall schedule and steel grade.

.. _floating-conventions:

Floating platform
^^^^^^^^^^^^^^^^^

This is where most frame mistakes happen. Read carefully.

* **Origin:** the tower base (as always).
* **BC:** ``hub_conn = 2`` (free-free); the platform is a
  :class:`~pybmodes.io.bmi.PlatformSupport`.
* **Vertical datum: mean sea level (MSL), z = 0.** Every vertical
  platform scalar is measured **from MSL**, *not* from the tower base:

  .. list-table::
     :header-rows: 1
     :widths: 16 26 58

     * - field
       - meaning
       - sign
     * - ``draft``
       - tower-base elevation relative to MSL
       - **negative = above MSL**. A tower base ``+15 m`` above MSL is
         ``draft = -15``.
     * - ``cm_pform``
       - platform CM depth below MSL
       - positive **downward** (CM below the waterline).
     * - ``ref_msl``
       - hydro/WAMIT reference-point depth below MSL
       - positive **downward**; usually ``0`` (reference at the
         waterline).

  pyBmodes forms the CM→tower-base lever **internally** as
  ``cm_pform − draft``. So the tower-base height enters the maths
  **exactly once, through** ``draft``. **Do not also add it to**
  ``cm_pform`` / ``ref_msl`` — that double-counts the lever and gives a
  wrong answer.

* **Horizontal CM offset:** ``cm_pform_x`` / ``cm_pform_y`` are the
  platform CM offset **from the tower axis** (x = fore-aft, y =
  lateral), *not* a coordinate in any global / WAMIT frame. For a
  tower on the platform centroid (the usual case) they are ``≈ 0``.

  .. important::

     The horizontal offset is applied **only to the structural
     inertia**. The added-mass (``hydro_M``), hydrostatic
     (``hydro_K``) and mooring (``mooring_K``) matrices are assumed to
     be referenced at a point **on the tower axis**
     (``PtfmRefxt = PtfmRefyt = 0``) and receive **no** horizontal
     arm. If your tower sits well off the platform centroid, you must
     transfer *those* matrices to the tower axis yourself (the same
     rigid-arm congruence transform, ``Tᵀ M T``, applied to each,
     built symmetric first) before handing them over — otherwise the
     inertia and the hydro/mooring end up in different horizontal
     frames. ``check_model`` warns when ``√(cm_pform_x² + cm_pform_y²)``
     exceeds the platform's yaw radius of gyration, which is the usual
     symptom of a coordinate-origin value leaking into the field.

* **Static equilibrium is assumed.** The modal problem is linearised
  about the platform's static equilibrium — it floats **upright at its
  design trim and heel**, with the restoring matrices taken there. An
  off-axis mass does not sit as a static horizontal offset; the
  platform *trims* (fore-aft tilt) or *heels* (transverse tilt) until
  the combined CG is over the centre of buoyancy. A correctly modelled
  floater is ballasted/moored to float at its design trim/heel, so the
  residual CG-to-tower-axis offset is **small**. Feeding a large,
  un-trimmed horizontal offset describes a configuration that would
  never float that way, and its rigid-body modes are not meaningful.

Blade (rotating)
^^^^^^^^^^^^^^^^

* **Origin:** the blade root at :math:`z = 0`; ``hub_rad`` sets the
  root's radial offset from the rotation axis.
* **x-axis:** chordwise (edgewise/lag) at the root before pre-twist.
* **y-axis:** flapwise at the root before pre-twist.
* **Pre-twist** is a per-section rotation about z; reported mode-shape
  ordinates are in the **section** frame (after pre-twist).
* **Rotation:** ``rot_rpm`` adds centrifugal stiffening; ``precone``
  the coning angle.


Worked example: OC3 Hywind (validated)
--------------------------------------

The OC3 Hywind spar (tower base **+10 m** above MSL, platform CM
**89.9155 m** below MSL) is encoded as:

.. code-block:: python

   PlatformSupport(
       draft    = -10.0,     # tower base 10 m ABOVE MSL  (negative = above)
       cm_pform =  89.9155,  # platform CM 89.9155 m BELOW MSL
       ref_msl  =  0.0,      # WAMIT reference at the waterline
       cm_pform_x = 0.0, cm_pform_y = 0.0,   # tower on the spar axis
       ...
   )
   # internal CM->tower-base lever = cm_pform - draft
   #                               = 89.9155 - (-10) = 99.9155 m   ✓

This deck reproduces BModes JJ to ≤ 0.0003 % on the first nine modes.


Common pitfalls
---------------

* **Adding the tower-base height to the offsets.** The ``+15 m`` (or
  ``+10 m``) base elevation goes in ``draft`` (negative), *not* added
  to ``cm_pform`` / ``ref_msl``. Doing both double-counts the lever.

* **Using ``cm_pform_x`` to "mount" the tower off-centre.** It moves
  only the inertia; hydro and mooring stay on the tower axis. Transfer
  all six matrices to the tower base consistently, or use the deck path
  (``from_windio_floating(..., hydrodyn_dat=…, moordyn_dat=…)``), which
  references everything for you.

* **Leaving ``mooring_K`` in the platform/hull frame.** Like hydro, it
  must be referenced to the tower axis.

* **Confusing OpenFAST's two frames.** ``PtfmCMzt`` (CM) and
  ``PtfmRefzt`` (reference) are both in the tower-base *t*-frame;
  pyBmodes' ``ref_msl`` is ``PtfmRefzt``. Don't add one to the other.

* **Comparing 6×6 columns to BModes JJ positionally.** Re-order by the
  modal-classifier labels, not by index (the DOF orders differ).

The hand-build path is error-prone for exactly these reasons; for any
non-trivial floater the **deck path is strongly recommended** because it
performs all of the referencing once, internally and consistently.

See also :doc:`units` (units + mode-shape normalisation), :doc:`theory`
(the modelling basis), and :doc:`limitations` (scope of the floating
model).
