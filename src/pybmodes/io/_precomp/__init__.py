# Copyright 2024-2026 Jae Hoon Seo
# Marine Structural Mechanics and Integrity Lab (SMI Lab), Inha University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private composite cross-section reduction sub-package (issue #35).

A WindIO blade is a thin-walled, multi-cell, laminated-composite beam.
Turning that layup into the 1-D distributed beam properties pyBmodes'
:class:`pybmodes.models.RotatingBlade` FEM consumes (mass/length,
EI_flap, EI_edge, GJ, EA, mass moments of inertia, c.g. / shear-centre
/ tension-centre offsets, structural twist) is a classical-lamination-
theory (CLT) thin-wall shear-flow reduction — the NREL *PreComp*
method (Bir 2006, NREL/TP-500-38929).

Layout (each module independently unit-testable):

* :mod:`pybmodes.io._precomp.laminate` — material → reduced stiffness
  ``Q``, ply rotation ``Qbar(theta)``, ABD assembly, membrane
  condensation ``Atilde = A - B D^-1 B``. Pure CLT, no geometry / IO.
* :mod:`pybmodes.io._precomp.geometry` — airfoil arc
  parameterisation, spanwise blend, chord/twist/offset application,
  region arc-band resolution across both WindIO dialects.
* :mod:`pybmodes.io._precomp.reduction` — segment assembly,
  EA / EI principal-axis diagonalisation, single- then multi-cell
  Bredt–Batho ``GJ``, centres, mass moments.

This sub-package is **internal** (underscore-prefixed, same contract as
:mod:`pybmodes.io._elastodyn`); the public entry points live in
``pybmodes.io.windio_blade``. Pure ``numpy``; the WindIO YAML
dependency stays behind the optional ``[windio]`` extra.
"""
