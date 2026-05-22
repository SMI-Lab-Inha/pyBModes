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

"""Bundled example inputs and reference decks.

This sub-package vendors two trees that ship inside the wheel so a
``pip install pybmodes`` user can copy them out to a working directory
without keeping a full git clone of the repository.

- ``sample_inputs/`` — pyBmodes-authored, Apache 2.0-licensed ``.bmi`` and
  section-property ``.dat`` files. Four analytical-reference cases
  (uniform blade, tower with top mass, rotating uniform blade,
  pinned-free cable) plus eleven reference-wind-turbine sub-cases under
  ``reference_turbines/``. ``verify.py`` runs the four analytical
  cases against closed-form references.
- ``reference_decks/`` — six pre-patched ElastoDyn decks (three
  land/monopile + three floating) whose polynomial blocks have been
  regenerated from the structural inputs via ``Tower.from_elastodyn``.

The trees are intentionally treated as opaque data; the ``.py``
helpers inside them (``verify.py``, ``reference_turbines/build.py``)
are intended to be run *after* vendoring out via
``pybmodes examples --copy <dir>``, not imported.

Users discover the bundles through three paths:

- ``pybmodes examples --copy DIR`` — CLI that copies one or both
  trees into a user-supplied directory.
- Browsing the wheel install directly under
  ``site-packages/pybmodes/_examples/``.
- For developers working from a source checkout, the GitHub source
  tree under the same path.
"""
