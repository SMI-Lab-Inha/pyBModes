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

"""Polynomial-vs-FEM comparison for the IEA-15-240-RWT monopile tower.

IEA-15 ships several sub-cases under ``OpenFAST/`` (Monopile,
UMaineSemi, OLAF, default-without-main); this script runs the
``IEA-15-240-RWT-Monopile`` one — the only fixed-bottom configuration
that ships a complete SubDyn deck. Delegates to
:mod:`visualise_polynomial_comparison_5mw_monopile.main` for the
shared comparison pipeline (combined-cantilever FEM via
``Tower.from_elastodyn_with_subdyn``, TP-rigid-motion subtraction,
amplitude / RMS metrics).

Run from the repo root::

    set PYTHONPATH=%CD%\\src
    python scripts\\visualise_polynomial_comparison_iea15_monopile.py
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for p in (SRC_DIR, SCRIPTS_DIR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from visualise_polynomial_comparison_5mw_monopile import main  # noqa: E402

_DECK_DIR = (
    REPO_ROOT / "external" / "OpenFAST_files" / "IEA-15-240-RWT" / "OpenFAST"
    / "IEA-15-240-RWT-Monopile"
)
_MAIN = _DECK_DIR / "IEA-15-240-RWT-Monopile_ElastoDyn.dat"
_SUBDYN = _DECK_DIR / "IEA-15-240-RWT-Monopile_SubDyn.dat"
_OUT = (
    REPO_ROOT / "scripts" / "outputs"
    / "polynomial_comparison_iea15_monopile_TwFA2_TwSS2.png"
)

if __name__ == "__main__":
    raise SystemExit(main([
        "--main", str(_MAIN),
        "--subdyn", str(_SUBDYN),
        "--out", str(_OUT),
        "--label", "IEA-15-240-RWT monopile",
    ]))
