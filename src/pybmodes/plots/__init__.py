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

"""Professional plotting utilities for pybmodes results.

Requires matplotlib >= 3.7.  Install with::

    pip install "pybmodes[plots]"

All functions return a :class:`matplotlib.figure.Figure` object; call
``fig.show()`` or ``fig.savefig(path)`` as needed.
"""

from .environmental import (
    jonswap_spectrum,
    kaimal_spectrum,
    plot_environmental_spectra,
)
from .mode_shapes import (
    bir_mode_shape_plot,
    bir_mode_shape_subplot,
    blade_fit_pairs,
    plot_fit_quality,
    plot_mode_shapes,
    tower_fit_pairs,
)
from .style import MATLAB_LINES, PALETTE, apply_style

__all__ = [
    "plot_mode_shapes",
    "plot_fit_quality",
    "blade_fit_pairs",
    "tower_fit_pairs",
    "bir_mode_shape_plot",
    "bir_mode_shape_subplot",
    "plot_environmental_spectra",
    "kaimal_spectrum",
    "jonswap_spectrum",
    "apply_style",
    "MATLAB_LINES",
    "PALETTE",
]
