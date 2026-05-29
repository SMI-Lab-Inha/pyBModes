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

"""Floating-tower frequency-gap diagnostic.

A floating ElastoDyn deck has two natural tower bending frequencies that
differ by design. The polynomial coefficient blocks in the deck encode
the cantilever tower mode shape, which is the only basis ElastoDyn's
``SHP = sum_i c_i * (h/H)^(i+1)`` ansatz can represent (the source-code
citations live in
``src/pybmodes/_examples/reference_decks/FLOATING_CASES.md``). The
coupled-system frequency that OpenFAST linearisation reports includes
platform 6-DOF participation, mooring restoring, and hydrostatic
restoring. The two can differ by 20-30 percent on floating platforms,
and the gap is expected rather than a bug.

``report_floating_frequency_gap`` runs both pyBmodes solves on the same
deck and reports the gap as a short text block, so users reconciling
pyBmodes-generated polynomials against OpenFAST linearisation output do
not have to re-derive the cantilever-vs-coupled architecture from
scratch.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass
class FloatingFrequencyGap:
    """Cantilever vs coupled tower bending frequencies for one deck.

    Cantilever frequencies come from a clamped-base
    :meth:`~pybmodes.models.Tower.from_elastodyn` solve. This is the
    modal basis ElastoDyn integrates into ``MTFA``/``KTFA`` at runtime
    and the basis the shipped polynomial coefficients describe.

    Coupled frequencies come from a free-base
    :meth:`~pybmodes.models.Tower.from_elastodyn_with_mooring` solve
    with mooring stiffness, hydrostatic restoring, and platform inertia
    engaged. These are the numbers an OpenFAST linearisation reports.
    """

    cantilever_fa_1: float
    cantilever_ss_1: float
    coupled_fa_1: float
    coupled_ss_1: float

    @property
    def gap_fa_1_pct(self) -> float:
        return 100.0 * (self.coupled_fa_1 - self.cantilever_fa_1) / self.cantilever_fa_1

    @property
    def gap_ss_1_pct(self) -> float:
        return 100.0 * (self.coupled_ss_1 - self.cantilever_ss_1) / self.cantilever_ss_1

    def format_report(self) -> str:
        """Return a short text block summarising the gap."""
        return (
            f"Cantilever 1st FA: {self.cantilever_fa_1:.3f} Hz (ElastoDyn polynomial basis)\n"
            f"Coupled 1st FA:    {self.coupled_fa_1:.3f} Hz (actual floating system frequency)\n"
            f"Gap: {self.gap_fa_1_pct:+.1f}% (platform restoring shifts apparent tower bending)\n"
            "\n"
            f"Cantilever 1st SS: {self.cantilever_ss_1:.3f} Hz\n"
            f"Coupled 1st SS:    {self.coupled_ss_1:.3f} Hz\n"
            f"Gap: {self.gap_ss_1_pct:+.1f}%"
        )


def report_floating_frequency_gap(
    main_dat_path: str | pathlib.Path,
    moordyn_dat_path: str | pathlib.Path,
    hydrodyn_dat_path: str | pathlib.Path | None = None,
    *,
    n_modes: int = 10,
) -> FloatingFrequencyGap:
    """Run cantilever and coupled solves on the same floating deck.

    The cantilever solve is the modal basis ElastoDyn consumes for its
    runtime tower-bending DOFs. The coupled solve is what OpenFAST
    linearisation produces when platform 6-DOF, mooring, and
    hydrostatic restoring are all engaged. The returned
    :class:`FloatingFrequencyGap` lets a user reconcile
    pyBmodes-generated polynomial coefficients against OpenFAST
    linearisation output without re-deriving the architectural reason
    they differ.

    On the coupled solve, ``n_modes + 6`` modes are requested so that
    after the six platform rigid-body modes (surge / sway / heave /
    roll / pitch / yaw) are filtered out, the tower-bending family
    selector still sees ``n_modes`` candidates.
    """
    from pybmodes.elastodyn.params import compute_tower_params_report
    from pybmodes.models.tower import Tower

    cantilever_model = Tower.from_elastodyn(main_dat_path)
    coupled_model = Tower.from_elastodyn_with_mooring(
        main_dat_path, moordyn_dat_path, hydrodyn_dat_path,
    )

    cant_modal = cantilever_model.run(n_modes=n_modes, check_model=False)
    coupled_modal = coupled_model.run(n_modes=n_modes + 6, check_model=False)
    coupled_tower_modal = _drop_rigid_body_shapes(coupled_modal)

    _, cant_report = compute_tower_params_report(cant_modal)
    _, coupled_report = compute_tower_params_report(coupled_tower_modal)

    cant_freqs = {s.mode_number: float(s.freq_hz) for s in cant_modal.shapes}
    coupled_freqs = {
        s.mode_number: float(s.freq_hz) for s in coupled_tower_modal.shapes
    }

    return FloatingFrequencyGap(
        cantilever_fa_1=cant_freqs[cant_report.selected_fa_modes[0]],
        cantilever_ss_1=cant_freqs[cant_report.selected_ss_modes[0]],
        coupled_fa_1=coupled_freqs[coupled_report.selected_fa_modes[0]],
        coupled_ss_1=coupled_freqs[coupled_report.selected_ss_modes[0]],
    )


_RIGID_BODY_FLOOR_HZ = 0.2
"""Tower-bending lower-frequency floor for the rigid-body filter.

Any unlabelled mode whose frequency lies below this floor is assumed
to be a platform rigid-body mode the classifier could not attribute
to a single DOF. The platform 6-DOF rigid-body modes on a realistic
floating wind turbine sit at 0.005 to 0.15 Hz; the 1st tower bending
pair sits at 0.3 Hz or higher (the OC3 Hywind cantilever-vs-coupled
gap of around 25 percent stiffens, never softens, the apparent tower
bending). The 0.2 Hz floor sits comfortably above the rigid-body
band and well below any realistic 1st tower bending frequency.
"""


def _drop_rigid_body_shapes(modal):
    """Filter out platform rigid-body modes from a floating modal result.

    On a free-base ``hub_conn = 2`` solve the pipeline tags rigid-body
    modes with ``mode_labels`` entries naming the DOF
    (``"surge"`` / ``"sway"`` / ``"heave"`` / ``"roll"`` / ``"pitch"``
    / ``"yaw"``). The tower-family classifier inside
    :func:`compute_tower_params_report` picks the lowest-frequency
    FA-dominated candidate and would otherwise land on surge for the
    coupled OC3 Hywind solve (~0.008 Hz), so labelled rigid-body
    shapes have to be removed before classification.

    :func:`pybmodes.fem.platform_modes.classify_platform_modes` may
    leave a strongly-coupled or rotated rigid-body pair tagged
    ``None``, in which case a pure label-based filter forwards those
    candidates into the tower-family classifier and the diagnostic
    could land on a low-frequency platform mode as the coupled 1st FA.
    The filter therefore additionally drops any unlabelled mode whose
    frequency lies below ``_RIGID_BODY_FLOOR_HZ`` (0.2 Hz). The
    rigid-body band on any realistic floating wind turbine sits at
    0.005 to 0.15 Hz, well below this floor; the 1st tower bending
    pair sits well above it, so this second cut catches the unlabelled
    rigid-body case Codex flagged on PR #114 without trimming any real
    tower mode. The eigensolver's asymmetric path on a soft-pitch spar
    such as OC3 Hywind drops some rigid-body modes from the returned
    spectrum, so we cannot rely on an index-based cut alone.

    On a cantilever solve ``mode_labels`` is ``None`` and the input is
    returned unchanged.
    """
    from pybmodes.models.result import ModalResult

    if modal.mode_labels is None:
        return modal
    n = len(modal.shapes)
    indices = [
        i
        for i in range(n)
        if modal.mode_labels[i] is None
        and float(modal.frequencies[i]) > _RIGID_BODY_FLOOR_HZ
    ]
    if not indices:
        return modal
    return ModalResult(
        frequencies=modal.frequencies[indices],
        shapes=[modal.shapes[i] for i in indices],
        mode_labels=None,
    )
