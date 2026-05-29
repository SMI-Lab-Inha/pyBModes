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


_ELASTIC_TIP_RATIO_FLOOR = 0.3
"""Minimum ratio of elastic tip displacement to raw tip displacement.

After the affine root-tangent subtraction (the Improved Direct Method
already used inside :func:`compute_tower_params_report` and
:func:`pybmodes.elastodyn.params._remove_root_rigid_motion`), the
residual tip displacement of a pure rigid-body mode is zero up to
numerical noise. A pure tower bending mode in a clamped-base frame
satisfies phi(0) = phi'(0) = 0 already, so the subtraction does
nothing and the elastic tip equals the raw tip. Mixed coupled modes
fall between the two extremes.

The 0.3 floor separates the two regimes cleanly. On the canonical
OC3 Hywind coupled solve, labelled rigid-body modes report ratios at
or below 0.007 (numerical noise) and the 1st FA / SS tower bending
pair reports ratios at 1.64 (the platform pitch motion opposes the
elastic tip swing, so the elastic component is larger than the raw
tip). Tower bending modes from the 2nd FA pair up through the 4th
pair report ratios from 0.69 to 4.57. The 0.3 floor sits well below
any of the tower band and well above the rigid-body band.
"""


def _has_tower_bending_content(shape) -> bool:
    """Return ``True`` when the mode shape's flap or lag axis carries
    significant tower bending content after the affine root-tangent
    subtraction. Used to distinguish coupled rigid-body modes
    (negligible elastic content) from tower bending modes regardless
    of frequency. Robust to a TLP-class stiff platform where a
    rigid-body frequency can sit above the spar-class 0.2 Hz band.
    """
    from pybmodes.elastodyn.params import _remove_root_rigid_motion

    for disp, slope in (
        (shape.flap_disp, shape.flap_slope),
        (shape.lag_disp, shape.lag_slope),
    ):
        raw_tip = abs(float(disp[-1]))
        if raw_tip < 1.0e-12:
            continue
        elastic_tip = abs(float(_remove_root_rigid_motion(
            shape.span_loc, disp, slope,
        )[-1]))
        if elastic_tip >= _ELASTIC_TIP_RATIO_FLOOR * raw_tip:
            return True
    return False


def _drop_rigid_body_shapes(modal):
    """Filter out platform rigid-body modes from a floating modal result.

    On a free-base ``hub_conn = 2`` solve the pipeline tags most
    rigid-body modes with ``mode_labels`` entries naming the DOF
    (``"surge"`` / ``"sway"`` / ``"heave"`` / ``"roll"`` / ``"pitch"``
    / ``"yaw"``). The tower-family classifier inside
    :func:`compute_tower_params_report` picks the lowest-frequency
    FA-dominated candidate and would otherwise land on surge for the
    coupled OC3 Hywind solve (~0.008 Hz), so labelled rigid-body
    shapes have to be removed before classification.

    :func:`pybmodes.fem.platform_modes.classify_platform_modes` is
    allowed to leave a strongly-coupled or rotated rigid-body pair
    tagged ``None``. A pure label-based filter would then forward
    those candidates into the tower-family classifier and the
    diagnostic could land on a platform mode as the coupled 1st FA.
    A frequency-floor cut catches the spar-class case (rigid-body
    modes below the 1st tower bending) but does not catch a
    TLP-class corner case (a stiff-platform rigid-body mode sitting
    above 0.2 Hz could still be unlabelled). The filter therefore
    keeps only the unlabelled modes whose shape content is
    *recognisably tower bending* via :func:`_has_tower_bending_content`,
    which compares the elastic tip displacement after the affine
    root-tangent subtraction against the raw tip displacement. This
    is the same projection :func:`_tower_candidate` already applies
    before polynomial fitting, so it does not introduce a new physics
    assumption.

    On a cantilever solve ``mode_labels`` is ``None`` and the input is
    returned unchanged.
    """
    from pybmodes.models.result import ModalResult

    if modal.mode_labels is None:
        return modal
    indices = [
        i
        for i, shape in enumerate(modal.shapes)
        if modal.mode_labels[i] is None
        and _has_tower_bending_content(shape)
    ]
    if not indices:
        return modal
    return ModalResult(
        frequencies=modal.frequencies[indices],
        shapes=[modal.shapes[i] for i in indices],
        mode_labels=None,
    )
