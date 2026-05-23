"""One test per check in :mod:`pybmodes.checks`.

Each test synthesises a small model that trips exactly one check and
asserts the corresponding ``ModelWarning`` appears in
``check_model(model)``'s output. A separate test confirms that a clean
model returns an empty list, and that ``Tower.run(..., check_model=False)``
suppresses the auto-run.

The synthetic-fixture helpers under :mod:`tests._synthetic_bmi` build
parseable ``.bmi`` and ``.dat`` files in ``tmp_path``; checks that
require an offending input the parser would reject (e.g. zero
mass density) bypass parsing by mutating the parsed ``SectionProperties``
in place before invoking ``check_model``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from pybmodes.checks import ModelWarning, check_model
from pybmodes.io.bmi import BMIFile, PlatformSupport, ScalingFactors, TipMassProps
from pybmodes.io.sec_props import SectionProperties
from pybmodes.models import Tower
from tests._synthetic_bmi import write_bmi, write_uniform_sec_props

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _build_synthetic_tower(tmp_path: pathlib.Path) -> Tower:
    """Build a clean uniform-tower model.

    The values are deliberately middle-of-the-road so the model trips
    no checks; individual tests then mutate the resulting model to
    produce exactly one offence.
    """
    write_uniform_sec_props(tmp_path / "secs.dat")
    write_bmi(
        tmp_path / "tower.bmi",
        beam_type=2,
        radius=90.0,
        hub_rad=0.0,
        hub_conn=1,
        sec_props_file="secs.dat",
        n_elements=10,
        tip_mass=200_000.0,
    )
    return Tower(tmp_path / "tower.bmi")


def _filter(out: list[ModelWarning], location_substr: str) -> list[ModelWarning]:
    return [w for w in out if location_substr in w.location]


# ---------------------------------------------------------------------------
# Each named check
# ---------------------------------------------------------------------------

class TestCheckModel:

    def test_clean_model_returns_empty_list(
        self, tmp_path: pathlib.Path
    ) -> None:
        tower = _build_synthetic_tower(tmp_path)
        # n_modes is conservatively below 6 × (n_elements + 1) so check
        # #7 won't fire.
        out = check_model(tower, n_modes=4)
        # INFO-severity findings are allowed (e.g. EI_FA/EI_SS = 10 on
        # the synthetic uniform tower could trip the boundary if it
        # were 10.0 exactly; the helper uses 1e9 / 1e8 = 10.0 which
        # is at the threshold). Filter to WARN + ERROR only.
        non_info = [w for w in out if w.severity in ("WARN", "ERROR")]
        assert non_info == [], (
            "clean synthetic model produced WARN/ERROR findings:\n  "
            + "\n  ".join(str(w) for w in non_info)
        )

    # --- 1: non-monotonic span -----------------------------------------
    def test_non_monotonic_span(self, tmp_path: pathlib.Path) -> None:
        tower = _build_synthetic_tower(tmp_path)
        # Force the parsed section properties so the cache is populated,
        # then mutate.
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        # Swap two adjacent stations to create a backwards step.
        sp.span_loc[2], sp.span_loc[3] = sp.span_loc[3], sp.span_loc[2]
        tower._sp = sp
        out = check_model(tower)
        hits = _filter(out, "span_loc")
        assert len(hits) == 1 and hits[0].severity == "WARN"
        assert "not strictly increasing" in hits[0].message

    # --- 2: zero or negative mass density -------------------------------
    def test_zero_mass_density(self, tmp_path: pathlib.Path) -> None:
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.mass_den[1] = 0.0  # one bad station mid-span
        tower._sp = sp
        out = check_model(tower)
        # Filter on the exact section-properties location so we don't
        # also pick up the unrelated RNA-vs-tower-mass INFO finding,
        # which fires because zeroing a station shrinks the integrated
        # tower mass below the tip mass.
        hits = [w for w in out if w.location == "section_properties.mass_den"]
        assert len(hits) == 1 and hits[0].severity == "ERROR"
        assert "≤ 0" in hits[0].message

    def test_negative_mass_density(self, tmp_path: pathlib.Path) -> None:
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.mass_den[0] = -1.0
        tower._sp = sp
        out = check_model(tower)
        hits = [w for w in out if w.location == "section_properties.mass_den"]
        assert len(hits) == 1 and hits[0].severity == "ERROR"

    # --- 3: stiffness jump > 5× -----------------------------------------
    def test_stiffness_jump_above_threshold(
        self, tmp_path: pathlib.Path
    ) -> None:
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.flp_stff[2] = sp.flp_stff[1] * 8.0  # 8× jump at station 1→2
        tower._sp = sp
        out = check_model(tower)
        hits = _filter(out, "flp_stff")
        assert len(hits) == 1 and hits[0].severity == "WARN"
        assert "EI_FA jumps by" in hits[0].message

    # --- 4: EI_FA / EI_SS ratio extreme ---------------------------------
    def test_ei_ratio_extreme(self, tmp_path: pathlib.Path) -> None:
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        # The synthetic uniform tower already has EI_FA = 1e8 and
        # EI_SS = 1e9, i.e. ratio = 0.1 — at the boundary. Push it
        # firmly past the trigger.
        sp.flp_stff[:] = sp.edge_stff * 0.02  # ratio = 0.02 < 0.1
        tower._sp = sp
        out = check_model(tower)
        hits = _filter(out, "flp_stff")
        info_hits = [w for w in hits if w.severity == "INFO"]
        assert len(info_hits) == 1
        assert "EI_FA / EI_SS extreme ratio" in info_hits[0].message

    # --- 5: RNA mass > tower mass ---------------------------------------
    def test_rna_mass_exceeds_tower_mass(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Use a deliberately light tower (10 kg/m) and a heavy RNA so
        # the integrated mass < tip mass.
        write_uniform_sec_props(tmp_path / "secs.dat", mass_den=10.0)
        write_bmi(
            tmp_path / "tower.bmi",
            beam_type=2, radius=90.0, hub_rad=0.0, hub_conn=1,
            sec_props_file="secs.dat", n_elements=10,
            tip_mass=1_000_000.0,  # 1000 kg/m equivalent for 90 m → 90,000 kg only
        )
        tower = Tower(tmp_path / "tower.bmi")
        out = check_model(tower)
        hits = [w for w in out if "tip_mass" in w.location]
        assert len(hits) == 1 and hits[0].severity == "INFO"
        assert "exceeds the integrated tower mass" in hits[0].message

    # --- 6: malformed support matrix ------------------------------------
    def _build_platform_tower(
        self, mooring_K: np.ndarray, hydro_K: np.ndarray | None = None,
    ) -> Tower:
        """Build an in-memory ``hub_conn=2`` PlatformSupport tower for
        the support-matrix checks. The parser doesn't reject degenerate
        support matrices, and the BMI fixture helper doesn't support
        ``tow_support=1`` yet, so we bypass parsing here."""
        n = 5
        span = np.linspace(0.0, 1.0, n)
        sp = SectionProperties(
            title="t", n_secs=n,
            span_loc=span, str_tw=np.zeros(n), tw_iner=np.zeros(n),
            mass_den=np.full(n, 100.0),
            flp_iner=np.full(n, 10.0), edge_iner=np.full(n, 10.0),
            flp_stff=np.full(n, 1.0e9), edge_stff=np.full(n, 1.0e9),
            tor_stff=np.full(n, 1.0e8), axial_stff=np.full(n, 1.0e10),
            cg_offst=np.zeros(n), sc_offst=np.zeros(n), tc_offst=np.zeros(n),
        )
        platform = PlatformSupport(
            draft=10.0, cm_pform=0.0, mass_pform=1.0e6,
            i_matrix=np.eye(3) * 1.0e9,
            ref_msl=0.0,
            hydro_M=np.eye(6) * 1.0e6,
            hydro_K=np.eye(6) * 1.0e7 if hydro_K is None else hydro_K,
            mooring_K=mooring_K,
            distr_m_z=np.array([]), distr_m=np.array([]),
            distr_k_z=np.array([]), distr_k=np.array([]),
            wires=None,
        )
        bmi = BMIFile(
            title="t", echo=False, beam_type=2, rot_rpm=0.0, rpm_mult=1.0,
            radius=90.0, hub_rad=0.0, precone=0.0, bl_thp=0.0, hub_conn=2,
            n_modes_print=20, tab_delim=True, mid_node_tw=False,
            tip_mass=TipMassProps(
                mass=100_000.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            ),
            id_mat=1, sec_props_file="", scaling=ScalingFactors(),
            n_elements=10, el_loc=np.linspace(0.0, 1.0, 11),
            tow_support=1, support=platform, source_file=None,
        )
        tower = Tower.__new__(Tower)
        tower._bmi = bmi
        tower._sp = sp
        return tower

    def test_rank_deficient_support_matrix_is_silent(self) -> None:
        """Rank-deficient support matrices are physically legitimate —
        surge/sway/yaw hydrostatic restoring is zero on most floaters
        and mooring layouts can be low-rank. The check must NOT flag
        these as errors."""
        # Diagonal mooring_K with only the heave entry populated —
        # rank-1 but symmetric and finite.
        mooring_K = np.diag([0.0, 0.0, 1.0e6, 0.0, 0.0, 0.0])
        tower = self._build_platform_tower(mooring_K=mooring_K)
        out = check_model(tower)
        hits = _filter(out, "mooring_K")
        assert hits == [], (
            f"rank-deficient mooring_K should pass silently, got: {hits}"
        )

    def test_asymmetric_support_matrix_warns(self) -> None:
        """An asymmetric mooring stiffness is non-physical (Maxwell-
        Betti reciprocity); flag with WARN."""
        mooring_K = np.diag([1.0e6] * 6)
        mooring_K[0, 4] = 1.0e5  # surge–pitch coupling, no symmetric partner
        tower = self._build_platform_tower(mooring_K=mooring_K)
        out = check_model(tower)
        hits = _filter(out, "mooring_K")
        assert len(hits) == 1 and hits[0].severity == "WARN"
        assert "symmetric" in hits[0].message.lower()

    def test_non_finite_support_matrix_errors(self) -> None:
        """A NaN or Inf in a support matrix is a transcription error;
        flag with ERROR."""
        mooring_K = np.diag([1.0e6] * 6)
        mooring_K[2, 2] = np.nan
        tower = self._build_platform_tower(mooring_K=mooring_K)
        out = check_model(tower)
        hits = _filter(out, "mooring_K")
        assert len(hits) == 1 and hits[0].severity == "ERROR"
        assert "non-finite" in hits[0].message.lower()

    # --- 7: n_modes > n_dof ---------------------------------------------
    def test_n_modes_exceeds_dof(self, tmp_path: pathlib.Path) -> None:
        tower = _build_synthetic_tower(tmp_path)
        # nselt=10, hub_conn=1 → n_free_dof = 9·10 + 6 − 6 = 90 exactly.
        # 200 genuinely exceeds the solvable count.
        out = check_model(tower, n_modes=200)
        hits = [w for w in out if "run(n_modes" in w.location]
        assert len(hits) == 1 and hits[0].severity == "ERROR"
        assert "exceeds the model's solvable DOF count" in hits[0].message
        assert "90 free DOFs" in hits[0].message

    def test_n_modes_in_old_false_error_window_is_clean(
        self, tmp_path: pathlib.Path
    ) -> None:
        """F3 regression: the pre-fix check used a ``6 × n_nodes`` (=66)
        per-node estimate and falsely ERRORed for n_modes in (66, 90].
        The FEM actually carries ``n_free_dof = 90`` solvable DOFs for
        this mesh, so n_modes=80 must NOT raise the n_modes ERROR."""
        tower = _build_synthetic_tower(tmp_path)
        out = check_model(tower, n_modes=80)
        hits = [w for w in out if "run(n_modes" in w.location]
        assert hits == [], (
            f"n_modes=80 ≤ n_free_dof(10,1)=90 must not ERROR; got {hits}"
        )

    # --- 8: polynomial-fit design-matrix conditioning -------------------
    def test_polyfit_cond_number_warn(self, tmp_path: pathlib.Path) -> None:
        """Cluster the mesh stations near the tip so the design matrix
        becomes ill-conditioned. With 11 stations all in [0.95, 1.0],
        the cond number jumps well above 1e4."""
        tower = _build_synthetic_tower(tmp_path)
        # Replace el_loc with a tip-clustered mesh.
        tower._bmi.el_loc = np.linspace(0.95, 1.0, 11)
        out = check_model(tower)
        hits = _filter(out, "el_loc")
        assert len(hits) == 1
        assert hits[0].severity in ("WARN", "ERROR")
        assert "polynomial-fit design matrix" in hits[0].message

    # --- platform CM offset (issue #95) ---------------------------------
    def _build_cm_offset_tower(self, cm_x: float, cm_y: float = 0.0) -> Tower:
        """A hub_conn=2 PlatformSupport tower with a full 6×6 inertia and
        a configurable horizontal CM offset. Yaw radius of gyration is
        √(I_yaw/m) = √(2.0e10 / 1.8e7) ≈ 33.3 m."""
        n = 5
        span = np.linspace(0.0, 1.0, n)
        sp = SectionProperties(
            title="t", n_secs=n,
            span_loc=span, str_tw=np.zeros(n), tw_iner=np.zeros(n),
            mass_den=np.full(n, 100.0),
            flp_iner=np.full(n, 10.0), edge_iner=np.full(n, 10.0),
            flp_stff=np.full(n, 1.0e9), edge_stff=np.full(n, 1.0e9),
            tor_stff=np.full(n, 1.0e8), axial_stff=np.full(n, 1.0e10),
            cg_offst=np.zeros(n), sc_offst=np.zeros(n), tc_offst=np.zeros(n),
        )
        m = 1.8e7
        i6 = np.zeros((6, 6))
        i6[0, 0] = i6[1, 1] = i6[2, 2] = m
        i6[3, 3] = i6[4, 4] = 1.3e10
        i6[5, 5] = 2.0e10                       # I_yaw → r_g ≈ 33.3 m
        platform = PlatformSupport(
            draft=10.0, cm_pform=0.0, mass_pform=m, i_matrix=i6, ref_msl=0.0,
            hydro_M=np.eye(6) * 1.0e6, hydro_K=np.eye(6) * 1.0e7,
            mooring_K=np.diag([1.0e6] * 6),
            distr_m_z=np.array([]), distr_m=np.array([]),
            distr_k_z=np.array([]), distr_k=np.array([]), wires=None,
            cm_pform_x=cm_x, cm_pform_y=cm_y,
        )
        bmi = BMIFile(
            title="t", echo=False, beam_type=2, rot_rpm=0.0, rpm_mult=1.0,
            radius=90.0, hub_rad=0.0, precone=0.0, bl_thp=0.0, hub_conn=2,
            n_modes_print=20, tab_delim=True, mid_node_tw=False,
            tip_mass=TipMassProps(
                mass=100_000.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            ),
            id_mat=1, sec_props_file="", scaling=ScalingFactors(),
            n_elements=10, el_loc=np.linspace(0.0, 1.0, 11),
            tow_support=1, support=platform, source_file=None,
        )
        tower = Tower.__new__(Tower)
        tower._bmi = bmi
        tower._sp = sp
        return tower

    def test_large_horizontal_cm_offset_warns(self) -> None:
        """A horizontal CM offset exceeding the platform's yaw radius of
        gyration (the issue #95 input error) flags WARN."""
        tower = self._build_cm_offset_tower(cm_x=-39.0)   # > r_g ≈ 33.3 m
        hits = _filter(check_model(tower), "cm_pform_x")
        assert len(hits) == 1 and hits[0].severity == "WARN"
        assert "radius of gyration" in hits[0].message
        assert "tower axis" in hits[0].message.lower()

    def test_small_horizontal_cm_offset_is_silent(self) -> None:
        """A modest CM offset (well under the gyration radius) is a
        legitimate asymmetric floater and must not warn."""
        tower = self._build_cm_offset_tower(cm_x=-4.0)     # < r_g ≈ 33.3 m
        assert _filter(check_model(tower), "cm_pform_x") == []

    def test_zero_cm_offset_is_silent(self) -> None:
        """The axisymmetric / symmetric default (offset 0) never warns."""
        tower = self._build_cm_offset_tower(cm_x=0.0, cm_y=0.0)
        assert _filter(check_model(tower), "cm_pform_x") == []

    def test_cm_offset_warning_suppressed_when_ref_offset_set(self) -> None:
        """Issue #100: setting ref_x/ref_y means the hydro/mooring are
        referenced off-axis too (intentional off-axis floater), so a large
        CM offset is consistent — the warning stands down."""
        tower = self._build_cm_offset_tower(cm_x=-39.0)   # would WARN alone
        assert _filter(check_model(tower), "cm_pform_x")  # sanity: it warns
        tower._bmi.support = dataclasses.replace(
            tower._bmi.support, ref_x=-39.0,
        )
        assert _filter(check_model(tower), "cm_pform_x") == []

    # --- floating-model readiness (issue #95) ---------------------------
    def test_zero_added_mass_warns(self) -> None:
        """A floating model with no added mass (hydro_M = 0) warns — the
        single most common seakeeping omission."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        tower._bmi.support = dataclasses.replace(
            tower._bmi.support, hydro_M=np.zeros((6, 6)),
        )
        hits = _filter(check_model(tower), "hydro_M")
        assert len(hits) == 1 and hits[0].severity == "WARN"
        assert "added mass" in hits[0].message.lower()

    def test_added_mass_present_is_silent(self) -> None:
        """A populated hydro_M does not warn."""
        tower = self._build_cm_offset_tower(cm_x=0.0)   # hydro_M = eye·1e6
        assert _filter(check_model(tower), "hydro_M") == []

    def test_no_restoring_warns(self) -> None:
        """No hydrostatic restoring AND no mooring stiffness → the
        rigid-body modes collapse to ~0 Hz; warn."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        tower._bmi.support = dataclasses.replace(
            tower._bmi.support,
            hydro_K=np.zeros((6, 6)), mooring_K=np.zeros((6, 6)),
        )
        hits = _filter(check_model(tower), "mooring_K")
        assert any(
            w.severity == "WARN" and "no restoring" in w.message.lower()
            for w in hits
        )

    def test_restoring_present_is_silent(self) -> None:
        """Mooring stiffness alone (no hydrostatic) still counts as
        restoring — no warning."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        tower._bmi.support = dataclasses.replace(
            tower._bmi.support, hydro_K=np.zeros((6, 6)),
        )  # mooring_K still populated
        hits = [w for w in check_model(tower) if "no restoring" in w.message.lower()]
        assert hits == []

    def test_non_physical_platform_inertia_errors(self) -> None:
        """A negative i_matrix diagonal entry is non-physical → ERROR."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        bad = tower._bmi.support.i_matrix.copy()
        bad[4, 4] = -1.0                               # negative pitch inertia
        tower._bmi.support = dataclasses.replace(tower._bmi.support, i_matrix=bad)
        hits = _filter(check_model(tower), "i_matrix")
        assert any(w.severity == "ERROR" for w in hits)

    def test_physical_platform_inertia_is_silent(self) -> None:
        """A well-formed positive inertia does not error."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        hits = [w for w in check_model(tower)
                if w.location in ("bmi.support.i_matrix", "bmi.support.mass_pform")]
        assert hits == []

    def test_floating_gates_skip_fixed_bottom_with_platform_block(self) -> None:
        """Codex P1: a fixed-bottom monopile (hub_conn=1) may carry an
        all-zero PlatformSupport by layout convention (the bundled
        monopile samples do, for CS_Monopile compatibility). The
        floating-readiness gates key on hub_conn==2 and must NOT fire on
        such a valid fixed-bottom deck, even though it has a
        PlatformSupport with zero hydro / mooring / inertia."""
        tower = self._build_cm_offset_tower(cm_x=0.0)
        tower._bmi.hub_conn = 1                       # fixed-bottom monopile
        z = np.zeros((6, 6))
        tower._bmi.support = dataclasses.replace(
            tower._bmi.support,
            hydro_M=z.copy(), hydro_K=z.copy(), mooring_K=z.copy(),
            i_matrix=z.copy(), mass_pform=0.0,
        )
        floating = [
            w for w in check_model(tower)
            if any(loc in w.location for loc in (
                "hydro_M", "mooring_K", "i_matrix", "mass_pform", "cm_pform"))
        ]
        assert floating == [], floating


# ---------------------------------------------------------------------------
# Auto-run integration on Tower.run / RotatingBlade.run
# ---------------------------------------------------------------------------

class TestAutoRunIntegration:

    def test_run_emits_warning_by_default(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A model with a stiffness jump triggers the auto-run check
        from .run(), surfaced as a UserWarning."""
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.flp_stff[2] = sp.flp_stff[1] * 8.0
        tower._sp = sp
        with pytest.warns(UserWarning, match="EI_FA jumps by"):
            tower.run(n_modes=4)

    def test_check_model_false_suppresses_auto_run(
        self, tmp_path: pathlib.Path, recwarn
    ) -> None:
        """check_model=False skips the auto-check entirely; no
        UserWarning should appear for the same offending input."""
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.flp_stff[2] = sp.flp_stff[1] * 8.0
        tower._sp = sp
        tower.run(n_modes=4, check_model=False)
        # No "EI_FA jumps by" warning should have been raised.
        relevant = [w for w in recwarn.list if "EI_FA jumps by" in str(w.message)]
        assert relevant == []

    def _tower_with_error_finding(self, tmp_path: pathlib.Path) -> Tower:
        """A tower whose section properties trigger an ERROR finding
        (a non-positive mass density)."""
        tower = _build_synthetic_tower(tmp_path)
        from pybmodes.io.sec_props import read_sec_props
        sp = read_sec_props(tower._bmi.resolve_sec_props_path())
        sp.mass_den[1] = 0.0          # non-physical → ERROR finding
        tower._sp = sp
        return tower

    def test_run_fails_closed_on_error_by_default(
        self, tmp_path: pathlib.Path
    ) -> None:
        """An ERROR-severity finding raises ModelValidationError from
        .run() by default (fail-closed, 1.14.0) rather than warning and
        feeding the eigensolver non-physical input."""
        from pybmodes.checks import ModelValidationError

        tower = self._tower_with_error_finding(tmp_path)
        with pytest.raises(ModelValidationError) as exc:
            tower.run(n_modes=4)
        # The exception carries the offending findings, all ERROR.
        assert exc.value.findings
        assert all(f.severity == "ERROR" for f in exc.value.findings)
        # And is a ValueError subclass for legacy callers.
        assert isinstance(exc.value, ValueError)

    def test_run_on_error_warn_downgrades_to_warning(
        self, tmp_path: pathlib.Path
    ) -> None:
        """on_error='warn' restores the pre-1.14.0 behaviour: the ERROR
        finding is emitted as a UserWarning and the solve continues."""
        tower = self._tower_with_error_finding(tmp_path)
        with pytest.warns(UserWarning, match="mass_den"):
            # The solve itself may further fail on the bad input, but the
            # check must warn (not raise ModelValidationError) first.
            try:
                tower.run(n_modes=4, on_error="warn")
            except Exception as exc:
                from pybmodes.checks import ModelValidationError
                assert not isinstance(exc, ModelValidationError)

    def test_run_rejects_unknown_on_error(
        self, tmp_path: pathlib.Path
    ) -> None:
        """An unrecognised on_error value is rejected up front."""
        tower = _build_synthetic_tower(tmp_path)
        with pytest.raises(ValueError, match="on_error must be"):
            tower.run(n_modes=4, on_error="explode")  # type: ignore[arg-type]


# ===========================================================================
# Non-finite section-property gate (runs before per-field checks)
# ===========================================================================

class TestCheckModelFiniteSectionProperties:
    """Non-finite (NaN / ±Inf) entries in any numeric section-property
    field produce an ERROR-severity ``ModelWarning`` *before* the per-
    field checks run. NaN / Inf would otherwise pass silently because
    every downstream comparison returns False on NaN.
    """

    def _build_tower_with_section_props(self, sp_overrides: dict):
        """Build an in-memory Tower with a synthetic SectionProperties.
        ``sp_overrides`` mutates specific fields after the clean build
        so each test can install exactly one offending value."""
        from pybmodes.io.bmi import BMIFile, ScalingFactors, TipMassProps
        from pybmodes.io.sec_props import SectionProperties
        from pybmodes.models import Tower

        n = 5
        span = np.linspace(0.0, 1.0, n)
        sp = SectionProperties(
            title="t", n_secs=n,
            span_loc=span, str_tw=np.zeros(n), tw_iner=np.zeros(n),
            mass_den=np.full(n, 100.0),
            flp_iner=np.full(n, 10.0), edge_iner=np.full(n, 10.0),
            flp_stff=np.full(n, 1.0e9), edge_stff=np.full(n, 1.0e9),
            tor_stff=np.full(n, 1.0e8), axial_stff=np.full(n, 1.0e10),
            cg_offst=np.zeros(n), sc_offst=np.zeros(n), tc_offst=np.zeros(n),
        )
        for fname, value in sp_overrides.items():
            setattr(sp, fname, value)
        bmi = BMIFile(
            title="t", echo=False, beam_type=2, rot_rpm=0.0, rpm_mult=1.0,
            radius=90.0, hub_rad=0.0, precone=0.0, bl_thp=0.0, hub_conn=1,
            n_modes_print=20, tab_delim=True, mid_node_tw=False,
            tip_mass=TipMassProps(
                mass=100_000.0, cm_offset=0.0, cm_axial=0.0,
                ixx=0.0, iyy=0.0, izz=0.0, ixy=0.0, izx=0.0, iyz=0.0,
            ),
            id_mat=1, sec_props_file="", scaling=ScalingFactors(),
            n_elements=10, el_loc=np.linspace(0.0, 1.0, 11),
            tow_support=0, support=None, source_file=None,
        )
        tower = Tower.__new__(Tower)
        tower._bmi = bmi
        tower._sp = sp
        return tower

    def test_nan_in_mass_den_raises_error(self) -> None:
        m = np.full(5, 100.0)
        m[2] = np.nan
        tower = self._build_tower_with_section_props({"mass_den": m})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.mass_den" in w.location]
        assert len(hits) == 1 and hits[0].severity == "ERROR"
        assert "non-finite" in hits[0].message.lower()

    def test_inf_in_flp_stff_raises_error(self) -> None:
        ei = np.full(5, 1.0e9)
        ei[3] = np.inf
        tower = self._build_tower_with_section_props({"flp_stff": ei})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.flp_stff" in w.location]
        assert len(hits) == 1 and hits[0].severity == "ERROR"

    def test_nan_in_span_loc_raises_error(self) -> None:
        span = np.linspace(0.0, 1.0, 5)
        span[1] = np.nan
        tower = self._build_tower_with_section_props({"span_loc": span})
        out = check_model(tower)
        hits = [w for w in out if "section_properties.span_loc" in w.location]
        assert len(hits) >= 1
        assert any(w.severity == "ERROR" for w in hits)
