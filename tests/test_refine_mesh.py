"""FE-mesh refinement on the deck / BMI readers (issue #58).

The geometry-derived constructors (``from_geometry``, ``from_windio*``)
have carried ``n_nodes`` since 1.6.0; this is the deck-reader half, plus
the ``refine_mesh`` method every constructor shares.

The two are deliberately different operations. Re-gridding continuous
geometry recomputes exact closed-form tube properties at each new
station; re-gridding a deck re-samples an already tabulated property
table, which smooths a deliberate step the deck put a node on. These
tests pin the convergence behaviour of the first and the warning that
makes the second visible.

Every model here is built from synthetic .bmi / .dat written into
``tmp_path``; nothing reads an external deck.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybmodes.models import RotatingBlade, Tower

from ._synthetic_bmi import write_bmi, write_uniform_sec_props

L = 60.0
EI = 5.0e9
MASS_DEN = 800.0

# Euler-Bernoulli uniform cantilever, beta_n L for the first three modes.
_BETA_L = np.array([1.87510407, 4.69409113, 7.85475744])


def _analytic(n: int = 3) -> np.ndarray:
    return _BETA_L[:n] ** 2 * np.sqrt(EI / (MASS_DEN * L**4)) / (2.0 * np.pi)


def _bending(freqs: np.ndarray, n: int = 3) -> np.ndarray:
    """EI_FA == EI_SS here, so every bending mode comes back as a
    degenerate fore-aft / side-side pair. Take one of each."""
    return np.asarray(freqs)[: 2 * n : 2]


def _uniform_tower(tmp_path, *, n_elements: int = 4, beam_type: int = 2,
                   **kw) -> Tower:
    write_uniform_sec_props(
        tmp_path / "secs.dat", mass_den=MASS_DEN,
        flp_stff=EI, edge_stff=EI, tor_stff=1.0e12, axial_stff=1.0e14,
    )
    path = write_bmi(
        tmp_path / "beam.bmi", beam_type=beam_type, radius=L, hub_conn=1,
        n_elements=n_elements, sec_props_file="secs.dat", **kw,
    )
    cls = Tower if beam_type == 2 else RotatingBlade
    return cls(path)


def _stepped_sec_props(path) -> None:
    """A table with a deliberate EI step at mid-span, encoded the way the
    adapters do it: two stations at the same normalised location."""
    span = [0.0, 0.5, 0.5, 1.0]
    stiff = [EI, EI, 0.25 * EI, 0.25 * EI]
    rows = "\n".join(
        f"{s:.4f}  0.0  0.0  {MASS_DEN}  {MASS_DEN * 0.1}  {MASS_DEN * 0.1}  "
        f"{k}  {k}  1.0e12  1.0e14  0.0  0.0  0.0"
        for s, k in zip(span, stiff)
    )
    path.write_text(
        "stepped section properties\n4  n_secs\n\n"
        "span_loc str_tw tw_iner mass_den flp_iner edge_iner "
        "flp_stff edge_stff tor_stff axial_stff cg_offst sc_offst tc_offst\n"
        "  -      deg    deg     kg/m     kg.m     kg.m     "
        "N.m^2   N.m^2     N.m^2   N        m        m        m\n"
        + rows + "\n",
        encoding="utf-8",
    )


class TestRefinementConverges:
    def test_coarse_deck_is_off_and_refinement_fixes_it(self, tmp_path):
        """The point of the feature: a 4-element deck resolves the higher
        modes badly, and asking for more nodes recovers them."""
        ref = _analytic(3)
        coarse = _bending(
            _uniform_tower(tmp_path).run(8, check_model=False).frequencies
        )
        fine = _bending(
            _uniform_tower(tmp_path, n_elements=4)
            .refine_mesh(60)
            .run(8, check_model=False).frequencies
        )
        err_coarse = abs(coarse[2] - ref[2]) / ref[2]
        err_fine = abs(fine[2] - ref[2]) / ref[2]
        assert err_fine < err_coarse
        assert err_fine < 0.01

    @pytest.mark.parametrize("n_nodes", [20, 40, 80])
    def test_refined_frequencies_match_euler_bernoulli(self, tmp_path, n_nodes):
        f = _bending(
            _uniform_tower(tmp_path).refine_mesh(n_nodes).run(
                8, check_model=False,
            ).frequencies
        )
        assert np.allclose(f, _analytic(3), rtol=0.01)

    def test_refinement_is_self_convergent(self, tmp_path):
        a = _bending(_uniform_tower(tmp_path).refine_mesh(100).run(
            8, check_model=False).frequencies)
        b = _bending(_uniform_tower(tmp_path).refine_mesh(200).run(
            8, check_model=False).frequencies)
        assert np.allclose(a, b, rtol=2.0e-3)


class TestConstructorKeyword:
    def test_bmi_constructor_keyword_matches_the_method(self, tmp_path):
        via_kw = Tower(
            _uniform_tower(tmp_path)._bmi.source_file, n_nodes=32,
        ).run(4, check_model=False).frequencies
        via_method = _uniform_tower(tmp_path).refine_mesh(32).run(
            4, check_model=False).frequencies
        assert np.allclose(via_kw, via_method)

    def test_from_bmi_forwards_n_nodes(self, tmp_path):
        path = _uniform_tower(tmp_path)._bmi.source_file
        assert Tower.from_bmi(path, n_nodes=25)._bmi.n_elements == 24

    def test_none_keeps_the_deck_mesh(self, tmp_path):
        assert _uniform_tower(tmp_path, n_elements=7)._bmi.n_elements == 7

    def test_blade_constructor_keyword(self, tmp_path):
        path = _uniform_tower(tmp_path, beam_type=1)._bmi.source_file
        assert RotatingBlade(path, n_nodes=17)._bmi.n_elements == 16

    def test_blade_refine_mesh_chains(self, tmp_path):
        blade = _uniform_tower(tmp_path, beam_type=1)
        assert blade.refine_mesh(21) is blade
        assert blade._bmi.n_elements == 20

    def test_tower_refine_mesh_chains(self, tmp_path):
        tower = _uniform_tower(tmp_path)
        assert tower.refine_mesh(11) is tower
        assert tower._bmi.el_loc.shape == (11,)
        assert tower._bmi.el_loc[0] == 0.0
        assert tower._bmi.el_loc[-1] == 1.0


class TestStepSmoothingWarning:
    def test_step_that_falls_between_nodes_warns(self, tmp_path):
        _stepped_sec_props(tmp_path / "secs.dat")
        path = write_bmi(
            tmp_path / "beam.bmi", beam_type=2, radius=L, hub_conn=1,
            n_elements=4, sec_props_file="secs.dat",
        )
        # 12 nodes -> spacing 1/11, so nothing lands on 0.5.
        with pytest.warns(UserWarning, match="property step"):
            Tower(path).refine_mesh(12)

    def test_step_landed_on_by_the_new_mesh_is_silent(self, tmp_path):
        _stepped_sec_props(tmp_path / "secs.dat")
        path = write_bmi(
            tmp_path / "beam.bmi", beam_type=2, radius=L, hub_conn=1,
            n_elements=4, sec_props_file="secs.dat",
        )
        # 11 nodes -> spacing 1/10, and 0.5 is a node.
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            Tower(path).refine_mesh(11)

    def test_smooth_table_never_warns(self, tmp_path):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            _uniform_tower(tmp_path).refine_mesh(37)


class TestValidation:
    @pytest.mark.parametrize("bad", [1, 0, -4, 2.5, True, None])
    def test_bad_node_count_rejected(self, tmp_path, bad):
        if bad is None:
            pytest.skip("None means 'keep the deck mesh' on the keyword path")
        with pytest.raises(ValueError, match="n_nodes"):
            _uniform_tower(tmp_path).refine_mesh(bad)

    def test_tension_wire_model_is_refused(self, tmp_path):
        """Wire attachments are FE node numbers, so a new mesh would move
        them to a different elevation."""
        write_uniform_sec_props(tmp_path / "secs.dat", mass_den=MASS_DEN)
        path = write_bmi(
            tmp_path / "wired.bmi", beam_type=2, radius=L, hub_conn=1,
            n_elements=8, sec_props_file="secs.dat", tow_support=1,
            wire_data=([3], [5], [1.0e6], [30.0]),
        )
        with pytest.raises(ValueError, match="tension-wire"):
            Tower(path).refine_mesh(30)
