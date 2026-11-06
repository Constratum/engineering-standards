"""Manufacturer-report values for the corrected ETAG engine and EN 1992-4.

The numbers are the ones printed in the anchor reports. c2 pairs that are
not a measured edge are marked in the test name as derived from the printed
area or psi factor.
"""

import math

import pytest

from en_standards.EN_1992_4_2018 import (
    AnchorProperties_EN1992_4,
    MechanicalAnchorCapacity_EN1992_4,
    interaction_from_capacities,
    interaction_table_7_3,
)
from en_standards.EOTA_TR029 import combined_pullout_concrete_cone
from en_standards.ETAG001C_R045_2013 import (
    AnchorProperties_ETAG,
    MetalAnchorCapacity_ETAG001_TR045,
)


def _rawlplug(**overrides):
    values = dict(
        diameter=12,
        embedment_depth=63,
        edge_distance=250,
        edge_distance_c2_1=375,
        edge_distance_c2_2=375,
        concrete_strength=20,
        concrete_strength_cube=25,
        concrete_thickness=150,
        is_cracked_concrete=True,
        N0_Rk_s=83.1,
        N0_Rk_p=17.2,
        V0_Rk_s=41.6,
        s_cr_N=189,
        c_cr_N=95,
        l_f=63,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        dense_reinforcement=True,
        splitting_check_required=False,
        number_of_anchors=2,
        spacing=200,
        alpha_gap=1.0,
        seismic_design=False,
    )
    values.update(overrides)
    return MetalAnchorCapacity_ETAG001_TR045(AnchorProperties_ETAG(**values))


def test_rawlplug_edge_uses_the_published_exponent_order():
    edge = _rawlplug().calculate_shear_capacity_ETAG_5_2_3()["concrete_edge_details"]
    assert edge["V0_Rk_c"] == pytest.approx(47.7, abs=0.05)
    assert edge["A_c_V"] == pytest.approx(142500, abs=1)
    assert edge["psi_h_V"] == pytest.approx(1.581, abs=0.001)


def _en(**overrides):
    values = dict(
        alpha_V_deg=0.0,
        edge_reinforcement=False,
        seismic_design=False,
    )
    values.update(overrides)
    return MechanicalAnchorCapacity_EN1992_4(
        AnchorProperties_EN1992_4(**values)
    ).calculate()


def test_fib_edge_area_needs_both_perpendicular_edges():
    """c2_2 = 164 mm is derived from the printed area 101760 mm2.

    The report prints psi_s,V = 0.900, which is the nearer edge c2_1 = 160 mm.
    Width 160 + 100 + 164 at height 240 mm is the printed area.
    """
    result = _en(
        diameter_mm=12,
        h_ef_mm=100,
        l_f_mm=100,
        concrete_thickness_mm=250,
        fck_cylinder_MPa=20,
        is_cracked=True,
        dense_reinforcement=True,
        number_of_anchors=2,
        spacing_mm=100,
        edge_distance_c1_mm=160,
        edge_distance_c2_1_mm=160,
        edge_distance_c2_2_mm=164,
        N_Rk_s_kN=42,
        V_Rk_s_kN=21,
        N_Rk_p_kN=32.044,
        psi_c=1.0,
        s_cr_N_mm=300,
        c_cr_N_mm=150,
        k_cr_N=7.7,
        k_ucr_N=11.0,
        k7=1.0,
        k8=2.0,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        splitting_check_required=False,
        seismic_design=False,
        e_N_x_mm=0,
        e_N_y_mm=0,
        e_V_mm=0,
    )
    edge = result["shear"]["concrete_edge_details"]
    assert edge["A_c_V"] == pytest.approx(101760, abs=1)
    assert edge["V0_Rk_c"] == pytest.approx(24.637, abs=0.01)
    assert result["shear"]["V_Rd_concrete"] == pytest.approx(13.058, abs=0.01)
    assert result["tension"]["N_Rd_concrete"] == pytest.approx(30.609, abs=0.01)


def test_hus4_edge_area_and_psi_from_the_printed_c2_pair():
    """c2_1 = 152.5 mm and c2_2 = 285 mm are derived from the printed factors.

    psi_s,V = 0.861 gives the nearer edge. The printed area 76875 mm2 then
    requires the far edge to be at least 1.5 c1.
    """
    result = _en(
        diameter_mm=10,
        h_ef_mm=42.5,
        l_f_mm=55,
        concrete_thickness_mm=150,
        fck_cylinder_MPa=25,
        is_cracked=True,
        dense_reinforcement=True,
        number_of_anchors=2,
        spacing_mm=75,
        edge_distance_c1_mm=190,
        edge_distance_c2_1_mm=152.5,
        edge_distance_c2_2_mm=285,
        N_Rk_s_kN=55,
        V_Rk_s_kN=28.8,
        N_Rk_p_kN=9.541,
        psi_c=1.118,
        s_cr_N_mm=127.5,
        c_cr_N_mm=63.8,
        k_cr_N=7.7,
        k_ucr_N=11.0,
        k7=0.8,
        k8=1.0,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.2,
        splitting_check_required=False,
        seismic_design=False,
        e_N_x_mm=0,
        e_N_y_mm=0,
        e_V_mm=0,
    )
    edge = result["shear"]["concrete_edge_details"]
    assert edge["A_c_V"] == pytest.approx(76875, abs=1)
    assert edge["psi_s_V"] == pytest.approx(0.861, abs=0.001)
    assert result["shear"]["V_Rd_concrete"] == pytest.approx(11.777, abs=0.01)


def test_alpha_gap_does_not_change_tension_and_does_change_seismic_shear():
    shared = dict(
        diameter=12,
        embedment_depth=63,
        edge_distance=9999,
        concrete_strength=20,
        concrete_strength_cube=25,
        concrete_thickness=150,
        is_cracked_concrete=True,
        N0_Rk_s=83.1,
        N0_Rk_p=17.2,
        V0_Rk_s=41.6,
        s_cr_N=189,
        c_cr_N=95,
        l_f=63,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        dense_reinforcement=True,
        splitting_check_required=False,
        number_of_anchors=1,
        seismic_design=True,
    )
    full = MetalAnchorCapacity_ETAG001_TR045(
        AnchorProperties_ETAG(**shared, alpha_gap=1.0)
    )
    halved = MetalAnchorCapacity_ETAG001_TR045(
        AnchorProperties_ETAG(**shared, alpha_gap=0.5)
    )
    assert full.calculate_tension_capacity_ETAG_5_2_2()["N_Rd_steel"] == pytest.approx(
        halved.calculate_tension_capacity_ETAG_5_2_2()["N_Rd_steel"]
    )
    assert halved.calculate_shear_capacity_ETAG_5_2_3()["V_Rd_steel"] == pytest.approx(
        0.5 * full.calculate_shear_capacity_ETAG_5_2_3()["V_Rd_steel"]
    )


def test_static_clearance_factor_is_rejected():
    with pytest.raises(ValueError, match="alpha_gap"):
        _rawlplug(alpha_gap=0.5)


def test_missing_partial_factor_is_rejected():
    values = dict(
        diameter=12,
        embedment_depth=63,
        edge_distance=9999,
        concrete_strength=20,
        concrete_strength_cube=25,
        concrete_thickness=150,
        is_cracked_concrete=True,
        N0_Rk_s=83.1,
        N0_Rk_p=17.2,
        V0_Rk_s=41.6,
        s_cr_N=189,
        c_cr_N=95,
        l_f=63,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        dense_reinforcement=True,
        splitting_check_required=False,
        alpha_gap=1.0,
        seismic_design=False,
    )
    with pytest.raises(TypeError):
        AnchorProperties_ETAG(**values)


def test_missing_reinforcement_flag_is_rejected():
    with pytest.raises(TypeError):
        _rawlplug(dense_reinforcement=None)


def test_splitting_matches_the_hilti_reports():
    def splitting(hef, h_min, s_cr_sp, c_cr_sp, gamma_inst):
        calc = MetalAnchorCapacity_ETAG001_TR045(
            AnchorProperties_ETAG(
                diameter=12,
                embedment_depth=hef,
                edge_distance=9999,
                concrete_strength=20,
                concrete_strength_cube=25,
                concrete_thickness=200,
                is_cracked_concrete=True,
                N0_Rk_s=55,
                N0_Rk_p=50,
                V0_Rk_s=27,
                s_cr_N=267 if hef == 89 else 363,
                c_cr_N=134 if hef == 89 else 182,
                l_f=hef,
                gamma_Ms_N=1.5,
                gamma_Ms_V=1.25,
                gamma_inst=gamma_inst,
                dense_reinforcement=True,
                splitting_check_required=True,
                number_of_anchors=2,
                spacing=100,
                alpha_gap=1.0,
                seismic_design=False,
                c_cr_sp=c_cr_sp,
                s_cr_sp=s_cr_sp,
                h_min=h_min,
            )
        )
        return calc.splitting_failure_ETAG_5_2_2_6()["N_Rd_sp"]

    assert splitting(89, 149, 304, 152, 1.0) == pytest.approx(30.160, abs=0.02)
    assert splitting(121, 151, 393, 197, 1.2) == pytest.approx(40.270, abs=0.02)


def test_hus4_en1992_resistances_and_interaction():
    result = _en(
        diameter_mm=10,
        h_ef_mm=42.5,
        l_f_mm=55,
        concrete_thickness_mm=150,
        fck_cylinder_MPa=25,
        is_cracked=True,
        dense_reinforcement=True,
        number_of_anchors=2,
        spacing_mm=75,
        edge_distance_c1_mm=190,
        edge_distance_c2_1_mm=152.5,
        edge_distance_c2_2_mm=285,
        N_Rk_s_kN=55,
        V_Rk_s_kN=28.8,
        N_Rk_p_kN=9.541,
        psi_c=1.118,
        s_cr_N_mm=127.5,
        c_cr_N_mm=63.8,
        k_cr_N=7.7,
        k_ucr_N=11.0,
        k7=0.8,
        k8=1.0,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.2,
        splitting_check_required=False,
        seismic_design=False,
        e_N_x_mm=0,
        e_N_y_mm=0,
        e_V_mm=0,
    )
    assert result["tension"]["concrete_details"]["N0_Rk_c"] == pytest.approx(10.667, abs=0.01)
    assert result["tension"]["N_Rd_concrete"] == pytest.approx(9.412, abs=0.01)
    assert result["tension"]["N_Rd_pullout"] == pytest.approx(5.926, abs=0.01)
    assert result["shear"]["V_Rd_steel"] == pytest.approx(18.432, abs=0.01)
    assert result["shear"]["V_Rd_pryout"] == pytest.approx(11.295, abs=0.01)
    interaction = interaction_from_capacities(result, tension_kN=5, shear_kN=4, number_of_anchors=2)
    assert interaction["concrete"] == pytest.approx(0.60, abs=0.01)
    assert interaction["steel"] == pytest.approx(0.02, abs=0.01)


def test_fib_table_7_3_uses_the_printed_ratios():
    """79% is the report's concrete interaction.

    Its tension ratio 0.779 is the bonded pull-out utilisation. This module
    does not calculate bonded pull-out, so the ratio is supplied here and
    only the Table 7.3 exponents are checked. 17% is the steel interaction.
    """
    concrete = interaction_table_7_3(0.404, 0.081, 0.779, 0.209)
    assert concrete["concrete"] == pytest.approx(0.79, abs=0.01)
    assert concrete["steel"] == pytest.approx(0.17, abs=0.01)


def _rhlx12_seismic(**overrides):
    """ETA 23/0707, diameter 12, h_nom 100 mm. One anchor, no concrete edge.

    C1 Table C7: NRk,s = 83.1 kN, NRk,p = 17.5 kN, VRk,s = 28.7 kN.
    C2 Table C8: NRk,p = 13.3 kN, VRk,s = 22.3 kN. Steel is the static value.
    alpha_gap is 1.0 with gap filling and 0.5 without, Tables C7 and C8.
    A concrete screw is "all other fasteners" in Table C.3.
    """
    values = dict(
        diameter_mm=12,
        h_ef_mm=80,
        l_f_mm=100,
        concrete_thickness_mm=200,
        fck_cylinder_MPa=20,
        is_cracked=True,
        dense_reinforcement=True,
        number_of_anchors=1,
        edge_distance_c1_mm=9999,
        edge_distance_c2_1_mm=None,
        edge_distance_c2_2_mm=None,
        N_Rk_s_kN=83.1,
        V_Rk_s_kN=41.6,
        N_Rk_p_kN=17.5,
        psi_c=1.0,
        s_cr_N_mm=240,
        c_cr_N_mm=120,
        k_cr_N=7.7,
        k_ucr_N=11.0,
        k7=1.0,
        k8=2.0,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        splitting_check_required=False,
        e_N_x_mm=0.0,
        e_N_y_mm=0.0,
        e_V_mm=0.0,
        alpha_V_deg=0.0,
        edge_reinforcement=False,
        seismic_design=True,
        seismic_performance_category="C1",
        annular_gap_filled=True,
        seismic_design_option="elastic",
        fastener_class="other",
        anchors_in_tension=1,
        alpha_gap_with_gap_filling=1.0,
        alpha_gap_without_gap_filling=0.5,
        N_Rk_s_eq_kN=83.1,
        N_Rk_p_eq_kN=17.5,
        V_Rk_s_eq_kN=28.7,
    )
    values.update(overrides)
    return _en(**values)


def test_rhlx_c1_resistances_follow_annex_c():
    """Formula (C.8) and Table C.3 for one concrete screw, gap filled.

    alpha_eq is 1.0 for steel and pull-out, and 0.85 for the cone and for
    pry-out. alpha_gap is 1.0, so it does not change the shear steel value.
    Partial factors stay at the static Table 4.1 values, clause 4.4.2.2.
    """
    result = _rhlx12_seismic()
    assert result["tension"]["N_Rd_steel"] == pytest.approx(83.1 / 1.5, abs=0.001)
    assert result["tension"]["N_Rd_pullout"] == pytest.approx(17.5 / 1.5, abs=0.001)
    n0 = result["tension"]["concrete_details"]["N0_Rk_c"]
    assert result["tension"]["N_Rd_concrete"] == pytest.approx(0.85 * n0 / 1.5, abs=0.001)
    assert result["shear"]["V_Rd_steel"] == pytest.approx(28.7 / 1.25, abs=0.001)
    assert result["shear"]["V_Rd_pryout"] == pytest.approx(0.85 * 2.0 * n0 / 1.5, abs=0.001)


def test_rhlx_c1_without_gap_filling_halves_shear_only():
    filled = _rhlx12_seismic()
    open_gap = _rhlx12_seismic(annular_gap_filled=False)
    assert open_gap["tension"]["N_Rd_steel"] == pytest.approx(
        filled["tension"]["N_Rd_steel"]
    )
    assert open_gap["shear"]["V_Rd_steel"] == pytest.approx(
        0.5 * filled["shear"]["V_Rd_steel"]
    )


def test_rhlx_c2_uses_table_c8_and_can_reduce_for_displacement():
    """Table C8 and Formula (C.11). The required displacement is half the
    Table C9 damage-limitation displacement, so the resistance is halved.
    """
    result = _rhlx12_seismic(
        seismic_performance_category="C2",
        N_Rk_p_eq_kN=13.3,
        V_Rk_s_eq_kN=22.3,
        delta_N_eq_DLS_mm=0.44,
        delta_V_eq_DLS_mm=5.00,
        delta_N_req_DLS_mm=0.22,
        delta_V_req_DLS_mm=2.50,
    )
    assert result["tension"]["N_Rd_pullout"] == pytest.approx(0.5 * 13.3 / 1.5, abs=0.001)
    assert result["shear"]["V_Rd_steel"] == pytest.approx(0.5 * 22.3 / 1.25, abs=0.001)
    assert result["displacement"]["tension_factor"] == pytest.approx(0.5)
    assert result["displacement"]["shear_factor"] == pytest.approx(0.5)


def test_annex_c_group_of_concrete_screws_uses_the_group_factors():
    result = _rhlx12_seismic(number_of_anchors=2, anchors_in_tension=2, spacing_mm=100)
    single = _rhlx12_seismic()
    assert result["tension"]["pullout_details"]["alpha_eq"] == pytest.approx(0.85)
    assert result["tension"]["N_Rd_pullout"] == pytest.approx(
        0.85 * single["tension"]["N_Rd_pullout"]
    )


def test_one_tensioned_fastener_in_a_group_keeps_the_single_factor():
    result = _rhlx12_seismic(number_of_anchors=2, anchors_in_tension=1, spacing_mm=100)
    assert result["tension"]["pullout_details"]["alpha_eq"] == pytest.approx(1.0)
    assert result["shear"]["steel_details"]["alpha_eq"] == pytest.approx(0.85)


def test_seismic_without_a_category_is_rejected():
    with pytest.raises(ValueError, match="seismic_performance_category"):
        _rhlx12_seismic(seismic_performance_category=None)


def test_seismic_without_gap_filling_is_rejected():
    with pytest.raises(TypeError, match="annular_gap_filled"):
        _rhlx12_seismic(annular_gap_filled=None)


def test_c2_without_table_c9_displacements_is_rejected():
    with pytest.raises(ValueError, match="delta_N_eq_DLS_mm"):
        _rhlx12_seismic(seismic_performance_category="C2")


def test_ductile_option_without_steel_properties_is_rejected():
    with pytest.raises(ValueError, match="f_uk_MPa"):
        _rhlx12_seismic(
            seismic_performance_category="C2",
            seismic_design_option="ductile_fastener",
            delta_N_eq_DLS_mm=0.44,
            delta_V_eq_DLS_mm=5.0,
            delta_N_req_DLS_mm=3.0,
            delta_V_req_DLS_mm=6.0,
        )


def test_finite_edge_without_c2_is_rejected():
    with pytest.raises(ValueError, match="c2"):
        _rawlplug(edge_distance_c2_1=None, edge_distance_c2_2=None)


def test_tr029_hit_v_bond_matches_the_printed_group():
    # No concrete edge. k = 2.3 and tau values are printed on the PROFIS sheet.
    tension = combined_pullout_concrete_cone(
        diameter_mm=12,
        h_ef_mm=121,
        bond_length_mm=121,
        tau_Rk=8.0,
        tau_Rk_ucr=20.0,
        fck_MPa=25,
        k_tau=2.3,
        n_anchors=2,
        edge_distance_mm=9999,
        spacing_mm=100,
        gamma_inst=1.2,
        dense_reinforcement=True,
        e_N_x_mm=0.0,
        e_N_y_mm=0.0,
    )
    assert tension["N_Rd_p"] == pytest.approx(27.584, abs=0.001)
    pryout = combined_pullout_concrete_cone(
        diameter_mm=12,
        h_ef_mm=121,
        bond_length_mm=121,
        tau_Rk=8.0,
        tau_Rk_ucr=20.0,
        fck_MPa=25,
        k_tau=2.3,
        n_anchors=4,
        edge_distance_mm=9999,
        spacing_x_mm=100,
        spacing_y_mm=100,
        gamma_inst=1.2,
        dense_reinforcement=True,
        e_N_x_mm=0.0,
        e_N_y_mm=0.0,
    )
    assert 2.0 * pryout["N_Rk_p"] / pryout["gamma_Mc_pryout"] == pytest.approx(
        91.904, abs=0.001
    )


def test_tr029_hit_z_bond_uses_the_helix_length():
    # hef,Helix = 60 mm is printed. The rod embedment is 89 mm and is not the bond length.
    result = combined_pullout_concrete_cone(
        diameter_mm=12,
        h_ef_mm=89,
        bond_length_mm=60,
        tau_Rk=22.0,
        tau_Rk_ucr=24.0,
        fck_MPa=25,
        k_tau=2.3,
        n_anchors=2,
        edge_distance_mm=9999,
        spacing_mm=100,
        gamma_inst=1.0,
        dense_reinforcement=True,
        e_N_x_mm=0.0,
        e_N_y_mm=0.0,
    )
    assert result["N_Rd_p"] == pytest.approx(51.606, abs=0.001)


def test_fib_bond_clause_7_2_1_6_matches_the_printed_pullout_and_pryout():
    """k3 = 7.7, Formula (7.19). psi_sus = 1 gives the sheet spacing cap 3 hef."""
    result = _en(
        diameter_mm=12,
        h_ef_mm=100,
        l_f_mm=100,
        concrete_thickness_mm=250,
        fck_cylinder_MPa=20,
        is_cracked=True,
        dense_reinforcement=True,
        number_of_anchors=2,
        spacing_mm=100,
        edge_distance_c1_mm=160,
        edge_distance_c2_1_mm=160,
        edge_distance_c2_2_mm=164,
        N_Rk_s_kN=42,
        V_Rk_s_kN=21,
        tau_Rk_cr=8.5,
        tau_Rk_ucr=18.0,
        psi_sus=1.0,
        s_cr_N_mm=300,
        c_cr_N_mm=150,
        k_cr_N=7.7,
        k_ucr_N=11.0,
        k7=1.0,
        k8=2.0,
        gamma_Ms_N=1.5,
        gamma_Ms_V=1.25,
        gamma_inst=1.0,
        splitting_check_required=False,
        e_N_x_mm=0,
        e_N_y_mm=0,
        e_V_mm=0,
    )
    assert result["tension"]["N_Rd_pullout"] == pytest.approx(28.994, abs=0.001)
    assert result["shear"]["V_Rd_pryout"] == pytest.approx(57.988, abs=0.001)


def test_fib_bond_matches_the_printed_pullout():
    # The sheet prints max tau_Rk,cr = 9.13, which is k1/pi with k1 = 7.7.
    result = combined_pullout_concrete_cone(
        diameter_mm=12,
        h_ef_mm=100,
        bond_length_mm=100,
        tau_Rk=8.5,
        tau_Rk_ucr=18.0,
        fck_MPa=20,
        k_tau=7.7 / math.pi,
        n_anchors=2,
        edge_distance_mm=160,
        spacing_mm=100,
        gamma_inst=1.0,
        dense_reinforcement=True,
        e_N_x_mm=0.0,
        e_N_y_mm=0.0,
    )
    assert result["N_Rd_p"] == pytest.approx(28.994, abs=0.001)
    assert 2.0 * result["N_Rk_p"] / result["gamma_Mc_pryout"] == pytest.approx(
        57.988, abs=0.001
    )
