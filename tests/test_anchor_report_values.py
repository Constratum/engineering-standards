"""Manufacturer-report values for the corrected ETAG engine and EN 1992-4.

The numbers are the ones printed in the anchor reports. c2 pairs that are
not a measured edge are marked in the test name as derived from the printed
area or psi factor.
"""

import math

import pytest

from en_standards.EN_1992_4_2018 import (
    calculate_mechanical_anchor,
    interaction_from_capacities,
    interaction_table_7_3,
)
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


def test_fib_edge_area_needs_both_perpendicular_edges():
    """c2_2 = 164 mm is derived from the printed area 101760 mm2.

    The report prints psi_s,V = 0.900, which is the nearer edge c2_1 = 160 mm.
    Width 160 + 100 + 164 at height 240 mm is the printed area.
    """
    result = calculate_mechanical_anchor(
        diameter_mm=12,
        embedment_depth_mm=100,
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
        k9=1.7,
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
    result = calculate_mechanical_anchor(
        diameter_mm=10,
        embedment_depth_mm=42.5,
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
        k9=1.7,
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
    result = calculate_mechanical_anchor(
        diameter_mm=10,
        embedment_depth_mm=42.5,
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
        k9=1.7,
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


def test_en1992_seismic_is_not_implemented():
    with pytest.raises(NotImplementedError, match="C1/C2"):
        calculate_mechanical_anchor(
            diameter_mm=10,
            embedment_depth_mm=42.5,
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
            k9=1.7,
            gamma_Ms_N=1.5,
            gamma_Ms_V=1.25,
            gamma_inst=1.2,
            splitting_check_required=False,
            seismic_design=True,
            e_N_x_mm=0,
            e_N_y_mm=0,
            e_V_mm=0,
        )


def test_finite_edge_without_c2_is_rejected():
    with pytest.raises(ValueError, match="c2"):
        _rawlplug(edge_distance_c2_1=None, edge_distance_c2_2=None)
