"""Static design of mechanical anchors to EN 1992-4:2018.

Seismic design is refused. It needs the ETA seismic category C1 or C2
factors, which this module does not contain.

Partial factors and the product coefficients k7, k8, k9, k_cr,N and
k_ucr,N come from the caller. Nothing here is filled in when they are
missing.
"""

import math

from en_standards.ETAG001C_R045_2013 import (
    NO_CONCRETE_EDGE_MM,
    projected_area_shear,
    projected_area_tension,
    psi_h_sp,
    psi_re_N,
)

# EN 1992-4, Table 4.1. The ETA installation factor multiplies the tension
# concrete and pull-out values. Concrete shear stays at the basic factor.
_GAMMA_C_BASIC = 1.5


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return number


def _flag(value, label):
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be true or false, got {value!r}")
    return value


def interaction_table_7_3(beta_n_steel, beta_v_steel, beta_n_concrete, beta_v_concrete):
    """EN 1992-4 Table 7.3. Steel exponent 2, concrete exponent 1.5."""
    steel = beta_n_steel**2 + beta_v_steel**2
    concrete = beta_n_concrete**1.5 + beta_v_concrete**1.5
    return {
        "steel": steel,
        "concrete": concrete,
        "governing": max(steel, concrete),
    }


def calculate_mechanical_anchor(
    *,
    diameter_mm,
    embedment_depth_mm,
    l_f_mm,
    concrete_thickness_mm,
    fck_cylinder_MPa,
    is_cracked,
    dense_reinforcement,
    number_of_anchors,
    edge_distance_c1_mm,
    N_Rk_s_kN,
    V_Rk_s_kN,
    N_Rk_p_kN,
    psi_c,
    s_cr_N_mm,
    c_cr_N_mm,
    k_cr_N,
    k_ucr_N,
    k7,
    k8,
    k9,
    gamma_Ms_N,
    gamma_Ms_V,
    gamma_inst,
    splitting_check_required,
    seismic_design,
    spacing_mm=None,
    spacing_x_mm=None,
    spacing_y_mm=None,
    edge_distance_c2_1_mm=None,
    edge_distance_c2_2_mm=None,
    c_cr_sp_mm=None,
    s_cr_sp_mm=None,
    h_min_mm=None,
    e_N_x_mm=None,
    e_N_y_mm=None,
    e_V_mm=None,
):
    """Characteristic and design resistance for one static mechanical anchor group.

    Resistances are in kN. Group modes are for the whole group. Steel and
    pull-out are for one anchor.
    """
    if _flag(seismic_design, "seismic_design"):
        raise NotImplementedError(
            "EN 1992-4 seismic design needs the ETA C1/C2 reduction factors. "
            "Those factors are not part of this calculation."
        )
    for label, value in (
        ("e_N_x_mm", e_N_x_mm),
        ("e_N_y_mm", e_N_y_mm),
        ("e_V_mm", e_V_mm),
    ):
        if value is None:
            raise ValueError(f"{label} is required. Pass 0 for a concentric load.")
    n = int(number_of_anchors)
    if n not in (1, 2, 4):
        raise ValueError(f"number_of_anchors must be 1, 2 or 4, got {number_of_anchors!r}")
    diameter = _positive(diameter_mm, "diameter_mm")
    h_ef = _positive(embedment_depth_mm, "embedment_depth_mm")
    l_f = _positive(l_f_mm, "l_f_mm")
    h = _positive(concrete_thickness_mm, "concrete_thickness_mm")
    fck = _positive(fck_cylinder_MPa, "fck_cylinder_MPa")
    cracked = _flag(is_cracked, "is_cracked")
    c1 = _positive(edge_distance_c1_mm, "edge_distance_c1_mm")
    s_cr_N = _positive(s_cr_N_mm, "s_cr_N_mm")
    c_cr_N = _positive(c_cr_N_mm, "c_cr_N_mm")
    k1 = _positive(k_cr_N if cracked else k_ucr_N, "k_cr_N" if cracked else "k_ucr_N")
    gamma_c = _GAMMA_C_BASIC * _positive(gamma_inst, "gamma_inst")
    gamma_c_shear = _GAMMA_C_BASIC
    gamma_ms_n = _positive(gamma_Ms_N, "gamma_Ms_N")
    gamma_ms_v = _positive(gamma_Ms_V, "gamma_Ms_V")

    n_rd_s = _positive(N_Rk_s_kN, "N_Rk_s_kN") / gamma_ms_n
    n_rd_p = (
        _positive(psi_c, "psi_c")
        * _positive(N_Rk_p_kN, "N_Rk_p_kN")
        / gamma_c
    )
    n0 = k1 * math.sqrt(fck) * h_ef**1.5 / 1000.0
    a0_n = s_cr_N * s_cr_N
    a_c_n = projected_area_tension(
        n, s_cr_N, c_cr_N, c1, spacing_mm, spacing_x_mm, spacing_y_mm
    )
    c_for_psi = min(c1, c_cr_N)
    psi_s = min(0.7 + 0.3 * (c_for_psi / c_cr_N), 1.0)
    psi_re = psi_re_N(h_ef, dense_reinforcement)
    psi_ec1 = (
        min(1.0 / (1.0 + 2.0 * abs(float(e_N_x_mm)) / s_cr_N), 1.0)
        if s_cr_N > 0
        else 1.0
    )
    psi_ec2 = (
        min(1.0 / (1.0 + 2.0 * abs(float(e_N_y_mm)) / s_cr_N), 1.0)
        if s_cr_N > 0
        else 1.0
    )
    n_rk_c = n0 * (a_c_n / a0_n) * psi_s * psi_re * psi_ec1 * psi_ec2
    n_rd_c = n_rk_c / gamma_c

    splitting = {"skipped": True, "N_Rd_sp": float("inf")}
    if _flag(splitting_check_required, "splitting_check_required"):
        if c_cr_sp_mm is None or s_cr_sp_mm is None or h_min_mm is None:
            raise ValueError(
                "splitting_check_required needs c_cr_sp_mm, s_cr_sp_mm and h_min_mm"
            )
        s_cr_sp = _positive(s_cr_sp_mm, "s_cr_sp_mm")
        c_cr_sp = _positive(c_cr_sp_mm, "c_cr_sp_mm")
        a0_sp = s_cr_sp * s_cr_sp
        a_c_sp = projected_area_tension(
            n, s_cr_sp, c_cr_sp, c1, spacing_mm, spacing_x_mm, spacing_y_mm
        )
        c_sp = min(c1, c_cr_sp)
        psi_s_sp = min(0.7 + 0.3 * (c_sp / c_cr_sp), 1.0)
        psi_h = psi_h_sp(h, h_min_mm, h_ef)
        n_rk_sp = n0 * (a_c_sp / a0_sp) * psi_s_sp * psi_re * psi_ec1 * psi_ec2 * psi_h
        splitting = {
            "skipped": False,
            "N_Rd_sp": n_rk_sp / gamma_c,
            "psi_h_sp": psi_h,
            "A_c_N": a_c_sp,
            "A0_c_N": a0_sp,
        }

    v_rd_s = _positive(k7, "k7") * _positive(V_Rk_s_kN, "V_Rk_s_kN") / gamma_ms_v
    v_rd_cp = _positive(k8, "k8") * n_rk_c / gamma_c_shear

    if c1 >= NO_CONCRETE_EDGE_MM:
        edge = {
            "V0_Rk_c": float("inf"),
            "V_Rd_c": float("inf"),
            "A_c_V": float("inf"),
            "A0_c_V": float("inf"),
        }
    else:
        if edge_distance_c2_1_mm is None or edge_distance_c2_2_mm is None:
            raise ValueError(
                "edge_distance_c2_1_mm and edge_distance_c2_2_mm are required "
                f"when c1={c1} mm is a concrete edge."
            )
        alpha = 0.1 * (l_f / c1) ** 0.5
        beta = 0.1 * (diameter / c1) ** 0.2
        v0 = (
            _positive(k9, "k9")
            * diameter**alpha
            * l_f**beta
            * math.sqrt(fck)
            * c1**1.5
            / 1000.0
        )
        a0_v = 4.5 * c1 * c1
        a_c_v = projected_area_shear(
            n,
            c1,
            edge_distance_c2_1_mm,
            edge_distance_c2_2_mm,
            h,
            spacing_mm,
            spacing_y_mm,
        )
        c2 = min(float(edge_distance_c2_1_mm), float(edge_distance_c2_2_mm))
        psi_s_v = min(0.7 + 0.3 * (c2 / (1.5 * c1)), 1.0)
        psi_h_v = math.sqrt(1.5 * c1 / h) if h < 1.5 * c1 else 1.0
        psi_ec_v = min(1.0 / (1.0 + 2.0 * abs(float(e_V_mm)) / (3.0 * c1)), 1.0)
        v_rk_c = v0 * (a_c_v / a0_v) * psi_s_v * psi_h_v * psi_ec_v
        edge = {
            "V0_Rk_c": v0,
            "V_Rd_c": v_rk_c / gamma_c_shear,
            "A_c_V": a_c_v,
            "A0_c_V": a0_v,
            "psi_s_V": psi_s_v,
            "psi_h_V": psi_h_v,
            "alpha_exp": alpha,
            "beta_exp": beta,
        }

    return {
        "tension": {
            "N_Rd_steel": n_rd_s,
            "N_Rd_pullout": n_rd_p,
            "N_Rd_concrete": n_rd_c,
            "N_Rd_splitting": splitting["N_Rd_sp"],
            "concrete_details": {
                "k1": k1,
                "N0_Rk_c": n0,
                "A_c_N": a_c_n,
                "A0_c_N": a0_n,
                "psi_s_N": psi_s,
                "psi_re_N": psi_re,
                "gamma_Mc": gamma_c,
            },
            "pullout_details": {"gamma_Mp": gamma_c, "psi_c": float(psi_c)},
            "steel_details": {"gamma_Ms": gamma_ms_n},
            "splitting_details": splitting,
        },
        "shear": {
            "V_Rd_steel": v_rd_s,
            "V_Rd_pryout": v_rd_cp,
            "V_Rd_concrete": edge["V_Rd_c"],
            "steel_details": {"gamma_Ms": gamma_ms_v, "k7": float(k7)},
            "pryout_details": {"k8": float(k8), "gamma_Mc": gamma_c_shear},
            "concrete_edge_details": edge,
        },
    }


def interaction_from_capacities(result, tension_kN, shear_kN, number_of_anchors):
    """Table 7.3 from the resistances and the group actions.

    Steel demand is the group action divided by the anchor count. Concrete
    demand is the group action. The concrete tension ratio is the largest of
    pull-out, cone and, when it was calculated, splitting.
    """
    n = int(number_of_anchors)
    if n < 1:
        raise ValueError(f"number_of_anchors must be >= 1, got {n}")
    tension = result["tension"]
    shear = result["shear"]
    beta_n_steel = (tension_kN / n) / tension["N_Rd_steel"]
    beta_v_steel = (shear_kN / n) / shear["V_Rd_steel"]
    concrete_tension = [
        tension_kN / tension["N_Rd_pullout"] / n,
        tension_kN / tension["N_Rd_concrete"],
    ]
    # Pull-out above is per anchor, so the group demand is divided by n.
    # The list entry already includes that division.
    if not tension["splitting_details"]["skipped"]:
        concrete_tension.append(tension_kN / tension["N_Rd_splitting"])
    concrete_shear = [
        shear_kN / shear["V_Rd_pryout"],
        shear_kN / shear["V_Rd_concrete"],
    ]
    return interaction_table_7_3(
        beta_n_steel,
        beta_v_steel,
        max(concrete_tension),
        max(concrete_shear),
    )
