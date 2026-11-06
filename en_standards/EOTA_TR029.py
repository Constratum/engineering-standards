"""Combined pull-out and concrete cone for a bonded anchor.

EOTA TR 029 section 5.2.2.3, which is the same proof the fib sheet prints
after ETAG bond testing. The basic resistance is the bond stress on the
bonded length. The group resistance is that value times the projected-area
ratio and the group factor psi_g,Np.

psi_g,Np uses tau_Rk,c = k * sqrt(h_ef * f_ck) / d. TR 029 prints k = 2.3.
The fib sheet does not print k; it prints the resulting max tau_Rk,cr, which
is the concrete-cone factor divided by pi.
"""

import math

from en_standards.ETAG001C_R045_2013 import (
    _positive,
    projected_area_tension,
    psi_re_N,
)

# TR 029 Eq. (5.3): s_cr,Np = 20 * d * (tau_Rk,ucr / 7.5)^0.5, and not more
# than three bonded lengths. c_cr,Np is half of that.
_S_CR_DIAMETER_FACTOR = 20.0
_TAU_UCR_REFERENCE = 7.5
_S_CR_BOND_LENGTH_CAP = 3.0

# TR 029 section 3.2.2. Concrete modes use 1.5. Tension bond then multiplies
# by the installation factor. Pryout stays at 1.5, which is what the sheets
# print as gM,c,p even when gM,p is 1.8.
_GAMMA_C = 1.5


def bond_critical_spacing(diameter_mm, tau_Rk_ucr, bond_length_mm):
    """s_cr,Np and c_cr,Np in mm."""
    diameter_mm = _positive(diameter_mm, "diameter_mm")
    tau_Rk_ucr = _positive(tau_Rk_ucr, "tau_Rk_ucr")
    bond_length_mm = _positive(bond_length_mm, "bond_length_mm")
    from_bond = (
        _S_CR_DIAMETER_FACTOR
        * diameter_mm
        * math.sqrt(tau_Rk_ucr / _TAU_UCR_REFERENCE)
    )
    s_cr = min(from_bond, _S_CR_BOND_LENGTH_CAP * bond_length_mm)
    return s_cr, s_cr / 2.0


def tau_Rk_c(k_tau, h_ef_mm, fck_MPa, diameter_mm):
    """Bond stress at which the concrete cone and the bond resistance meet."""
    k_tau = _positive(k_tau, "k_tau")
    h_ef_mm = _positive(h_ef_mm, "h_ef_mm")
    fck_MPa = _positive(fck_MPa, "fck_MPa")
    diameter_mm = _positive(diameter_mm, "diameter_mm")
    return k_tau * math.sqrt(h_ef_mm * fck_MPa) / diameter_mm


def psi_g_np(n_anchors, spacing_mm, s_cr_np, tau_Rk, tau_c):
    """TR 029 group factor. One anchor has no group factor."""
    n_anchors = int(n_anchors)
    if n_anchors < 1:
        raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")
    tau_Rk = _positive(tau_Rk, "tau_Rk")
    tau_c = _positive(tau_c, "tau_Rk_c")
    s_cr_np = _positive(s_cr_np, "s_cr_Np")
    if n_anchors == 1:
        return 1.0, 1.0
    spacing_mm = _positive(spacing_mm, "spacing_mm")
    root_n = math.sqrt(n_anchors)
    ratio = tau_Rk / tau_c
    psi0 = root_n - (root_n - 1.0) * ratio ** 1.5
    # The equation is written as psi >= 1. A ratio above 1 floors it.
    if psi0 < 1.0:
        psi0 = 1.0
    if spacing_mm >= s_cr_np:
        return 1.0, psi0
    psi = psi0 - math.sqrt(spacing_mm / s_cr_np) * (psi0 - 1.0)
    if psi < 1.0:
        psi = 1.0
    return psi, psi0


def _mean_spacing(n_anchors, spacing_mm, spacing_x_mm, spacing_y_mm):
    if n_anchors == 1:
        return None
    if n_anchors == 2:
        if spacing_mm is None:
            raise ValueError("spacing_mm is required for a 2-anchor bond group")
        return _positive(spacing_mm, "spacing_mm")
    if n_anchors == 4:
        if spacing_x_mm is None or spacing_y_mm is None:
            raise ValueError(
                "spacing_x_mm and spacing_y_mm are required for a 4-anchor bond group"
            )
        return (
            _positive(spacing_x_mm, "spacing_x_mm")
            + _positive(spacing_y_mm, "spacing_y_mm")
        ) / 2.0
    raise ValueError(f"Bond group supports 1, 2 or 4 anchors, got {n_anchors}")


def combined_pullout_concrete_cone(
    *,
    diameter_mm,
    h_ef_mm,
    bond_length_mm,
    tau_Rk,
    tau_Rk_ucr,
    fck_MPa,
    k_tau,
    n_anchors,
    edge_distance_mm,
    gamma_inst,
    dense_reinforcement,
    e_N_x_mm,
    e_N_y_mm,
    spacing_mm=None,
    spacing_x_mm=None,
    spacing_y_mm=None,
):
    """Group characteristic and design resistance, in kN.

    N_Rd,p divides by 1.5 * gamma_inst. The returned N_Rk,p is what pryout
    multiplies by k and then divides by 1.5, without gamma_inst.
    """
    diameter_mm = _positive(diameter_mm, "diameter_mm")
    h_ef_mm = _positive(h_ef_mm, "h_ef_mm")
    bond_length_mm = _positive(bond_length_mm, "bond_length_mm")
    tau_Rk = _positive(tau_Rk, "tau_Rk")
    gamma_inst = _positive(gamma_inst, "gamma_inst")
    if not isinstance(dense_reinforcement, bool):
        raise TypeError(
            "dense_reinforcement must be true or false, "
            f"got {dense_reinforcement!r}"
        )
    n_anchors = int(n_anchors)
    s_cr, c_cr = bond_critical_spacing(diameter_mm, tau_Rk_ucr, bond_length_mm)
    edge_distance_mm = _positive(edge_distance_mm, "edge_distance_mm")
    area = projected_area_tension(
        n_anchors,
        s_cr,
        c_cr,
        edge_distance_mm,
        spacing_mm,
        spacing_x_mm,
        spacing_y_mm,
    )
    area0 = s_cr * s_cr
    spacing = _mean_spacing(n_anchors, spacing_mm, spacing_x_mm, spacing_y_mm)
    tau_c = tau_Rk_c(k_tau, h_ef_mm, fck_MPa, diameter_mm)
    psi_g, psi0 = psi_g_np(n_anchors, spacing if spacing is not None else s_cr, s_cr, tau_Rk, tau_c)
    c_min = min(edge_distance_mm, c_cr)
    psi_s = min(0.7 + 0.3 * (c_min / c_cr), 1.0)
    psi_re = psi_re_N(h_ef_mm, dense_reinforcement)
    e_N_x_mm = float(e_N_x_mm)
    e_N_y_mm = float(e_N_y_mm)
    if e_N_x_mm < 0.0 or e_N_y_mm < 0.0:
        raise ValueError("e_N_x_mm and e_N_y_mm must be >= 0")
    psi_ec1 = min(1.0 / (1.0 + 2.0 * e_N_x_mm / s_cr), 1.0)
    psi_ec2 = min(1.0 / (1.0 + 2.0 * e_N_y_mm / s_cr), 1.0)
    n0 = tau_Rk * math.pi * diameter_mm * bond_length_mm / 1000.0
    n_rk = n0 * (area / area0) * psi_s * psi_re * psi_ec1 * psi_ec2 * psi_g
    gamma_mp = _GAMMA_C * gamma_inst
    return {
        "N0_Rk_p": n0,
        "N_Rk_p": n_rk,
        "N_Rd_p": n_rk / gamma_mp,
        "A_p_N": area,
        "A0_p_N": area0,
        "s_cr_Np": s_cr,
        "c_cr_Np": c_cr,
        "psi_g_Np": psi_g,
        "psi_g_Np0": psi0,
        "psi_s_Np": psi_s,
        "tau_Rk_c": tau_c,
        "gamma_Mp": gamma_mp,
        "gamma_Mc_pryout": _GAMMA_C,
    }
