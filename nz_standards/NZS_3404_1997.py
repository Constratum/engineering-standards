"""NZS_3404_1997 Method

This module references the following standard:
NZS 3404:Parts 1 and 2:1997, incorporating Amendments 1 and 2.

Covers practical steel member design helpers for sections 3, 4, 5, 6, 7, 8
and 12. The functions follow the style used in NZS_AS_1720_1_2022.py:
material/standard tables live in the module, section properties are passed
explicitly by the caller, and action/capacity comparison is normally done
outside this module.

Method developed June 2026
(c) Constratum Ltd

Developed - Nima Shokrollahi
Reviewed  - TBD
"""

import numpy as np

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _residual_key(residual_stress):
    """Normalise residual stress labels used in NZS 3404 tables."""
    key = str(residual_stress).upper().replace(" ", "").strip()
    aliases = {
        "HOTROLLED": "HR",
        "HOT-FINISHED": "HR",
        "HOTFINISHED": "HR",
        "COLD-FORMED": "CF",
        "COLDFORMED": "CF",
        "LIGHTLYWELDED": "LW",
        "LIGHTWELDED": "LW",
        "HEAVILYWELDED": "HW",
        "STRESSRELIEVED": "SR",
        "ANY": "ANY",
    }
    return aliases.get(key, key)


def _check_positive(**values):
    """Raise ValueError when a required engineering input is not positive."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value!r}")


# ------------------------------------------------------------------
# Section 3 - General design requirements
# ------------------------------------------------------------------
class Section3:
    """
    General design requirements and strength reduction factors for NZS 3404.

    Table 3.3(1): ULS strength reduction factors for bare steel members and
    connections. Table 3.3(2): SLS bolt slip factor for friction-type
    connections.
    """

    TABLE_3_3_1_PHI = {
        "bending_shear": 0.90,
        "axial_compression": 0.90,
        "axial_tension": 0.90,
        "combined_actions": 0.90,
        "connection_component": 0.90,
        "bolt_shear": 0.80,
        "bolt_tension": 0.80,
        "bolt_combined": 0.80,
        "ply_bearing_bolt": 0.90,
        "bolt_group": 0.80,
        "pin_shear": 0.80,
        "pin_bearing": 0.80,
        "pin_bending": 0.80,
        "ply_bearing_pin": 0.90,
        "complete_penetration_butt_weld_sp": 0.90,
        "complete_penetration_butt_weld_gp": 0.60,
        "rhs_longitudinal_fillet_weld_t_lt_3": 0.70,
        "fillet_weld_sp": 0.80,
        "fillet_weld_gp": 0.60,
        "plug_or_slot_weld_sp": 0.80,
        "plug_or_slot_weld_gp": 0.60,
        "weld_group_sp": 0.80,
        "weld_group_gp": 0.60,
    }

    TABLE_3_3_2_PHI_SLS = {
        "friction_type_bolted_connection": 0.70,
    }

    def get_phi(self, design_case="bending_shear"):
        """
        Strength reduction factor phi (Table 3.3(1)).

        Args:
            design_case (str): Key from TABLE_3_3_1_PHI.

        Returns:
            float: strength reduction factor.
        """
        key = str(design_case).lower().strip()
        if key not in self.TABLE_3_3_1_PHI:
            raise ValueError(
                f"Design case '{design_case}' not recognised. "
                f"Valid: {list(self.TABLE_3_3_1_PHI.keys())}"
            )
        return self.TABLE_3_3_1_PHI[key]

    def get_phi_sls(self, design_case="friction_type_bolted_connection"):
        """
        Serviceability strength reduction factor phi (Table 3.3(2)).

        Args:
            design_case (str): Key from TABLE_3_3_2_PHI_SLS.

        Returns:
            float: serviceability strength reduction factor.
        """
        key = str(design_case).lower().strip()
        if key not in self.TABLE_3_3_2_PHI_SLS:
            raise ValueError(
                f"SLS design case '{design_case}' not recognised. "
                f"Valid: {list(self.TABLE_3_3_2_PHI_SLS.keys())}"
            )
        return self.TABLE_3_3_2_PHI_SLS[key]

    def design_capacity(self, nominal_capacity, design_case="bending_shear"):
        """
        ULS design capacity phi*Ru (Cl 3.3).

        Args:
            nominal_capacity (float): Nominal capacity Ru.
            design_case (str): Table 3.3(1) design case.

        Returns:
            float: phi*Ru.
        """
        return self.get_phi(design_case) * nominal_capacity

    @staticmethod
    def utilisation(action, design_capacity):
        """
        Utilisation ratio S*/(phi*Ru) (Cl 3.3).

        Args:
            action (float): Design action S*.
            design_capacity (float): Design capacity phi*Ru.

        Returns:
            float: utilisation ratio. Pass when <= 1.0.
        """
        _check_positive(design_capacity=design_capacity)
        return action / design_capacity


# ------------------------------------------------------------------
# Section 4 - Structural analysis
# ------------------------------------------------------------------
class Section4:
    """
    Structural analysis helpers for NZS 3404 section 4.

    Includes elastic buckling loads, simple moment-amplification utilities,
    frame buckling approximations and plastic hinge rotation limits.
    """

    TABLE_4_7_ROTATION_LIMITS_MRAD = {
        "negligible": {
            1: {"earthquake": 40, "non_earthquake": 60},
            2: {"earthquake": 40, "non_earthquake": 60},
            3: {"earthquake": 30, "non_earthquake": 45},
        },
        "low": {
            1: {"earthquake": 30, "non_earthquake": 45},
            2: {"earthquake": 30, "non_earthquake": 45},
            3: {"earthquake": 20, "non_earthquake": 30},
        },
        "moderate": {
            1: {"earthquake": 13, "non_earthquake": 20},
            2: {"earthquake": 13, "non_earthquake": 20},
            3: {"earthquake": 10, "non_earthquake": 15},
        },
        "high": {
            1: {"earthquake": 8, "non_earthquake": 15},
            2: {"earthquake": 8, "non_earthquake": 15},
            3: {"earthquake": 5, "non_earthquake": 10},
        },
    }

    TABLE_4_8_3_2_KE_IDEALISED = {
        1: 0.70,
        2: 0.85,
        3: 1.00,
        4: 1.20,
        5: 2.20,
        6: 2.20,
    }

    TABLE_4_8_3_4_BETA_E = {
        "pinned": {"braced": 1.50, "sway": 0.50},
        "rigid_column": {"braced": 1.00, "sway": 1.00},
        "fixed": {"braced": 2.00, "sway": 0.67},
    }

    @staticmethod
    def elastic_buckling_load(E_MPa, I_mm4, L_mm, k_e=1.0):
        """
        Member elastic buckling load Nom (Cl 4.8.2).

        Nom = pi^2*E*I/(k_e*L)^2 [N]
        """
        _check_positive(E_MPa=E_MPa, I_mm4=I_mm4, L_mm=L_mm, k_e=k_e)
        return np.pi**2 * E_MPa * I_mm4 / (k_e * L_mm) ** 2

    def get_ke_idealised(self, case_number):
        """
        Effective length factor for idealised end restraints (Fig 4.8.3.2).

        Args:
            case_number (int): Case 1 to 6.

        Returns:
            float: k_e.
        """
        if case_number not in self.TABLE_4_8_3_2_KE_IDEALISED:
            raise ValueError("case_number must be one of 1, 2, 3, 4, 5, 6")
        return self.TABLE_4_8_3_2_KE_IDEALISED[case_number]

    def get_beta_e(self, far_end_fixity, member_type="braced"):
        """
        Beam far-end modifying factor beta_e (Table 4.8.3.4).

        Args:
            far_end_fixity (str): 'pinned', 'rigid_column', or 'fixed'.
            member_type (str): 'braced' or 'sway'.

        Returns:
            float: beta_e.
        """
        fixity = str(far_end_fixity).lower().strip()
        mtype = str(member_type).lower().strip()
        if fixity not in self.TABLE_4_8_3_4_BETA_E:
            raise ValueError(
                f"far_end_fixity must be one of {list(self.TABLE_4_8_3_4_BETA_E)}"
            )
        if mtype not in ("braced", "sway"):
            raise ValueError("member_type must be 'braced' or 'sway'")
        return self.TABLE_4_8_3_4_BETA_E[fixity][mtype]

    @staticmethod
    def stiffness_ratio_gamma(
        sum_I_over_L_columns, sum_beta_I_over_L_beams, base_condition=None
    ):
        """
        Stiffness ratio gamma in rectangular frames (Cl 4.8.3.4).

        Args:
            sum_I_over_L_columns (float): Sum of column stiffnesses.
            sum_beta_I_over_L_beams (float): Sum of beam stiffnesses.
            base_condition (str, optional): 'not_rigid' enforces gamma >= 10,
                'rigid' enforces gamma >= 0.6.

        Returns:
            float: gamma.
        """
        _check_positive(sum_beta_I_over_L_beams=sum_beta_I_over_L_beams)
        gamma = sum_I_over_L_columns / sum_beta_I_over_L_beams
        if base_condition == "not_rigid":
            return max(gamma, 10.0)
        if base_condition == "rigid":
            return max(gamma, 0.6)
        return gamma

    @staticmethod
    def frame_buckling_factor_braced(N_omb, N_star):
        """Column/frame elastic buckling factor lambda_mb (Cl 4.9.2.2)."""
        _check_positive(N_star=N_star)
        return N_omb / N_star

    @staticmethod
    def frame_buckling_factor_sway(N_oms_over_L, N_star_over_L):
        """
        Storey sway buckling factor lambda_ms (Cl 4.9.2.3).

        Args:
            N_oms_over_L (iterable): Values of N_oms/L for storey columns.
            N_star_over_L (iterable): Values of N*/L; tension may be negative.
        """
        denominator = float(sum(N_star_over_L))
        _check_positive(sum_N_star_over_L=denominator)
        return float(sum(N_oms_over_L)) / denominator

    @staticmethod
    def portal_frame_lambda_pinned(
        E_MPa, I_pr_mm4, s_r_mm, N_c_star_N, h_e_mm, N_r_star_N
    ):
        """Portal frame buckling factor for pinned-base frames (Cl 4.9.2.4)."""
        denominator = s_r_mm * (N_c_star_N * h_e_mm + 0.3 * N_r_star_N * s_r_mm)
        _check_positive(E_MPa=E_MPa, I_pr_mm4=I_pr_mm4, denominator=denominator)
        return 3.0 * E_MPa * I_pr_mm4 / denominator

    @staticmethod
    def portal_frame_lambda_fixed(
        E_MPa, I_pc_mm4, I_pr_mm4, h_e_mm, s_r_mm, N_c_star_N, N_r_star_N
    ):
        """Portal frame buckling factor for fixed-base frames (Cl 4.9.2.4)."""
        _check_positive(
            E_MPa=E_MPa,
            I_pc_mm4=I_pc_mm4,
            I_pr_mm4=I_pr_mm4,
            h_e_mm=h_e_mm,
            s_r_mm=s_r_mm,
        )
        psi_f = I_pc_mm4 * s_r_mm / (I_pr_mm4 * h_e_mm)
        denominator = (5.0 * N_r_star_N * s_r_mm**2 / I_pr_mm4) + (
            2.0 * psi_f * N_c_star_N * h_e_mm**2 / I_pc_mm4
        )
        _check_positive(denominator=denominator)
        return 5.0 * E_MPa * (10.0 + psi_f) / denominator

    @staticmethod
    def plastic_analysis_delta_p(lambda_c):
        """
        Moment amplification factor delta_p for plastic analysis (Cl 4.6.4).

        Valid for 5 <= lambda_c < 10.
        """
        if lambda_c < 5.0:
            raise ValueError("lambda_c < 5 requires second-order plastic analysis")
        if lambda_c >= 10.0:
            return 1.0
        return 0.9 / (1.0 - 1.0 / lambda_c)

    def plastic_hinge_rotation_limit(
        self, N_star, phi_NS, member_category, includes_earthquake=True
    ):
        """
        Limiting plastic hinge rotation theta_p (Cl 4.7.2).

        Args:
            N_star (float): Design axial force N* [N].
            phi_NS (float): Design compression section capacity phi*Ns [N].
            member_category (int): 1, 2 or 3.
            includes_earthquake (bool): Selects earthquake/non-earthquake column.

        Returns:
            float: limiting plastic rotation in radians.
        """
        _check_positive(phi_NS=phi_NS)
        if member_category not in (1, 2, 3):
            raise ValueError("member_category must be 1, 2 or 3")
        ratio = N_star / phi_NS
        if ratio <= 0.15:
            band = "negligible"
        elif ratio <= 0.30:
            band = "low"
        elif ratio <= 0.50:
            band = "moderate"
        elif ratio <= 0.80:
            band = "high"
        else:
            raise ValueError("N*/(phi*Ns) exceeds 0.8; Table 4.7 limit not provided")
        column = "earthquake" if includes_earthquake else "non_earthquake"
        return (
            self.TABLE_4_7_ROTATION_LIMITS_MRAD[band][member_category][column] / 1000.0
        )


# ------------------------------------------------------------------
# Section 5 - Members subject to bending and shear
# ------------------------------------------------------------------
class Section5(Section3):
    """
    Bending, lateral buckling and web shear helpers for NZS 3404 section 5.

    Section properties (Z, S, Iy, Iw, J, Aw, etc.) are supplied by the caller.
    Units are MPa, N, mm, mm2, mm3 and mm4 unless noted.
    """

    TABLE_5_2_LIMITS = {
        1: {
            "SR": (10, 16, 35),
            "HR": (9, 16, 35),
            "LW": (8, 15, 35),
            "CF": (8, 15, 35),
            "HW": (8, 14, 35),
        },
        2: {
            "SR": (10, 25, None),
            "HR": (9, 25, None),
            "LW": (8, 22, None),
            "CF": (8, 22, None),
            "HW": (8, 22, None),
        },
        3: {
            "SR": (30, 45, 90),
            "HR": (30, 45, 90),
            "LW": (30, 40, 90),
            "CF": (30, 40, 90),
            "HW": (30, 35, 90),
        },
        4: {
            "RHS": (45, 60, None),
            "OTHER": (82, 130, None),
            "I_WEB": (None, 180, None),
            "ANY": (82, 130, None),
        },
        5: {
            "SR": (50, 120, None),
            "HR": (50, 120, None),
            "CF": (50, 120, None),
            "LW": (42, 120, None),
            "HW": (42, 120, None),
        },
    }

    TABLE_5_6_2_ALPHA_M_UNRESTRAINED = {1: 0.25, 2: 1.25, 3: 2.25}

    def get_plate_limits_5_2(self, case_number, residual_stress="HR", subtype=None):
        """
        Plate element slenderness limits (Table 5.2).

        Returns:
            tuple: (lambda_ep, lambda_ey, lambda_ed).
        """
        if case_number not in self.TABLE_5_2_LIMITS:
            raise ValueError("case_number must be one of 1, 2, 3, 4, 5")
        if case_number == 4:
            key = str(subtype or "ANY").upper().strip()
        else:
            key = _residual_key(residual_stress)
        table = self.TABLE_5_2_LIMITS[case_number]
        if key not in table:
            raise ValueError(
                f"Limit key '{key}' not valid for Table 5.2 case {case_number}"
            )
        return table[key]

    @staticmethod
    def plate_slenderness(b_mm, t_mm, fy_MPa):
        """Flat plate element slenderness lambda_e (Cl 5.2.2.1)."""
        _check_positive(b_mm=b_mm, t_mm=t_mm, fy_MPa=fy_MPa)
        return (b_mm / t_mm) * np.sqrt(fy_MPa / 250.0)

    @staticmethod
    def chs_slenderness(d_o_mm, t_mm, fy_MPa):
        """Circular hollow section slenderness lambda_s (Cl 5.2.2.4)."""
        _check_positive(d_o_mm=d_o_mm, t_mm=t_mm, fy_MPa=fy_MPa)
        return (d_o_mm / t_mm) * (fy_MPa / 250.0)

    @staticmethod
    def effective_section_modulus(
        Z_mm3, S_mm3, lambda_s, lambda_sp, lambda_sy, slender_case="uniform"
    ):
        """
        Effective section modulus Ze (Cl 5.2.3 to 5.2.5).

        Args:
            Z_mm3 (float): Elastic section modulus.
            S_mm3 (float): Plastic section modulus.
            lambda_s (float): Section slenderness parameter.
            lambda_sp (float): Plasticity slenderness limit.
            lambda_sy (float): Yield slenderness limit.
            slender_case (str): 'uniform', 'outstand_gradient', or 'chs'.

        Returns:
            float: effective section modulus Ze.
        """
        _check_positive(Z_mm3=Z_mm3, S_mm3=S_mm3, lambda_sy=lambda_sy)
        Zc = min(S_mm3, 1.5 * Z_mm3)
        if lambda_sp is not None and lambda_s <= lambda_sp:
            return Zc
        if lambda_sp is not None and lambda_s <= lambda_sy:
            return Z_mm3 + ((lambda_sy - lambda_s) / (lambda_sy - lambda_sp)) * (
                Zc - Z_mm3
            )
        if slender_case == "uniform":
            return Z_mm3 * (lambda_sy / lambda_s)
        if slender_case == "outstand_gradient":
            return Z_mm3 * (lambda_sy / lambda_s) ** 2
        if slender_case == "chs":
            return min(
                Z_mm3 * np.sqrt(lambda_sy / lambda_s),
                Z_mm3 * (2.0 * lambda_sy / lambda_s) ** 2,
            )
        raise ValueError(
            "slender_case must be 'uniform', 'outstand_gradient', or 'chs'"
        )

    @staticmethod
    def nominal_section_moment_capacity(fy_MPa, Ze_mm3):
        """Nominal section moment capacity Ms = fy*Ze (Cl 5.2.1) [N.mm]."""
        _check_positive(fy_MPa=fy_MPa, Ze_mm3=Ze_mm3)
        return fy_MPa * Ze_mm3

    def design_section_moment_capacity(self, fy_MPa, Ze_mm3):
        """Design section moment capacity phi*Ms (Cl 5.1, 5.2) [N.mm]."""
        return self.get_phi("bending_shear") * self.nominal_section_moment_capacity(
            fy_MPa, Ze_mm3
        )

    @staticmethod
    def alpha_m_quarter_point(Mm_star, M2_star, M3_star, M4_star):
        """
        Moment modification factor alpha_m from quarter-point moments
        (Eq 5.6.1.1(2)).
        """
        _check_positive(Mm_star=Mm_star)
        denom = np.sqrt(M2_star**2 + M3_star**2 + M4_star**2)
        _check_positive(denominator=denom)
        return min(1.7 * abs(Mm_star) / denom, 2.5)

    @staticmethod
    def alpha_m_table_5_6_1(
        case_number, beta_m=None, two_a_over_L=None, contains_yielding_region=False
    ):
        """
        Moment modification factor alpha_m for both-end restrained segments
        (Table 5.6.1).
        """
        c = int(case_number)
        if c == 1:
            if beta_m is None:
                raise ValueError("beta_m is required for case 1")
            alpha = 1.75 + 1.05 * beta_m + 0.3 * beta_m**2 if beta_m <= 0.6 else 2.5
        elif c == 2:
            if two_a_over_L is None:
                raise ValueError("two_a_over_L is required for case 2")
            alpha = 1.0 + 0.35 * (1.0 - two_a_over_L) ** 2
        elif c == 3:
            if two_a_over_L is None:
                raise ValueError("two_a_over_L is required for case 3")
            alpha = 1.35 + 0.4 * two_a_over_L**2
        elif c == 4:
            if beta_m is None:
                raise ValueError("beta_m is required for case 4")
            alpha = 1.35 + 0.15 * beta_m if beta_m < 0.9 else -1.2 + 3.0 * beta_m
        elif c == 5:
            if beta_m is None:
                raise ValueError("beta_m is required for case 5")
            alpha = 1.35 + 0.36 * beta_m
        elif c == 6:
            if beta_m is None:
                raise ValueError("beta_m is required for case 6")
            alpha = 1.13 + 0.10 * beta_m if beta_m <= 0.7 else -1.25 + 3.5 * beta_m
        elif c == 7:
            if beta_m is None:
                raise ValueError("beta_m is required for case 7")
            alpha = 1.13 + 0.12 * beta_m if beta_m <= 0.75 else -2.38 + 4.8 * beta_m
        elif c == 8:
            alpha = 1.0
        elif c == 9:
            alpha = 1.75
        elif c == 10:
            alpha = 3.5
        else:
            raise ValueError("case_number must be in the range 1..10")
        return min(alpha, 1.75) if contains_yielding_region else alpha

    def alpha_m_table_5_6_2(self, case_number, contains_yielding_region=False):
        """Moment modification factor for one-end unrestrained segments (Table 5.6.2)."""
        if case_number not in self.TABLE_5_6_2_ALPHA_M_UNRESTRAINED:
            raise ValueError("case_number must be 1, 2 or 3")
        alpha = self.TABLE_5_6_2_ALPHA_M_UNRESTRAINED[case_number]
        return min(alpha, 1.75) if contains_yielding_region else alpha

    @staticmethod
    def effective_length_ltb(L_mm, k_t=1.0, k_l=1.0, k_r=1.0):
        """Lateral buckling effective length Le = kt*kl*kr*L (Cl 5.6.3)."""
        _check_positive(L_mm=L_mm, k_t=k_t, k_l=k_l, k_r=k_r)
        return k_t * k_l * k_r * L_mm

    @staticmethod
    def reference_buckling_moment(E_MPa, G_MPa, Iy_mm4, J_mm4, Iw_mm6, Le_mm):
        """
        Reference lateral buckling moment M0 for open equal-flange sections
        (Eq 5.6.1.1(4)) [N.mm].
        """
        _check_positive(E_MPa=E_MPa, G_MPa=G_MPa, Iy_mm4=Iy_mm4, Le_mm=Le_mm)
        return np.sqrt(
            (np.pi**2 * E_MPa * Iy_mm4 / Le_mm**2)
            * (G_MPa * J_mm4 + np.pi**2 * E_MPa * Iw_mm6 / Le_mm**2)
        )

    @staticmethod
    def alpha_s_ltb(Msx_Nmm, Moa_Nmm):
        """Slenderness reduction factor alpha_s (Eq 5.6.1.1(3))."""
        _check_positive(Msx_Nmm=Msx_Nmm, Moa_Nmm=Moa_Nmm)
        ratio = Msx_Nmm / Moa_Nmm
        return 0.6 * (np.sqrt(ratio**2 + 3.0) - ratio)

    @staticmethod
    def nominal_member_moment_capacity(Msx_Nmm, alpha_m=1.0, alpha_s=1.0):
        """Nominal major-axis member moment capacity Mbx (Eq 5.6.1.1(1))."""
        _check_positive(Msx_Nmm=Msx_Nmm, alpha_m=alpha_m, alpha_s=alpha_s)
        return min(alpha_m * alpha_s * Msx_Nmm, Msx_Nmm)

    @staticmethod
    def web_slenderness(d_p_mm, t_w_mm, fy_MPa):
        """Web slenderness (d_p/t_w)*sqrt(fy/250) used in section 5."""
        _check_positive(d_p_mm=d_p_mm, t_w_mm=t_w_mm, fy_MPa=fy_MPa)
        return (d_p_mm / t_w_mm) * np.sqrt(fy_MPa / 250.0)

    @staticmethod
    def unstiffened_web_ok(d1_mm, t_mm, fy_MPa, bounded_both_sides=True):
        """
        Unstiffened web slenderness check (Cl 5.10.1).

        Returns:
            bool: True when the web slenderness is within the clause limit.
        """
        limit = 180.0 if bounded_both_sides else 90.0
        return Section5.web_slenderness(d1_mm, t_mm, fy_MPa) <= limit

    @staticmethod
    def shear_yield_capacity(Aw_mm2, fy_MPa, section_type="flat_plate"):
        """
        Nominal shear yield capacity Vw (Cl 5.11.4).

        section_type: 'flat_plate' -> 0.6*fy*Aw; 'chs' -> 0.36*fy*Ae.
        """
        _check_positive(Aw_mm2=Aw_mm2, fy_MPa=fy_MPa)
        if section_type == "chs":
            return 0.36 * fy_MPa * Aw_mm2
        return 0.6 * fy_MPa * Aw_mm2

    @staticmethod
    def alpha_v_unstiffened(d_p_mm, t_w_mm, fy_MPa):
        """Shear buckling factor alpha_v for unstiffened webs (Cl 5.11.5.1)."""
        slenderness = Section5.web_slenderness(d_p_mm, t_w_mm, fy_MPa)
        return min((82.0 / slenderness) ** 2, 1.0)

    @staticmethod
    def shear_buckling_capacity_unstiffened(Vw_N, d_p_mm, t_w_mm, fy_MPa):
        """Nominal shear buckling capacity Vb for unstiffened webs (Cl 5.11.5.1)."""
        return min(Section5.alpha_v_unstiffened(d_p_mm, t_w_mm, fy_MPa) * Vw_N, Vw_N)

    @staticmethod
    def shear_capacity_uniform(
        Aw_mm2, fy_MPa, d_p_mm, t_w_mm, section_type="flat_plate"
    ):
        """
        Nominal web shear capacity Vvu for effectively uniform shear (Cl 5.11.2).
        """
        Vw = Section5.shear_yield_capacity(Aw_mm2, fy_MPa, section_type)
        stocky_limit = 82.0 / np.sqrt(fy_MPa / 250.0)
        if d_p_mm / t_w_mm <= stocky_limit:
            return Vw
        return Section5.shear_buckling_capacity_unstiffened(Vw, d_p_mm, t_w_mm, fy_MPa)

    @staticmethod
    def shear_capacity_non_uniform(Vvu_N, f_vm_star_MPa, f_va_star_MPa):
        """Nominal shear capacity Vvn for non-uniform shear (Cl 5.11.3)."""
        _check_positive(Vvu_N=Vvu_N, f_va_star_MPa=f_va_star_MPa)
        return min(2.0 * Vvu_N / (0.9 + f_vm_star_MPa / f_va_star_MPa), Vvu_N)

    @staticmethod
    def shear_bending_interaction_capacity(Vv_N, M_star_Nmm, phi, Ms_Nmm):
        """
        Nominal web shear capacity in the presence of bending Vvm (Cl 5.12.2).
        """
        _check_positive(Vv_N=Vv_N, phi=phi, Ms_Nmm=Ms_Nmm)
        if M_star_Nmm <= 0.75 * phi * Ms_Nmm:
            return Vv_N
        if M_star_Nmm <= phi * Ms_Nmm:
            return Vv_N * (2.2 - 1.6 * M_star_Nmm / (phi * Ms_Nmm))
        raise ValueError("M* exceeds phi*Ms; bending capacity is exceeded")


# ------------------------------------------------------------------
# Section 6 - Members subject to axial compression
# ------------------------------------------------------------------
class Section6(Section3):
    """Axial compression capacity helpers for NZS 3404 section 6."""

    TABLE_6_2_4_LAMBDA_EY = {
        1: {"SR": 16, "HR": 16, "LW": 15, "CF": 15, "HW": 14},
        2: {"SR": 45, "HR": 45, "LW": 40, "CF": 40, "HW": 35},
        3: {"SR": 82, "HR": 82, "CF": 82, "LW": 82, "HW": 82},
    }

    TABLE_6_3_3_1_ALPHA_B = {
        "ub_uc_hot_rolled_tf_le_40": 0.0,
        "box_welded": 0.0,
        "ub_uc_hot_rolled_tf_gt_40": 1.0,
        "rhs_chs_hot_formed_kf_1": -1.0,
        "rhs_chs_hot_formed_kf_lt_1": -0.5,
        "rhs_chs_cold_formed_stress_relieved_kf_1": -1.0,
        "rhs_chs_cold_formed_stress_relieved_kf_lt_1": -0.5,
        "rhs_chs_cold_formed_non_stress_relieved": -0.5,
        "welded_i_flame_cut_tf_le_40_kf_1": 0.0,
        "welded_i_flame_cut_tf_le_40_kf_lt_1": 0.5,
        "welded_i_flame_cut_tf_gt_40_kf_1": 0.0,
        "welded_i_flame_cut_tf_gt_40_kf_lt_1": 1.0,
        "welded_i_as_rolled_tf_le_40": 0.5,
        "welded_i_as_rolled_tf_gt_40": 1.0,
        "tee_angle_channel_hot_rolled_kf_1": 0.5,
        "tee_angle_channel_hot_rolled_kf_lt_1": 1.0,
    }

    def get_lambda_ey_6_2_4(self, case_number, residual_stress="HR"):
        """Plate element yield slenderness limit lambda_ey (Table 6.2.4)."""
        if case_number not in self.TABLE_6_2_4_LAMBDA_EY:
            raise ValueError("case_number must be 1, 2 or 3")
        key = _residual_key(residual_stress)
        if key not in self.TABLE_6_2_4_LAMBDA_EY[case_number]:
            raise ValueError(f"Residual stress '{residual_stress}' not valid")
        return self.TABLE_6_2_4_LAMBDA_EY[case_number][key]

    @staticmethod
    def plate_slenderness(b_mm, t_mm, fy_MPa):
        """Flat plate slenderness lambda_e (Cl 6.2.3.1)."""
        return Section5.plate_slenderness(b_mm, t_mm, fy_MPa)

    @staticmethod
    def chs_slenderness(d_o_mm, t_mm, fy_MPa):
        """CHS element slenderness lambda_e (Cl 6.2.3.2)."""
        return Section5.chs_slenderness(d_o_mm, t_mm, fy_MPa)

    @staticmethod
    def effective_width(b_mm, lambda_e, lambda_ey, k_b=None, supported_edges="both"):
        """
        Effective width be for a flat plate element (Cl 6.2.4.2 / 6.2.4.3).
        """
        _check_positive(b_mm=b_mm, lambda_e=lambda_e, lambda_ey=lambda_ey)
        factor = lambda_ey / lambda_e
        if k_b is not None:
            k_bo = 4.0 if supported_edges == "both" else 0.425
            factor *= np.sqrt(k_b / k_bo)
        return min(b_mm * factor, b_mm)

    @staticmethod
    def effective_diameter_chs(d_o_mm, lambda_e, lambda_ey):
        """Effective outside diameter de for CHS (Cl 6.2.4.4)."""
        _check_positive(d_o_mm=d_o_mm, lambda_e=lambda_e, lambda_ey=lambda_ey)
        return min(
            d_o_mm * np.sqrt(lambda_ey / lambda_e),
            d_o_mm * (3.0 * lambda_ey / lambda_e) ** 2,
            d_o_mm,
        )

    @staticmethod
    def form_factor(Ae_mm2, Ag_mm2):
        """Form factor kf = Ae/Ag (Cl 6.2.2)."""
        _check_positive(Ae_mm2=Ae_mm2, Ag_mm2=Ag_mm2)
        return min(Ae_mm2 / Ag_mm2, 1.0)

    @staticmethod
    def nominal_section_capacity(An_mm2, fy_MPa, k_f=1.0):
        """Nominal compression section capacity Ns = kf*An*fy (Cl 6.2.1)."""
        _check_positive(An_mm2=An_mm2, fy_MPa=fy_MPa, k_f=k_f)
        return k_f * An_mm2 * fy_MPa

    @staticmethod
    def modified_member_slenderness(Le_mm, r_mm, k_f, fy_MPa):
        """Modified member slenderness lambda_n (Cl 6.3.3)."""
        _check_positive(Le_mm=Le_mm, r_mm=r_mm, k_f=k_f, fy_MPa=fy_MPa)
        return (Le_mm / r_mm) * np.sqrt(k_f) * np.sqrt(fy_MPa / 250.0)

    def get_alpha_b(self, section_type):
        """Compression member section constant alpha_b (Table 6.3.3(1))."""
        key = str(section_type).lower().strip()
        if key not in self.TABLE_6_3_3_1_ALPHA_B:
            raise ValueError(
                f"section_type '{section_type}' not recognised. "
                f"Valid: {list(self.TABLE_6_3_3_1_ALPHA_B.keys())}"
            )
        return self.TABLE_6_3_3_1_ALPHA_B[key]

    @staticmethod
    def alpha_c(lambda_n, alpha_b):
        """
        Member slenderness reduction factor alpha_c (Cl 6.3.3).
        """
        if lambda_n <= 0:
            return 1.0
        alpha_a = 2100.0 * (lambda_n - 13.5) / (lambda_n**2 - 15.3 * lambda_n + 2050.0)
        lam = lambda_n + alpha_a * alpha_b
        if lam <= 0:
            return 1.0
        eta = max(0.00326 * (lam - 13.5), 0.0)
        xi_m = ((lam / 90.0) ** 2 + 1.0 + eta) / (2.0 * (lam / 90.0) ** 2)
        root_term = max(0.0, 1.0 - (90.0 / (xi_m * lam)) ** 2)
        return min(xi_m * (1.0 - np.sqrt(root_term)), 1.0)

    @staticmethod
    def nominal_member_capacity(Ns_N, alpha_c):
        """Nominal compression member capacity Nc = alpha_c*Ns (Cl 6.3.3)."""
        _check_positive(Ns_N=Ns_N, alpha_c=alpha_c)
        return alpha_c * Ns_N

    @staticmethod
    def laced_battened_transverse_shear(Ns_N, Nc_N, N_star_N, lambda_n):
        """
        Design transverse shear force for laced/battened compression members
        (Cl 6.4.1). The clause expression is subject to a minimum of 0.01*N*.
        """
        _check_positive(Ns_N=Ns_N, Nc_N=Nc_N, N_star_N=N_star_N, lambda_n=lambda_n)
        value = np.pi * (Ns_N / Nc_N - 1.0) * N_star_N / lambda_n
        return max(value, 0.01 * N_star_N)


# ------------------------------------------------------------------
# Section 7 - Members subject to axial tension
# ------------------------------------------------------------------
class Section7(Section3):
    """Axial tension capacity helpers for NZS 3404 section 7."""

    TABLE_7_3_2_KTE = {
        1: 0.85,
        "1_unequal_angle_short_leg": 0.75,
        2: 0.85,
        3: 0.90,
        4: 1.00,
        5: 1.00,
        6: 1.00,
        7: 0.75,
        "both_flanges_only": 0.85,
        "uniform": 1.00,
    }

    def get_kte(self, case="uniform"):
        """
        Correction factor for distribution of tensile forces k_te (Cl 7.3).

        Args:
            case: 'uniform', 1..7, '1_unequal_angle_short_leg',
                or 'both_flanges_only'.
        """
        key = case
        if isinstance(case, str):
            key = case.lower().strip()
        if key not in self.TABLE_7_3_2_KTE:
            raise ValueError(f"k_te case '{case}' not recognised")
        return self.TABLE_7_3_2_KTE[key]

    @staticmethod
    def nominal_tension_capacity(Ag_mm2, An_mm2, fy_MPa, fu_MPa, k_te=1.0):
        """
        Nominal section tension capacity Nt (Cl 7.2).

        Nt = min(Ag*fy, 0.85*k_te*An*fu) [N].
        """
        _check_positive(
            Ag_mm2=Ag_mm2, An_mm2=An_mm2, fy_MPa=fy_MPa, fu_MPa=fu_MPa, k_te=k_te
        )
        return min(Ag_mm2 * fy_MPa, 0.85 * k_te * An_mm2 * fu_MPa)

    def design_tension_capacity(self, Ag_mm2, An_mm2, fy_MPa, fu_MPa, k_te=1.0):
        """Design axial tension capacity phi*Nt (Cl 7.1, 7.2)."""
        Nt = self.nominal_tension_capacity(Ag_mm2, An_mm2, fy_MPa, fu_MPa, k_te)
        return self.get_phi("axial_tension") * Nt


# ------------------------------------------------------------------
# Section 8 - Members subject to combined actions
# ------------------------------------------------------------------
class Section8(Section3):
    """Combined bending and axial action helpers for NZS 3404 section 8."""

    TABLE_8_1_LIMITS = {
        1: {"SR": 11, "HR": 10, "LW": 9, "CF": 9, "HW": 9},
        2: {"SR": 11, "HR": 10, "LW": 9, "CF": 9, "HW": 9},
        3: {"ANY": 40},
        4: {"ANY": 82},
    }

    def significant_axial_force(self, N_star_N, reference_capacity_N, phi=None):
        """
        Significant axial force test using 0.05*phi*capacity (Cl 8.1.4).

        Returns:
            bool: True when axial force is significant and section 8 applies.
        """
        if phi is None:
            phi = self.get_phi("combined_actions")
        _check_positive(reference_capacity_N=reference_capacity_N, phi=phi)
        return N_star_N > 0.05 * phi * reference_capacity_N

    @staticmethod
    def axial_ratio(N_star_N, phi, Ns_N):
        """Convenience ratio N*/(phi*Ns) used throughout section 8."""
        _check_positive(phi=phi, Ns_N=Ns_N)
        return N_star_N / (phi * Ns_N)

    @staticmethod
    def reduced_moment_major(Msx_Nmm, N_star_N, phi, Ns_N, alternative=False):
        """
        Nominal reduced major-axis section moment capacity Mrx (Cl 8.3.2).
        """
        ratio = Section8.axial_ratio(N_star_N, phi, Ns_N)
        base = Msx_Nmm * (1.0 - ratio)
        if alternative:
            base = 1.18 * base
        return min(base, Msx_Nmm)

    @staticmethod
    def reduced_moment_minor(
        Msy_Nmm, N_star_N, phi, Ns_N, alternative=False, section_type="i"
    ):
        """
        Nominal reduced minor-axis section moment capacity Mry (Cl 8.3.3).
        """
        ratio = Section8.axial_ratio(N_star_N, phi, Ns_N)
        if not alternative:
            return Msy_Nmm * (1.0 - ratio)
        if section_type in ("i", "i_section", "i-section"):
            return min(1.19 * Msy_Nmm * (1.0 - ratio**2), Msy_Nmm)
        if section_type in ("hollow", "rhs", "shs", "chs"):
            return min(1.18 * Msy_Nmm * (1.0 - ratio), Msy_Nmm)
        raise ValueError("section_type must be 'i' or a hollow section key")

    @staticmethod
    def biaxial_section_utilisation(
        Mx_star_Nmm, My_star_Nmm, N_star_N, phi, Ns_N, Msx_Nmm, Msy_Nmm
    ):
        """
        Biaxial section utilisation, general provision (Cl 8.3.4.1).
        """
        return (
            N_star_N / (phi * Ns_N)
            + Mx_star_Nmm / (phi * Msx_Nmm)
            + My_star_Nmm / (phi * Msy_Nmm)
        )

    @staticmethod
    def biaxial_section_utilisation_alt(
        Mx_star_Nmm, My_star_Nmm, N_star_N, phi, Ns_N, Mrx_Nmm, Mry_Nmm
    ):
        """
        Biaxial section utilisation, alternative provision (Cl 8.3.4.2).
        """
        gamma = min(1.4 + Section8.axial_ratio(N_star_N, phi, Ns_N), 2.0)
        return (Mx_star_Nmm / (phi * Mrx_Nmm)) ** gamma + (
            My_star_Nmm / (phi * Mry_Nmm)
        ) ** gamma

    @staticmethod
    def in_plane_moment_capacity(Ms_Nmm, N_star_N, phi, Nc_N):
        """
        Nominal in-plane member moment capacity Mi for compression members
        (Cl 8.4.2.2.1).
        """
        return Ms_Nmm * (1.0 - N_star_N / (phi * Nc_N))

    @staticmethod
    def in_plane_moment_capacity_alt(Ms_Nmm, N_star_N, phi, Nc_N, beta_m, Mr_Nmm):
        """
        Alternative nominal in-plane member moment capacity Mi (Cl 8.4.2.2.2).
        """
        ratio = N_star_N / (phi * Nc_N)
        c = ((1.0 + beta_m) / 2.0) ** 3
        value = Ms_Nmm * ((1.0 - c) * (1.0 - ratio) + 1.18 * c * np.sqrt(1.0 - ratio))
        return min(value, Mr_Nmm)

    @staticmethod
    def plastic_hinge_member_slenderness_limit(beta_m, lambda_EYC):
        """Allowable N*/(phi*Ns) for plastic hinge member slenderness (Cl 8.4.3.2)."""
        base = 0.263 * (beta_m + 1.0) ** 0.88 / np.exp(0.19 / (beta_m + 1.0))
        return base**lambda_EYC

    @staticmethod
    def lambda_EYC(Ns_N, E_MPa, I_mm4, L_mm):
        """lambda_EYC = sqrt(Ns/Nol), Nol = pi^2*E*I/L^2 (Cl 8.4.3.2)."""
        _check_positive(Ns_N=Ns_N, E_MPa=E_MPa, I_mm4=I_mm4, L_mm=L_mm)
        N_ol = np.pi**2 * E_MPa * I_mm4 / L_mm**2
        return np.sqrt(Ns_N / N_ol)

    @staticmethod
    def plastic_hinge_web_slenderness_limit(d1_mm, t_mm, fy_MPa):
        """
        Allowable N*/(phi*Ns) from web slenderness (Cl 8.4.3.3).
        """
        slender = Section5.web_slenderness(d1_mm, t_mm, fy_MPa)
        if slender <= 30.0:
            return 1.0
        if slender < 45.0:
            return min(1.91 - slender / 27.4, 1.0)
        if slender <= 82.0:
            return 0.60 - slender / 137.0
        raise ValueError("Web slenderness exceeds 82; plastic hinges are not permitted")

    @staticmethod
    def design_plastic_moment_major(phi_Mspx_Nmm, N_star_N, phi_Ns_N):
        """Design plastic moment capacity phi*Mprx (Cl 8.4.3.4)."""
        return min(1.18 * phi_Mspx_Nmm * (1.0 - N_star_N / phi_Ns_N), phi_Mspx_Nmm)

    @staticmethod
    def design_plastic_moment_minor(phi_Mspy_Nmm, N_star_N, phi_Ns_N):
        """Design plastic moment capacity phi*Mpry (Cl 8.4.3.4)."""
        ratio = N_star_N / phi_Ns_N
        return min(1.19 * phi_Mspy_Nmm * (1.0 - ratio**2), phi_Mspy_Nmm)

    @staticmethod
    def out_of_plane_moment_capacity_compression(Mbx_Nmm, N_star_N, phi, Ncy_N):
        """Nominal out-of-plane moment capacity Mox, compression (Cl 8.4.4.1.1)."""
        return Mbx_Nmm * (1.0 - N_star_N / (phi * Ncy_N))

    @staticmethod
    def torsional_buckling_capacity(
        E_MPa, G_MPa, Iw_mm6, J_mm4, Lz_mm, Ix_mm4, Iy_mm4, A_mm2
    ):
        """Elastic torsional buckling capacity Noz (Cl 8.4.4.1.2)."""
        _check_positive(
            E_MPa=E_MPa,
            G_MPa=G_MPa,
            Lz_mm=Lz_mm,
            Ix_mm4=Ix_mm4,
            Iy_mm4=Iy_mm4,
            A_mm2=A_mm2,
        )
        numerator = G_MPa * J_mm4 + np.pi**2 * E_MPa * Iw_mm6 / Lz_mm**2
        return numerator / ((Ix_mm4 + Iy_mm4) / A_mm2)

    @staticmethod
    def out_of_plane_moment_capacity_tension(Mbx_Nmm, N_star_N, phi, Nt_N, Mrx_Nmm):
        """Nominal out-of-plane moment capacity Mox, tension (Cl 8.4.4.2)."""
        return min(Mbx_Nmm * (1.0 + N_star_N / (phi * Nt_N)), Mrx_Nmm)

    @staticmethod
    def biaxial_member_utilisation_compression(
        Mx_star_Nmm, My_star_Nmm, phi, Mcx_Nmm, Miy_Nmm
    ):
        """Biaxial member utilisation for compression or zero axial force (Cl 8.4.5.1)."""
        return (Mx_star_Nmm / (phi * Mcx_Nmm)) ** 1.4 + (
            My_star_Nmm / (phi * Miy_Nmm)
        ) ** 1.4

    @staticmethod
    def biaxial_member_utilisation_tension(
        Mx_star_Nmm, My_star_Nmm, phi, Mtx_Nmm, Mry_Nmm
    ):
        """Biaxial member utilisation for tension (Cl 8.4.5.2)."""
        return (Mx_star_Nmm / (phi * Mtx_Nmm)) ** 1.4 + (
            My_star_Nmm / (phi * Mry_Nmm)
        ) ** 1.4

    @staticmethod
    def single_angle_eccentric_compression_utilisation(
        N_star_N, Mh_star_Nmm, phi, Nch_N, Mbx_Nmm, alpha_rad
    ):
        """Single angle eccentric compression utilisation (Eq 8.4.6)."""
        return N_star_N / (phi * Nch_N) + Mh_star_Nmm / (
            phi * Mbx_Nmm * np.cos(alpha_rad)
        )


# ------------------------------------------------------------------
# Section 12 - Seismic design
# ------------------------------------------------------------------
class Section12(Section3):
    """Seismic design tables and helpers for NZS 3404 section 12."""

    TABLE_12_2_4_MU = {
        1: {"description": "fully ductile", "mu": ">3.0"},
        2: {"description": "limited ductile", "mu": "1.25<mu<=3.0"},
        3: {"description": "nominally ductile", "mu": 1.25},
        4: {"description": "elastic", "mu": 1.0},
    }

    TABLE_12_2_6_MEMBER_CATEGORY = {
        1: {
            "structure_category": 1,
            "capacity_design": True,
            "primary": 1,
            "secondary": 2,
        },
        2: {
            "structure_category": 2,
            "capacity_design": True,
            "primary": 2,
            "secondary": 2,
        },
        3: {
            "structure_category": 3,
            "capacity_design": True,
            "primary": 3,
            "secondary": 3,
        },
        4: {
            "structure_category": 3,
            "capacity_design": False,
            "columns": 2,
            "other": 3,
        },
        5: {
            "structure_category": 4,
            "capacity_design": False,
            "columns": 3,
            "other": 4,
        },
    }

    TABLE_12_2_8_1_OVERSTRENGTH = {
        (1, "250"): {"phi_os": 1.15, "phi_om": 1.30, "phi_oms": 1.35},
        (1, "300"): {"phi_os": 1.15, "phi_om": 1.30, "phi_oms": 1.35},
        (1, "300plus"): {"phi_os": 1.15, "phi_om": 1.20, "phi_oms": 1.25},
        (2, "250"): {"phi_os": 1.05, "phi_om": 1.30, "phi_oms": 1.25},
        (2, "300"): {"phi_os": 1.05, "phi_om": 1.30, "phi_oms": 1.25},
        (2, "300plus"): {"phi_os": 1.05, "phi_om": 1.20, "phi_oms": 1.15},
        (2, "350"): {"phi_os": 1.10, "phi_om": 1.30, "phi_oms": 1.30},
        (2, "350_aust"): {"phi_os": 1.10, "phi_om": 1.25, "phi_oms": 1.25},
        (3, "250"): {"phi_os": 1.00, "phi_om": 1.30, "phi_oms": 1.15},
        (3, "300"): {"phi_os": 1.00, "phi_om": 1.30, "phi_oms": 1.15},
        (3, "300plus"): {"phi_os": 1.00, "phi_om": 1.20, "phi_oms": 1.10},
        (3, "350"): {"phi_os": 1.00, "phi_om": 1.30, "phi_oms": 1.15},
        (3, "350_aust"): {"phi_os": 1.00, "phi_om": 1.25, "phi_oms": 1.15},
        (3, "450"): {"phi_os": 1.10, "phi_om": 1.30, "phi_oms": 1.30},
    }

    TABLE_12_2_9_DAMPING_PERCENT = {
        ("clad", "elastic", "welded"): (4, 6),
        ("clad", "elastic", "bolted"): (5, 10),
        ("clad", "inelastic", "welded"): (5, 7),
        ("clad", "inelastic", "bolted"): (10, 15),
        ("unclad", "elastic", "welded"): (2, 3),
        ("unclad", "elastic", "bolted"): (5, 7),
        ("unclad", "inelastic", "welded"): (5, 7),
        ("unclad", "inelastic", "bolted"): (10, 15),
    }

    TABLE_12_4_MATERIAL_REQUIREMENTS = {
        "category_1_2_3": {
            "max_specified_grade_reference_yield_MPa": 360,
            "min_total_actual_elongation_percent": 25,
            "max_actual_yield_ratio": 0.80,
            "max_actual_yield_stress_factor": 1.33,
            "min_charpy_average_J_at_0C": 70,
            "min_charpy_individual_J_at_0C": 50,
        },
        "category_4": {
            "max_specified_grade_reference_yield_MPa": 450,
            "min_total_actual_elongation_percent": 15,
            "max_actual_yield_ratio": 0.90,
            "max_actual_yield_stress_factor": None,
            "min_charpy_average_J_at_0C": None,
            "min_charpy_individual_J_at_0C": None,
        },
    }

    TABLE_12_5_LIMITS = {
        1: {
            "SR": (10, 10, 11, 25),
            "HR": (9, 9, 10, 25),
            "LW": (8, 8, 9, 22),
            "CF": (8, 8, 9, 22),
            "HW": (8, 8, 9, 22),
        },
        2: {
            "SR": (10, 10, 11, 25),
            "HR": (9, 9, 10, 25),
            "LW": (8, 8, 9, 22),
            "CF": (8, 8, 9, 22),
            "HW": (8, 8, 9, 22),
        },
        3: {
            "SR": (25, 30, 40, 60),
            "HR": (25, 30, 40, 60),
            "LW": (25, 30, 40, 60),
            "CF": (25, 30, 40, 60),
            "HW": (25, 30, 35, 60),
        },
        4: {"NON_UNIFORM": (82, 82, 101, 161), "RHS_WEB": (30, 40, 55, 75)},
        5: {
            "SR": (35, 50, 65, 170),
            "HR": (35, 50, 65, 170),
            "CF": (35, 50, 65, 170),
            "LW": (30, 42, 60, 170),
            "HW": (30, 42, 60, 170),
        },
    }

    @staticmethod
    def structural_performance_factor(
        structure_category, limit_state="ULS", category_2_detailing=False
    ):
        """
        Structural performance factor Sp (Cl 12.2.2.1).
        """
        if str(limit_state).upper() == "SLS":
            return 0.7
        cat = int(structure_category)
        if cat in (1, 2):
            return 0.7
        if cat in (3, 4):
            return 0.7 if category_2_detailing else 0.9
        raise ValueError("structure_category must be 1, 2, 3 or 4")

    def get_member_category_from_table_12_2_6(self, case_number, member_type="primary"):
        """Minimum member ductility category from Table 12.2.6."""
        if case_number not in self.TABLE_12_2_6_MEMBER_CATEGORY:
            raise ValueError("case_number must be in the range 1..5")
        row = self.TABLE_12_2_6_MEMBER_CATEGORY[case_number]
        key = str(member_type).lower().strip()
        if key not in row:
            raise ValueError(
                f"member_type '{member_type}' not valid for case {case_number}"
            )
        return row[key]

    def get_overstrength_factors(self, member_category, steel_grade):
        """
        Overstrength factors for beams, braces and columns (Table 12.2.8(1)).

        Args:
            member_category (int): 1, 2 or 3.
            steel_grade (str): '250', '300', '300plus', '350', '350_aust', '450'.
        """
        key = (int(member_category), str(steel_grade).lower().strip())
        if key not in self.TABLE_12_2_8_1_OVERSTRENGTH:
            raise ValueError(f"Overstrength factors not available for {key}")
        return self.TABLE_12_2_8_1_OVERSTRENGTH[key].copy()

    def get_damping_range(
        self, cladding="clad", behaviour="elastic", connection="welded"
    ):
        """Initial viscous damping range in percent (Table 12.2.9)."""
        key = (
            str(cladding).lower().strip(),
            str(behaviour).lower().strip(),
            str(connection).lower().strip(),
        )
        if key not in self.TABLE_12_2_9_DAMPING_PERCENT:
            raise ValueError(f"Damping case '{key}' not recognised")
        return self.TABLE_12_2_9_DAMPING_PERCENT[key]

    @staticmethod
    def damping_modified_Ch_ratio(eta_percent):
        """
        Ratio Ch_eta/Ch_5 for damping > 5% where applicable (Cl 12.2.9.2).
        """
        _check_positive(eta_percent=eta_percent)
        return 0.5 + 1.5 / (0.4 * eta_percent + 1.0)

    def get_plate_limit_12_5(
        self, case_number, member_category, residual_stress="HR", subtype=None
    ):
        """
        Plate element slenderness limit for seismic design (Table 12.5).

        Args:
            case_number (int): Table 12.5 case 1 to 5.
            member_category (int): Member category 1 to 4.
            residual_stress (str): SR, HR, LW, CF, HW.
            subtype (str, optional): For case 4, 'non_uniform' or 'rhs_web'.
        """
        if member_category not in (1, 2, 3, 4):
            raise ValueError("member_category must be 1, 2, 3 or 4")
        if case_number not in self.TABLE_12_5_LIMITS:
            raise ValueError("case_number must be one of 1, 2, 3, 4, 5")
        if case_number == 4:
            key = str(subtype or "NON_UNIFORM").upper().strip()
        else:
            key = _residual_key(residual_stress)
        if key not in self.TABLE_12_5_LIMITS[case_number]:
            raise ValueError(
                f"Limit key '{key}' not valid for Table 12.5 case {case_number}"
            )
        return self.TABLE_12_5_LIMITS[case_number][key][member_category - 1]

    def material_compliance(
        self,
        member_category,
        specified_fy_MPa,
        actual_elongation_percent=None,
        actual_yield_ratio=None,
        actual_fy_MPa=None,
        specified_fy_for_ratio_MPa=None,
        charpy_average_J=None,
        charpy_individual_J=None,
        thickness_mm=None,
    ):
        """
        Material requirement compliance summary (Table 12.4).

        Returns:
            dict: checks keyed by requirement; None means not assessed.
        """
        key = "category_4" if int(member_category) == 4 else "category_1_2_3"
        req = self.TABLE_12_4_MATERIAL_REQUIREMENTS[key]
        checks = {
            "specified_grade_reference_yield": (
                specified_fy_MPa <= req["max_specified_grade_reference_yield_MPa"]
            ),
            "total_actual_elongation": None,
            "actual_yield_ratio": None,
            "actual_yield_stress": None,
            "charpy_average": None,
            "charpy_individual": None,
        }
        if actual_elongation_percent is not None:
            checks["total_actual_elongation"] = (
                actual_elongation_percent >= req["min_total_actual_elongation_percent"]
            )
        if actual_yield_ratio is not None:
            checks["actual_yield_ratio"] = (
                actual_yield_ratio <= req["max_actual_yield_ratio"]
            )
        if (
            actual_fy_MPa is not None
            and req["max_actual_yield_stress_factor"] is not None
        ):
            fy_ref = specified_fy_for_ratio_MPa or specified_fy_MPa
            checks["actual_yield_stress"] = (
                actual_fy_MPa <= req["max_actual_yield_stress_factor"] * fy_ref
            )
        charpy_required = key == "category_1_2_3" and (
            thickness_mm is None or thickness_mm > 12
        )
        if charpy_required and charpy_average_J is not None:
            checks["charpy_average"] = (
                charpy_average_J >= req["min_charpy_average_J_at_0C"]
            )
        if charpy_required and charpy_individual_J is not None:
            checks["charpy_individual"] = (
                charpy_individual_J >= req["min_charpy_individual_J_at_0C"]
            )
        return checks


# ------------------------------------------------------------------
# Combined convenience class
# ------------------------------------------------------------------
class NZS_3404_1997(Section4, Section5, Section6, Section7, Section8, Section12):
    """
    Convenience class exposing helpers for NZS 3404 sections 3, 4, 5, 6, 7,
    8 and 12 from a single object.
    """

    def __init__(self, E_MPa=200000.0, G_MPa=80000.0):
        self.E_MPa = E_MPa
        self.G_MPa = G_MPa

    def tables_as_dataframes(self):
        """
        Return selected standard tables as pandas DataFrames for notebooks.

        Returns:
            dict[str, pandas.DataFrame]: common lookup tables.
        """
        import pandas as pd

        return {
            "table_3_3_1_phi": pd.DataFrame.from_dict(
                self.TABLE_3_3_1_PHI, orient="index", columns=["phi"]
            ),
            "table_12_2_8_1_overstrength": pd.DataFrame.from_dict(
                self.TABLE_12_2_8_1_OVERSTRENGTH, orient="index"
            ),
            "table_12_2_9_damping": pd.DataFrame.from_dict(
                self.TABLE_12_2_9_DAMPING_PERCENT,
                orient="index",
                columns=["lower_percent", "upper_percent"],
            ),
        }
