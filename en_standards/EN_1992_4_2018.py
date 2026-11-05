"""Mechanical and bonded anchors to EN 1992-4:2018, including Annex C.

Static tension is clause 7.2.1 and static shear is clause 7.2.2. Combined
tension and shear is Table 7.3. Seismic resistance is Annex C, Formula (C.8),
using the product's C1 or C2 characteristic values. Partial factors under
seismic loading are the static values, clause 4.4.2.2.

Nothing in this module is filled in when a required product value or a
required design choice is missing.
"""

import math
from dataclasses import dataclass

from en_standards.ETAG001C_R045_2013 import (
    NO_CONCRETE_EDGE_MM,
    projected_area_shear,
    projected_area_tension,
    psi_h_sp,
    psi_re_N,
)

# EN 1992-4 Table 4.1. gamma_inst multiplies tension concrete and pull-out.
# Concrete shear uses gamma_inst = 1.0, so it stays at gamma_c.
_GAMMA_C = 1.5

# Clause 7.2.2.5 (6). These are code factors, not ETA values.
_K9_CRACKED = 1.7
_K9_UNCRACKED = 2.4

# Clause 7.2.1.6 (4), Formula (7.19).
_K3_CRACKED = 7.7
_K3_UNCRACKED = 11.0

# Clause 7.2.1.6 (3), Formula (7.15).
_S_CR_NP_FACTOR = 7.3
_S_CR_NP_HEF_CAP = 3.0

# Clause 7.2.1.7 (2) b) 1). Groups need the larger edge distance.
_SPLITTING_EDGE_FACTOR_SINGLE = 1.0
_SPLITTING_EDGE_FACTOR_GROUP = 1.2

# Table C.3. Concrete screws are "all other fasteners": they are not headed
# and they are not undercut anchors whose k1 matches a headed fastener.
_ALPHA_EQ = {
    ("tension", "steel", "headed_k1", "single"): 1.0,
    ("tension", "steel", "headed_k1", "group"): 1.0,
    ("tension", "steel", "other", "single"): 1.0,
    ("tension", "steel", "other", "group"): 1.0,
    ("tension", "cone", "headed_k1", "single"): 1.0,
    ("tension", "cone", "headed_k1", "group"): 0.85,
    ("tension", "cone", "other", "single"): 0.85,
    ("tension", "cone", "other", "group"): 0.75,
    ("tension", "pullout", "headed_k1", "single"): 1.0,
    ("tension", "pullout", "headed_k1", "group"): 0.85,
    ("tension", "pullout", "other", "single"): 1.0,
    ("tension", "pullout", "other", "group"): 0.85,
    ("tension", "splitting", "headed_k1", "single"): 1.0,
    ("tension", "splitting", "headed_k1", "group"): 0.85,
    ("tension", "splitting", "other", "single"): 1.0,
    ("tension", "splitting", "other", "group"): 0.85,
    ("shear", "steel", "headed_k1", "single"): 1.0,
    ("shear", "steel", "headed_k1", "group"): 0.85,
    ("shear", "steel", "other", "single"): 1.0,
    ("shear", "steel", "other", "group"): 0.85,
    ("shear", "pryout", "headed_k1", "single"): 1.0,
    ("shear", "pryout", "headed_k1", "group"): 0.85,
    ("shear", "pryout", "other", "single"): 0.85,
    ("shear", "pryout", "other", "group"): 0.75,
    ("shear", "edge", "headed_k1", "single"): 1.0,
    ("shear", "edge", "headed_k1", "group"): 0.85,
    ("shear", "edge", "other", "single"): 1.0,
    ("shear", "edge", "other", "group"): 0.85,
}

_FASTENER_CLASSES = ("headed_k1", "other")
_DESIGN_OPTIONS = ("capacity", "elastic", "ductile_fastener")
_SEISMIC_CATEGORIES = ("C1", "C2")


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return number


def _non_negative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0, got {value!r}")
    return number


def _flag(value, label):
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be true or false, got {value!r}")
    return value


def cylinder_strength_from_grade(concrete_grade):
    """Cylinder strength, in MPa, from a grade such as C20/25."""
    text = str(concrete_grade).strip()
    if len(text) < 4 or text[0] not in ("C", "c") or "/" not in text:
        raise ValueError(
            "concrete grade must look like C20/25, "
            f"got {concrete_grade!r}"
        )
    cylinder = text[1:].split("/", 1)[0].strip()
    if not cylinder:
        raise ValueError(
            f"concrete grade {concrete_grade!r} has no cylinder strength"
        )
    return _positive(float(cylinder), f"cylinder strength of {concrete_grade}")


def k9_7_2_2_5(is_cracked):
    """Clause 7.2.2.5 (6)."""
    if not isinstance(is_cracked, bool):
        raise TypeError(f"is_cracked must be true or false, got {is_cracked!r}")
    return _K9_CRACKED if is_cracked else _K9_UNCRACKED


def psi_alpha_V_7_2_2_5(alpha_V_deg):
    """Formula (7.48). alpha is 0 when the shear is perpendicular to the edge."""
    alpha = _non_negative(alpha_V_deg, "alpha_V_deg")
    if alpha > 90.0:
        raise ValueError(f"alpha_V_deg must be from 0 to 90, got {alpha_V_deg!r}")
    radians = math.radians(alpha)
    denominator = math.cos(radians) ** 2 + 0.5 * math.sin(radians) ** 2
    if denominator <= 0.0:
        raise ValueError("Formula (7.48) has a zero denominator")
    return max(1.0 / denominator, 1.0)


def psi_re_V_7_2_2_5(
    *,
    is_cracked,
    edge_reinforcement,
    edge_reinforcement_spacing_mm,
    edge_reinforcement_cover_mm,
    h_ef_mm,
    c1_mm,
):
    """Clause 7.2.2.5 (13).

    The factor is 1.0 in uncracked concrete and in cracked concrete without
    edge reinforcement. It is 1.4 only when the cracked member has edge
    reinforcement that meets the spacing and cover conditions. A present
    reinforcement that misses those conditions is an error, not 1.0.
    """
    cracked = _flag(is_cracked, "is_cracked")
    reinforced = _flag(edge_reinforcement, "edge_reinforcement")
    if not cracked or not reinforced:
        return 1.0
    if edge_reinforcement_spacing_mm is None or edge_reinforcement_cover_mm is None:
        raise ValueError(
            "cracked concrete with edge reinforcement needs "
            "edge_reinforcement_spacing_mm and edge_reinforcement_cover_mm "
            "for clause 7.2.2.5 (13)"
        )
    spacing = _positive(edge_reinforcement_spacing_mm, "edge_reinforcement_spacing_mm")
    cover = _positive(edge_reinforcement_cover_mm, "edge_reinforcement_cover_mm")
    h_ef = _positive(h_ef_mm, "h_ef_mm")
    c1 = _positive(c1_mm, "c1_mm")
    close = spacing <= 100.0 and spacing <= 2.0 * c1
    covered = h_ef >= 2.5 * cover
    if close and covered:
        return 1.4
    raise ValueError(
        "edge reinforcement does not meet clause 7.2.2.5 (13) for "
        "psi_re,V = 1.4 (spacing at most 100 mm and 2 c1, and h_ef at least "
        "2.5 times the reinforcement cover). The clause gives no other value."
    )


def splitting_verification_required_7_2_1_7(
    *,
    number_of_anchors,
    edges_mm,
    thickness_mm,
    h_min_mm,
    c_cr_sp_mm,
    reinforcement_limits_cracks,
):
    """Clause 7.2.1.7 (2) b).

    Verification is omitted when every edge is at least 1.0 c_cr,sp for one
    fastener, or 1.2 c_cr,sp for a group, and h is at least h_min. It is also
    omitted when the caller states that reinforcement limits the crack width
    to 0.3 mm and the cracked-concrete resistances are the ones being used.
    """
    n = int(number_of_anchors)
    if n < 1:
        raise ValueError(f"number_of_anchors must be >= 1, got {number_of_anchors!r}")
    if not isinstance(edges_mm, (list, tuple)) or len(edges_mm) < 1:
        raise ValueError("edges_mm must list every concrete edge")
    edges = [_positive(edge, "concrete edge") for edge in edges_mm]
    thickness = _positive(thickness_mm, "thickness_mm")
    h_min = _positive(h_min_mm, "h_min_mm")
    c_cr_sp = _positive(c_cr_sp_mm, "c_cr_sp_mm")
    if _flag(reinforcement_limits_cracks, "reinforcement_limits_cracks"):
        return False
    factor = _SPLITTING_EDGE_FACTOR_SINGLE if n == 1 else _SPLITTING_EDGE_FACTOR_GROUP
    exempt = thickness >= h_min and all(edge >= factor * c_cr_sp for edge in edges)
    return not exempt


def alpha_eq_table_c_3(loading, failure_mode, fastener_class, group):
    """Table C.3. ``group`` is false for one fastener and for the one tensioned
    fastener of a group, note (a).
    """
    if fastener_class not in _FASTENER_CLASSES:
        raise ValueError(
            "fastener_class must be 'headed_k1' or 'other', "
            f"got {fastener_class!r}"
        )
    if not isinstance(group, bool):
        raise TypeError(f"group must be true or false, got {group!r}")
    key = (loading, failure_mode, fastener_class, "group" if group else "single")
    if key not in _ALPHA_EQ:
        raise ValueError(f"Table C.3 has no alpha_eq for {key}")
    return _ALPHA_EQ[key]


def interaction_table_7_3(beta_n_steel, beta_v_steel, beta_n_concrete, beta_v_concrete):
    """Table 7.3, Formula (7.54) for steel and Formula (7.55) for other modes."""
    for label, value in (
        ("beta_n_steel", beta_n_steel),
        ("beta_v_steel", beta_v_steel),
        ("beta_n_concrete", beta_n_concrete),
        ("beta_v_concrete", beta_v_concrete),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number, got {value!r}")
        if value < 0.0:
            raise ValueError(f"{label} must be >= 0, got {value!r}")
    steel = beta_n_steel**2 + beta_v_steel**2
    concrete = beta_n_concrete**1.5 + beta_v_concrete**1.5
    return {
        "steel": steel,
        "concrete": concrete,
        "governing": max(steel, concrete),
        "steel_clause": "7.54",
        "concrete_clause": "7.55",
    }


def interaction_annex_c(beta_n_steel, beta_v_steel, beta_n_concrete, beta_v_concrete):
    """Formula (C.9). k15 = 1 for steel and for modes other than steel when
    there is no supplementary reinforcement.
    """
    for label, value in (
        ("beta_n_steel", beta_n_steel),
        ("beta_v_steel", beta_v_steel),
        ("beta_n_concrete", beta_n_concrete),
        ("beta_v_concrete", beta_v_concrete),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number, got {value!r}")
        if value < 0.0:
            raise ValueError(f"{label} must be >= 0, got {value!r}")
    steel = beta_n_steel + beta_v_steel
    concrete = beta_n_concrete + beta_v_concrete
    return {
        "steel": steel,
        "concrete": concrete,
        "governing": max(steel, concrete),
        "k15": 1.0,
        "clause": "C.9",
    }


@dataclass
class AnchorProperties_EN1992_4:
    """Inputs for one mechanical or bonded anchor group.

    Seismic fields are required only when ``seismic_design`` is true. Pass
    them as None for a static check; a static check that is given seismic
    product data still does not apply Annex C.
    """

    diameter_mm: float
    h_ef_mm: float
    l_f_mm: float
    concrete_thickness_mm: float
    fck_cylinder_MPa: float
    is_cracked: bool
    dense_reinforcement: bool
    number_of_anchors: int
    edge_distance_c1_mm: float
    edge_distance_c2_1_mm: float
    edge_distance_c2_2_mm: float
    N_Rk_s_kN: float
    V_Rk_s_kN: float
    s_cr_N_mm: float
    c_cr_N_mm: float
    k_cr_N: float
    k_ucr_N: float
    k7: float
    k8: float
    gamma_Ms_N: float
    gamma_Ms_V: float
    gamma_inst: float
    splitting_check_required: bool
    e_N_x_mm: float
    e_N_y_mm: float
    e_V_mm: float
    alpha_V_deg: float
    edge_reinforcement: bool
    seismic_design: bool
    spacing_mm: float = None
    spacing_x_mm: float = None
    spacing_y_mm: float = None
    c_cr_sp_mm: float = None
    s_cr_sp_mm: float = None
    h_min_mm: float = None
    edge_reinforcement_spacing_mm: float = None
    edge_reinforcement_cover_mm: float = None
    # Mechanical pull-out. None when the anchor is bonded.
    N_Rk_p_kN: float = None
    psi_c: float = None
    # Bonded 7.2.1.6. None when the anchor is mechanical.
    tau_Rk_cr: float = None
    tau_Rk_ucr: float = None
    psi_sus: float = None
    # Annex C. None unless seismic_design is true.
    seismic_performance_category: str = None
    annular_gap_filled: bool = None
    seismic_design_option: str = None
    fastener_class: str = None
    anchors_in_tension: int = None
    alpha_gap_with_gap_filling: float = None
    alpha_gap_without_gap_filling: float = None
    N_Rk_s_eq_kN: float = None
    N_Rk_p_eq_kN: float = None
    V_Rk_s_eq_kN: float = None
    delta_N_eq_DLS_mm: float = None
    delta_V_eq_DLS_mm: float = None
    delta_N_req_DLS_mm: float = None
    delta_V_req_DLS_mm: float = None
    # Clause C.3 (3), option b). Required only for ductile_fastener.
    f_uk_MPa: float = None
    f_yk_MPa: float = None
    rupture_elongation: float = None
    stretch_length_mm: float = None

    def __post_init__(self):
        self.number_of_anchors = int(self.number_of_anchors)
        if self.number_of_anchors not in (1, 2, 4):
            raise ValueError(
                "number_of_anchors must be 1, 2 or 4, "
                f"got {self.number_of_anchors!r}"
            )
        self.diameter_mm = _positive(self.diameter_mm, "diameter_mm")
        self.h_ef_mm = _positive(self.h_ef_mm, "h_ef_mm")
        self.l_f_mm = _positive(self.l_f_mm, "l_f_mm")
        self.concrete_thickness_mm = _positive(
            self.concrete_thickness_mm, "concrete_thickness_mm"
        )
        self.fck_cylinder_MPa = _positive(self.fck_cylinder_MPa, "fck_cylinder_MPa")
        self.is_cracked = _flag(self.is_cracked, "is_cracked")
        self.dense_reinforcement = _flag(self.dense_reinforcement, "dense_reinforcement")
        self.edge_distance_c1_mm = _positive(self.edge_distance_c1_mm, "edge_distance_c1_mm")
        self.s_cr_N_mm = _positive(self.s_cr_N_mm, "s_cr_N_mm")
        self.c_cr_N_mm = _positive(self.c_cr_N_mm, "c_cr_N_mm")
        self.k_cr_N = _positive(self.k_cr_N, "k_cr_N")
        if self.is_cracked:
            if self.k_ucr_N is not None:
                self.k_ucr_N = _positive(self.k_ucr_N, "k_ucr_N")
        elif self.k_ucr_N is None:
            raise ValueError("k_ucr_N is required for uncracked concrete")
        else:
            self.k_ucr_N = _positive(self.k_ucr_N, "k_ucr_N")
        self.k7 = _positive(self.k7, "k7")
        self.k8 = _positive(self.k8, "k8")
        self.gamma_Ms_N = _positive(self.gamma_Ms_N, "gamma_Ms_N")
        self.gamma_Ms_V = _positive(self.gamma_Ms_V, "gamma_Ms_V")
        self.gamma_inst = _positive(self.gamma_inst, "gamma_inst")
        self.splitting_check_required = _flag(
            self.splitting_check_required, "splitting_check_required"
        )
        for label in ("e_N_x_mm", "e_N_y_mm", "e_V_mm"):
            setattr(self, label, _non_negative(getattr(self, label), label))
        self.alpha_V_deg = _non_negative(self.alpha_V_deg, "alpha_V_deg")
        self.edge_reinforcement = _flag(self.edge_reinforcement, "edge_reinforcement")
        self.seismic_design = _flag(self.seismic_design, "seismic_design")
        if self.edge_distance_c1_mm < NO_CONCRETE_EDGE_MM and (
            self.edge_distance_c2_1_mm is None or self.edge_distance_c2_2_mm is None
        ):
            raise ValueError(
                "edge_distance_c2_1_mm and edge_distance_c2_2_mm are required "
                f"when c1={self.edge_distance_c1_mm} mm is a concrete edge."
            )
        mechanical = self.N_Rk_p_kN is not None or self.psi_c is not None
        bonded = (
            self.tau_Rk_cr is not None
            or self.tau_Rk_ucr is not None
            or self.psi_sus is not None
        )
        if mechanical and bonded:
            raise ValueError(
                "pass either mechanical pull-out (N_Rk_p_kN and psi_c) or "
                "bonded 7.2.1.6 (tau_Rk_cr, tau_Rk_ucr and psi_sus), not both"
            )
        if not mechanical and not bonded:
            raise ValueError(
                "mechanical pull-out needs N_Rk_p_kN and psi_c; "
                "bonded 7.2.1.6 needs tau_Rk_cr, tau_Rk_ucr and psi_sus"
            )
        if mechanical:
            if self.N_Rk_p_kN is None or self.psi_c is None:
                raise ValueError("mechanical pull-out needs both N_Rk_p_kN and psi_c")
            self.N_Rk_p_kN = _positive(self.N_Rk_p_kN, "N_Rk_p_kN")
            self.psi_c = _positive(self.psi_c, "psi_c")
            self.fastener_kind = "mechanical"
        else:
            if self.tau_Rk_cr is None or self.tau_Rk_ucr is None or self.psi_sus is None:
                raise ValueError(
                    "bonded 7.2.1.6 needs tau_Rk_cr, tau_Rk_ucr and psi_sus"
                )
            self.tau_Rk_cr = _positive(self.tau_Rk_cr, "tau_Rk_cr")
            self.tau_Rk_ucr = _positive(self.tau_Rk_ucr, "tau_Rk_ucr")
            self.psi_sus = _positive(self.psi_sus, "psi_sus")
            self.fastener_kind = "bonded"
        self.N_Rk_s_kN = _positive(self.N_Rk_s_kN, "N_Rk_s_kN")
        self.V_Rk_s_kN = _positive(self.V_Rk_s_kN, "V_Rk_s_kN")
        if self.seismic_design:
            self._validate_seismic()

    def _validate_seismic(self):
        if self.seismic_performance_category not in _SEISMIC_CATEGORIES:
            raise ValueError(
                "seismic_performance_category must be 'C1' or 'C2', "
                f"got {self.seismic_performance_category!r}"
            )
        self.annular_gap_filled = _flag(self.annular_gap_filled, "annular_gap_filled")
        if self.seismic_design_option not in _DESIGN_OPTIONS:
            raise ValueError(
                "seismic_design_option must be 'capacity', 'elastic' or "
                f"'ductile_fastener', got {self.seismic_design_option!r}"
            )
        if self.fastener_class not in _FASTENER_CLASSES:
            raise ValueError(
                "fastener_class must be 'headed_k1' or 'other' for Annex C, "
                f"got {self.fastener_class!r}"
            )
        if self.anchors_in_tension is None:
            raise ValueError(
                "anchors_in_tension is required for Annex C. Table C.3 note (a) "
                "uses the single-fastener factors when only one fastener is in tension."
            )
        self.anchors_in_tension = int(self.anchors_in_tension)
        if not 1 <= self.anchors_in_tension <= self.number_of_anchors:
            raise ValueError(
                "anchors_in_tension must be from 1 to number_of_anchors, "
                f"got {self.anchors_in_tension}"
            )
        self.alpha_gap_with_gap_filling = _positive(
            self.alpha_gap_with_gap_filling, "alpha_gap_with_gap_filling"
        )
        self.alpha_gap_without_gap_filling = _positive(
            self.alpha_gap_without_gap_filling, "alpha_gap_without_gap_filling"
        )
        self.N_Rk_s_eq_kN = _positive(self.N_Rk_s_eq_kN, "N_Rk_s_eq_kN")
        self.N_Rk_p_eq_kN = _positive(self.N_Rk_p_eq_kN, "N_Rk_p_eq_kN")
        self.V_Rk_s_eq_kN = _positive(self.V_Rk_s_eq_kN, "V_Rk_s_eq_kN")
        if self.fastener_kind == "bonded":
            raise ValueError(
                "Annex C bonded resistance needs tau_Rk,eq from the ETA. "
                "This product path has no seismic bond stress."
            )
        if self.seismic_performance_category == "C2":
            for label in (
                "delta_N_eq_DLS_mm",
                "delta_V_eq_DLS_mm",
                "delta_N_req_DLS_mm",
                "delta_V_req_DLS_mm",
            ):
                setattr(self, label, _positive(getattr(self, label), label))
        if self.seismic_design_option == "ductile_fastener":
            if self.seismic_performance_category != "C2":
                raise ValueError(
                    "clause C.3 (3) a) requires performance category C2 for "
                    "design by yielding of the fastener"
                )
            for label in (
                "f_uk_MPa",
                "f_yk_MPa",
                "rupture_elongation",
                "stretch_length_mm",
            ):
                if getattr(self, label) is None:
                    raise ValueError(
                        f"{label} is required for seismic design option "
                        "ductile_fastener, clause C.3 (3)"
                    )
            self.f_uk_MPa = _positive(self.f_uk_MPa, "f_uk_MPa")
            self.f_yk_MPa = _positive(self.f_yk_MPa, "f_yk_MPa")
            self.rupture_elongation = _positive(
                self.rupture_elongation, "rupture_elongation"
            )
            self.stretch_length_mm = _positive(
                self.stretch_length_mm, "stretch_length_mm"
            )


class MechanicalAnchorCapacity_EN1992_4:
    """One method per clause for a mechanical or bonded anchor group."""

    def __init__(self, props: AnchorProperties_EN1992_4):
        if not isinstance(props, AnchorProperties_EN1992_4):
            raise TypeError(
                "props must be AnchorProperties_EN1992_4, "
                f"got {type(props).__name__}"
            )
        self.p = props

    def _gamma_mc_tension(self):
        return _GAMMA_C * self.p.gamma_inst

    def _gamma_mc_shear(self):
        return _GAMMA_C

    def _k1(self):
        return self.p.k_cr_N if self.p.is_cracked else self.p.k_ucr_N

    def _alpha_gap_shear(self):
        """Clause C.5 (2). alpha_gap acts on shear only."""
        if not self.p.seismic_design:
            return 1.0
        if self.p.annular_gap_filled:
            return self.p.alpha_gap_with_gap_filling
        return self.p.alpha_gap_without_gap_filling

    def _alpha_eq(self, loading, failure_mode):
        if not self.p.seismic_design:
            return 1.0
        if loading == "tension":
            group = self.p.anchors_in_tension > 1
        else:
            group = self.p.number_of_anchors > 1
        return alpha_eq_table_c_3(
            loading, failure_mode, self.p.fastener_class, group
        )

    def _basic_characteristic(self, loading, failure_mode, static_or_eta_kN):
        """Formula (C.8) for a seismic check, or the static value itself."""
        if not self.p.seismic_design:
            return static_or_eta_kN
        gap = self._alpha_gap_shear() if loading == "shear" else 1.0
        return gap * self._alpha_eq(loading, failure_mode) * static_or_eta_kN

    def steel_failure_tension_7_2_1_3(self):
        characteristic = (
            self.p.N_Rk_s_eq_kN if self.p.seismic_design else self.p.N_Rk_s_kN
        )
        r_k = self._basic_characteristic("tension", "steel", characteristic)
        return {
            "N_Rk_s": r_k,
            "N_Rd_s": r_k / self.p.gamma_Ms_N,
            "gamma_Ms": self.p.gamma_Ms_N,
            "alpha_eq": self._alpha_eq("tension", "steel"),
        }

    def concrete_cone_failure_7_2_1_4(self):
        """Formula (7.1). k1 is the cracked or uncracked product factor."""
        p = self.p
        k1 = self._k1()
        n0 = k1 * math.sqrt(p.fck_cylinder_MPa) * p.h_ef_mm**1.5 / 1000.0
        a0 = p.s_cr_N_mm * p.s_cr_N_mm
        area = projected_area_tension(
            p.number_of_anchors,
            p.s_cr_N_mm,
            p.c_cr_N_mm,
            p.edge_distance_c1_mm,
            p.spacing_mm,
            p.spacing_x_mm,
            p.spacing_y_mm,
        )
        c_for_psi = min(p.edge_distance_c1_mm, p.c_cr_N_mm)
        psi_s = min(0.7 + 0.3 * (c_for_psi / p.c_cr_N_mm), 1.0)
        psi_re = psi_re_N(p.h_ef_mm, p.dense_reinforcement)
        psi_ec1 = min(1.0 / (1.0 + 2.0 * p.e_N_x_mm / p.s_cr_N_mm), 1.0)
        psi_ec2 = min(1.0 / (1.0 + 2.0 * p.e_N_y_mm / p.s_cr_N_mm), 1.0)
        n_rk_static = n0 * (area / a0) * psi_s * psi_re * psi_ec1 * psi_ec2
        n_rk = self._basic_characteristic("tension", "cone", n_rk_static)
        return {
            "k1": k1,
            "N0_Rk_c": n0,
            "N_Rk_c_static": n_rk_static,
            "N_Rk_c": n_rk,
            "N_Rd_c": n_rk / self._gamma_mc_tension(),
            "A_c_N": area,
            "A0_c_N": a0,
            "psi_s_N": psi_s,
            "psi_re_N": psi_re,
            "gamma_Mc": self._gamma_mc_tension(),
            "alpha_eq": self._alpha_eq("tension", "cone"),
        }

    def pullout_failure_7_2_1_5(self):
        """Mechanical pull-out. psi_c raises the C20/25 ETA value to the grade."""
        if self.p.fastener_kind != "mechanical":
            raise ValueError("clause 7.2.1.5 is the mechanical pull-out check")
        basic = self.p.psi_c * self.p.N_Rk_p_kN
        if self.p.seismic_design:
            # The ETA seismic pull-out already replaces the static value.
            # psi_c is not applied again: the ETA table does not say to.
            basic = self.p.N_Rk_p_eq_kN
        n_rk = self._basic_characteristic("tension", "pullout", basic)
        return {
            "N_Rk_p": n_rk,
            "N_Rd_p": n_rk / self._gamma_mc_tension(),
            "gamma_Mp": self._gamma_mc_tension(),
            "psi_c": None if self.p.seismic_design else self.p.psi_c,
            "alpha_eq": self._alpha_eq("tension", "pullout"),
        }

    def combined_pullout_cone_bonded_7_2_1_6(self):
        """Formula (7.13). The concrete cone stress uses Formula (7.19)."""
        if self.p.fastener_kind != "bonded":
            raise ValueError("clause 7.2.1.6 is the bonded pull-out check")
        p = self.p
        tau = p.tau_Rk_cr if p.is_cracked else p.tau_Rk_ucr
        k3 = _K3_CRACKED if p.is_cracked else _K3_UNCRACKED
        s_from_bond = _S_CR_NP_FACTOR * p.diameter_mm * math.sqrt(p.psi_sus * p.tau_Rk_ucr)
        s_cr = min(s_from_bond, _S_CR_NP_HEF_CAP * p.h_ef_mm)
        c_cr = s_cr / 2.0
        area = projected_area_tension(
            p.number_of_anchors,
            s_cr,
            c_cr,
            p.edge_distance_c1_mm,
            p.spacing_mm,
            p.spacing_x_mm,
            p.spacing_y_mm,
        )
        area0 = s_cr * s_cr
        tau_c = k3 * math.sqrt(p.h_ef_mm * p.fck_cylinder_MPa) / (math.pi * p.diameter_mm)
        spacing = self._mean_spacing()
        psi_g, psi0 = self._psi_g(spacing if spacing is not None else s_cr, s_cr, tau, tau_c)
        c_min = min(p.edge_distance_c1_mm, c_cr)
        psi_s = min(0.7 + 0.3 * (c_min / c_cr), 1.0)
        psi_re = psi_re_N(p.h_ef_mm, p.dense_reinforcement)
        psi_ec1 = min(1.0 / (1.0 + 2.0 * p.e_N_x_mm / s_cr), 1.0)
        psi_ec2 = min(1.0 / (1.0 + 2.0 * p.e_N_y_mm / s_cr), 1.0)
        n0 = p.psi_sus * tau * math.pi * p.diameter_mm * p.h_ef_mm / 1000.0
        n_rk_static = n0 * (area / area0) * psi_s * psi_re * psi_ec1 * psi_ec2 * psi_g
        n_rk = self._basic_characteristic("tension", "pullout", n_rk_static)
        return {
            "N0_Rk_p": n0,
            "N_Rk_p": n_rk,
            "N_Rd_p": n_rk / self._gamma_mc_tension(),
            "A_p_N": area,
            "A0_p_N": area0,
            "s_cr_Np": s_cr,
            "c_cr_Np": c_cr,
            "psi_g_Np": psi_g,
            "psi_g_Np0": psi0,
            "psi_s_Np": psi_s,
            "tau_Rk_c": tau_c,
            "k3": k3,
            "gamma_Mp": self._gamma_mc_tension(),
            "alpha_eq": self._alpha_eq("tension", "pullout"),
        }

    def _mean_spacing(self):
        p = self.p
        if p.number_of_anchors == 1:
            return None
        if p.number_of_anchors == 2:
            if p.spacing_mm is None:
                raise ValueError("spacing_mm is required for two anchors")
            return _positive(p.spacing_mm, "spacing_mm")
        if p.spacing_x_mm is None or p.spacing_y_mm is None:
            raise ValueError("spacing_x_mm and spacing_y_mm are required for four anchors")
        return (
            _positive(p.spacing_x_mm, "spacing_x_mm")
            + _positive(p.spacing_y_mm, "spacing_y_mm")
        ) / 2.0

    def _psi_g(self, spacing, s_cr, tau, tau_c):
        n = self.p.number_of_anchors
        if n == 1:
            return 1.0, 1.0
        root_n = math.sqrt(n)
        psi0 = root_n - (root_n - 1.0) * (tau / tau_c) ** 1.5
        if psi0 < 1.0:
            psi0 = 1.0
        if spacing >= s_cr:
            return 1.0, psi0
        psi = psi0 - math.sqrt(spacing / s_cr) * (psi0 - 1.0)
        if psi < 1.0:
            psi = 1.0
        return psi, psi0

    def splitting_failure_7_2_1_7(self, cone):
        p = self.p
        if not p.splitting_check_required:
            return {"skipped": True, "N_Rk_sp": float("inf"), "N_Rd_sp": float("inf")}
        if p.c_cr_sp_mm is None or p.s_cr_sp_mm is None or p.h_min_mm is None:
            raise ValueError(
                "splitting needs c_cr_sp_mm, s_cr_sp_mm and h_min_mm"
            )
        s_cr = _positive(p.s_cr_sp_mm, "s_cr_sp_mm")
        c_cr = _positive(p.c_cr_sp_mm, "c_cr_sp_mm")
        area0 = s_cr * s_cr
        area = projected_area_tension(
            p.number_of_anchors,
            s_cr,
            c_cr,
            p.edge_distance_c1_mm,
            p.spacing_mm,
            p.spacing_x_mm,
            p.spacing_y_mm,
        )
        c_sp = min(p.edge_distance_c1_mm, c_cr)
        psi_s = min(0.7 + 0.3 * (c_sp / c_cr), 1.0)
        psi_h = psi_h_sp(p.concrete_thickness_mm, p.h_min_mm, p.h_ef_mm)
        n_rk_static = (
            cone["N0_Rk_c"]
            * (area / area0)
            * psi_s
            * cone["psi_re_N"]
            * psi_h
        )
        # Eccentricity factors are the cone factors. They are already inside
        # N_Rk_c_static but not inside N0, so apply them here too.
        psi_ec1 = min(1.0 / (1.0 + 2.0 * p.e_N_x_mm / p.s_cr_N_mm), 1.0)
        psi_ec2 = min(1.0 / (1.0 + 2.0 * p.e_N_y_mm / p.s_cr_N_mm), 1.0)
        n_rk_static *= psi_ec1 * psi_ec2
        n_rk = self._basic_characteristic("tension", "splitting", n_rk_static)
        return {
            "skipped": False,
            "N_Rk_sp": n_rk,
            "N_Rd_sp": n_rk / self._gamma_mc_tension(),
            "psi_h_sp": psi_h,
            "A_c_N": area,
            "A0_c_N": area0,
            "alpha_eq": self._alpha_eq("tension", "splitting"),
        }

    def calculate_tension_capacity_7_2_1(self):
        steel = self.steel_failure_tension_7_2_1_3()
        cone = self.concrete_cone_failure_7_2_1_4()
        if self.p.fastener_kind == "bonded":
            pullout = self.combined_pullout_cone_bonded_7_2_1_6()
        else:
            pullout = self.pullout_failure_7_2_1_5()
        splitting = self.splitting_failure_7_2_1_7(cone)
        modes = {
            "steel": steel["N_Rd_s"],
            "pullout": pullout["N_Rd_p"],
            "concrete_cone": cone["N_Rd_c"],
        }
        if not splitting["skipped"]:
            modes["splitting"] = splitting["N_Rd_sp"]
        governing = min(modes, key=modes.get)
        return {
            "N_Rd_steel": steel["N_Rd_s"],
            "N_Rd_pullout": pullout["N_Rd_p"],
            "N_Rd_concrete": cone["N_Rd_c"],
            "N_Rd_splitting": splitting["N_Rd_sp"],
            "N_Rd": modes[governing],
            "governing_mode": governing,
            "steel_details": steel,
            "pullout_details": pullout,
            "concrete_details": cone,
            "splitting_details": splitting,
        }

    def steel_failure_shear_7_2_2_3(self):
        characteristic = (
            self.p.V_Rk_s_eq_kN if self.p.seismic_design else self.p.V_Rk_s_kN
        )
        # k7 accounts for group ductility on the static value. The ETA seismic
        # steel shear is used as given, without a second k7.
        if self.p.seismic_design:
            basic = characteristic
        else:
            basic = self.p.k7 * characteristic
        r_k = self._basic_characteristic("shear", "steel", basic)
        return {
            "V_Rk_s": r_k,
            "V_Rd_s": r_k / self.p.gamma_Ms_V,
            "gamma_Ms": self.p.gamma_Ms_V,
            "k7": None if self.p.seismic_design else self.p.k7,
            "alpha_gap": self._alpha_gap_shear(),
            "alpha_eq": self._alpha_eq("shear", "steel"),
        }

    def concrete_pryout_failure_7_2_2_4(self, tension):
        """Formula (7.39). Bonded pryout uses the clause 7.2.1.6 resistance."""
        if self.p.fastener_kind == "bonded":
            n_rk = tension["pullout_details"]["N_Rk_p"]
        else:
            n_rk = tension["concrete_details"]["N_Rk_c_static"]
        basic = self.p.k8 * n_rk
        # Bonded N_Rk_p is already the group value. Seismic alpha_eq for
        # pryout is applied to that static resistance. For a seismic mechanical
        # anchor the cone static value is scaled here, not the already-scaled
        # seismic cone, so alpha_eq is not applied twice.
        if self.p.seismic_design and self.p.fastener_kind == "bonded":
            r_k = basic
        else:
            r_k = self._basic_characteristic("shear", "pryout", basic)
        return {
            "V_Rk_cp": r_k,
            "V_Rd_cp": r_k / self._gamma_mc_shear(),
            "k8": self.p.k8,
            "gamma_Mc": self._gamma_mc_shear(),
            "alpha_eq": self._alpha_eq("shear", "pryout"),
        }

    def concrete_edge_failure_7_2_2_5(self):
        """Formula (7.40). No edge means this mode is not verified."""
        p = self.p
        if p.edge_distance_c1_mm >= NO_CONCRETE_EDGE_MM:
            return {
                "V0_Rk_c": float("inf"),
                "V_Rk_c": float("inf"),
                "V_Rd_c": float("inf"),
                "A_c_V": float("inf"),
                "A0_c_V": float("inf"),
                "psi_re_V": None,
                "psi_alpha_V": None,
            }
        c1 = p.edge_distance_c1_mm
        k9 = k9_7_2_2_5(p.is_cracked)
        alpha = 0.1 * (p.l_f_mm / c1) ** 0.5
        beta = 0.1 * (p.diameter_mm / c1) ** 0.2
        v0 = (
            k9
            * p.diameter_mm**alpha
            * p.l_f_mm**beta
            * math.sqrt(p.fck_cylinder_MPa)
            * c1**1.5
            / 1000.0
        )
        a0 = 4.5 * c1 * c1
        area = projected_area_shear(
            p.number_of_anchors,
            c1,
            p.edge_distance_c2_1_mm,
            p.edge_distance_c2_2_mm,
            p.concrete_thickness_mm,
            p.spacing_mm,
            p.spacing_y_mm,
        )
        c2 = min(float(p.edge_distance_c2_1_mm), float(p.edge_distance_c2_2_mm))
        psi_s = min(0.7 + 0.3 * (c2 / (1.5 * c1)), 1.0)
        psi_h = math.sqrt(1.5 * c1 / p.concrete_thickness_mm) if p.concrete_thickness_mm < 1.5 * c1 else 1.0
        psi_ec = min(1.0 / (1.0 + 2.0 * p.e_V_mm / (3.0 * c1)), 1.0)
        psi_alpha = psi_alpha_V_7_2_2_5(p.alpha_V_deg)
        psi_re = psi_re_V_7_2_2_5(
            is_cracked=p.is_cracked,
            edge_reinforcement=p.edge_reinforcement,
            edge_reinforcement_spacing_mm=p.edge_reinforcement_spacing_mm,
            edge_reinforcement_cover_mm=p.edge_reinforcement_cover_mm,
            h_ef_mm=p.h_ef_mm,
            c1_mm=c1,
        )
        v_rk_static = v0 * (area / a0) * psi_s * psi_h * psi_ec * psi_alpha * psi_re
        v_rk = self._basic_characteristic("shear", "edge", v_rk_static)
        return {
            "k9": k9,
            "V0_Rk_c": v0,
            "V_Rk_c": v_rk,
            "V_Rd_c": v_rk / self._gamma_mc_shear(),
            "A_c_V": area,
            "A0_c_V": a0,
            "psi_s_V": psi_s,
            "psi_h_V": psi_h,
            "psi_alpha_V": psi_alpha,
            "psi_re_V": psi_re,
            "alpha_exp": alpha,
            "beta_exp": beta,
            "alpha_eq": self._alpha_eq("shear", "edge"),
            "alpha_gap": self._alpha_gap_shear(),
        }

    def calculate_shear_capacity_7_2_2(self, tension):
        steel = self.steel_failure_shear_7_2_2_3()
        pryout = self.concrete_pryout_failure_7_2_2_4(tension)
        edge = self.concrete_edge_failure_7_2_2_5()
        modes = {
            "steel": steel["V_Rd_s"],
            "pryout": pryout["V_Rd_cp"],
            "concrete_edge": edge["V_Rd_c"],
        }
        finite = {name: value for name, value in modes.items() if math.isfinite(value)}
        if not finite:
            raise ValueError("shear capacity has no finite failure mode")
        governing = min(finite, key=finite.get)
        return {
            "V_Rd_steel": steel["V_Rd_s"],
            "V_Rd_pryout": pryout["V_Rd_cp"],
            "V_Rd_concrete": edge["V_Rd_c"],
            "V_Rd": finite[governing],
            "governing_mode": governing,
            "steel_details": steel,
            "pryout_details": pryout,
            "concrete_edge_details": edge,
        }

    def check_interaction_7_2_3(self, tension_kN, shear_kN, tension, shear):
        """Table 7.3 for static design. Annex C Formula (C.9) for seismic."""
        ratios = self._ratios(tension_kN, shear_kN, tension, shear)
        if self.p.seismic_design:
            return interaction_annex_c(
                ratios["beta_n_steel"],
                ratios["beta_v_steel"],
                ratios["beta_n_concrete"],
                ratios["beta_v_concrete"],
            )
        return interaction_table_7_3(
            ratios["beta_n_steel"],
            ratios["beta_v_steel"],
            ratios["beta_n_concrete"],
            ratios["beta_v_concrete"],
        )

    def _ratios(self, tension_kN, shear_kN, tension, shear):
        n = self.p.number_of_anchors
        tension_kN = _non_negative(tension_kN, "tension_kN")
        shear_kN = _non_negative(shear_kN, "shear_kN")
        concrete_tension = [
            tension_kN / tension["N_Rd_pullout"] / n,
            tension_kN / tension["N_Rd_concrete"],
        ]
        if not tension["splitting_details"]["skipped"]:
            concrete_tension.append(tension_kN / tension["N_Rd_splitting"])
        concrete_shear = [shear_kN / shear["V_Rd_pryout"]]
        if math.isfinite(shear["V_Rd_concrete"]):
            concrete_shear.append(shear_kN / shear["V_Rd_concrete"])
        return {
            "beta_n_steel": (tension_kN / n) / tension["N_Rd_steel"],
            "beta_v_steel": (shear_kN / n) / shear["V_Rd_steel"],
            "beta_n_concrete": max(concrete_tension),
            "beta_v_concrete": max(concrete_shear),
        }

    def displacement_check_c_6(self, tension, shear):
        """Formula (C.11). Reduce a C2 resistance when the ETA displacement
        exceeds the displacement the application can accept. C1 has no
        displacement capacity.
        """
        if not self.p.seismic_design or self.p.seismic_performance_category != "C2":
            return tension, shear, {"applied": False}
        n_factor = min(1.0, self.p.delta_N_req_DLS_mm / self.p.delta_N_eq_DLS_mm)
        v_factor = min(1.0, self.p.delta_V_req_DLS_mm / self.p.delta_V_eq_DLS_mm)
        if n_factor < 1.0:
            for key in ("N_Rd_steel", "N_Rd_pullout", "N_Rd_concrete", "N_Rd_splitting", "N_Rd"):
                tension[key] = tension[key] * n_factor
        if v_factor < 1.0:
            for key in ("V_Rd_steel", "V_Rd_pryout", "V_Rd_concrete", "V_Rd"):
                if math.isfinite(shear[key]):
                    shear[key] = shear[key] * v_factor
        return tension, shear, {
            "applied": True,
            "clause": "C.11",
            "tension_factor": n_factor,
            "shear_factor": v_factor,
        }

    def ductile_fastener_check_c_3(self, tension):
        """Clause C.3 (3). Option b) only."""
        if not self.p.seismic_design or self.p.seismic_design_option != "ductile_fastener":
            return {"applied": False}
        p = self.p
        if p.f_uk_MPa > 800.0 or p.f_yk_MPa / p.f_uk_MPa > 0.8 or p.rupture_elongation < 0.12:
            raise ValueError(
                "the fastener is not ductile under clause C.3 (3) c): "
                "f_uk must be at most 800 MPa, f_yk/f_uk at most 0.8, and "
                "the rupture elongation at least 12%"
            )
        if p.stretch_length_mm < 8.0 * p.diameter_mm:
            raise ValueError(
                "clause C.3 (3) c) requires a stretch length of at least 8d "
                f"({8.0 * p.diameter_mm} mm), got {p.stretch_length_mm} mm"
            )
        if p.anchors_in_tension > 1:
            raise ValueError(
                "Formula (C.2) for a ductile group needs the design action "
                "per fastener, which this check is not given"
            )
        concrete_modes = [
            tension["concrete_details"]["N_Rk_c"],
            tension["pullout_details"]["N_Rk_p"],
        ]
        if not tension["splitting_details"]["skipped"]:
            concrete_modes.append(tension["splitting_details"]["N_Rk_sp"])
        r_conc = min(concrete_modes)
        r_steel = tension["steel_details"]["N_Rk_s"]
        if p.anchors_in_tension == 1:
            limit = 0.7 * r_conc / p.gamma_inst
            if r_steel > limit:
                raise ValueError(
                    "Formula (C.1) is not met: seismic steel resistance "
                    f"{r_steel} kN exceeds 0.7 * Rk,conc,eq / gamma_inst "
                    f"({limit} kN)"
                )
        return {"applied": True, "clause": "C.1", "R_k_s_eq": r_steel, "R_k_conc_eq": r_conc}

    def calculate(self):
        tension = self.calculate_tension_capacity_7_2_1()
        shear = self.calculate_shear_capacity_7_2_2(tension)
        tension, shear, displacement = self.displacement_check_c_6(tension, shear)
        ductility = self.ductile_fastener_check_c_3(tension)
        return {
            "tension": tension,
            "shear": shear,
            "displacement": displacement,
            "ductility": ductility,
        }


def _newton_layout(result):
    def newtons(kN):
        if not math.isfinite(kN):
            return float("inf")
        return float(kN) * 1000.0

    tension = result["tension"]
    shear = result["shear"]
    return {
        "tension": {
            "governing_capacity_N": newtons(tension["N_Rd"]),
            "governing_mode": tension["governing_mode"],
            "steel_failure": {"N_Rd_s_N": newtons(tension["N_Rd_steel"])},
            "pullout_failure": {"N_Rd_p_N": newtons(tension["N_Rd_pullout"])},
            "concrete_breakout": {"N_Rd_c_N": newtons(tension["N_Rd_concrete"])},
            "splitting": {"N_Rd_sp_N": newtons(tension["N_Rd_splitting"])},
        },
        "shear": {
            "governing_capacity_N": newtons(shear["V_Rd"]),
            "governing_mode": shear["governing_mode"],
            "steel_failure": {"V_Rd_s_N": newtons(shear["V_Rd_steel"])},
            "pryout_failure": {"V_Rd_cp_N": newtons(shear["V_Rd_pryout"])},
            "concrete_edge": {"V_Rd_c_N": newtons(shear["V_Rd_concrete"])},
        },
        "summary": {
            "N_Rd_kN": tension["N_Rd"],
            "V_Rd_kN": shear["V_Rd"],
            "tension_governing": tension["governing_mode"],
            "shear_governing": shear["governing_mode"],
        },
        "interaction_inputs": result,
        "displacement": result["displacement"],
    }


def calculate_anchor_capacity_en1992_4_detailed(**kwargs):
    """Resistances in newtons, in the same layout as the ETAG detailed result."""
    props = AnchorProperties_EN1992_4(**kwargs)
    result = MechanicalAnchorCapacity_EN1992_4(props).calculate()
    return _newton_layout(result)


def interaction_from_capacities(result, tension_kN, shear_kN, number_of_anchors):
    """Table 7.3 from a static result of this module.

    ``result`` is the dict returned by ``MechanicalAnchorCapacity_EN1992_4.calculate``
    or the same shape stored under ``interaction_inputs`` of the detailed result.
    """
    if "interaction_inputs" in result:
        result = result["interaction_inputs"]
    tension = result["tension"]
    shear = result["shear"]
    n = int(number_of_anchors)
    if n < 1:
        raise ValueError(f"number_of_anchors must be >= 1, got {n}")
    beta_n_steel = (tension_kN / n) / tension["N_Rd_steel"]
    beta_v_steel = (shear_kN / n) / shear["V_Rd_steel"]
    concrete_tension = [
        tension_kN / tension["N_Rd_pullout"] / n,
        tension_kN / tension["N_Rd_concrete"],
    ]
    if not tension["splitting_details"]["skipped"]:
        concrete_tension.append(tension_kN / tension["N_Rd_splitting"])
    concrete_shear = [shear_kN / shear["V_Rd_pryout"]]
    if math.isfinite(shear["V_Rd_concrete"]):
        concrete_shear.append(shear_kN / shear["V_Rd_concrete"])
    return interaction_table_7_3(
        beta_n_steel,
        beta_v_steel,
        max(concrete_tension),
        max(concrete_shear),
    )
