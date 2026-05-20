"""AS_1720_1_2022 Method

This module references the following standard:
AS 1720.1:2010 (incorporating Amendments 1, 2, 3) — Australian standard only.

Covers timber member design for F-grade and MGP-grade sawn timber.

Method developed May 2026
(c) Constratum Ltd

Developed - NSh
Reviewed  - MB

###Initialise Dependents and Libraries
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Section 3 - Basic structural members
# ------------------------------------------------------------------
class Section3:
    """
    Timber member design capacity functions for F-grade and MGP-grade sawn timber
    in accordance with AS 1720.1:2010 (Australian standard only).

    Material properties:   Table H2.1 (F4–F34, MGP10–MGP15, A17)
    Stability constants:   Table 3.1 (rho_b), Table 3.3 (rho_c)
    Capacity factor:       Table 2.1 — varies by member category (1/2/3) and grade tier
    Duration of load:      Table 2.3
    Temperature factor:    k6 — user-supplied (0.9 for tropical AU, otherwise 1.0)

    Key AS-specific features:
      - phi varies by category AND grade tier (high grades F17+/MGP15/A17 get higher phi)
      - Grade system: F4–F34, MGP10–MGP15, A17
      - k4 formula for seasoned timber at EMC > 15% (Cl 2.4.2.3)
      - k4 partial seasoning table for unseasoned timber (Table 2.5)
      - k6 applies for coastal Queensland north of 25°S, or regions north of 16°S
      - ft' has separate hardwood / softwood columns in Table H2.1
      - rho_b from Table 3.1 (separate seasoned/unseasoned values)
      - rho_c from Table 3.3 (separate seasoned/unseasoned values)

    Design philosophy:
      - Material properties (fb, fc, ft, fs, E, G, phi, rho_b, rho_c) are read from
        standard tables via get_* methods inside this module.
      - Section properties (A, Z, I, As, At, b, d) are defined in the notebook and
        passed explicitly to capacity functions — not calculated inside this module.
      - Each capacity function returns the design capacity only (N or N·mm).
      - Action vs capacity comparison is done externally via
        AS_NZS_1170_0_2002.uls_strength(capacity, action*).
      - The ONLY exception is combined_bending_compression / combined_bending_tension
        (Cl 3.5.1 / 3.5.2) which are interaction equations by definition.
    """

    # ------------------------------------------------------------------
    # Table H2.1 — Characteristic values for F-grade sawn timber (MPa)
    # Tuple: (fb, ft_hardwood, ft_softwood, fs, fc, E, G)
    #
    # Notes:
    #   Note 1: for beam depth d > 300mm, fb *= (300/d)^0.167
    #   Note 2: for largest tension dimension > 150mm, ft *= (150/d)^0.167
    # ------------------------------------------------------------------
    TABLE_H2_1 = {
        #          fb    ft_hard  ft_soft   fs    fc      E       G
        "F34": ( 84,    51,      42,      6.1,  63,  21500,  1430),
        "F27": ( 67,    42,      34,      5.1,  51,  18500,  1230),
        "F22": ( 55,    34,      29,      4.2,  42,  16000,  1070),
        "F17": ( 42,    25,      22,      3.6,  34,  14000,   930),
        "F14": ( 36,    22,      19,      3.3,  27,  12000,   800),
        "F11": ( 31,    18,      15,      2.8,  22,  10500,   700),
        "F8":  ( 22,    13,      12,      2.2,  18,   9100,   610),
        "F7":  ( 18,    11,       8.9,    1.9,  13,   7900,   530),
        "F5":  ( 14,     9,       7.3,    1.6,  11,   6900,   460),
        "F4":  ( 12,     7,       5.8,    1.3,   8.6, 6100,   410),
    }

    # ------------------------------------------------------------------
    # Table 3.1 — Material constant rho_b for bending stability
    # Tuple: (seasoned, unseasoned). None = not defined for that condition.
    # ------------------------------------------------------------------
    TABLE_3_1_RHO_B = {
        "F34":  (1.12, 1.21), "F27":  (1.08, 1.17), "F22":  (1.05, 1.15),
        "F17":  (0.98, 1.08), "F14":  (0.98, 1.08), "F11":  (0.98, 1.07),
        "F8":   (0.89, 0.99), "F7":   (0.86, 0.96), "F5":   (0.82, 0.91),
        "F4":   (0.80, 0.90),
        "MGP15": (0.91, None), "MGP12": (0.85, None), "MGP10": (0.75, None),
        "A17":   (0.95, None),
    }

    # ------------------------------------------------------------------
    # Table 3.3 — Material constant rho_c for compression stability
    # Tuple: (seasoned, unseasoned). None = not defined for that condition.
    # ------------------------------------------------------------------
    TABLE_3_3_RHO_C = {
        "F34":  (1.17, 1.34), "F27":  (1.14, 1.31), "F22":  (1.12, 1.28),
        "F17":  (1.08, 1.25), "F14":  (1.05, 1.21), "F11":  (1.02, 1.18),
        "F8":   (1.00, 1.16), "F7":   (0.92, 1.08), "F5":   (0.91, 1.07),
        "F4":   (0.87, 1.02),
        "MGP15": (0.99, None), "MGP12": (0.98, None), "MGP10": (0.96, None),
        "A17":   (1.10, None),
    }

    # ------------------------------------------------------------------
    # Table 2.3 — k1 duration of load factor
    # ------------------------------------------------------------------
    TABLE_2_3_K1 = {
        "5 seconds": 1.00,
        "5 minutes": 1.00,
        "5 hours":   0.97,
        "5 days":    0.94,
        "5 months":  0.80,
        "50+ years": 0.57,
    }

    # ------------------------------------------------------------------
    # Table 2.1 — Capacity factor phi
    # High grades: F17 and above, MGP15, A17 (upper section of Table 2.1)
    # Keys: (is_high_grade, category) -> phi
    # ------------------------------------------------------------------
    _HIGH_GRADES = {"F34", "F27", "F22", "F17", "MGP15", "A17"}
    TABLE_2_1_PHI = {
        (True,  1): 0.95, (True,  2): 0.85, (True,  3): 0.75,
        (False, 1): 0.90, (False, 2): 0.70, (False, 3): 0.60,
    }

    # ------------------------------------------------------------------
    # Table 2.7 — g31/g32 geometric factors for k9 strength sharing
    # ------------------------------------------------------------------
    TABLE_2_7_G = {1: 1.00, 2: 1.14, 3: 1.20, 4: 1.24, 5: 1.26,
                   6: 1.28, 7: 1.30, 8: 1.31, 9: 1.32}

    @staticmethod
    def _parse_seasoned(seasoned):
        """
        Parse seasoned input to bool.

        Accepts 'Yes'/'No' (case-insensitive) or bool for programmatic use.
        """
        if isinstance(seasoned, bool):
            return seasoned
        if not isinstance(seasoned, str):
            raise ValueError(
                f"seasoned must be 'Yes', 'No', or bool; got {seasoned!r}"
            )
        key = seasoned.strip().lower()
        if key == "yes":
            return True
        if key == "no":
            return False
        raise ValueError(
            f"seasoned must be 'Yes' or 'No'; got '{seasoned}'"
        )

    def __init__(self, grade="F11", seasoned="Yes", species_type="softwood",
                 member_category=1):
        """
        Initialise an AS 1720.1:2010 F-grade timber member design object.

        Args:
            grade (str): Stress grade — 'F4'–'F34', 'MGP10'/'MGP12'/'MGP15',
                         or 'A17'. Case-insensitive.
            seasoned (str | bool): 'Yes' = seasoned (EMC <= 15%). 'No' = unseasoned.
            species_type (str): 'softwood' or 'hardwood'.
                Affects ft' selection from Table H2.1.
            member_category (int): Structural category from Table 2.1.
                1 = secondary member (e.g. partition stud)
                2 = primary structural member
                3 = post-disaster critical
        """
        self.grade           = grade.upper().strip()
        self.seasoned        = self._parse_seasoned(seasoned)
        self.species_type    = species_type.lower()
        self.member_category = member_category

        if (self.grade not in self.TABLE_H2_1
                and self.grade not in self.TABLE_3_1_RHO_B):
            raise ValueError(
                f"Grade '{grade}' not found in AS 1720.1:2010 Table H2.1. "
                f"Valid F-grades: {sorted(self.TABLE_H2_1.keys())}. "
                f"For NZ SG-grades use NZS_1720_1_2022."
            )

    # ── Table 2.1 — Capacity factor ───────────────────────────────────

    def get_phi(self):
        """
        Capacity factor phi (Table 2.1, Cl 2.3).
        Depends on member category (1/2/3) and grade tier (high/standard).

        High grades (F17+, MGP15, A17): Category 1=0.95, 2=0.85, 3=0.75
        Standard grades (F14 and below): Category 1=0.90, 2=0.70, 3=0.60

        Returns:
            float: phi
        """
        is_high = self.grade in self._HIGH_GRADES
        return self.TABLE_2_1_PHI[(is_high, self.member_category)]

    # ── Modification factors ─────────────────────────────────────────

    def get_k1(self, duration="5 days"):
        """
        Duration of load factor k1 (Table 2.3, Cl 2.4.1.1).

        Args:
            duration (str): '5 seconds', '5 minutes', '5 hours',
                            '5 days', '5 months', '50+ years'.

        Returns:
            float: k1
        """
        if duration not in self.TABLE_2_3_K1:
            raise ValueError(
                f"Duration '{duration}' not recognised. "
                f"Valid: {list(self.TABLE_2_3_K1.keys())}"
            )
        return self.TABLE_2_3_K1[duration]

    def get_k4(self, emc_percent=15, least_dimension_mm=None):
        """
        Moisture condition factor k4 (Cl 2.4.2).

        Seasoned (Cl 2.4.2.3):
          EMC <= 15%: k4 = 1.0
          EMC >  15%: k4 = max(1 - 0.3*(EMC-15)/10, 0.7)

        Unseasoned (Cl 2.4.2.2, Table 2.5):
          Base: k4 = 1.0
          Partial seasoning when least_dimension_mm provided:
            <= 38 mm: k4 = 1.15
            <= 50 mm: k4 = 1.10
            <= 75 mm: k4 = 1.05
            >  75 mm: k4 = 1.00

        Args:
            emc_percent (float): Expected in-service annual average EMC (%).
            least_dimension_mm (float, optional): Smallest section dimension (mm).
                Only used for unseasoned timber partial seasoning (Table 2.5).

        Returns:
            float: k4
        """
        if self.seasoned:
            if emc_percent <= 15:
                return 1.0
            return max(1.0 - 0.3 * (emc_percent - 15) / 10, 0.7)
        else:
            if least_dimension_mm is not None:
                if least_dimension_mm <= 38:   return 1.15
                elif least_dimension_mm <= 50: return 1.10
                elif least_dimension_mm <= 75: return 1.05
            return 1.0

    def get_k6(self, k6=1.0):
        """
        Temperature factor k6 (Cl 2.4.3).

        1.0 for covered structures under ambient conditions (most AU locations).
        0.9 for seasoned timber in coastal Queensland north of 25°S,
            or all regions north of 16°S latitude.

        Args:
            k6 (float): User-supplied temperature factor. Default 1.0.

        Returns:
            float: k6
        """
        return k6

    def get_k9(self, n_com=1, n_mem=1, spacing_mm=None, span_mm=None):
        """
        Strength sharing factor k9 for parallel member systems (Cl 2.4.5.3, Table 2.7).
        Applies to bending capacity only.

        Args:
            n_com (int): Members fastened together (combined group). Default 1.
            n_mem (int): Number of discrete parallel members. Default 1.
            spacing_mm (float, optional): Centre-to-centre spacing (mm).
            span_mm (float, optional): Effective span of parallel members (mm).

        Returns:
            float: k9 (>= 1.0)
        """
        def _g(n):
            return self.TABLE_2_7_G.get(min(n, 9), 1.33)
        g31 = _g(n_com)
        g32 = _g(n_com * n_mem)
        if n_mem <= 1 or spacing_mm is None or span_mm is None:
            return max(g31, 1.0)
        return max(g31 + (g32 - g31) * (1.0 - 2.0 * spacing_mm / span_mm), 1.0)

    # ── Material properties — Table H2.1 ─────────────────────────────

    def get_fb(self, depth_mm=None):
        """
        Characteristic bending strength fb' (MPa), Table H2.1.
        For beam depth > 300 mm: fb' *= (300/d)^0.167.

        Args:
            depth_mm (float, optional): Beam depth in mm.

        Returns:
            float: fb' (MPa)
        """
        fb = self.TABLE_H2_1[self.grade][0]
        if depth_mm is not None and depth_mm > 300:
            fb *= (300 / depth_mm) ** 0.167
        return fb

    def get_ft(self, largest_dim_mm=None):
        """
        Characteristic tension strength ft' (MPa), Table H2.1.
        Selects hardwood or softwood column based on species_type.
        For largest dimension > 150 mm: ft' *= (150/d)^0.167.

        Args:
            largest_dim_mm (float, optional): Largest cross-section dimension (mm).

        Returns:
            float: ft' (MPa)
        """
        idx = 1 if self.species_type == "hardwood" else 2
        ft = self.TABLE_H2_1[self.grade][idx]
        if largest_dim_mm is not None and largest_dim_mm > 150:
            ft *= (150 / largest_dim_mm) ** 0.167
        return ft

    def get_fs(self):
        """
        Characteristic shear strength fs' (MPa), Table H2.1.

        Returns:
            float: fs' (MPa)
        """
        return self.TABLE_H2_1[self.grade][3]

    def get_fc(self):
        """
        Characteristic compression strength fc' (MPa), Table H2.1.

        Returns:
            float: fc' (MPa)
        """
        return self.TABLE_H2_1[self.grade][4]

    def get_E(self):
        """
        Short-duration average modulus of elasticity E (MPa), Table H2.1.

        Returns:
            float: E (MPa)
        """
        return self.TABLE_H2_1[self.grade][5]

    def get_G(self):
        """
        Short-duration modulus of rigidity G (MPa), Table H2.1.

        Returns:
            float: G (MPa)
        """
        return self.TABLE_H2_1[self.grade][6]

    def get_rho_b(self):
        """
        Material constant rho_b for bending stability (Table 3.1).

        Returns:
            float: rho_b
        """
        seasoned_val, unseasoned_val = self.TABLE_3_1_RHO_B[self.grade]
        val = seasoned_val if self.seasoned else unseasoned_val
        if val is None:
            raise ValueError(
                f"rho_b for grade '{self.grade}' is only defined for seasoned timber."
            )
        return val

    def get_rho_c(self):
        """
        Material constant rho_c for compression stability (Table 3.3).

        Returns:
            float: rho_c
        """
        seasoned_val, unseasoned_val = self.TABLE_3_3_RHO_C[self.grade]
        val = seasoned_val if self.seasoned else unseasoned_val
        if val is None:
            raise ValueError(
                f"rho_c for grade '{self.grade}' is only defined for seasoned timber."
            )
        return val

    # ── Cl 3.2.3 — Bending slenderness S1 ────────────────────────────

    def calc_S1(self, d_mm, b_mm, restraint_type="discrete",
                restraint_face="compression", Lay_mm=None, L_phi_mm=None):
        """
        Slenderness coefficient S1 for lateral buckling under bending (Cl 3.2.3.2).
        Uses AS rho_b from Table 3.1.

        For minor-axis bending: S2 = 0 — call calc_k12_bending(0.0) -> k12 = 1.0.

        Args:
            d_mm (float): Depth (mm) — major-axis dimension.
            b_mm (float): Breadth (mm).
            restraint_type (str): 'discrete' or 'continuous'.
            restraint_face (str): 'compression' or 'tension'.
            Lay_mm (float, optional): Unbraced length (mm). Required for discrete.
            L_phi_mm (float, optional): Torsional restraint spacing (mm).

        Returns:
            float: S1
        """
        rho_b = self.get_rho_b()
        if restraint_type == "continuous":
            if restraint_face == "compression":
                if Lay_mm is not None:
                    limit = 64.0 * (b_mm / (rho_b * d_mm)) ** 2
                    if (Lay_mm / d_mm) <= limit:
                        return 0.0
                return 0.0
            else:
                if L_phi_mm is not None:
                    return (1.5 * d_mm / b_mm) / (
                        ((np.pi * d_mm / L_phi_mm) ** 2 + 0.4) ** 0.5)
                return 2.25 * d_mm / b_mm
        else:
            if Lay_mm is None:
                raise ValueError("Lay_mm is required for discrete restraint.")
            if restraint_face == "compression":
                return 1.25 * (d_mm / b_mm) * (Lay_mm / d_mm) ** 0.5
            else:
                return (d_mm / b_mm) ** 1.35 * (Lay_mm / d_mm) ** 0.25

    # ── Cl 3.2.4 — Bending stability factor k12b ─────────────────────

    def calc_k12_bending(self, S1):
        """
        Stability factor k12 for bending (Cl 3.2.4).
        Uses AS rho_b from Table 3.1.

        rho_b*S1 <= 10  -> k12 = 1.0
        rho_b*S1 <= 20  -> k12 = 1.5 - 0.05*rho_b*S1
        rho_b*S1 >  20  -> k12 = 200 / (rho_b*S1)²

        Args:
            S1 (float): Slenderness coefficient from calc_S1().

        Returns:
            float: k12 (0 < k12 <= 1.0)
        """
        rho_S = self.get_rho_b() * S1
        if rho_S <= 10:   return 1.0
        elif rho_S <= 20: return 1.5 - 0.05 * rho_S
        else:             return 200.0 / rho_S ** 2

    # ── Cl 3.3.2 — Compression slenderness S3, S4 ───────────────────

    def calc_S3(self, d_mm, Laz_mm, g13=1.0, L_mm=None):
        """
        Slenderness S3 for major-axis compression buckling (Cl 3.3.2.2).
        S3 = min(Laz/d, g13*L/d)

        Args:
            d_mm (float): Depth (mm).
            Laz_mm (float): Unbraced length for major-axis buckling (mm).
            g13 (float): Effective length factor (Table 3.2). Default 1.0.
                         Studs in light framing = 0.9.
            L_mm (float, optional): Total member length (mm).

        Returns:
            float: S3
        """
        if L_mm is not None:
            return min(Laz_mm / d_mm, g13 * L_mm / d_mm)
        return Laz_mm / d_mm

    def calc_S4(self, b_mm, Lay_mm, g13=1.0, L_mm=None):
        """
        Slenderness S4 for minor-axis compression buckling (Cl 3.3.2.2).
        S4 = min(Lay/b, g13*L/b)

        Args:
            b_mm (float): Breadth (mm).
            Lay_mm (float): Unbraced length for minor-axis buckling (mm).
            g13 (float): Effective length factor (Table 3.2). Default 1.0.
            L_mm (float, optional): Total member length (mm).

        Returns:
            float: S4
        """
        if L_mm is not None:
            return min(Lay_mm / b_mm, g13 * L_mm / b_mm)
        return Lay_mm / b_mm

    # ── Cl 3.3.3 — Compression stability factor k12c ─────────────────

    def calc_k12_compression(self, S):
        """
        Stability factor k12 for compression (Cl 3.3.3).
        Uses AS rho_c from Table 3.3. Apply separately for S3 and S4.

        rho_c*S <= 10  -> k12 = 1.0
        rho_c*S <= 20  -> k12 = 1.5 - 0.05*rho_c*S
        rho_c*S >  20  -> k12 = 200 / (rho_c*S)²

        Args:
            S (float): Slenderness coefficient S3 or S4.

        Returns:
            float: k12 (0 < k12 <= 1.0)
        """
        rho_S = self.get_rho_c() * S
        if rho_S <= 10:   return 1.0
        elif rho_S <= 20: return 1.5 - 0.05 * rho_S
        else:             return 200.0 / rho_S ** 2

    # ── Cl 3.2.1 — Bending capacity ──────────────────────────────────

    def bending_capacity_major(self, Zz_mm3, depth_mm, k1, k4, k6, k9, k12b):
        """
        Design bending capacity Mdz about the major z-axis (Cl 3.2.1.1).

        Mdz = phi * k1 * k4 * k6 * k9 * k12b * fb' * Zz   [N·mm]

        Args:
            Zz_mm3 (float): Elastic section modulus about major z-axis (mm³).
            depth_mm (float): Stud depth (mm) — for fb' depth adjustment (Table H2.1).
            k1 (float): Duration factor from get_k1().
            k4 (float): Moisture factor from get_k4().
            k6 (float): Temperature factor from get_k6().
            k9 (float): Strength sharing factor from get_k9().
            k12b (float): Bending stability factor from calc_k12_bending().

        Returns:
            float: Mdz (N·mm)
        """
        return (self.get_phi() * k1 * k4 * k6 * k9 * k12b
                * self.get_fb(depth_mm=depth_mm) * Zz_mm3)

    def bending_capacity_minor(self, Zy_mm3, k1, k4, k6, k9):
        """
        Design bending capacity Mdy about the minor y-axis (Cl 3.2.1.1).
        Minor axis: S2 = 0 always -> k12by = 1.0.

        Mdy = phi * k1 * k4 * k6 * k9 * 1.0 * fb' * Zy   [N·mm]

        Args:
            Zy_mm3 (float): Elastic section modulus about minor y-axis (mm³).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k6 (float): Temperature factor.
            k9 (float): Strength sharing factor.

        Returns:
            float: Mdy (N·mm)
        """
        return self.get_phi() * k1 * k4 * k6 * k9 * self.get_fb() * Zy_mm3

    # ── Cl 3.2.5 — Shear capacity ─────────────────────────────────────

    def shear_capacity(self, As_mm2, k1, k4, k6):
        """
        Design shear capacity Vd (Cl 3.2.5).
        Note: k9 is not applied to shear.

        Vd = phi * k1 * k4 * k6 * fs' * As   [N]

        Args:
            As_mm2 (float): Shear plane area (mm²), Cl 3.2.5.
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k6 (float): Temperature factor.

        Returns:
            float: Vd (N)
        """
        return self.get_phi() * k1 * k4 * k6 * self.get_fs() * As_mm2

    # ── Cl 3.3.1 — Compression capacity ──────────────────────────────

    def compression_capacity_major(self, A_mm2, k1, k4, k6, k12c_z):
        """
        Design compression capacity for major-axis (z-axis) buckling Ndc_z (Cl 3.3.1.1).

        Ndc_z = phi * k1 * k4 * k6 * k12c_z * fc' * Ac   [N]

        Args:
            A_mm2 (float): Gross cross-sectional area Ac (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k6 (float): Temperature factor.
            k12c_z (float): Compression stability factor for S3 (major axis).

        Returns:
            float: Ndc_z (N)
        """
        return self.get_phi() * k1 * k4 * k6 * k12c_z * self.get_fc() * A_mm2

    def compression_capacity_minor(self, A_mm2, k1, k4, k6, k12c_y):
        """
        Design compression capacity for minor-axis (y-axis) buckling Ndc_y (Cl 3.3.1.1).

        Ndc_y = phi * k1 * k4 * k6 * k12c_y * fc' * Ac   [N]

        Args:
            A_mm2 (float): Gross cross-sectional area Ac (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k6 (float): Temperature factor.
            k12c_y (float): Compression stability factor for S4 (minor axis).

        Returns:
            float: Ndc_y (N)
        """
        return self.get_phi() * k1 * k4 * k6 * k12c_y * self.get_fc() * A_mm2

    # ── Cl 3.4.1 — Tension capacity ──────────────────────────────────

    def tension_capacity(self, At_mm2, k1, k4, k6, largest_dim_mm=None):
        """
        Design tension capacity Ndt (Cl 3.4.1).

        Ndt = phi * k1 * k4 * k6 * ft' * At   [N]

        Args:
            At_mm2 (float): Net tension area (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k6 (float): Temperature factor.
            largest_dim_mm (float, optional): Largest cross-section dimension (mm)
                for ft' size adjustment.

        Returns:
            float: Ndt (N)
        """
        return (self.get_phi() * k1 * k4 * k6
                * self.get_ft(largest_dim_mm=largest_dim_mm) * At_mm2)

    # ── Cl 3.5.1 — Combined bending and compression ───────────────────

    def combined_bending_compression(self, Mz_star, My_star, Nc_star,
                                     Mdz, Mdy, Ndc_z, Ndc_y):
        """
        Combined bending and compression interaction check (Cl 3.5.1).

        This function takes both action effects and capacities because the
        standard defines interaction equations — not a simple capacity >= action
        comparison. This is the ONLY capacity function where action effects are
        passed as arguments.

        Criteria 1 — Eq 3.5(1): (Mz*/Mdz)² + My*/Mdy + Nc*/Ndc_y <= 1
        Criteria 2 — Eq 3.5(2): Mz*/Mdz + (My*/Mdy)² + Nc*/Ndc_z <= 1

        Args:
            Mz_star (float): Design moment about major z-axis (N·mm).
            My_star (float): Design moment about minor y-axis (N·mm).
            Nc_star (float): Design compression force (N, positive = compression).
            Mdz (float): Bending capacity major axis (N·mm).
            Mdy (float): Bending capacity minor axis (N·mm).
            Ndc_z (float): Compression capacity major-axis buckling (N).
            Ndc_y (float): Compression capacity minor-axis buckling (N).

        Returns:
            tuple: (UR1, UR2) — both must be <= 1.0 for pass.
        """
        UR1 = (Mz_star / Mdz) ** 2 + (My_star / Mdy) + (Nc_star / Ndc_y)
        UR2 = (Mz_star / Mdz) + (My_star / Mdy) ** 2 + (Nc_star / Ndc_z)
        return UR1, UR2

    # ── Cl 3.5.2 — Combined bending and tension ───────────────────────

    def combined_bending_tension(self, Mx_star, Nt_star, Mdx, Ndt,
                                 k12b, Zx_mm3, A_mm2):
        """
        Combined bending and tension interaction check (Cl 3.5.2).

        Criteria 1 — Eq 3.5(3): k12b * Mx*/Mdx + Nt*/Ndt <= 1
        Criteria 2 — Eq 3.5(4): Mx*/Mdx - (Z/A) * Nt*/Mdx <= 1

        Args:
            Mx_star (float): Design bending moment about relevant axis (N·mm).
            Nt_star (float): Design tension force (N).
            Mdx (float): Bending capacity about relevant axis (N·mm).
            Ndt (float): Tension capacity from tension_capacity() (N).
            k12b (float): Bending stability factor used in bending capacity.
            Zx_mm3 (float): Section modulus about relevant axis (mm³).
            A_mm2 (float): Gross cross-sectional area (mm²).

        Returns:
            tuple: (UR1, UR2) — both must be <= 1.0 for pass.
        """
        UR1 = k12b * Mx_star / Mdx + Nt_star / Ndt
        UR2 = Mx_star / Mdx - (Zx_mm3 / A_mm2) * Nt_star / Mdx
        return UR1, UR2


# Backwards-compatible alias for existing notebooks and engineering cells.
AS_1720_1_2022 = Section3


# ------------------------------------------------------------------
# Section 4 - Timber joints
# ------------------------------------------------------------------

class CoachScrewedJoint:
    """
    Class for calculating design capacities of coach screwed joints according to
    NZS AS 1720.1:2022 standard, section 4.5.
    """

    # Table 4.12 - Factor for multiple coach screwed joints (k17)
    TABLE_4_3_A = 1.0  # Default value, replace with actual table

    TABLE_2_3 = 1.14  # Factor for multiple coach screwed joints (k1) for 5 second effective duration

    def __init__(self, joint_type=None, joint_group=None, is_seasoned=False):
        """
        Initialize the CoachScrewedJoint class.

        Args:
            joint_type (int): Type of joint (1 or 2)
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            is_seasoned (bool): Whether the timber is seasoned
        """
        self.joint_type = joint_type
        self.is_seasoned = is_seasoned

        # Validate joint_group is compatible with is_seasoned
        if joint_group is not None:
            if is_seasoned and not joint_group.startswith("JD"):
                # Convert J1-J6 to JD1-JD6 for seasoned timber
                if joint_group in ["J1", "J2", "J3", "J4", "J5", "J6"]:
                    self.joint_group = "JD" + joint_group[1:]
                    print(
                        f"Warning: Joint group '{joint_group}' converted to '{self.joint_group}' for seasoned timber"
                    )
                else:
                    self.joint_group = joint_group
                    print(
                        f"Warning: Joint group '{joint_group}' may not be compatible with seasoned timber"
                    )
            elif not is_seasoned and joint_group.startswith("JD"):
                # Convert JD1-JD6 to J1-J6 for unseasoned timber
                if joint_group in ["JD1", "JD2", "JD3", "JD4", "JD5", "JD6"]:
                    self.joint_group = "J" + joint_group[2:]
                    print(
                        f"Warning: Joint group '{joint_group}' converted to '{self.joint_group}' for unseasoned timber"
                    )
                else:
                    self.joint_group = joint_group
                    print(
                        f"Warning: Joint group '{joint_group}' may not be compatible with unseasoned timber"
                    )
            else:
                self.joint_group = joint_group
        else:
            self.joint_group = joint_group

        self.create_characteristic_capacity_tables()
        self._create_tables()  # Add this line to create the bolt capacity tables

    def set_joint_properties(self, joint_type, joint_group, is_seasoned=False):
        """
        Set the joint properties.

        Args:
            joint_type (int): Type of joint (1 or 2)
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            is_seasoned (bool): Whether the timber is seasoned
        """
        self.joint_type = joint_type
        self.is_seasoned = is_seasoned

        # Validate joint_group is compatible with is_seasoned
        if joint_group is not None:
            if is_seasoned and not joint_group.startswith("JD"):
                # Convert J1-J6 to JD1-JD6 for seasoned timber
                if joint_group in ["J1", "J2", "J3", "J4", "J5", "J6"]:
                    self.joint_group = "JD" + joint_group[1:]
                    print(
                        f"Warning: Joint group '{joint_group}' converted to '{self.joint_group}' for seasoned timber"
                    )
                else:
                    self.joint_group = joint_group
                    print(
                        f"Warning: Joint group '{joint_group}' may not be compatible with seasoned timber"
                    )
            elif not is_seasoned and joint_group.startswith("JD"):
                # Convert JD1-JD6 to J1-J6 for unseasoned timber
                if joint_group in ["JD1", "JD2", "JD3", "JD4", "JD5", "JD6"]:
                    self.joint_group = "J" + joint_group[2:]
                    print(
                        f"Warning: Joint group '{joint_group}' converted to '{self.joint_group}' for unseasoned timber"
                    )
                else:
                    self.joint_group = joint_group
                    print(
                        f"Warning: Joint group '{joint_group}' may not be compatible with unseasoned timber"
                    )
            else:
                self.joint_group = joint_group
        else:
            self.joint_group = joint_group

        self.create_characteristic_capacity_tables()
        self._create_tables()  # Add this line to create the bolt capacity tables

    def create_characteristic_capacity_tables(self):
        """
        Create pandas DataFrames for the characteristic capacity tables.
        """
        # Table 4.13(A) - Unseasoned timber
        table_4_13A = {
            "Joint Group": ["J1", "J2", "J3", "J4", "J5", "J6"],
            "6": [149, 118, 83, 66, 50, 36],
            "8": [168, 133, 98, 69, 52, 39],
            "10": [189, 152, 112, 77, 58, 42],
            "12": [208, 168, 124, 83, 68, 48],
            "16": [241, 193, 143, 98, 77, 58],
            "20": [270, 218, 162, 112, 83, 64],
        }

        # Table 4.13(B) - Seasoned timber
        table_4_13B = {
            "Joint Group": ["JD1", "JD2", "JD3", "JD4", "JD5", "JD6"],
            "6": [185, 147, 104, 83, 62, 48],
            "8": [210, 166, 122, 87, 66, 48],
            "10": [232, 191, 141, 97, 73, 54],
            "12": [261, 210, 154, 104, 85, 60],
            "16": [301, 241, 179, 124, 97, 73],
            "20": [338, 272, 205, 139, 104, 79],
        }

        # Create DataFrames
        self.df_unseasoned = pd.DataFrame(table_4_13A)
        self.df_seasoned = pd.DataFrame(table_4_13B)

        # Set 'Joint Group' as index for easier lookup
        self.df_unseasoned.set_index("Joint Group", inplace=True)
        self.df_seasoned.set_index("Joint Group", inplace=True)

        # Create DataFrame for maximum tensile capacity (Table 4.14)
        data_tensile = {
            "Diameter": [6, 8, 10, 12, 16, 20],
            "Maximum Tensile Load": [3900, 7700, 11600, 17400, 39000, 61000],
        }
        self.df_tensile = pd.DataFrame(data_tensile)
        self.df_tensile.set_index("Diameter", inplace=True)

    def get_characteristic_capacity_withdrawal_df(
        self, screw_diameter, joint_group=None
    ):
        """
        Get the characteristic capacity for a single coach screw axially loaded in withdrawal
        using pandas DataFrames.

        Args:
            screw_diameter (int or str): Diameter of the coach screw in mm or as string (e.g., "M12")
            joint_group (str, optional): Joint group. If None, uses the instance's joint_group

        Returns:
            float: Characteristic capacity in N per mm penetration of thread
        """
        if joint_group is None:
            joint_group = self.joint_group

        # Handle string format with 'M' prefix (e.g., "M12")
        if isinstance(screw_diameter, str) and screw_diameter.startswith("M"):
            screw_diameter = int(screw_diameter[1:])

        if screw_diameter not in [6, 8, 10, 12, 16, 20]:
            raise ValueError("Screw diameter must be one of: 6, 8, 10, 12, 16, 20 mm")

        # Convert screw_diameter to string for DataFrame lookup
        screw_diameter_str = str(screw_diameter)

        try:
            if self.is_seasoned:
                return self.df_seasoned.loc[joint_group, screw_diameter_str]
            else:
                return self.df_unseasoned.loc[joint_group, screw_diameter_str]
        except KeyError:
            valid_groups = (
                self.df_seasoned.index.tolist()
                if self.is_seasoned
                else self.df_unseasoned.index.tolist()
            )
            raise ValueError(
                f"Joint group '{joint_group}' not valid. Valid groups are: {valid_groups}"
            )

    def get_maximum_tensile_capacity_df(self, screw_diameter):
        """
        Get the maximum tensile capacity for a coach screw subject to direct axial loading
        using pandas DataFrame.

        Args:
            screw_diameter (int or str): Diameter of the coach screw in mm or as string (e.g., "M12")

        Returns:
            float: Maximum tensile capacity in N
        """
        # Handle string format with 'M' prefix (e.g., "M12")
        if isinstance(screw_diameter, str) and screw_diameter.startswith("M"):
            screw_diameter = int(screw_diameter[1:])

        if screw_diameter not in [6, 8, 10, 12, 16, 20]:
            raise ValueError("Screw diameter must be one of: 6, 8, 10, 12, 16, 20 mm")

        return self.df_tensile.loc[screw_diameter, "Maximum Tensile Load"]

    def _create_tables(self):
        """Create all the tables needed for capacity calculations."""
        # Table 4.9(B) - Characteristic capacity for single bolts parallel to grain - unseasoned timber
        self.table_4_9B = self._create_table_4_9B()

        # Table 4.9(C) - Characteristic capacity for single bolts parallel to grain - seasoned timber
        self.table_4_9C = self._create_table_4_9C()

        # Table 4.10(B) - Characteristic capacity for single bolts perpendicular to grain - unseasoned timber
        self.table_4_10B = self._create_table_4_10B()

        # Table 4.10(C) - Characteristic capacity for single bolts perpendicular to grain - seasoned timber
        self.table_4_10C = self._create_table_4_10C()

    def _create_table_4_9B(self):
        """Create Table 4.9(B) - Characteristic capacity for single bolts parallel to grain - unseasoned timber."""
        # Joint group J1
        j1_data = {
            25: [3300, 5900, 9200, 13300, 23400, 36600, 50000, 82400, 118700],
            38: [3300, 5900, 9200, 13300, 23400, 36600, 50000, 82400, 118700],
            50: [3300, 5900, 9200, 13300, 23400, 36600, 50000, 82400, 118700],
            75: [3300, 5900, 9200, 13300, 23400, 36600, 52700, 82400, 118700],
            100: [3300, 5900, 9200, 13300, 23400, 36600, 52700, 82400, 118700],
            150: [3300, 5900, 9200, 13300, 23400, 36600, 52700, 82400, 118700],
            200: [3300, 5900, 9200, 13300, 23400, 36600, 52700, 82400, 118700],
        }

        # Joint group J2
        j2_data = {
            25: [2800, 4400, 7500, 10600, 18800, 31000, 43200, 65300, 95800],
            38: [2800, 4900, 7700, 10600, 18800, 31000, 43200, 65300, 95800],
            50: [2800, 4900, 7700, 11100, 19700, 30800, 43200, 65300, 95800],
            75: [2800, 4900, 7700, 11100, 19700, 30800, 43200, 65300, 95800],
            100: [2800, 4900, 7700, 11100, 19700, 30800, 43200, 65300, 95800],
            150: [2800, 4900, 7700, 11100, 19700, 30800, 44400, 69300, 99800],
            200: [2800, 4900, 7700, 11100, 19700, 30800, 44400, 69300, 99800],
        }

        # Joint group J3
        j3_data = {
            25: [2600, 4600, 6400, 9300, 16200, 25800, 39900, 53900, 76000],
            38: [2600, 4500, 7100, 10200, 16200, 25800, 39900, 53900, 76000],
            50: [2600, 4500, 7100, 10200, 16200, 25800, 39900, 53900, 76000],
            75: [2600, 4500, 7100, 10200, 16200, 25800, 39900, 53900, 76000],
            100: [2600, 4500, 7100, 10200, 16200, 28400, 40900, 53900, 76000],
            150: [2600, 4500, 7100, 10200, 16200, 28400, 40900, 53900, 76000],
            200: [2600, 4500, 7100, 10500, 18200, 28400, 40900, 63900, 92000],
        }

        # Joint group J4
        j4_data = {
            25: [2000, 3600, 5500, 8100, 14300, 21000, 31500, 42000, 60000],
            38: [2000, 3600, 5500, 8100, 14300, 21000, 31500, 42000, 60000],
            50: [2000, 3600, 5600, 8100, 14300, 21000, 31500, 42000, 60000],
            75: [2000, 3600, 5600, 8100, 14300, 21000, 31500, 42000, 60000],
            100: [2000, 3600, 5600, 8100, 14300, 22400, 32300, 42000, 60000],
            150: [2000, 3600, 5600, 8100, 14300, 22400, 32300, 50400, 72600],
            200: [2000, 3600, 5600, 8100, 14300, 22400, 32300, 50400, 72600],
        }

        # Joint group J5
        j5_data = {
            25: [1700, 3200, 4800, 7000, 12400, 19400, 27900, 37000, 53000],
            38: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 37000, 53000],
            50: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 37000, 53000],
            75: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 37000, 53000],
            100: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 37000, 53000],
            150: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 43600, 62700],
            200: [1700, 3100, 4800, 7000, 12400, 19400, 27900, 43600, 62700],
        }

        # Joint group J6
        j6_data = {
            25: [1400, 1800, 2300, 2700, 3600, 4500, 5400, 6800, 8100],
            38: [1600, 2700, 4300, 6100, 10800, 16900, 24200, 32000, 46000],
            50: [1600, 2800, 4300, 6200, 10800, 16900, 24200, 32000, 46000],
            75: [1600, 2800, 4300, 6200, 10800, 16900, 24200, 32000, 46000],
            100: [1600, 2800, 4300, 6200, 11100, 17300, 24600, 32000, 46000],
            150: [1600, 2800, 4300, 6200, 11100, 17300, 24600, 37800, 54600],
            200: [1600, 2800, 4300, 6200, 11100, 17300, 24900, 37800, 54600],
        }

        # Create a multi-level dictionary
        table_data = {
            "J1": j1_data,
            "J2": j2_data,
            "J3": j3_data,
            "J4": j4_data,
            "J5": j5_data,
            "J6": j6_data,
        }

        # Create a multi-index DataFrame
        bolt_diameters = ["M6", "M8", "M10", "M12", "M16", "M20", "M24", "M30", "M36"]

        # Initialize an empty DataFrame
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [table_data.keys(), table_data["J1"].keys()],
                names=["Joint group", "Effective timber thickness (beff)"],
            ),
            columns=bolt_diameters,
        )

        # Fill the DataFrame
        for joint_group, thickness_data in table_data.items():
            for thickness, values in thickness_data.items():
                for i, bolt_diameter in enumerate(bolt_diameters):
                    df.loc[(joint_group, thickness), bolt_diameter] = values[i]

        return df

    def _create_table_4_9C(self):
        """Create Table 4.9(C) - Characteristic capacity for single bolts parallel to grain - seasoned timber."""
        # Joint group JD1
        jd1_data = {
            25: [4100, 6900, 8600, 10400, 13800, 17300, 20700, 25900, 31100],
            35: [4100, 6900, 8600, 10400, 13800, 17300, 20700, 25900, 31100],
            45: [4100, 7300, 11400, 16400, 24800, 31600, 37300, 46600, 55900],
            70: [4100, 7300, 11400, 16400, 29100, 45500, 58600, 72500, 111800],
            90: [4100, 7300, 11400, 16400, 29100, 45500, 65600, 93200, 111800],
            105: [4100, 7300, 11400, 16400, 29100, 45500, 65600, 102500, 139400],
            120: [4100, 7300, 11400, 16400, 29100, 45500, 65600, 102500, 147500],
        }

        # Joint group JD2
        jd2_data = {
            25: [3500, 6200, 9700, 11700, 15500, 19500, 23300, 29100, 35000],
            35: [3500, 6200, 9700, 11700, 15500, 19500, 23300, 29100, 35000],
            45: [3500, 6200, 9700, 14000, 20000, 25000, 30000, 37500, 45000],
            70: [3500, 6200, 9700, 14000, 24900, 38900, 46600, 58300, 69900],
            90: [3500, 6200, 9700, 14000, 24900, 38900, 55900, 74700, 89900],
            105: [3500, 6200, 9700, 14000, 24900, 38900, 55900, 87400, 104900],
            120: [3500, 6200, 9700, 14000, 24900, 38900, 55900, 87400, 118100],
        }

        # Joint group JD3
        jd3_data = {
            25: [3200, 4400, 5500, 6600, 8800, 11000, 13200, 16500, 19800],
            35: [3200, 5600, 7700, 9200, 12300, 15400, 18500, 23100, 27700],
            45: [3200, 5600, 8800, 10800, 14400, 18000, 21600, 27000, 32400],
            70: [3200, 5600, 8800, 12700, 22500, 30800, 37000, 46200, 55400],
            90: [3200, 5600, 8800, 12700, 22500, 35200, 47500, 59400, 71300],
            105: [3200, 5600, 8800, 12700, 22500, 35200, 50700, 69300, 83200],
            120: [3200, 5600, 8800, 12700, 22500, 35200, 50700, 79200, 95000],
        }

        # Joint group JD4
        jd4_data = {
            25: [2600, 3600, 4400, 5300, 7100, 8900, 10700, 13300, 16000],
            35: [2600, 4500, 6200, 7500, 9900, 12400, 14900, 18600, 22300],
            45: [2600, 4500, 7100, 8500, 11400, 14200, 17000, 21300, 25600],
            70: [2600, 4500, 7100, 10200, 18200, 24900, 29800, 37300, 44700],
            90: [2600, 4500, 7100, 10200, 18200, 28400, 38300, 47900, 57500],
            105: [2600, 4500, 7100, 10200, 18200, 28400, 40900, 55900, 67100],
            120: [2600, 4500, 7100, 10200, 18200, 28400, 40900, 63900, 76700],
        }

        # Joint group JD5
        jd5_data = {
            25: [2100, 2800, 3500, 4200, 5600, 7000, 8400, 10500, 12600],
            35: [2100, 3500, 4900, 5900, 7800, 9800, 11800, 14700, 17600],
            45: [2100, 3500, 5600, 6700, 9000, 11200, 13400, 16800, 20200],
            70: [2100, 3500, 5600, 8000, 14300, 19600, 23500, 29400, 35300],
            90: [2100, 3500, 5600, 8000, 14300, 22400, 30200, 37800, 45400],
            105: [2100, 3500, 5600, 8000, 14300, 22400, 32300, 44100, 52900],
            120: [2100, 3500, 5600, 8000, 14300, 22400, 32300, 50400, 60500],
        }

        # Joint group JD6
        jd6_data = {
            25: [1700, 2300, 2800, 3400, 4500, 5600, 6700, 8400, 10100],
            35: [1700, 2800, 3900, 4700, 6200, 7800, 9400, 11800, 14100],
            45: [1700, 2800, 4500, 5400, 7200, 9000, 10800, 13500, 16200],
            70: [1700, 2800, 4500, 6400, 11500, 15700, 18900, 23600, 28300],
            90: [1700, 2800, 4500, 6400, 11500, 18000, 24300, 30300, 36400],
            105: [1700, 2800, 4500, 6400, 11500, 18000, 25900, 35400, 42500],
            120: [1700, 2800, 4500, 6400, 11500, 18000, 25900, 40400, 48500],
        }

        # Create a multi-level dictionary
        table_data = {
            "JD1": jd1_data,
            "JD2": jd2_data,
            "JD3": jd3_data,
            "JD4": jd4_data,
            "JD5": jd5_data,
            "JD6": jd6_data,
        }

        # Create a multi-index DataFrame
        bolt_diameters = ["M6", "M8", "M10", "M12", "M16", "M20", "M24", "M30", "M36"]

        # Initialize an empty DataFrame
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [table_data.keys(), table_data["JD1"].keys()],
                names=["Joint group", "Effective timber thickness (beff)"],
            ),
            columns=bolt_diameters,
        )

        # Fill the DataFrame
        for joint_group, thickness_data in table_data.items():
            for thickness, values in thickness_data.items():
                for i, bolt_diameter in enumerate(bolt_diameters):
                    df.loc[(joint_group, thickness), bolt_diameter] = values[i]

        return df

    def _create_table_4_10B(self):
        """Create Table 4.10(B) - Characteristic capacity for single bolts perpendicular to grain - unseasoned timber."""
        # Joint group J1
        j1_data = {
            25: [1650, 2200, 2750, 3300, 4400, 5500, 6600, 8250, 9900],
            38: [2530, 3380, 4190, 5020, 6690, 8360, 10030, 12540, 15050],
            50: [3230, 4300, 5360, 6430, 8560, 10700, 13200, 16500, 19800],
            75: [3230, 4980, 6960, 9150, 13380, 16600, 19800, 24750, 29700],
            100: [3230, 4980, 6960, 9150, 13380, 16600, 19800, 24750, 29700],
            150: [3230, 4980, 6960, 9150, 14080, 19690, 25870, 36150, 47520],
            200: [3230, 4980, 6960, 9150, 14080, 19690, 25870, 36150, 45360],
        }

        # Joint group J2
        j2_data = {
            25: [1310, 1750, 2190, 2630, 3500, 4380, 5250, 6560, 7880],
            38: [2000, 2660, 3330, 3990, 5320, 6650, 7980, 9980, 11970],
            50: [2630, 3510, 4380, 5260, 7010, 8750, 10500, 13130, 15750],
            75: [3090, 4750, 6560, 7880, 10500, 13130, 15750, 19690, 23630],
            100: [3090, 4750, 6560, 7880, 10500, 13130, 15750, 19690, 23630],
            150: [3090, 4750, 6560, 8750, 13440, 18780, 24690, 34510, 45360],
            200: [3090, 4750, 6560, 8750, 13440, 18780, 24690, 34510, 45360],
        }

        # Joint group J3
        j3_data = {
            25: [830, 1100, 1380, 1650, 2200, 2750, 3300, 4130, 4950],
            38: [1250, 1670, 2080, 2500, 3330, 4160, 5000, 6250, 7500],
            50: [1650, 2200, 2750, 3300, 4400, 5500, 6600, 8250, 9900],
            75: [2420, 3730, 5230, 6600, 8800, 11000, 13200, 16500, 19800],
            100: [2420, 3730, 5230, 6600, 8800, 11000, 13200, 16500, 19800],
            150: [2420, 3730, 5230, 6600, 8800, 11000, 13200, 19800, 27110],
            200: [2420, 3730, 5230, 6600, 10560, 14760, 19400, 27110, 35640],
        }

        # Joint group J4
        j4_data = {
            25: [530, 710, 890, 1070, 1420, 1780, 2130, 2660, 3200],
            38: [810, 1080, 1350, 1620, 2160, 2700, 3240, 4050, 4860],
            50: [1070, 1420, 1780, 2130, 2840, 3550, 4260, 5330, 6390],
            75: [1600, 2130, 2660, 3200, 4260, 5330, 6390, 7990, 9590],
            100: [1770, 2730, 3820, 4260, 5680, 7100, 8520, 10650, 12780],
            150: [1770, 2730, 3820, 5020, 7720, 10650, 14780, 19170, 25780],
            200: [1770, 2730, 3820, 5020, 7720, 10800, 14190, 19830, 26070],
        }

        # Joint group J5
        j5_data = {
            25: [350, 470, 590, 710, 940, 1180, 1410, 1760, 2120],
            38: [540, 710, 890, 1070, 1430, 1790, 2140, 2680, 3210],
            50: [710, 940, 1180, 1410, 1880, 2350, 2820, 3530, 4230],
            75: [1060, 1410, 1760, 2120, 2820, 3530, 4230, 5290, 6350],
            100: [1180, 1820, 2540, 2820, 3760, 4700, 5640, 7050, 8460],
            150: [1310, 2020, 2820, 3710, 5640, 7050, 9460, 12690, 17090],
            200: [1310, 2020, 2820, 3710, 5640, 7990, 10520, 14720, 19320],
        }

        # Joint group J6
        j6_data = {
            25: [180, 240, 300, 360, 480, 600, 720, 900, 1080],
            38: [270, 360, 460, 550, 730, 910, 1090, 1370, 1640],
            50: [360, 480, 600, 720, 960, 1200, 1440, 1800, 2160],
            75: [540, 720, 900, 1080, 1440, 1800, 2160, 2700, 3240],
            100: [720, 960, 1200, 1440, 1920, 2400, 2880, 3600, 4320],
            150: [780, 1190, 1670, 2160, 2880, 3600, 4320, 5400, 6480],
            200: [780, 1190, 1670, 2160, 3380, 4720, 6200, 8640, 11340],
        }

        # Create a multi-level dictionary
        table_data = {
            "J1": j1_data,
            "J2": j2_data,
            "J3": j3_data,
            "J4": j4_data,
            "J5": j5_data,
            "J6": j6_data,
        }

        # Create a multi-index DataFrame
        bolt_diameters = ["M6", "M8", "M10", "M12", "M16", "M20", "M24", "M30", "M36"]

        # Initialize an empty DataFrame
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [table_data.keys(), table_data["J1"].keys()],
                names=["Joint group", "Effective timber thickness (beff)"],
            ),
            columns=bolt_diameters,
        )

        # Fill the DataFrame
        for joint_group, thickness_data in table_data.items():
            for thickness, values in thickness_data.items():
                for i, bolt_diameter in enumerate(bolt_diameters):
                    df.loc[(joint_group, thickness), bolt_diameter] = values[i]

        return df

    def _create_table_4_10C(self):
        """Create Table 4.10(C) - Characteristic capacity for single bolts perpendicular to grain - seasoned timber."""
        # Joint group JD1
        jd1_data = {
            25: [2310, 3080, 3850, 4620, 6160, 7700, 9240, 11550, 13860],
            35: [3100, 4130, 5160, 6200, 8260, 10330, 12390, 15490, 18590],
            45: [3540, 4720, 5900, 7080, 9440, 11800, 14160, 17700, 21240],
            70: [4340, 6680, 9330, 12260, 16520, 20650, 24780, 30980, 37170],
            90: [4340, 6680, 9330, 12260, 18880, 26390, 34780, 47790, 63720],
            105: [4340, 6680, 9330, 12260, 18880, 26390, 34780, 46460, 55760],
            120: [4340, 6680, 9330, 12260, 18880, 26390, 34780, 48470, 63720],
        }

        # Joint group JD2
        jd2_data = {
            25: [1690, 2250, 2810, 3380, 4500, 5630, 6750, 8440, 10130],
            35: [2260, 3020, 3770, 4530, 6040, 7550, 9060, 11320, 13590],
            45: [2590, 3450, 4310, 5170, 6900, 8620, 10350, 12940, 15520],
            70: [3970, 6110, 8540, 11220, 15120, 18900, 22680, 28350, 34020],
            90: [3970, 6110, 8540, 11220, 17280, 24150, 31830, 42530, 50240],
            105: [3970, 6110, 8540, 11220, 17280, 24150, 31830, 42530, 50240],
            120: [3970, 6110, 8540, 11220, 17280, 24150, 31830, 42530, 50240],
        }

        # Joint group JD3
        jd3_data = {
            25: [1280, 1700, 2130, 2550, 3400, 4250, 5100, 6380, 7650],
            35: [1700, 2270, 2840, 3410, 4540, 5680, 6810, 8510, 10220],
            45: [1950, 2600, 3250, 3900, 5200, 6500, 7800, 9750, 11700],
            70: [3570, 4760, 5950, 7140, 9520, 11900, 14280, 17850, 21420],
            90: [3750, 5770, 8080, 10660, 14280, 17850, 21420, 26780, 32130],
            105: [3750, 5770, 8080, 10660, 16320, 22850, 30120, 40160, 48190],
            120: [3750, 5770, 8080, 10660, 16320, 22850, 30120, 40160, 48190],
        }

        # Joint group JD4
        jd4_data = {
            25: [940, 1250, 1560, 1880, 2500, 3130, 3750, 4690, 5630],
            35: [1310, 1750, 2190, 2630, 3500, 4380, 5250, 6560, 7880],
            45: [1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 9000],
            70: [2830, 3500, 4380, 5250, 7000, 8750, 10500, 13130, 15750],
            90: [3120, 4810, 6720, 8830, 11900, 14880, 17850, 22310, 26780],
            105: [3120, 4810, 6720, 8830, 13600, 19040, 25090, 33450, 40140],
            120: [3120, 4810, 6720, 8830, 13600, 19040, 25090, 33450, 40140],
        }

        # Joint group JD5
        jd5_data = {
            25: [680, 900, 1130, 1350, 1800, 2250, 2700, 3380, 4050],
            35: [950, 1270, 1590, 1900, 2540, 3170, 3810, 4760, 5710],
            45: [1080, 1440, 1800, 2160, 2880, 3600, 4320, 5400, 6480],
            70: [1220, 1880, 2630, 3460, 5200, 7800, 10400, 13000, 15600],
            90: [2430, 3240, 4050, 4860, 6480, 8100, 9720, 12150, 14580],
            105: [2510, 3770, 5280, 6940, 9720, 12150, 14580, 18230, 21870],
            120: [2510, 3770, 5280, 6940, 10880, 15230, 20070, 26760, 32110],
        }

        # Joint group JD6
        jd6_data = {
            25: [460, 610, 760, 920, 1220, 1530, 1830, 2290, 2750],
            35: [640, 850, 1070, 1280, 1710, 2140, 2560, 3200, 3840],
            45: [730, 970, 1220, 1460, 1950, 2430, 2920, 3650, 4380],
            70: [1280, 1710, 2140, 2560, 3410, 4270, 5120, 6400, 7680],
            90: [1650, 2200, 2750, 3290, 4390, 5490, 6590, 8230, 9880],
            105: [1970, 2930, 4100, 5390, 7680, 9600, 11520, 14400, 17280],
            120: [1970, 2930, 4100, 5390, 8460, 11840, 15600, 20800, 24960],
        }

        # Create a multi-level dictionary
        table_data = {
            "JD1": jd1_data,
            "JD2": jd2_data,
            "JD3": jd3_data,
            "JD4": jd4_data,
            "JD5": jd5_data,
            "JD6": jd6_data,
        }

        # Create a multi-index DataFrame
        bolt_diameters = ["M6", "M8", "M10", "M12", "M16", "M20", "M24", "M30", "M36"]

        # Initialize an empty DataFrame
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [table_data.keys(), table_data["JD1"].keys()],
                names=["Joint group", "Effective timber thickness (beff)"],
            ),
            columns=bolt_diameters,
        )

        # Fill the DataFrame
        for joint_group, thickness_data in table_data.items():
            for thickness, values in thickness_data.items():
                for i, bolt_diameter in enumerate(bolt_diameters):
                    df.loc[(joint_group, thickness), bolt_diameter] = values[i]

        return df

    def get_characteristic_capacity_parallel(
        self, joint_group, effective_thickness, bolt_diameter, is_seasoned=False
    ):
        """
        Get the characteristic capacity for a single bolt loaded parallel to the grain.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: Characteristic capacity in N
        """
        # Select the appropriate table
        table = self.table_4_9C if is_seasoned else self.table_4_9B

        # Find the closest thickness value in the table
        available_thicknesses = [
            t
            for t in table.index.get_level_values(1).unique()
            if t <= effective_thickness
        ]
        if not available_thicknesses:
            raise ValueError(
                f"Effective thickness {effective_thickness} mm is too small for the available data"
            )

        closest_thickness = max(available_thicknesses)

        # Get the capacity
        try:
            return table.loc[(joint_group, closest_thickness), bolt_diameter]
        except KeyError:
            valid_groups = (
                self.table_4_9C.index.tolist()
                if is_seasoned
                else self.table_4_9B.index.tolist()
            )
            raise ValueError(
                f"Joint group '{joint_group}' not valid. Valid groups are: {valid_groups}"
            )

    def get_characteristic_capacity_perpendicular(
        self, joint_group, effective_thickness, bolt_diameter, is_seasoned=False
    ):
        """
        Get the characteristic capacity for a single bolt loaded perpendicular to the grain.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: Characteristic capacity in N
        """
        # Select the appropriate table
        table = self.table_4_10C if is_seasoned else self.table_4_10B

        # Find the closest thickness value in the table
        available_thicknesses = [
            t
            for t in table.index.get_level_values(1).unique()
            if t <= effective_thickness
        ]
        if not available_thicknesses:
            raise ValueError(
                f"Effective thickness {effective_thickness} mm is too small for the available data"
            )

        closest_thickness = max(available_thicknesses)

        # Get the capacity
        try:
            return table.loc[(joint_group, closest_thickness), bolt_diameter]
        except KeyError:
            valid_groups = (
                self.table_4_10C.index.tolist()
                if is_seasoned
                else self.table_4_10B.index.tolist()
            )
            raise ValueError(
                f"Joint group '{joint_group}' not valid. Valid groups are: {valid_groups}"
            )

    def get_characteristic_capacity_angle(
        self,
        joint_group,
        effective_thickness,
        bolt_diameter,
        angle_degrees,
        is_seasoned=None,
    ):
        """
        Get the characteristic capacity for a single bolt loaded at an angle to the grain
        using Hankinson's formula.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            angle_degrees (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool, optional): Whether the timber is seasoned. If None, uses the instance's is_seasoned.

        Returns:
            float: Characteristic capacity in N
        """
        # Use instance is_seasoned if not provided
        if is_seasoned is None:
            is_seasoned = self.is_seasoned

        # Get capacities parallel and perpendicular to grain
        Q_parallel = self.get_characteristic_capacity_parallel(
            joint_group, effective_thickness, bolt_diameter, is_seasoned
        )
        Q_perpendicular = self.get_characteristic_capacity_perpendicular(
            joint_group, effective_thickness, bolt_diameter, is_seasoned
        )

        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)

        # Apply Hankinson's formula (Equation 4.4(1))
        numerator = Q_parallel * Q_perpendicular
        denominator = Q_parallel * (np.sin(angle_radians) ** 2) + Q_perpendicular * (
            np.cos(angle_radians) ** 2
        )

        return numerator / denominator

    def get_system_capacity(
        self,
        joint_configuration,
        effective_thickness,
        bolt_diameter,
        joint_group,
        angle_degrees=0,
        is_seasoned=False,
    ):
        """
        Calculate the system capacity for a bolted joint system.

        Args:
            joint_configuration (str): Type of joint configuration ('two_member', 'three_member_A',
                                      'three_member_B', 'multiple_member')
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            angle_degrees (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: System capacity in N
        """
        # Get the characteristic capacity for a single bolt at the specified angle
        if angle_degrees == 0:
            # Parallel to grain
            Q_k = self.get_characteristic_capacity_parallel(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        elif angle_degrees == 90:
            # Perpendicular to grain
            Q_k = self.get_characteristic_capacity_perpendicular(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        else:
            # At an angle to grain
            Q_k = self.get_characteristic_capacity_angle(
                joint_group,
                effective_thickness,
                bolt_diameter,
                angle_degrees,
                is_seasoned,
            )

        # Calculate system capacity based on joint configuration
        if joint_configuration == "two_member":
            system_capacity = Q_k
        elif joint_configuration == "three_member":
            system_capacity = 2 * Q_k
        elif joint_configuration == "multiple_member":
            # For multiple member joints, the system capacity is the sum of the basic loads
            system_capacity = (
                Q_k  # This should be modified based on the actual number of interfaces
            )
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        return system_capacity

    def calculate_joint_capacity(
        self,
        joint_configuration,
        member_thicknesses,
        bolt_diameter,
        joint_group,
        load_angle=0,
        is_seasoned=False,
    ):
        """
        Calculate the capacity of a bolted joint system.

        Args:
            joint_configuration (str): Type of joint configuration ('two_member', 'three_member',
                                      'multiple_member')
            member_thicknesses (list): List of thicknesses for each member in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            load_angle (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: Joint capacity in N
        """
        # Determine effective thickness based on joint configuration and member thicknesses
        if joint_configuration == "two_member":
            # For two-member joints, effective thickness is the smaller of t1 and t2
            effective_thickness = min(member_thicknesses)
        elif joint_configuration == "three_member":
            # For three-member joints, effective thickness is t1
            effective_thickness = member_thicknesses[0]
        elif joint_configuration == "multiple_member":
            # For multiple member joints, effective thickness depends on the specific configuration
            # This is a simplification
            effective_thickness = min(member_thicknesses)
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        # Calculate system capacity
        system_capacity = self.get_system_capacity(
            joint_configuration,
            effective_thickness,
            bolt_diameter,
            joint_group,
            load_angle,
            is_seasoned,
        )

        return system_capacity

    def design_capacity_type1_joint_4_5_3_1(
        self,
        n_screws,
        screw_diameter,
        is_side_grain=True,
        has_metal_side_plates=False,
        joint_configuration="two_member",
        member_thicknesses=[50],
        joint_group=None,
        angle_degrees=0,
        is_seasoned=False,
    ):
        """
        Calculate the design capacity for Type 1 joints to resist shear loads.

        Args:
            n_screws (int): Number of coach screws in the connection
            screw_diameter (int or str): Diameter of the coach screw in mm or as string (e.g., "M12")
            is_side_grain (bool): Whether the coach screws are in side grain (True) or end grain (False)
            has_metal_side_plates (bool): Whether the load is applied through metal side plates
            joint_configuration (str): Type of joint configuration ('two_member', 'three_member', etc.)
            member_thicknesses (list): List of thicknesses for each member in mm
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            angle_degrees (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: Design capacity (Nd,1) in N
        """
        phi = 0.9
        k1 = 1.14
        if self.joint_type != 1:
            raise ValueError("This method is only applicable for Type 1 joints")

        if joint_group is None:
            joint_group = self.joint_group

        # Determine effective thickness based on joint configuration
        if joint_configuration == "two_member":
            # For two-member joints, effective thickness is the smaller of t1 and t2
            effective_thickness = min(member_thicknesses)
        elif joint_configuration == "three_member":
            # For three-member joints, effective thickness is t1
            effective_thickness = member_thicknesses[1]
        elif joint_configuration == "multiple_member":
            # For multiple member joints, effective thickness depends on the specific configuration
            effective_thickness = min(member_thicknesses)
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        # Format bolt diameter string
        # Handle string format with 'M' prefix (e.g., "M12")
        if isinstance(screw_diameter, str) and screw_diameter.startswith("M"):
            bolt_diameter = screw_diameter
        else:
            bolt_diameter = f"M{screw_diameter}"

        # Calculate characteristic capacity Qk using the appropriate tables
        if angle_degrees == 0:
            # Parallel to grain
            Qk = self.get_characteristic_capacity_parallel(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        elif angle_degrees == 90:
            # Perpendicular to grain
            Qk = self.get_characteristic_capacity_perpendicular(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        else:
            # At an angle to grain
            Qk = self.get_characteristic_capacity_angle(
                joint_group,
                effective_thickness,
                bolt_diameter,
                angle_degrees,
                is_seasoned,
            )

        # Calculate system capacity based on joint configuration
        if joint_configuration == "two_member":
            system_capacity = Qk
        elif joint_configuration == "three_member":
            system_capacity = 2 * Qk
        elif joint_configuration == "multiple_member":
            # For multiple member joints, the system capacity is the sum of the basic loads
            system_capacity = (
                Qk  # This should be modified based on the actual number of interfaces
            )
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        # Factor k13 based on grain direction
        k13 = 1.0 if is_side_grain else 0.6

        # Factor k16 based on metal side plates
        k16 = 2.0 if has_metal_side_plates else 1.0

        # Factor k17 for multiple coach screwed joints
        k17 = self.TABLE_4_3_A  # This would need to be looked up from Table 4.12

        # Calculate design capacity according to Equation 4.5(2)
        Nd1 = phi * k1 * k13 * k16 * k17 * n_screws * system_capacity

        return Nd1

    def design_capacity_type2_joint_4_5_3_2(
        self,
        n_screws,
        screw_diameter,
        penetration_depth,
        is_side_grain,
    ):
        """
        Calculate the design capacity for Type 2 joints axially loaded in withdrawal
        using pandas DataFrames.

        Args:
            n_screws (int): Number of coach screws in the connection
            screw_diameter (int or str): Diameter of the coach screw in mm or as string (e.g., "M12")
            penetration_depth (float): Depth of penetration of the threaded portion in mm
            is_side_grain (bool): Whether withdrawal is from side grain (True) or end grain (False)

        Returns:
            float: Design capacity (Nd,j) in N
        """
        phi = 0.9
        if self.joint_type != 2:
            raise ValueError("This method is only applicable for Type 2 joints")

        # Get characteristic capacity per mm penetration
        Qk_per_mm = self.get_characteristic_capacity_withdrawal_df(screw_diameter)

        # Calculate total characteristic capacity
        Qk = Qk_per_mm * penetration_depth

        # Factor k13 based on grain direction
        k13 = 1.0 if is_side_grain else 0.6

        # Maximum tensile capacity
        Ntm = self.get_maximum_tensile_capacity_df(screw_diameter)

        # Calculate design capacity according to Equations 4.5(4), 4.5(5), and 4.5(6)
        Ndj_1 = n_screws * Ntm
        Ndj_2 = phi * k13 * n_screws * Qk

        # Design capacity is the lesser of the three values
        Ndj = min(Ndj_1, Ndj_2)

        return Ndj

    def check_lateral_load_conditions(
        self, screw_diameter, washer_diameter=None, t1=None, t2=None, hole_diameter=None
    ):
        """
        Check if the conditions for lateral loads in Type 1 joints are met.

        Args:
            screw_diameter (int): Diameter of the coach screw in mm
            washer_diameter (float): Diameter of the washer in mm
            t1 (float): Thickness of outermost member in mm
            t2 (float): Thickness of second member in mm
            hole_diameter (float): Diameter of the hole in mm

        Returns:
            bool: Whether all conditions are met
        """
        conditions_met = True
        messages = []

        # Condition (i): Coach screw shall be considered to be a bolt of diameter equal to the shank diameter
        # This is just informational

        # Condition (ii): Coach screws shall be fitted with washers as specified in Clause 4.4.5
        if washer_diameter is None:
            conditions_met = False
            messages.append("Washers must be fitted as specified in Clause 4.4.5")

        # Condition (iii): In a two-member joint, the thinner member shall have a minimum thickness of three times the shank diameter
        if t1 is not None and t2 is not None:
            thinner_member = min(t1, t2)
            if thinner_member < 3 * screw_diameter:
                conditions_met = False
                messages.append(
                    f"Thinner member thickness ({thinner_member} mm) is less than required minimum ({3 * screw_diameter} mm)"
                )

        # Condition (iv): The diameter of the hole shall be not less than the shank diameter nor exceed it by more than 1 mm or 10%
        if hole_diameter is not None:
            if hole_diameter < screw_diameter:
                conditions_met = False
                messages.append(
                    f"Hole diameter ({hole_diameter} mm) is less than shank diameter ({screw_diameter} mm)"
                )
            max_allowed = max(screw_diameter + 1, screw_diameter * 1.1)
            if hole_diameter > max_allowed:
                conditions_met = False
                messages.append(
                    f"Hole diameter ({hole_diameter} mm) exceeds maximum allowed ({max_allowed} mm)"
                )

        # Return results
        return conditions_met, messages


class BoltedJointSystem:
    """
    Class for calculating system capacities of bolted joints according to
    NZS AS 1720.1:2022 standard, section 4.4.2.4.
    """

    # Table 4.12 - Factor for multiple bolted joints (k17)
    TABLE_4_3_A = (
        1.0  # Default value for k17, should be updated based on the number of rows
    )

    # Table C6 - Values of f'pj for bolted joints
    TABLE_C6 = {
        # Unseasoned timber
        "J1": 22.0,
        "J2": 17.5,
        "J3": 11.0,
        "J4": 7.1,
        "J5": 4.7,
        "J6": 2.4,
        # Seasoned timber
        "JD1": 29.0,
        "JD2": 22.5,
        "JD3": 17.0,
        "JD4": 12.5,
        "JD5": 9.0,
        "JD6": 6.1,
    }

    def __init__(self, joint_type=None, joint_group=None, is_seasoned=False):
        """
        Initialize the BoltedJointSystem class.

        Args:
            joint_type (int): Type of joint (1 or 2)
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            is_seasoned (bool): Whether the timber is seasoned
        """
        self.joint_type = joint_type
        self.joint_group = joint_group
        self.is_seasoned = is_seasoned

        # Table 4.11 - Maximum tensile load capacity
        self.max_tensile_capacity = {
            "M6": 6400,
            "M8": 11700,
            "M10": 18600,
            "M12": 27000,
            "M16": 50200,
            "M20": 78400,
            "M24": 113000,
            "M30": 180000,
            "M36": 312000,
        }

        # Table 2.6 - Length of bearing factor
        self.k7_table = {
            12: 1.75,
            25: 1.40,
            50: 1.20,
            75: 1.15,
            125: 1.10,
            150: 1.00,  # 150 or more
        }

        # Initialize tables for characteristic capacities
        self._create_tables()

    def _create_tables(self):
        """Create tables for characteristic bolt capacities."""
        # Table 4.9(B) - Characteristic capacities for bolts loaded parallel to grain in unseasoned timber
        self.table_4_9B = {
            # This would contain actual values from the standard
            # For simplicity, using placeholder values here
            "J1": {
                "M6": 2000,
                "M8": 2500,
                "M10": 3000,
                "M12": 3500,
                "M16": 4500,
                "M20": 5500,
            },
            "J2": {
                "M6": 1700,
                "M8": 2100,
                "M10": 2500,
                "M12": 3000,
                "M16": 3800,
                "M20": 4700,
            },
            "J3": {
                "M6": 1400,
                "M8": 1700,
                "M10": 2100,
                "M12": 2500,
                "M16": 3100,
                "M20": 3800,
            },
            "J4": {
                "M6": 1000,
                "M8": 1300,
                "M10": 1600,
                "M12": 1900,
                "M16": 2400,
                "M20": 3000,
            },
            "J5": {
                "M6": 800,
                "M8": 1000,
                "M10": 1200,
                "M12": 1500,
                "M16": 1900,
                "M20": 2300,
            },
            "J6": {
                "M6": 600,
                "M8": 800,
                "M10": 1000,
                "M12": 1200,
                "M16": 1500,
                "M20": 1800,
            },
        }

        # Table 4.9(C) - Characteristic capacities for bolts loaded parallel to grain in seasoned timber
        self.table_4_9C = {
            # Placeholder values
            "JD1": {
                "M6": 2400,
                "M8": 3000,
                "M10": 3600,
                "M12": 4200,
                "M16": 5400,
                "M20": 6600,
            },
            "JD2": {
                "M6": 2000,
                "M8": 2500,
                "M10": 3000,
                "M12": 3600,
                "M16": 4600,
                "M20": 5600,
            },
            "JD3": {
                "M6": 1700,
                "M8": 2100,
                "M10": 2500,
                "M12": 3000,
                "M16": 3800,
                "M20": 4600,
            },
            "JD4": {
                "M6": 1200,
                "M8": 1600,
                "M10": 1900,
                "M12": 2300,
                "M16": 2900,
                "M20": 3600,
            },
            "JD5": {
                "M6": 1000,
                "M8": 1200,
                "M10": 1500,
                "M12": 1800,
                "M16": 2300,
                "M20": 2800,
            },
            "JD6": {
                "M6": 700,
                "M8": 900,
                "M10": 1200,
                "M12": 1400,
                "M16": 1800,
                "M20": 2200,
            },
        }

        # Table 4.10(B) - Characteristic capacities for bolts loaded perpendicular to grain in unseasoned timber
        self.table_4_10B = {
            # Placeholder values
            "J1": {
                "M6": 1600,
                "M8": 2000,
                "M10": 2400,
                "M12": 2800,
                "M16": 3600,
                "M20": 4400,
            },
            "J2": {
                "M6": 1360,
                "M8": 1680,
                "M10": 2000,
                "M12": 2400,
                "M16": 3040,
                "M20": 3760,
            },
            "J3": {
                "M6": 1120,
                "M8": 1360,
                "M10": 1680,
                "M12": 2000,
                "M16": 2480,
                "M20": 3040,
            },
            "J4": {
                "M6": 800,
                "M8": 1040,
                "M10": 1280,
                "M12": 1520,
                "M16": 1920,
                "M20": 2400,
            },
            "J5": {
                "M6": 640,
                "M8": 800,
                "M10": 960,
                "M12": 1200,
                "M16": 1520,
                "M20": 1840,
            },
            "J6": {
                "M6": 480,
                "M8": 640,
                "M10": 800,
                "M12": 960,
                "M16": 1200,
                "M20": 1440,
            },
        }

        # Table 4.10(C) - Characteristic capacities for bolts loaded perpendicular to grain in seasoned timber
        self.table_4_10C = {
            # Placeholder values
            "JD1": {
                "M6": 1920,
                "M8": 2400,
                "M10": 2880,
                "M12": 3360,
                "M16": 4320,
                "M20": 5280,
            },
            "JD2": {
                "M6": 1600,
                "M8": 2000,
                "M10": 2400,
                "M12": 2880,
                "M16": 3680,
                "M20": 4480,
            },
            "JD3": {
                "M6": 1360,
                "M8": 1680,
                "M10": 2000,
                "M12": 2400,
                "M16": 3040,
                "M20": 3680,
            },
            "JD4": {
                "M6": 960,
                "M8": 1280,
                "M10": 1520,
                "M12": 1840,
                "M16": 2320,
                "M20": 2880,
            },
            "JD5": {
                "M6": 800,
                "M8": 960,
                "M10": 1200,
                "M12": 1440,
                "M16": 1840,
                "M20": 2240,
            },
            "JD6": {
                "M6": 560,
                "M8": 720,
                "M10": 960,
                "M12": 1120,
                "M16": 1440,
                "M20": 1760,
            },
        }

    def get_k7_factor(self, bearing_length):
        """
        Get the k7 (length of bearing) factor from Table 2.6.

        Args:
            bearing_length (float): Length of bearing in mm (for washers, this is the diameter or side length)

        Returns:
            float: k7 factor
        """
        if bearing_length <= 0:
            raise ValueError("Bearing length must be positive")

        if bearing_length <= 12:
            return 1.75
        elif bearing_length >= 150:
            return 1.00

        # Find the closest values in the table
        lengths = sorted(self.k7_table.keys())
        for i in range(len(lengths) - 1):
            if bearing_length <= lengths[i + 1]:
                # Linear interpolation between table values
                l1, l2 = lengths[i], lengths[i + 1]
                k1, k2 = self.k7_table[l1], self.k7_table[l2]
                return k1 + (k2 - k1) * (bearing_length - l1) / (l2 - l1)

        return 1.00  # Default for lengths > 150mm

    def get_effective_thickness(self, joint_configuration, member_thicknesses):
        """
        Determine the effective timber thickness (beff) based on joint configuration.

        Args:
            joint_configuration (str): Type of joint configuration
            member_thicknesses (list): List of thicknesses for each member in mm

        Returns:
            float: Effective timber thickness in mm
        """
        if not member_thicknesses:
            raise ValueError("Member thicknesses list cannot be empty")

        if joint_configuration == "two_member":
            return min(member_thicknesses)
        elif joint_configuration == "three_member":
            if len(member_thicknesses) < 1:
                raise ValueError(
                    "Three member joints require at least one member thickness"
                )
            return member_thicknesses[0]
        elif joint_configuration == "multiple_member":
            return min(member_thicknesses)
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

    def get_characteristic_capacity_parallel(
        self, joint_group, effective_thickness, bolt_diameter, is_seasoned=None
    ):
        """
        Get the characteristic capacity for a single bolt loaded parallel to the grain.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            is_seasoned (bool, optional): Whether the timber is seasoned. If None, uses the instance's is_seasoned.

        Returns:
            float: Characteristic capacity in N
        """
        # Use instance is_seasoned if not provided
        if is_seasoned is None:
            is_seasoned = self.is_seasoned

        # Normalize bolt diameter format
        if isinstance(bolt_diameter, (int, float)):
            bolt_diameter = f"M{int(bolt_diameter)}"
        elif isinstance(bolt_diameter, str) and not bolt_diameter.startswith("M"):
            bolt_diameter = f"M{bolt_diameter}"

        # Get the appropriate table
        table = self.table_4_9C if is_seasoned else self.table_4_9B

        # Validate inputs
        if joint_group not in table:
            valid_groups = list(table.keys())
            raise ValueError(
                f"Invalid joint group: {joint_group}. Valid groups are: {valid_groups}"
            )

        if bolt_diameter not in table[joint_group]:
            valid_sizes = list(table[joint_group].keys())
            raise ValueError(
                f"Invalid bolt diameter: {bolt_diameter}. Valid sizes are: {valid_sizes}"
            )

        # Return the characteristic capacity
        return table[joint_group][bolt_diameter]

    def get_characteristic_capacity_perpendicular(
        self, joint_group, effective_thickness, bolt_diameter, is_seasoned=None
    ):
        """
        Get the characteristic capacity for a single bolt loaded perpendicular to the grain.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            is_seasoned (bool, optional): Whether the timber is seasoned. If None, uses the instance's is_seasoned.

        Returns:
            float: Characteristic capacity in N
        """
        # Use instance is_seasoned if not provided
        if is_seasoned is None:
            is_seasoned = self.is_seasoned

        # Normalize bolt diameter format
        if isinstance(bolt_diameter, (int, float)):
            bolt_diameter = f"M{int(bolt_diameter)}"
        elif isinstance(bolt_diameter, str) and not bolt_diameter.startswith("M"):
            bolt_diameter = f"M{bolt_diameter}"

        # Get the appropriate table
        table = self.table_4_10C if is_seasoned else self.table_4_10B

        # Validate inputs
        if joint_group not in table:
            valid_groups = list(table.keys())
            raise ValueError(
                f"Invalid joint group: {joint_group}. Valid groups are: {valid_groups}"
            )

        if bolt_diameter not in table[joint_group]:
            valid_sizes = list(table[joint_group].keys())
            raise ValueError(
                f"Invalid bolt diameter: {bolt_diameter}. Valid sizes are: {valid_sizes}"
            )

        # Return the characteristic capacity
        return table[joint_group][bolt_diameter]

    def get_characteristic_capacity_angle(
        self,
        joint_group,
        effective_thickness,
        bolt_diameter,
        angle_degrees,
        is_seasoned=None,
    ):
        """
        Get the characteristic capacity for a single bolt loaded at an angle to the grain
        using Hankinson's formula.

        Args:
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            angle_degrees (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool, optional): Whether the timber is seasoned. If None, uses the instance's is_seasoned.

        Returns:
            float: Characteristic capacity in N
        """
        # Use instance is_seasoned if not provided
        if is_seasoned is None:
            is_seasoned = self.is_seasoned

        # Get capacities parallel and perpendicular to grain
        Q_parallel = self.get_characteristic_capacity_parallel(
            joint_group, effective_thickness, bolt_diameter, is_seasoned
        )
        Q_perpendicular = self.get_characteristic_capacity_perpendicular(
            joint_group, effective_thickness, bolt_diameter, is_seasoned
        )

        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)

        # Apply Hankinson's formula (Equation 4.4(1))
        numerator = Q_parallel * Q_perpendicular
        denominator = Q_parallel * (np.sin(angle_radians) ** 2) + Q_perpendicular * (
            np.cos(angle_radians) ** 2
        )

        return numerator / denominator

    def get_system_capacity(
        self,
        joint_configuration,
        effective_thickness,
        bolt_diameter,
        joint_group,
        angle_degrees=0,
        is_seasoned=False,
    ):
        """
        Calculate the system capacity for a bolted joint system.

        Args:
            joint_configuration (str): Type of joint configuration ('two_member', 'three_member_A',
                                      'three_member_B', 'multiple_member')
            effective_thickness (int): Effective timber thickness in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            angle_degrees (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: System capacity in N
        """
        # Get the characteristic capacity for a single bolt at the specified angle
        if angle_degrees == 0:
            # Parallel to grain
            Q_k = self.get_characteristic_capacity_parallel(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        elif angle_degrees == 90:
            # Perpendicular to grain
            Q_k = self.get_characteristic_capacity_perpendicular(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        else:
            # At an angle to grain
            Q_k = self.get_characteristic_capacity_angle(
                joint_group,
                effective_thickness,
                bolt_diameter,
                angle_degrees,
                is_seasoned,
            )

        # Calculate system capacity based on joint configuration
        if joint_configuration == "two_member":
            system_capacity = Q_k
        elif joint_configuration == "three_member":
            system_capacity = 2 * Q_k
        elif joint_configuration == "multiple_member":
            # For multiple member joints, the system capacity is the sum of the basic loads
            system_capacity = (
                Q_k  # This should be modified based on the actual number of interfaces
            )
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        return system_capacity

    def calculate_joint_capacity(
        self,
        joint_configuration,
        member_thicknesses,
        bolt_diameter,
        joint_group,
        load_angle=0,
        is_seasoned=False,
    ):
        """
        Calculate the capacity of a bolted joint system.

        Args:
            joint_configuration (str): Type of joint configuration ('two_member', 'three_member',
                                      'multiple_member')
            member_thicknesses (list): List of thicknesses for each member in mm
            bolt_diameter (str): Bolt diameter (e.g., 'M12')
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            load_angle (float): Angle between load direction and grain direction in degrees
            is_seasoned (bool): Whether the timber is seasoned

        Returns:
            float: Joint capacity in N
        """
        # Determine effective thickness based on joint configuration and member thicknesses
        if joint_configuration == "two_member":
            # For two-member joints, effective thickness is the smaller of t1 and t2
            effective_thickness = min(member_thicknesses)
        elif joint_configuration == "three_member":
            # For three-member joints, effective thickness is t1
            effective_thickness = member_thicknesses[0]
        elif joint_configuration == "multiple_member":
            # For multiple member joints, effective thickness depends on the specific configuration
            # This is a simplification
            effective_thickness = min(member_thicknesses)
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        # Calculate system capacity
        system_capacity = self.get_system_capacity(
            joint_configuration,
            effective_thickness,
            bolt_diameter,
            joint_group,
            load_angle,
            is_seasoned,
        )

        return system_capacity

    def design_capacity_type1_joint_4_4_3_2(
        self,
        n_bolts,
        bolt_diameter,
        joint_group,
        joint_configuration,
        member_thicknesses,
        angle_degrees=0,
        has_metal_side_plates=False,
        is_seasoned=False,
    ):
        """
        Calculate the design capacity for Type 1 joints containing n bolts in shear to resist lateral loads.

        Args:
            n_bolts (int): Number of bolts in the joint
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
            joint_configuration (str): Type of joint configuration
            member_thicknesses (list): List of thicknesses for each member in mm
            load_direction (str): Direction of load ('parallel', 'perpendicular', or 'angle')
            angle_degrees (float): Angle between load direction and grain direction in degrees
            has_metal_side_plates (bool): Whether the joint has metal side plates
            n_rows (int): Number of rows of fasteners per interface
            has_transverse_restraint (bool): Whether the joint has transverse restraint
            k1 (float): Duration of load factor (default 1.14 for 5-second load)
            k16 (float): Factor for bolts with metal side plates (default 2.0 if applicable, 1.0 otherwise)
            k17 (float): Factor for multiple bolted joint
            phi (float): Capacity factor (default 0.8)

        Returns:
            float: Design capacity (Nd,1) in N
        """
        # Set default values if not provided
        phi = 0.8
        k1 = 1.14  # For 5-second duration load
        k16 = 2.0 if has_metal_side_plates else 1.0
        k17 = self.TABLE_4_3_A

        # Get effective thickness based on joint configuration
        # Determine effective thickness based on joint configuration
        if joint_configuration == "two_member":
            # For two-member joints, effective thickness is the smaller of t1 and t2
            effective_thickness = min(member_thicknesses)
        elif joint_configuration == "three_member":
            # For three-member joints, effective thickness is t1
            effective_thickness = member_thicknesses[0]
        elif joint_configuration == "multiple_member":
            # For multiple member joints, effective thickness depends on the specific configuration
            effective_thickness = min(member_thicknesses)
        else:
            raise ValueError(f"Invalid joint configuration: {joint_configuration}")

        # Use the CoachScrewedJoint's get_characteristic_capacity method for consistent calculation
        if angle_degrees == 0:
            # Parallel to grain
            Qk = self.get_characteristic_capacity_parallel(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        elif angle_degrees == 90:
            # Perpendicular to grain
            Qk = self.get_characteristic_capacity_perpendicular(
                joint_group, effective_thickness, bolt_diameter, is_seasoned
            )
        else:
            # At an angle to grain
            Qk = self.get_characteristic_capacity_angle(
                joint_group,
                effective_thickness,
                bolt_diameter,
                angle_degrees,
                is_seasoned,
            )

        # Calculate design capacity according to equation 4.4(3)
        N_d = phi * k1 * k16 * k17 * n_bolts * Qk

        return N_d

    def design_capacity_type2_joint_4_4_3_3(
        self,
        n_bolts,
        bolt_diameter,
        washer_diameter,
        joint_group=None,
        washer_area=None,
    ):
        """
        Calculate the design capacity for Type 2 joints in which bolts are loaded in direct tension.

        Args:
            n_bolts (int): Number of bolts in the joint
            bolt_diameter (str or int): Bolt diameter (e.g., 'M12' or 12)
            washer_diameter (float): Diameter or side length of washer in mm
            joint_group (str, optional): Joint group. If None, uses the instance's joint_group
            washer_area (float, optional): Area of the washer in mm². If None, calculated from washer_diameter

        Returns:
            float: Design capacity in N
        """
        # Set default values
        phi = 0.8
        k1 = 1.14
        k7 = self.get_k7_factor(washer_diameter)

        # Use instance joint_group if not provided
        if joint_group is None:
            joint_group = self.joint_group
            if joint_group is None:
                raise ValueError(
                    "Joint group must be provided either at initialization or as a function parameter"
                )

        # Get characteristic bearing capacity f'pj from Table C6
        if joint_group not in self.TABLE_C6:
            valid_groups = list(self.TABLE_C6.keys())
            raise ValueError(
                f"Invalid joint group: {joint_group}. Valid groups are: {valid_groups}"
            )

        f_p = self.TABLE_C6[joint_group]

        # Normalize bolt diameter format
        if isinstance(bolt_diameter, (int, float)):
            bolt_diameter = f"M{int(bolt_diameter)}"
        elif isinstance(bolt_diameter, str) and not bolt_diameter.startswith("M"):
            bolt_diameter = f"M{bolt_diameter}"

        # Get maximum tensile capacity from Table 4.11
        if bolt_diameter not in self.max_tensile_capacity:
            valid_sizes = list(self.max_tensile_capacity.keys())
            raise ValueError(
                f"Invalid bolt diameter: {bolt_diameter}. Valid sizes are: {valid_sizes}"
            )

        N_dt = self.max_tensile_capacity[bolt_diameter]

        # Calculate design capacity according to equations 4.4(4) and 4.4(6)
        N_d_eq4 = n_bolts * N_dt  # Equation 4.4(4)
        N_d_eq6 = phi * k1 * k7 * f_p * washer_area  # Equation 4.4(6)

        # Design capacity is the lesser of the two values
        if N_d_eq4 <= N_d_eq6:
            return N_d_eq4
        else:
            return N_d_eq6


def create_joint_system(
    joint_connection_type, joint_type=None, joint_group=None, is_seasoned=False
):
    """
    Factory function to create the appropriate joint system based on connection type.

    Args:
        joint_connection_type (str): Type of connection - either "CoachScrew" or "Bolt"
        joint_type (int): Type of joint (1 or 2)
        joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
        is_seasoned (bool): Whether the timber is seasoned

    Returns:
        Either a CoachScrewedJoint or BoltedJointSystem instance

    Raises:
        ValueError: If joint_connection_type is not recognized
    """
    if joint_connection_type.lower() in (
        "coachscrew",
        "coach screw",
        "coach-screw",
        "coach",
    ):
        return CoachScrewedJoint(joint_type, joint_group, is_seasoned)
    elif joint_connection_type.lower() in ("bolt", "bolted"):
        return BoltedJointSystem(joint_type, joint_group, is_seasoned)
    else:
        valid_types = ["CoachScrew", "Bolt"]
        raise ValueError(
            f"Invalid joint connection type: {joint_connection_type}. Valid types are: {valid_types}"
        )


def calculate_joint_capacities(
    joint_connection_type,
    n_fasteners,
    fastener_diameter,
    joint_group,
    is_side_grain=True,
    has_metal_side_plates=False,
    joint_configuration="two_member",
    member_thicknesses=[45],
    angle_degrees=0,
    is_seasoned=False,
    washer_diameter=None,
    penetration_depth=None,
    washer_area=None,
    **kwargs,
):
    """
    Calculate both tension (Nt) and shear (Ns) capacities for a joint using either coach screws or bolts.

    Args:
        joint_connection_type (str): Type of connection - either "coach_screw" or "bolt"
        n_fasteners (int): Number of fasteners in the joint
        fastener_diameter (str or int): Diameter of the fastener (e.g., 'M12' or 12)
        joint_group (str): Joint group (J1-J6 for unseasoned, JD1-JD6 for seasoned)
        is_side_grain (bool): Whether the fasteners are in side grain (True) or end grain (False)
        has_metal_side_plates (bool): Whether the joint has metal side plates
        joint_configuration (str): Type of joint configuration ('two_member', 'three_member', etc.)
        member_thicknesses (list): List of thicknesses for each member in mm
        angle_degrees (float): Angle between load direction and grain direction in degrees
        is_seasoned (bool): Whether the timber is seasoned
        washer_diameter (float): Diameter of the washer in mm (required for bolts in tension)
        penetration_depth (float): Penetration depth for coach screws in mm (required for coach screws in tension)
        **kwargs: Additional arguments to pass to the specific joint system methods

    Returns:
        tuple: (Nt, Ns) where:
            - Nt is the design capacity for tension (N)
            - Ns is the design capacity for shear (N)

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Create appropriate joint systems for tension and shear
    if joint_connection_type == "coach_screw":
        # For coach screws
        joint_tension = CoachScrewedJoint(
            joint_type=2,  # Type 2 for tension/withdrawal capacity
            joint_group=joint_group,
            is_seasoned=is_seasoned,
        )

        joint_shear = CoachScrewedJoint(
            joint_type=1,  # Type 1 for lateral/shear capacity
            joint_group=joint_group,
            is_seasoned=is_seasoned,
        )

        # Calculate tension capacity
        if penetration_depth is None:
            raise ValueError(
                "Penetration depth is required for coach screws in tension"
            )

        Nt = joint_tension.design_capacity_type2_joint_4_5_3_2(
            n_fasteners, fastener_diameter, penetration_depth, is_side_grain
        )

        # Calculate shear capacity
        Ns = joint_shear.design_capacity_type1_joint_4_5_3_1(
            n_fasteners,
            fastener_diameter,
            is_side_grain,
            has_metal_side_plates,
            joint_configuration,
            member_thicknesses,
            joint_group,
            angle_degrees,
            is_seasoned,
        )

    elif joint_connection_type == "bolt":
        # For bolts
        joint_system = BoltedJointSystem(
            joint_type=None,  # Will be set per operation
            joint_group=joint_group,
            is_seasoned=is_seasoned,
        )

        # For tension capacity
        if washer_diameter is None:
            raise ValueError("Washer diameter is required for bolts in tension")

        joint_system.joint_type = 2  # Set to Type 2 for tension
        Nt = joint_system.design_capacity_type2_joint_4_4_3_3(
            n_fasteners, fastener_diameter, washer_diameter, joint_group, washer_area
        )

        # For shear capacity
        joint_system.joint_type = 1  # Set to Type 1 for shear
        Ns = joint_system.design_capacity_type1_joint_4_4_3_2(
            n_fasteners,
            fastener_diameter,
            joint_group,
            joint_configuration,
            member_thicknesses,
            angle_degrees=angle_degrees,
            has_metal_side_plates=has_metal_side_plates,
            is_seasoned=is_seasoned,
        )
    else:
        valid_types = ["coach_screw", "bolt"]
        raise ValueError(
            f"Invalid joint connection type: {joint_connection_type}. Valid types are: {valid_types}"
        )

    return Nt, Ns


# ------------------------------------------------------------------
# Section 8 - LVL design values
# ------------------------------------------------------------------
class LVLDesignValues:
    """
    Class for LVL (Laminated Veneer Lumber) design values according to
    NZS AS 1720.1:2022 standard, Table ZZ8.1.

    This class provides characteristic values for design of Structural LVL
    used 'on-edge' with moisture content 15% or less.
    """

    def __init__(self):
        """Initialize the LVL design values class with Table ZZ8.1 data."""
        self.create_lvl_table()

    def create_lvl_table(self):
        """
        Create Table ZZ8.1 - Characteristic values for design - Structural LVL
        used 'on-edge' (moisture content 15% or less).
        """
        # Table ZZ8.1 data
        lvl_data = {
            "LVL_grade": ["LVL16", "LVL13", "LVL11", "LVL10", "LVL8"],
            "Design_density_kg_m3": [660, 620, 590, 580, 550],
            "Characteristic_density_kg_m3": [480, 480, 480, 480, 480],
            "Bending_f_b_MPa": [50, 45, 38, 35, 30],
            "Shear_in_beams_f_s_MPa": [4.5, 4.0, 3.5, 3.5, 3.5],
            "Bearing_perpendicular_f_p_MPa": [10, 10, 10, 10, 10],
            "Tension_parallel_to_grain_f_t_MPa": [25, 25, 16, 15, 15],
            "Compression_parallel_to_grain_f_c_MPa": [45, 38, 32, 30, 30],
            "Short_duration_modulus_of_elasticity_E_MPa": [
                16000,
                13200,
                11000,
                10000,
                8000,
            ],
            "Short_duration_modulus_of_rigidity_G_MPa": [800, 660, 550, 500, 400],
        }

        # Create DataFrame
        self.df_lvl = pd.DataFrame(lvl_data)
        self.df_lvl.set_index("LVL_grade", inplace=True)

        # Create notes dictionary for additional information
        self.notes = {
            "tension_perpendicular": "Tension perpendicular to the grain shall be taken as f_tp = 0.5 MPa",
            "characteristic_density_usage": "The characteristic density is to be used for the dead load calculations using the detailed method",
            "beam_depth_adjustment": "For beams with a depth exceeding 95 mm, multiply the value for bending (f_b) in Table 8.1 in AS 1720.1 by (95/d)^0.167, where d is the depth of the beam",
            "bearing_perpendicular_note": "The bearing strength perpendicular to the grain has been determined in accordance with AS/NZS 4063.1 clause 2.8 and includes stress-spreading and hanging-edge effects. Care should be taken in the application of the length of bearing factor, k_7, to this value if there is no guidance from the manufacturer. Should the perpendicular-to-grain strength without the effects of stress spreading and hanging edge be required, refer to the manufacturer's data or Franke and Quenneville (2010)",
            "tension_members_150mm": "For tension members with a width of 150 mm or less, Table 8.1 shall be used, with no adjustment required",
            "tension_members_larger": "For tension members with the larger cross-sectional dimension exceeding 150 mm, multiply the value for tension (f_t) in Table ZZ8.1 by (150/d)^0.167, where d is the larger cross-sectional dimension of the tension member",
        }

    def get_lvl_properties(self, lvl_grade):
        """
        Get all properties for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade (e.g., 'LVL16', 'LVL13', etc.)

        Returns:
            dict: Dictionary containing all properties for the specified grade
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        properties = self.df_lvl.loc[lvl_grade].to_dict()
        return properties

    def get_design_density(self, lvl_grade):
        """
        Get design density for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Design density in kg/m³
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Design_density_kg_m3"]

    def get_characteristic_density(self, lvl_grade):
        """
        Get characteristic density for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Characteristic density in kg/m³
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Characteristic_density_kg_m3"]

    def get_bending_strength(self, lvl_grade, beam_depth=None):
        """
        Get bending strength for a specific LVL grade with optional depth adjustment.

        Args:
            lvl_grade (str): LVL grade
            beam_depth (float, optional): Beam depth in mm. If > 95mm, applies depth adjustment factor

        Returns:
            float: Bending strength (f'b) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        f_b = self.df_lvl.loc[lvl_grade, "Bending_f_b_MPa"]

        # Apply depth adjustment if beam depth > 95mm
        if beam_depth is not None and beam_depth > 95:
            adjustment_factor = (95 / beam_depth) ** 0.167
            f_b = f_b * adjustment_factor

        return f_b

    def get_shear_strength(self, lvl_grade):
        """
        Get shear strength in beams for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Shear strength (f's) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Shear_in_beams_f_s_MPa"]

    def get_bearing_perpendicular_strength(self, lvl_grade):
        """
        Get bearing strength perpendicular to grain for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Bearing strength perpendicular to grain (f'p) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Bearing_perpendicular_f_p_MPa"]

    def get_tension_parallel_strength(self, lvl_grade, member_dimension=None):
        """
        Get tension strength parallel to grain for a specific LVL grade with optional size adjustment.

        Args:
            lvl_grade (str): LVL grade
            member_dimension (float, optional): Larger cross-sectional dimension in mm.
                                              If > 150mm, applies size adjustment factor

        Returns:
            float: Tension strength parallel to grain (f't) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        f_t = self.df_lvl.loc[lvl_grade, "Tension_parallel_to_grain_f_t_MPa"]

        # Apply size adjustment if member dimension > 150mm
        if member_dimension is not None and member_dimension > 150:
            adjustment_factor = (150 / member_dimension) ** 0.167
            f_t = f_t * adjustment_factor

        return f_t

    def get_tension_perpendicular_strength(self):
        """
        Get tension strength perpendicular to grain (constant for all LVL grades).

        Returns:
            float: Tension strength perpendicular to grain (f'tp) in MPa
        """
        return 0.5  # As per note a in the table

    def get_compression_parallel_strength(self, lvl_grade):
        """
        Get compression strength parallel to grain for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Compression strength parallel to grain (f'c) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Compression_parallel_to_grain_f_c_MPa"]

    def get_modulus_of_elasticity(self, lvl_grade):
        """
        Get short-duration average modulus of elasticity for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Modulus of elasticity (E) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Short_duration_modulus_of_elasticity_E_MPa"]

    def get_modulus_of_rigidity(self, lvl_grade):
        """
        Get short-duration average modulus of rigidity for beams for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            float: Modulus of rigidity (G) in MPa
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        return self.df_lvl.loc[lvl_grade, "Short_duration_modulus_of_rigidity_G_MPa"]

    def get_available_grades(self):
        """
        Get list of available LVL grades.

        Returns:
            list: List of available LVL grades
        """
        return list(self.df_lvl.index)

    def get_table_notes(self):
        """
        Get the notes associated with Table ZZ8.1.

        Returns:
            dict: Dictionary containing all table notes
        """
        return self.notes

    def display_lvl_table(self):
        """
        Display the complete LVL table.

        Returns:
            pd.DataFrame: The complete LVL properties table
        """
        return self.df_lvl

    def get_lvl_properties_summary(self, lvl_grade):
        """
        Get a formatted summary of all properties for a specific LVL grade.

        Args:
            lvl_grade (str): LVL grade

        Returns:
            str: Formatted string with all properties
        """
        if lvl_grade not in self.df_lvl.index:
            valid_grades = list(self.df_lvl.index)
            raise ValueError(
                f"Invalid LVL grade: {lvl_grade}. Valid grades are: {valid_grades}"
            )

        properties = self.get_lvl_properties(lvl_grade)

        summary = f"""
            LVL Grade: {lvl_grade}
            ===================
            Design Density: {properties['Design_density_kg_m3']} kg/m³
            Characteristic Density: {properties['Characteristic_density_kg_m3']} kg/m³
            Bending Strength (f'b): {properties['Bending_f_b_MPa']} MPa
            Shear Strength (f's): {properties['Shear_in_beams_f_s_MPa']} MPa
            Bearing Perpendicular (f'p): {properties['Bearing_perpendicular_f_p_MPa']} MPa
            Tension Parallel (f't): {properties['Tension_parallel_to_grain_f_t_MPa']} MPa
            Tension Perpendicular (f'tp): 0.5 MPa
            Compression Parallel (f'c): {properties['Compression_parallel_to_grain_f_c_MPa']} MPa
            Modulus of Elasticity (E): {properties['Short_duration_modulus_of_elasticity_E_MPa']} MPa
            Modulus of Rigidity (G): {properties['Short_duration_modulus_of_rigidity_G_MPa']} MPa
        """

        return summary.strip()
