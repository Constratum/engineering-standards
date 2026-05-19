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


class AS_1720_1_2022:
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
