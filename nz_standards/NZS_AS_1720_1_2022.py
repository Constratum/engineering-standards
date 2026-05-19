"""NZS_1720_1_2022 Method

This module references the following standard:
NZS AS 1720.1:2022 (AS 1720.1:2010, MOD) — New Zealand modifications only.

Covers timber member design for New Zealand-verified SG-grade sawn timber.

Method developed May 2026
(c) Constratum Ltd

Developed - NSh
Reviewed  - MB

###Initialise Dependents and Libraries
"""

import numpy as np
import pandas as pd


class NZS_1720_1_2022:
    """
    Timber member design capacity functions for NZ-verified SG-grade sawn timber
    in accordance with NZS AS 1720.1:2022 (New Zealand modifications only).

    Material properties:   Table ZZ2.1 (SG6, SG8, SG10, SG12, SG15, No.1 Framing)
    Stability constants:   Table ZZ3.1 (rho_b), Table ZZ3.2 (rho_c)
    Capacity factor:       ZZ2.3 — phi = 0.8 for all timber members (flat, no categories)
    Duration of load:      Table 2.3 (shared with AS)
    Temperature factor:    k6 = 1.0 (NZS does not specify temperature reduction for NZ)

    Key NZS-specific differences vs AS 1720.1:
      - phi = 0.8 flat (ZZ2.3) — no member category split
      - Grade system: SG6–SG15, not F-grades
      - Dry/wet conditions handled by grade selection, not k4 formula
      - G = E'/15 (Table ZZ2.1 Note 3)
      - fs = 3.8 MPa radiata pine, 3.0 MPa Douglas fir (Table ZZ2.1 Note 1)
      - fp = 6.9 MPa bearing perpendicular (Table ZZ2.1 Note 2)
      - rho_b from Table ZZ3.1 (lower than AS Table 3.1 for SG grades)
      - rho_c from Table ZZ3.2 (replaces AS Table 3.3)

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
    # Table ZZ2.1 — Characteristic properties for NZ-verified SG-grade timber
    # Tuple: (design_density_kg_m3, char_density_kg_m3, fb_MPa, fc_MPa,
    #         ft_MPa, ftp_MPa, E_MPa, E_lb_MPa)
    #
    # Notes from Table ZZ2.1:
    #   Note 1: fs = 3.8 MPa radiata pine (seasoned); 3.0 MPa Douglas fir
    #   Note 2: fp = 6.9 MPa bearing perpendicular to grain
    #   Note 3: G = E'/15
    #   Note 5: design_density for dead load calculations only
    #   Note 6: char_density for connection design (detailed method)
    # ------------------------------------------------------------------
    TABLE_ZZ2_1 = {
        #              ρ_d   ρ'    fb    fc    ft   ftp     E'     E_lb
        "SG15":  (570, 475, 41.0, 35.0, 23.0, 0.5, 15200, 11500),
        "SG12":  (540, 450, 28.0, 25.0, 14.0, 0.5, 12000,  9000),
        "SG10":  (500, 415, 20.0, 20.0,  8.0, 0.5, 10000,  7500),
        "SG8":   (450, 375, 14.0, 18.0,  6.0, 0.4,  8000,  5400),
        "SG6":   (400, 330, 10.0, 15.0,  4.0, 0.4,  6000,  4000),
        # Table ZZ2.2 — Non-verified No.1 Framing (same values as SG6)
        "NO1FRAMING": (400, 330, 10.0, 15.0, 4.0, 0.4, 6000, 4000),
    }

    # Shear strength — Table ZZ2.1 Note 1 (species-specific, not in table columns)
    FS_RADIATA_PINE = 3.8   # MPa — seasoned radiata pine
    FS_DOUGLAS_FIR  = 3.0   # MPa — seasoned Douglas fir

    # Bearing perpendicular to grain — Table ZZ2.1 Note 2
    FP_PERPENDICULAR = 6.9  # MPa

    # ------------------------------------------------------------------
    # Table ZZ3.1 — Material constant rho_b for sawn timber beams (seasoned)
    # Note: NZS does not provide unseasoned rho_b for SG grades.
    # Values correspond to seasoned timber, r = 0.25.
    # ------------------------------------------------------------------
    TABLE_ZZ3_1_RHO_B = {
        "SG15": 0.94,
        "SG12": 0.87,
        "SG10": 0.81,
        "SG8":  0.76,
        "SG6":  0.74,
        "NO1FRAMING": 0.74,
    }

    # ------------------------------------------------------------------
    # Table ZZ3.2 — Material constant rho_c for sawn timber columns (seasoned)
    # Note: NZS replaces AS Table 3.3 entirely.
    # Values correspond to seasoned timber, r = 0.25.
    # ------------------------------------------------------------------
    TABLE_ZZ3_2_RHO_C = {
        "SG15": 1.06,
        "SG12": 1.02,
        "SG10": 1.00,
        "SG8":  1.05,
        "SG6":  1.10,
        "NO1FRAMING": 1.10,
    }

    # ------------------------------------------------------------------
    # Table 2.3 — k1 duration of load factor (shared with AS 1720.1)
    # ------------------------------------------------------------------
    TABLE_2_3_K1 = {
        "5 seconds": 1.00,
        "5 minutes": 1.00,
        "5 hours":   0.97,
        "5 days":    0.94,
        "5 months":  0.80,
        "50+ years": 0.57,
    }

    # NZS phi = 0.8 flat for all timber members (Cl ZZ2.3)
    PHI = 0.8

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

    def __init__(self, grade="SG8", seasoned="Yes", species="radiata_pine"):
        """
        Initialise a NZS 1720.1:2022 SG-grade timber member design object.

        Args:
            grade (str): NZ stress grade — 'SG6', 'SG8', 'SG10', 'SG12', 'SG15',
                         or 'No1Framing'. Case-insensitive. 'dry'/'wet' suffixes
                         are stripped (e.g. 'SG8 (dry)' -> 'SG8').
            seasoned (str | bool): 'Yes' = dry/seasoned (EMC <= 15%), k4 formula
                             applies if EMC > 15%. 'No' = wet/unseasoned, k4 = 1.0
                             (design directly to wet characteristic values).
            species (str): 'radiata_pine' (fs = 3.8 MPa) or 'douglas_fir'
                           (fs = 3.0 MPa). Affects shear strength only.
        """
        # Normalise grade — strip dry/wet suffixes and whitespace
        raw = (grade.upper()
               .replace("(DRY)", "").replace("(WET)", "")
               .replace(" DRY", "").replace(" WET", "")
               .replace(" ", "").strip())
        _alias = {
            "NO.1FRAMING": "NO1FRAMING",
            "NO1": "NO1FRAMING",
            "NUMBER1FRAMING": "NO1FRAMING",
        }
        self.grade = _alias.get(raw, raw)
        if self.grade not in self.TABLE_ZZ2_1:
            raise ValueError(
                f"Grade '{grade}' not found in NZS 1720.1:2022 Table ZZ2.1. "
                f"Valid NZS grades: {list(self.TABLE_ZZ2_1.keys())}. "
                f"For Australian F-grades use AS_1720_1_2022."
            )
        self.seasoned = self._parse_seasoned(seasoned)
        self.species  = species.lower()

    # ── Cl ZZ2.3 — Capacity factor ────────────────────────────────────

    def get_phi(self):
        """
        Capacity factor phi for timber members (Cl ZZ2.3).
        phi = 0.8 for all timber members in NZS — no category split.

        Returns:
            float: 0.8
        """
        return self.PHI

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

    def get_k4(self, emc_percent=15):
        """
        Moisture condition factor k4 (Cl 2.4.2).

        NZS approach: dry grades use characteristic dry values directly.
        If the member is expected to be at EMC > 15% in service, either:
          (a) use the wet grade characteristic values (seasoned=False), or
          (b) apply k4 reduction to dry values (this function, Cl 2.4.2.3).

        Seasoned (dry, Cl 2.4.2.3):
          EMC <= 15%: k4 = 1.0
          EMC >  15%: k4 = max(1 - 0.3*(EMC-15)/10, 0.7)

        Unseasoned (wet): k4 = 1.0 (use wet characteristic values directly).

        Args:
            emc_percent (float): Expected in-service annual average EMC (%).

        Returns:
            float: k4
        """
        if self.seasoned:
            if emc_percent <= 15:
                return 1.0
            return max(1.0 - 0.3 * (emc_percent - 15) / 10, 0.7)
        return 1.0

    def get_k6(self):
        """
        Temperature factor k6 (Cl 2.4.3).
        NZS 1720.1:2022 does not specify k6 for NZ conditions.
        Returns 1.0 (assumed for all NZ locations per standard note).

        Returns:
            float: 1.0
        """
        return 1.0

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
        _G = {1: 1.00, 2: 1.14, 3: 1.20, 4: 1.24, 5: 1.26,
              6: 1.28, 7: 1.30, 8: 1.31, 9: 1.32}
        def _g(n):
            return _G.get(min(n, 9), 1.33)
        g31 = _g(n_com)
        g32 = _g(n_com * n_mem)
        if n_mem <= 1 or spacing_mm is None or span_mm is None:
            return max(g31, 1.0)
        return max(g31 + (g32 - g31) * (1.0 - 2.0 * spacing_mm / span_mm), 1.0)

    # ── Material properties — Table ZZ2.1 ────────────────────────────

    def get_fb(self, depth_mm=None):
        """
        Characteristic bending strength fb' (MPa), Table ZZ2.1.
        For beam depth > 300 mm: fb' *= (300/d)^0.167 (AS 1720.1 Cl applies).

        Args:
            depth_mm (float, optional): Beam depth in mm.

        Returns:
            float: fb' (MPa)
        """
        fb = self.TABLE_ZZ2_1[self.grade][2]
        if depth_mm is not None and depth_mm > 300:
            fb *= (300 / depth_mm) ** 0.167
        return fb

    def get_fc(self):
        """
        Characteristic compression strength fc' (MPa), Table ZZ2.1.

        Returns:
            float: fc' (MPa)
        """
        return self.TABLE_ZZ2_1[self.grade][3]

    def get_ft(self, largest_dim_mm=None):
        """
        Characteristic tension strength ft' (MPa), Table ZZ2.1.
        For largest cross-section dimension > 150 mm: ft' *= (150/d)^0.167.

        Args:
            largest_dim_mm (float, optional): Largest cross-section dimension (mm).

        Returns:
            float: ft' (MPa)
        """
        ft = self.TABLE_ZZ2_1[self.grade][4]
        if largest_dim_mm is not None and largest_dim_mm > 150:
            ft *= (150 / largest_dim_mm) ** 0.167
        return ft

    def get_ftp(self):
        """
        Tension perpendicular-to-grain strength ftp' (MPa), Table ZZ2.1.

        Returns:
            float: ftp' (MPa) — 0.4 for SG6/SG8, 0.5 for SG10/SG12/SG15
        """
        return self.TABLE_ZZ2_1[self.grade][5]

    def get_fs(self):
        """
        Characteristic shear strength fs' (MPa), Table ZZ2.1 Note 1.
        Species-specific: 3.8 MPa for radiata pine, 3.0 MPa for Douglas fir.

        Returns:
            float: fs' (MPa)
        """
        return (self.FS_RADIATA_PINE
                if "douglas" not in self.species
                else self.FS_DOUGLAS_FIR)

    def get_fp(self):
        """
        Bearing strength perpendicular-to-grain fp' (MPa), Table ZZ2.1 Note 2.

        Returns:
            float: 6.9 MPa
        """
        return self.FP_PERPENDICULAR

    def get_E(self):
        """
        Short-duration average modulus of elasticity E' (MPa), Table ZZ2.1.

        Returns:
            float: E' (MPa)
        """
        return self.TABLE_ZZ2_1[self.grade][6]

    def get_E_lb(self):
        """
        Lower-bound short-duration modulus of elasticity E_lb (MPa), Table ZZ2.1.
        Used for isolated members or systems without sufficient load sharing.

        Returns:
            float: E_lb (MPa)
        """
        return self.TABLE_ZZ2_1[self.grade][7]

    def get_G(self):
        """
        Short-duration modulus of rigidity G (MPa), Table ZZ2.1 Note 3.
        G = E'/15.

        Returns:
            float: G (MPa)
        """
        return self.get_E() / 15.0

    def get_design_density(self):
        """
        Design density (kg/m³), Table ZZ2.1 Note 5.
        For dead load calculations due to self-weight of timber only.

        Returns:
            float: kg/m³
        """
        return self.TABLE_ZZ2_1[self.grade][0]

    def get_characteristic_density(self):
        """
        Characteristic density rho' (kg/m³), Table ZZ2.1 Note 6.
        For connection design using the detailed method.

        Returns:
            float: kg/m³
        """
        return self.TABLE_ZZ2_1[self.grade][1]

    def get_rho_b(self):
        """
        Material constant rho_b for bending stability, Table ZZ3.1.

        Returns:
            float: rho_b
        """
        return self.TABLE_ZZ3_1_RHO_B[self.grade]

    def get_rho_c(self):
        """
        Material constant rho_c for compression stability, Table ZZ3.2.

        Returns:
            float: rho_c
        """
        return self.TABLE_ZZ3_2_RHO_C[self.grade]

    # ── Cl 3.2.3 — Bending slenderness S1 ────────────────────────────

    def calc_S1(self, d_mm, b_mm, restraint_type="discrete",
                restraint_face="compression", Lay_mm=None, L_phi_mm=None):
        """
        Slenderness coefficient S1 for lateral buckling under bending (Cl 3.2.3.2).
        Uses NZS rho_b from Table ZZ3.1.

        For minor-axis bending: S2 = 0 — call calc_k12_bending(0.0) -> k12 = 1.0.

        Args:
            d_mm (float): Depth (mm) — major-axis dimension.
            b_mm (float): Breadth (mm).
            restraint_type (str): 'discrete' or 'continuous'.
            restraint_face (str): 'compression' or 'tension'.
            Lay_mm (float, optional): Unbraced length (mm). Required for discrete.
            L_phi_mm (float, optional): Torsional restraint spacing (mm).
                Only used for continuous tension-edge + torsional restraints.

        Returns:
            float: S1
        """
        rho_b = self.get_rho_b()
        if restraint_type == "continuous":
            if restraint_face == "compression":
                # Check Eq 3.2(6): Lay/d <= 64*(b/(rho_b*d))^2
                if Lay_mm is not None:
                    limit = 64.0 * (b_mm / (rho_b * d_mm)) ** 2
                    if (Lay_mm / d_mm) <= limit:
                        return 0.0
                return 0.0
            else:
                # Continuous tension edge
                if L_phi_mm is not None:
                    # With torsional restraints — Eq 3.2(8)
                    return (1.5 * d_mm / b_mm) / (
                        ((np.pi * d_mm / L_phi_mm) ** 2 + 0.4) ** 0.5)
                # Tension edge only — Eq 3.2(7)
                return 2.25 * d_mm / b_mm
        else:
            if Lay_mm is None:
                raise ValueError("Lay_mm is required for discrete restraint.")
            if restraint_face == "compression":
                return 1.25 * (d_mm / b_mm) * (Lay_mm / d_mm) ** 0.5   # Eq 3.2(4)
            else:
                return (d_mm / b_mm) ** 1.35 * (Lay_mm / d_mm) ** 0.25  # Eq 3.2(5)

    # ── Cl 3.2.4 — Bending stability factor k12b ─────────────────────

    def calc_k12_bending(self, S1):
        """
        Stability factor k12 for bending (Cl 3.2.4).
        Uses NZS rho_b from Table ZZ3.1.

        rho_b*S1 <= 10  -> k12 = 1.0
        rho_b*S1 <= 20  -> k12 = 1.5 - 0.05*rho_b*S1
        rho_b*S1 >  20  -> k12 = 200 / (rho_b*S1)²

        For minor-axis bending S2 = 0 -> k12 = 1.0 (do not call this function).

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
        Slenderness coefficient S3 for major-axis compression buckling (Cl 3.3.2.2).
        S3 = min(Laz/d, g13*L/d)

        Args:
            d_mm (float): Depth (mm) — major-axis dimension.
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
        Slenderness coefficient S4 for minor-axis compression buckling (Cl 3.3.2.2).
        S4 = min(Lay/b, g13*L/b)

        Args:
            b_mm (float): Breadth (mm) — minor-axis dimension.
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
        Uses NZS rho_c from Table ZZ3.2. Apply separately for S3 and S4.

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

    def bending_capacity_major(self, Zz_mm3, depth_mm, k1, k4, k9, k12b):
        """
        Design bending capacity Mdz about the major z-axis (Cl 3.2.1.1).
        NZS: k6 = 1.0 always (not passed as argument).

        Mdz = phi * k1 * k4 * k9 * k12b * fb' * Zz   [N·mm]

        Args:
            Zz_mm3 (float): Elastic section modulus about major z-axis (mm³).
            depth_mm (float): Stud depth (mm) — for fb' depth adjustment (Table ZZ2.1).
            k1 (float): Duration factor from get_k1().
            k4 (float): Moisture factor from get_k4().
            k9 (float): Strength sharing factor from get_k9().
            k12b (float): Bending stability factor from calc_k12_bending().

        Returns:
            float: Mdz (N·mm)
        """
        return (self.PHI * k1 * k4 * k9 * k12b
                * self.get_fb(depth_mm=depth_mm) * Zz_mm3)

    def bending_capacity_minor(self, Zy_mm3, k1, k4, k9):
        """
        Design bending capacity Mdy about the minor y-axis (Cl 3.2.1.1).
        Minor axis: S2 = 0 always -> k12by = 1.0.

        Mdy = phi * k1 * k4 * k9 * 1.0 * fb' * Zy   [N·mm]

        Args:
            Zy_mm3 (float): Elastic section modulus about minor y-axis (mm³).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k9 (float): Strength sharing factor.

        Returns:
            float: Mdy (N·mm)
        """
        return self.PHI * k1 * k4 * k9 * self.get_fb() * Zy_mm3

    # ── Cl 3.2.5 — Shear capacity ─────────────────────────────────────

    def shear_capacity(self, As_mm2, k1, k4):
        """
        Design shear capacity Vd (Cl 3.2.5).
        Note: k9 is not applied to shear.

        Vd = phi * k1 * k4 * fs' * As   [N]

        Args:
            As_mm2 (float): Shear plane area (mm²), Cl 3.2.5.
            k1 (float): Duration factor.
            k4 (float): Moisture factor.

        Returns:
            float: Vd (N)
        """
        return self.PHI * k1 * k4 * self.get_fs() * As_mm2

    # ── Cl 3.3.1 — Compression capacity ──────────────────────────────

    def compression_capacity_major(self, A_mm2, k1, k4, k12c_z):
        """
        Design compression capacity for major-axis (z-axis) buckling Ndc_z (Cl 3.3.1.1).

        Ndc_z = phi * k1 * k4 * k12c_z * fc' * Ac   [N]

        Args:
            A_mm2 (float): Gross cross-sectional area Ac (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k12c_z (float): Compression stability factor for S3 (major axis).

        Returns:
            float: Ndc_z (N)
        """
        return self.PHI * k1 * k4 * k12c_z * self.get_fc() * A_mm2

    def compression_capacity_minor(self, A_mm2, k1, k4, k12c_y):
        """
        Design compression capacity for minor-axis (y-axis) buckling Ndc_y (Cl 3.3.1.1).

        Ndc_y = phi * k1 * k4 * k12c_y * fc' * Ac   [N]

        Args:
            A_mm2 (float): Gross cross-sectional area Ac (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            k12c_y (float): Compression stability factor for S4 (minor axis).

        Returns:
            float: Ndc_y (N)
        """
        return self.PHI * k1 * k4 * k12c_y * self.get_fc() * A_mm2

    # ── Cl 3.4.1 — Tension capacity ──────────────────────────────────

    def tension_capacity(self, At_mm2, k1, k4, largest_dim_mm=None):
        """
        Design tension capacity Ndt (Cl 3.4.1).

        Ndt = phi * k1 * k4 * ft' * At   [N]

        Args:
            At_mm2 (float): Net tension area (mm²).
            k1 (float): Duration factor.
            k4 (float): Moisture factor.
            largest_dim_mm (float, optional): Largest cross-section dimension (mm)
                for ft' size adjustment.

        Returns:
            float: Ndt (N)
        """
        return (self.PHI * k1 * k4
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
            Mdz (float): Bending capacity major axis from bending_capacity_major() (N·mm).
            Mdy (float): Bending capacity minor axis from bending_capacity_minor() (N·mm).
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
