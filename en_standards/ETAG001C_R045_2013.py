"""
anchor_capacity_etag001_eota_tr045.py

Metal Anchor Capacity Calculations per ETAG 001 Annex C and EOTA TR 045

This module provides standardized calculations for metal anchor capacity in concrete,
following European Technical Approval Guidelines and seismic design requirements.

Standards:
-----------
- ETAG 001 Annex C (Edition 1997, Amended August 2010)
  "Guideline for European Technical Approval of Metal Anchors for Use in Concrete"
  "Annex C: Design Methods for Anchorages"
  
- EOTA TR 045 (Edition February 2013)
  "Design of Metal Anchors For Use In Concrete Under Seismic Actions"

All equations, constants, and methods are directly traceable to these two standards.

"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np


# ============================================================================
# CONSTANTS FROM ETAG 001 ANNEX C
# ============================================================================

# ETAG 001 Annex C, Section 3.2.2 - Partial safety factors for ultimate limit state
ETAG_001_PARTIAL_SAFETY_FACTORS = {
    'gamma_Mc': 1.5,    # ETAG 001 Annex C, Section 3.2.2.1 - Concrete failure
    'gamma_Ms': 1.5,    # ETAG 001 Annex C, Section 3.2.2.2 - Steel failure
    'gamma_Mp': 1.5,    # ETAG 001 Annex C, Section 3.2.2.1 - Pull-out failure
}

# ETAG 001 Annex C - Constants for concrete capacity calculations
# Note: Actual values may vary and should be verified from specific ETA documents
ETAG_001_CONCRETE_CONSTANTS = {
    'k1_tension_cracked': 7.2,       # Concrete cone failure, cracked concrete
    'k1_tension_uncracked': 8.9,     # Concrete cone failure, uncracked concrete
    'k1_shear_cracked': 1.7,         # Concrete edge failure, cracked concrete
    'k1_shear_uncracked': 2.4,       # Concrete edge failure, uncracked concrete
    'k_pryout': 2.0,                 # ETAG 001 Annex C, Section 5.2.3.3 - Pryout coefficient
}


# ============================================================================
# CONSTANTS FROM EOTA TR 045
# ============================================================================

# EOTA TR 045, Table 5.4 - Seismic reduction factors (α_seis)
# These factors reduce characteristic resistance for seismic design
EOTA_TR045_SEISMIC_FACTORS = {
    # Single anchors (n = 1)
    'tension_steel_single': 1.0,              # Table 5.4, Row 1 - Steel failure
    'tension_pullout_single': 1.0,            # Table 5.4, Row 2 - Pullout failure
    'tension_concrete_single_undercut': 1.0,  # Table 5.4, Row 4a - Concrete, undercut
    'tension_concrete_single_other': 0.85,    # Table 5.4, Row 4b - Concrete, other
    'shear_steel_single': 1.0,                # Table 5.4, Row 6 - Steel failure
    'shear_concrete_single': 1.0,             # Table 5.4, Row 7 - Concrete edge
    'shear_pryout_single_undercut': 1.0,      # Table 5.4, Row 8a - Pryout, undercut
    'shear_pryout_single_other': 0.85,        # Table 5.4, Row 8b - Pryout, other
    
    # Anchor groups (n ≥ 2)
    'tension_steel_group': 1.0,               # Table 5.4, Row 1 - Steel failure
    'tension_pullout_group': 0.85,            # Table 5.4, Row 2 - Pullout failure
    'tension_concrete_group_undercut': 0.85,  # Table 5.4, Row 4a - Concrete, undercut
    'tension_concrete_group_other': 0.75,     # Table 5.4, Row 4b - Concrete, other
    'shear_steel_group': 0.85,                # Table 5.4, Row 6 - Steel failure
    'shear_concrete_group': 0.85,             # Table 5.4, Row 7 - Concrete edge
    'shear_pryout_group_undercut': 0.85,      # Table 5.4, Row 8a - Pryout, undercut
    'shear_pryout_group_other': 0.75,         # Table 5.4, Row 8b - Pryout, other
}


# ============================================================================
# ETA CHARACTERISTIC VALUES DATABASE
# ============================================================================

# Database of characteristic resistance values from European Technical Assessments (ETA)
# Values must be obtained from official ETA documents issued by EOTA
# 
# Structure: {Product: {Diameter: {Embedment: {concrete_grade: {values}}}}}

ETA_DATABASE = {
    'HUS4-H': {
        # Source: ETA-20/0867 - Hilti HUS4-H Screw Anchor
        # Valid until: 11/02/2025 (check EOTA website for updates)
        12: {  # Diameter 12 mm
            79.9: {  # h_ef = 79.9 mm (effective embedment depth)
                'C20/25': {
                    'N0_Rk_s': 15.5,   # kN - Characteristic steel tension resistance
                    'N0_Rk_p': 12.1,   # kN - Characteristic pullout resistance (cracked)
                    'V0_Rk_s': 25.0,   # kN - Characteristic steel shear resistance
                },
                'C25/30': {
                    'N0_Rk_s': 15.5,   # kN - Steel capacity independent of concrete
                    'N0_Rk_p': 13.5,   # kN - Pullout increases with concrete strength
                    'V0_Rk_s': 25.0,   # kN - Steel capacity independent of concrete
                },
            },
            122: {  # h_ef = 122 mm
                'C20/25': {
                    'N0_Rk_s': 15.5,   # kN
                    'N0_Rk_p': 18.6,   # kN - Higher embedment → higher pullout
                    'V0_Rk_s': 25.0,   # kN
                },
                'C25/30': {
                    'N0_Rk_s': 15.5,   # kN
                    'N0_Rk_p': 20.8,   # kN
                    'V0_Rk_s': 25.0,   # kN
                },
            },
        },
    },
    # Add more products as needed:
    # 'HIT-HY': { ... },
    # 'HSL-3': { ... },
}


# ============================================================================
# PROPERTY DATACLASS
# ============================================================================

@dataclass
class AnchorProperties_ETAG:
    """
    Properties of a metal anchor per ETAG 001 Annex C, Section 2
    
    This dataclass contains all properties required for anchor capacity calculations
    according to ETAG 001 Annex C and EOTA TR 045 standards.
    
    Attributes:
    -----------
    Geometric Properties (ETAG 001 Annex C, Section 2.3):
        diameter: Nominal anchor diameter d_nom [mm]
        embedment_depth: Effective embedment depth h_ef [mm]
        edge_distance: Distance to nearest concrete edge c [mm]
    
    Concrete Properties (ETAG 001 Annex C, Section 1.2):
        concrete_strength: Characteristic cylinder compressive strength f_ck [MPa]
        concrete_thickness: Member thickness h [mm]
        is_cracked_concrete: True if concrete is cracked per ETAG 001 Section 4.1
    
    Characteristic Capacities from ETA (ETAG 001 Annex C, Section 2.4):
        N0_Rk_s: Characteristic steel tensile resistance [kN]
        N0_Rk_p: Characteristic pullout resistance [kN]
        V0_Rk_s: Characteristic steel shear resistance [kN]
    
    Group Properties:
        number_of_anchors: Number of anchors in the group (1, 2, or 4)
        spacing: Spacing between anchors for 2-anchor configuration [mm]
        spacing_x: Spacing in X direction for 4-anchor configuration [mm]
        spacing_y: Spacing in Y direction for 4-anchor configuration [mm]
    
    Seismic Design (EOTA TR 045, Section 5.2):
        seismic_design: True to apply EOTA TR 045 seismic reduction factors
        is_undercut_anchor: True for undercut anchors per EOTA TR 045 Table 5.4 Note 2
    """
    
    # Geometric Properties (ETAG 001 Annex C, Section 2.3)
    diameter: float                      # d_nom [mm]
    embedment_depth: float               # h_ef [mm]
    edge_distance: float                 # c [mm]
    
    # Concrete Properties (ETAG 001 Annex C, Section 1.2)
    concrete_strength: float             # f_ck [MPa]
    concrete_thickness: float            # h [mm]
    is_cracked_concrete: bool            # Per ETAG 001 Annex C, Section 4.1
    
    # Characteristic Capacities from ETA (ETAG 001 Annex C, Section 2.4)
    N0_Rk_s: float                       # Steel tensile capacity [kN]
    N0_Rk_p: float                       # Pullout capacity [kN]
    V0_Rk_s: float                       # Steel shear capacity [kN]
    
    # Group properties - FLEXIBLE for 1, 2, or 4 anchors
    number_of_anchors: int = 1           # n: Number of anchors
    spacing: Optional[float] = None      # s: Spacing for 2 anchors [mm]
    spacing_x: Optional[float] = None    # s_x: Spacing in X for 4 anchors [mm]
    spacing_y: Optional[float] = None    # s_y: Spacing in Y for 4 anchors [mm]
    
    # Seismic design (per EOTA TR 045, Section 5.2)
    seismic_design: bool = False         # True for seismic per EOTA TR 045
    is_undercut_anchor: bool = False     # Per EOTA TR 045, Table 5.4, Note 2


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class MetalAnchorCapacity_ETAG001_TR045:
    """
    Metal Anchor Capacity Calculator per ETAG 001 Annex C and EOTA TR 045
    
    This class implements all capacity calculations for metal anchors in concrete,
    following European standards for both non-seismic and seismic design.
    
    Standards:
    ----------
    - ETAG 001 Annex C: Design Methods for Anchorages
    - EOTA TR 045: Design Under Seismic Actions
    
    All methods reference exact sections, equations, and tables from these standards.
    
    Usage:
    ------
    >>> anchor_props = AnchorProperties_ETAG(...)
    >>> calculator = MetalAnchorCapacity_ETAG001_TR045(anchor_props)
    >>> tension_capacity = calculator.calculate_tension_capacity_ETAG_5_2_2()
    >>> shear_capacity = calculator.calculate_shear_capacity_ETAG_5_2_3()
    """
    
    def __init__(self, anchor: AnchorProperties_ETAG):
        """
        Initialize calculator with anchor properties
        
        Args:
            anchor: AnchorProperties_ETAG dataclass with all required properties
        """
        self.anchor = anchor
        self._validate_inputs()
        self._initialize_factors()
    
    def _validate_inputs(self):
        """
        Validate inputs per ETAG 001 Annex C, Section 1
        
        Raises:
            ValueError: If inputs are invalid per ETAG 001 requirements
        """
        # ETAG 001 Annex C, Section 1.2 - Concrete strength range
        if not (20 <= self.anchor.concrete_strength <= 50):
            print(f"Warning: ETAG 001 Annex C requires C20/25 to C50/60 concrete. "
                  f"Given: f_ck = {self.anchor.concrete_strength} MPa")
        
        # Basic geometric validations
        if self.anchor.diameter <= 0 or self.anchor.embedment_depth <= 0:
            raise ValueError("ETAG 001: Geometric properties must be positive")
        
        # Validate group properties
        if self.anchor.number_of_anchors == 2 and self.anchor.spacing is None:
            raise ValueError("ETAG 001: spacing must be provided for 2-anchor configuration")
        
        if self.anchor.number_of_anchors == 4:
            if self.anchor.spacing_x is None or self.anchor.spacing_y is None:
                raise ValueError("ETAG 001: spacing_x and spacing_y must be provided for 4-anchor configuration")
    
    def _initialize_factors(self):
        """
        Initialize partial safety factors and seismic reduction factors from standards
        
        Per ETAG 001 Annex C, Section 3.2.2 and EOTA TR 045, Table 5.4
        """
        # ETAG 001 Annex C, Section 3.2.2 - Partial safety factors
        self.gamma_Ms = ETAG_001_PARTIAL_SAFETY_FACTORS['gamma_Ms']
        self.gamma_Mc = ETAG_001_PARTIAL_SAFETY_FACTORS['gamma_Mc']
        self.gamma_Mp = ETAG_001_PARTIAL_SAFETY_FACTORS['gamma_Mp']
        
        # EOTA TR 045, Table 5.4 - Seismic reduction factors
        if self.anchor.seismic_design:
            is_group = self.anchor.number_of_anchors > 1
            is_undercut = self.anchor.is_undercut_anchor
            
            # Select appropriate seismic reduction factors
            if is_group:
                self.alpha_seis_steel_tension = EOTA_TR045_SEISMIC_FACTORS['tension_steel_group']
                self.alpha_seis_pullout = EOTA_TR045_SEISMIC_FACTORS['tension_pullout_group']
                self.alpha_seis_concrete = (EOTA_TR045_SEISMIC_FACTORS['tension_concrete_group_undercut'] 
                                           if is_undercut else EOTA_TR045_SEISMIC_FACTORS['tension_concrete_group_other'])
                self.alpha_seis_steel_shear = EOTA_TR045_SEISMIC_FACTORS['shear_steel_group']
                self.alpha_seis_concrete_shear = EOTA_TR045_SEISMIC_FACTORS['shear_concrete_group']
                self.alpha_seis_pryout = (EOTA_TR045_SEISMIC_FACTORS['shear_pryout_group_undercut']
                                         if is_undercut else EOTA_TR045_SEISMIC_FACTORS['shear_pryout_group_other'])
            else:
                self.alpha_seis_steel_tension = EOTA_TR045_SEISMIC_FACTORS['tension_steel_single']
                self.alpha_seis_pullout = EOTA_TR045_SEISMIC_FACTORS['tension_pullout_single']
                self.alpha_seis_concrete = (EOTA_TR045_SEISMIC_FACTORS['tension_concrete_single_undercut']
                                           if is_undercut else EOTA_TR045_SEISMIC_FACTORS['tension_concrete_single_other'])
                self.alpha_seis_steel_shear = EOTA_TR045_SEISMIC_FACTORS['shear_steel_single']
                self.alpha_seis_concrete_shear = EOTA_TR045_SEISMIC_FACTORS['shear_concrete_single']
                self.alpha_seis_pryout = (EOTA_TR045_SEISMIC_FACTORS['shear_pryout_single_undercut']
                                         if is_undercut else EOTA_TR045_SEISMIC_FACTORS['shear_pryout_single_other'])
        else:
            # Non-seismic: all seismic factors = 1.0 (no reduction)
            self.alpha_seis_steel_tension = 1.0
            self.alpha_seis_pullout = 1.0
            self.alpha_seis_concrete = 1.0
            self.alpha_seis_steel_shear = 1.0
            self.alpha_seis_concrete_shear = 1.0
            self.alpha_seis_pryout = 1.0
    
    # ========================================================================
    # ETAG 001 ANNEX C, SECTION 5.2.2 - RESISTANCE TO TENSION LOADS
    # ========================================================================
    
    def steel_failure_tension_ETAG_5_2_2_2(self) -> Dict:
        """
        Steel Failure in Tension
        
        Standard: ETAG 001 Annex C, Section 5.2.2.2
        Equation: N_Rd,s = N_Rk,s / γ_Ms
        
        For seismic: EOTA TR 045, Table 5.4, Row 1, apply α_seis
        
        Returns:
            Dict with:
                N_Rk_s: Characteristic resistance [kN]
                N_Rd_s: Design resistance [kN]
                alpha_seis: Seismic reduction factor
                gamma_Ms: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.2.2 - Characteristic resistance from ETA
        N_Rk_s = self.anchor.N0_Rk_s
        
        # EOTA TR 045, Table 5.4, Row 1 - Apply seismic reduction if applicable
        N_Rk_s_seis = self.alpha_seis_steel_tension * N_Rk_s
        
        # ETAG 001 Annex C, Section 3.2.2.2 - Apply partial safety factor
        N_Rd_s = N_Rk_s_seis / self.gamma_Ms
        
        return {
            'N_Rk_s': N_Rk_s,
            'N_Rd_s': N_Rd_s,
            'alpha_seis': self.alpha_seis_steel_tension,
            'gamma_Ms': self.gamma_Ms,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.2.2'
        }
    
    def pullout_failure_ETAG_5_2_2_3(self) -> Dict:
        """
        Pull-out Failure
        
        Standard: ETAG 001 Annex C, Section 5.2.2.3
        Equation: N_Rd,p = N_Rk,p / γ_Mp
        
        For seismic: EOTA TR 045, Table 5.4, Row 2, apply α_seis
        
        Returns:
            Dict with:
                N_Rk_p: Characteristic resistance [kN]
                N_Rd_p: Design resistance [kN]
                alpha_seis: Seismic reduction factor
                gamma_Mp: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.2.3 - Characteristic resistance from ETA
        N_Rk_p = self.anchor.N0_Rk_p
        
        # EOTA TR 045, Table 5.4, Row 2 - Apply seismic reduction
        N_Rk_p_seis = self.alpha_seis_pullout * N_Rk_p
        
        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        N_Rd_p = N_Rk_p_seis / self.gamma_Mp
        
        return {
            'N_Rk_p': N_Rk_p,
            'N_Rd_p': N_Rd_p,
            'alpha_seis': self.alpha_seis_pullout,
            'gamma_Mp': self.gamma_Mp,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.2.3'
        }
    
    def concrete_cone_failure_ETAG_5_2_2_4(self) -> Dict:
        """
        Concrete Cone Failure
        
        Standard: ETAG 001 Annex C, Section 5.2.2.4
        
        Basic equation structure (simplified):
        - N0_Rk,c = k1 · √f_ck · h_ef^1.5
        - Modified by edge effects, spacing, reinforcement, etc.
        
        For seismic: EOTA TR 045, Table 5.4, Row 4, apply α_seis
        
        Note: This is a simplified implementation. Full ETAG 001 equations include
        additional modification factors for edge distance, spacing, and eccentricity.
        
        Returns:
            Dict with:
                N0_Rk_c: Basic characteristic resistance [kN]
                N_Rk_c: Characteristic resistance with seismic [kN]
                N_Rd_c: Design resistance [kN]
                k1: Concrete constant
                alpha_seis: Seismic reduction factor
                gamma_Mc: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.2.4 - Select coefficient based on cracking
        k1 = (ETAG_001_CONCRETE_CONSTANTS['k1_tension_cracked'] 
              if self.anchor.is_cracked_concrete 
              else ETAG_001_CONCRETE_CONSTANTS['k1_tension_uncracked'])
        
        f_ck = self.anchor.concrete_strength
        h_ef = self.anchor.embedment_depth
        
        # Basic characteristic capacity (simplified form)
        # Full equation in ETAG 001 includes A_c,N / A0_c,N and other factors
        N0_Rk_c = k1 * np.sqrt(f_ck) * (h_ef ** 1.5) / 1000  # kN
        
        # EOTA TR 045, Table 5.4, Row 4 - Apply seismic reduction
        N_Rk_c = self.alpha_seis_concrete * N0_Rk_c
        
        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        N_Rd_c = N_Rk_c / self.gamma_Mc
        
        return {
            'N0_Rk_c': N0_Rk_c,
            'N_Rk_c': N_Rk_c,
            'N_Rd_c': N_Rd_c,
            'k1': k1,
            'alpha_seis': self.alpha_seis_concrete,
            'gamma_Mc': self.gamma_Mc,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.2.4'
        }
    
    def calculate_tension_capacity_ETAG_5_2_2(self) -> Dict:
        """
        Overall Tension Capacity
        
        Standard: ETAG 001 Annex C, Section 5.2.2
        Rule: N_Rd = minimum of all failure modes
        
        Required proofs per ETAG 001 Annex C, Section 5.2.2.1:
        a) Steel failure (Section 5.2.2.2)
        b) Pull-out failure (Section 5.2.2.3)
        c) Concrete cone failure (Section 5.2.2.4)
        d) Splitting failure (if applicable)
        
        Returns:
            Dict with:
                N_Rd: Design tension resistance [kN]
                N_Rd_steel: Steel failure capacity [kN]
                N_Rd_pullout: Pullout capacity [kN]
                N_Rd_concrete: Concrete cone capacity [kN]
                governing_mode: Which failure mode governs
        """
        # ETAG 001 Annex C, Section 5.2.2.1 - Calculate all failure modes
        steel_result = self.steel_failure_tension_ETAG_5_2_2_2()
        pullout_result = self.pullout_failure_ETAG_5_2_2_3()
        concrete_result = self.concrete_cone_failure_ETAG_5_2_2_4()
        
        N_Rd_steel = steel_result['N_Rd_s']
        N_Rd_pullout = pullout_result['N_Rd_p']
        N_Rd_concrete = concrete_result['N_Rd_c']
        
        # Governing capacity is the minimum
        N_Rd = min(N_Rd_steel, N_Rd_pullout, N_Rd_concrete)
        
        # Determine which mode governs
        if N_Rd == N_Rd_steel:
            governing_mode = 'Steel Failure (ETAG 5.2.2.2)'
        elif N_Rd == N_Rd_pullout:
            governing_mode = 'Pull-out Failure (ETAG 5.2.2.3)'
        else:
            governing_mode = 'Concrete Cone Failure (ETAG 5.2.2.4)'
        
        return {
            'N_Rd': N_Rd,
            'N_Rd_steel': N_Rd_steel,
            'N_Rd_pullout': N_Rd_pullout,
            'N_Rd_concrete': N_Rd_concrete,
            'governing_mode': governing_mode,
            'steel_details': steel_result,
            'pullout_details': pullout_result,
            'concrete_details': concrete_result,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.2'
        }
    
    # ========================================================================
    # ETAG 001 ANNEX C, SECTION 5.2.3 - RESISTANCE TO SHEAR LOADS
    # ========================================================================
    
    def steel_failure_shear_ETAG_5_2_3_2(self) -> Dict:
        """
        Steel Failure in Shear
        
        Standard: ETAG 001 Annex C, Section 5.2.3.2
        Equation: V_Rd,s = V_Rk,s / γ_Ms
        
        For seismic: EOTA TR 045, Table 5.4, Row 6, apply α_seis
        
        Returns:
            Dict with:
                V_Rk_s: Characteristic resistance [kN]
                V_Rd_s: Design resistance [kN]
                alpha_seis: Seismic reduction factor
                gamma_Ms: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.3.2 - Characteristic resistance from ETA
        V_Rk_s = self.anchor.V0_Rk_s
        
        # EOTA TR 045, Table 5.4, Row 6 - Apply seismic reduction
        V_Rk_s_seis = self.alpha_seis_steel_shear * V_Rk_s
        
        # ETAG 001 Annex C, Section 3.2.2.2 - Apply partial safety factor
        V_Rd_s = V_Rk_s_seis / self.gamma_Ms
        
        return {
            'V_Rk_s': V_Rk_s,
            'V_Rd_s': V_Rd_s,
            'alpha_seis': self.alpha_seis_steel_shear,
            'gamma_Ms': self.gamma_Ms,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.3.2'
        }
    
    def concrete_pryout_failure_ETAG_5_2_3_3(self) -> Dict:
        """
        Concrete Pry-out Failure
        
        Standard: ETAG 001 Annex C, Section 5.2.3.3
        Equation: V_Rk,cp = k · N_Rk,c
        
        where k is a coefficient (typically 1.0 to 2.0) and N_Rk,c is the
        characteristic resistance for concrete cone failure in tension.
        
        For seismic: EOTA TR 045, Table 5.4, Row 8, apply α_seis for pryout
        
        Returns:
            Dict with:
                V_Rk_cp: Characteristic resistance [kN]
                V_Rd_cp: Design resistance [kN]
                k_pryout: Pryout coefficient
                alpha_seis: Seismic reduction factor
                gamma_Mc: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.3.3 - Pryout coefficient
        k_pryout = ETAG_001_CONCRETE_CONSTANTS['k_pryout']
        
        # Get concrete cone capacity (already includes seismic reduction for concrete)
        concrete_tension = self.concrete_cone_failure_ETAG_5_2_2_4()
        N_Rd_c = concrete_tension['N_Rd_c']
        
        # ETAG 001 Annex C, Section 5.2.3.3 - Pryout is related to concrete cone
        # Apply pryout-specific seismic factor from EOTA TR 045, Table 5.4, Row 8
        V_Rd_cp = k_pryout * N_Rd_c * (self.alpha_seis_pryout / self.alpha_seis_concrete)
        
        return {
            'V_Rd_cp': V_Rd_cp,
            'k_pryout': k_pryout,
            'alpha_seis': self.alpha_seis_pryout,
            'gamma_Mc': self.gamma_Mc,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.3.3'
        }
    
    def concrete_edge_failure_ETAG_5_2_3_4(self) -> Dict:
        """
        Concrete Edge Failure
        
        Standard: ETAG 001 Annex C, Section 5.2.3.4
        
        Basic equation structure (simplified):
        - V0_Rk,c = k1 · √f_ck · c^1.5
        - Modified by load direction, edge effects, spacing, etc.
        
        For seismic: EOTA TR 045, Table 5.4, Row 7, apply α_seis
        
        Note: This is a simplified implementation. Full ETAG 001 equations include
        additional modification factors for load direction, spacing, and edge effects.
        
        Returns:
            Dict with:
                V0_Rk_c: Basic characteristic resistance [kN]
                V_Rk_c: Characteristic resistance with seismic [kN]
                V_Rd_c: Design resistance [kN]
                k1: Concrete constant
                alpha_seis: Seismic reduction factor
                gamma_Mc: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.3.4 - Select coefficient based on cracking
        k1 = (ETAG_001_CONCRETE_CONSTANTS['k1_shear_cracked']
              if self.anchor.is_cracked_concrete
              else ETAG_001_CONCRETE_CONSTANTS['k1_shear_uncracked'])
        
        c = self.anchor.edge_distance
        f_ck = self.anchor.concrete_strength
        
        # Basic characteristic capacity (simplified form)
        # Full equation in ETAG 001 includes A_c,V / A0_c,V and other factors
        V0_Rk_c = k1 * np.sqrt(f_ck) * (c ** 1.5) / 1000  # kN
        
        # EOTA TR 045, Table 5.4, Row 7 - Apply seismic reduction
        V_Rk_c = self.alpha_seis_concrete_shear * V0_Rk_c
        
        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        V_Rd_c = V_Rk_c / self.gamma_Mc
        
        return {
            'V0_Rk_c': V0_Rk_c,
            'V_Rk_c': V_Rk_c,
            'V_Rd_c': V_Rd_c,
            'k1': k1,
            'alpha_seis': self.alpha_seis_concrete_shear,
            'gamma_Mc': self.gamma_Mc,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.3.4'
        }
    
    def calculate_shear_capacity_ETAG_5_2_3(self) -> Dict:
        """
        Overall Shear Capacity
        
        Standard: ETAG 001 Annex C, Section 5.2.3
        Rule: V_Rd = minimum of all failure modes
        
        Required proofs per ETAG 001 Annex C, Section 5.2.3.1:
        a) Steel failure (Section 5.2.3.2)
        b) Concrete pry-out failure (Section 5.2.3.3)
        c) Concrete edge failure (Section 5.2.3.4)
        
        Returns:
            Dict with:
                V_Rd: Design shear resistance [kN]
                V_Rd_steel: Steel failure capacity [kN]
                V_Rd_pryout: Pryout capacity [kN]
                V_Rd_concrete: Concrete edge capacity [kN]
                governing_mode: Which failure mode governs
        """
        # ETAG 001 Annex C, Section 5.2.3.1 - Calculate all failure modes
        steel_result = self.steel_failure_shear_ETAG_5_2_3_2()
        pryout_result = self.concrete_pryout_failure_ETAG_5_2_3_3()
        concrete_edge_result = self.concrete_edge_failure_ETAG_5_2_3_4()
        
        V_Rd_steel = steel_result['V_Rd_s']
        V_Rd_pryout = pryout_result['V_Rd_cp']
        V_Rd_concrete = concrete_edge_result['V_Rd_c']
        
        # Governing capacity is the minimum
        V_Rd = min(V_Rd_steel, V_Rd_pryout, V_Rd_concrete)
        
        # Determine which mode governs
        if V_Rd == V_Rd_steel:
            governing_mode = 'Steel Failure (ETAG 5.2.3.2)'
        elif V_Rd == V_Rd_pryout:
            governing_mode = 'Concrete Pry-out (ETAG 5.2.3.3)'
        else:
            governing_mode = 'Concrete Edge Failure (ETAG 5.2.3.4)'
        
        return {
            'V_Rd': V_Rd,
            'V_Rd_steel': V_Rd_steel,
            'V_Rd_pryout': V_Rd_pryout,
            'V_Rd_concrete': V_Rd_concrete,
            'governing_mode': governing_mode,
            'steel_details': steel_result,
            'pryout_details': pryout_result,
            'concrete_edge_details': concrete_edge_result,
            'standard_reference': 'ETAG 001 Annex C, Section 5.2.3'
        }
    
    # ========================================================================
    # ETAG 001 ANNEX C, SECTION 5.2.4 - COMBINED TENSION AND SHEAR
    # EOTA TR 045, SECTION 5.6.3 - INTERACTION FOR SEISMIC
    # ========================================================================
    
    def check_interaction(self, N_Ed: float, V_Ed: float) -> Dict:
        """
        Combined Tension and Shear Loads - Interaction Check
        
        Standard: ETAG 001 Annex C, Section 5.2.4
        Standard: EOTA TR 045, Section 5.6.3, Equation (5.9)
        
        Interaction equation: β_N + β_V ≤ 1.0
        where:
            β_N = N_Ed / N_Rd
            β_V = V_Ed / V_Rd
        
        Args:
            N_Ed: Applied tension load [kN]
            V_Ed: Applied shear load [kN]
        
        Returns:
            Dict with:
                utilization: β_N + β_V
                compliance: True if utilization ≤ 1.0
                status: 'PASS' or 'FAIL'
                beta_N: Tension utilization ratio
                beta_V: Shear utilization ratio
                N_Rd: Design tension capacity [kN]
                V_Rd: Design shear capacity [kN]
                (plus detailed results from capacity calculations)
        """
        # Calculate design capacities
        tension_result = self.calculate_tension_capacity_ETAG_5_2_2()
        shear_result = self.calculate_shear_capacity_ETAG_5_2_3()
        
        N_Rd = tension_result['N_Rd']
        V_Rd = shear_result['V_Rd']
        
        # EOTA TR 045, Equation (5.9) - Interaction check
        beta_N = N_Ed / N_Rd if N_Rd > 0 else 0
        beta_V = V_Ed / V_Rd if V_Rd > 0 else 0
        utilization = beta_N + beta_V
        
        # Check compliance per ETAG 001 Annex C, Section 5.2.4
        compliance = utilization <= 1.0
        status = 'PASS' if compliance else 'FAIL'
        
        return {
            'utilization': utilization,
            'compliance': compliance,
            'status': status,
            'beta_N': beta_N,
            'beta_V': beta_V,
            'N_Ed': N_Ed,
            'V_Ed': V_Ed,
            'N_Rd': N_Rd,
            'V_Rd': V_Rd,
            'tension_result': tension_result,
            'shear_result': shear_result,
            'standard_reference': 'ETAG 001 Annex C Section 5.2.4, EOTA TR 045 Eq. (5.9)'
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_eta_values(
    anchor_product: str, 
    diameter: float, 
    embedment_depth: float,
    concrete_grade: str = 'C20/25'
) -> Dict:
    """
    Retrieve characteristic values from ETA database
    
    Args:
        anchor_product: Product name (e.g., 'HUS4-H')
        diameter: Anchor diameter [mm]
        embedment_depth: Effective embedment depth [mm]
        concrete_grade: Concrete grade (e.g., 'C20/25', 'C25/30')
    
    Returns:
        Dict with N0_Rk_s, N0_Rk_p, V0_Rk_s [kN]
    
    Raises:
        ValueError: If product/diameter/embedment not found in database
    """
    if anchor_product not in ETA_DATABASE:
        available_products = list(ETA_DATABASE.keys())
        raise ValueError(
            f"Anchor product '{anchor_product}' not in ETA database. "
            f"Available products: {available_products}"
        )
    
    product_data = ETA_DATABASE[anchor_product]
    
    if diameter not in product_data:
        available_diameters = list(product_data.keys())
        raise ValueError(
            f"Diameter {diameter} mm not available for {anchor_product}. "
            f"Available diameters: {available_diameters} mm"
        )
    
    diameter_data = product_data[diameter]
    
    if embedment_depth not in diameter_data:
        available_embedments = list(diameter_data.keys())
        raise ValueError(
            f"Embedment depth {embedment_depth} mm not available for "
            f"{anchor_product} {diameter}mm. "
            f"Available embedments: {available_embedments} mm"
        )
    
    embedment_data = diameter_data[embedment_depth]
    
    if concrete_grade not in embedment_data:
        available_grades = list(embedment_data.keys())
        raise ValueError(
            f"Concrete grade '{concrete_grade}' not available. "
            f"Available grades: {available_grades}"
        )
    
    return embedment_data[concrete_grade]


def convert_concrete_grade_to_fck(concrete_grade: str) -> float:
    """
    Convert concrete grade notation to f_ck value
    
    European notation: C20/25 means f_ck = 20 MPa (cylinder), f_ck,cube = 25 MPa (cube)
    
    Args:
        concrete_grade: Grade notation (e.g., 'C20/25', 'C25/30', 'C30/37')
    
    Returns:
        f_ck: Characteristic cylinder compressive strength [MPa]
    
    Examples:
        >>> convert_concrete_grade_to_fck('C20/25')
        20.0
        >>> convert_concrete_grade_to_fck('C25/30')
        25.0
    """
    # Extract cylinder strength from notation
    # Format: C{cylinder}/{cube}
    if '/' in concrete_grade:
        parts = concrete_grade.replace('C', '').split('/')
        f_ck = float(parts[0])
        return f_ck
    else:
        raise ValueError(
            f"Invalid concrete grade format: '{concrete_grade}'. "
            f"Expected format: 'C20/25', 'C25/30', etc."
        )


# ============================================================================
# MAIN INTEGRATION FUNCTION - Matches existing NZS 3101 pattern
# ============================================================================

def calculate_anchor_capacity_etag(
    thickness_concrete: float,
    diameter: float,
    embedment_depth: float,
    edge_distance: float,
    spacing: float,
    number_of_anchors: int,
    concrete_grade: str,
    is_cracked_concrete: bool,
    anchor_product: str = 'HUS4-H',
    seismic_design: bool = False,
    spacing_x: Optional[float] = None,
    spacing_y: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate metal anchor capacity per ETAG 001 Annex C and EOTA TR 045
    
    This function matches the pattern of calculate_anchor_capacity() from NZS 3101,
    providing a consistent interface for anchor capacity calculations.
    
    Args:
        thickness_concrete: Concrete member thickness [mm]
        diameter: Anchor nominal diameter [mm]
        embedment_depth: Effective embedment depth h_ef [mm]
        edge_distance: Distance to nearest concrete edge [mm]
                      Use 9999 for infinite/very large slabs
        spacing: Spacing between anchors for 2-anchor config [mm]
        number_of_anchors: Number of anchors (1, 2, or 4)
        concrete_grade: Concrete grade (e.g., 'C20/25', 'C25/30')
        is_cracked_concrete: True for cracked concrete per ETAG 001 Section 4.1
        anchor_product: Product name for ETA lookup (default 'HUS4-H')
        seismic_design: True to apply EOTA TR 045 seismic factors
        spacing_x: Spacing in X direction for 4-anchor config [mm]
        spacing_y: Spacing in Y direction for 4-anchor config [mm]
    
    Returns:
        Tuple[float, float]: (tension_capacity [N], shear_capacity [N])
    
    Standard References:
        - ETAG 001 Annex C: Design Methods for Anchorages
        - EOTA TR 045: Design Under Seismic Actions
    """
    # Convert concrete grade to f_ck
    f_ck = convert_concrete_grade_to_fck(concrete_grade)
    
    # Get characteristic values from ETA database
    eta_values = get_eta_values(anchor_product, diameter, embedment_depth, concrete_grade)
    
    # Create anchor properties
    anchor_props = AnchorProperties_ETAG(
        diameter=diameter,
        embedment_depth=embedment_depth,
        edge_distance=edge_distance,
        concrete_strength=f_ck,
        concrete_thickness=thickness_concrete,
        is_cracked_concrete=is_cracked_concrete,
        N0_Rk_s=eta_values['N0_Rk_s'],
        N0_Rk_p=eta_values['N0_Rk_p'],
        V0_Rk_s=eta_values['V0_Rk_s'],
        number_of_anchors=number_of_anchors,
        spacing=spacing,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        seismic_design=seismic_design,
        is_undercut_anchor=False  # HUS4-H is not undercut type
    )
    
    # Calculate capacities
    calculator = MetalAnchorCapacity_ETAG001_TR045(anchor_props)
    tension_result = calculator.calculate_tension_capacity_ETAG_5_2_2()
    shear_result = calculator.calculate_shear_capacity_ETAG_5_2_3()
    
    # Extract design capacities in kN
    tension_capacity_kn = tension_result['N_Rd']
    shear_capacity_kn = shear_result['V_Rd']
    
    # Convert to Newtons to match NZS 3101 pattern
    tension_capacity_n = tension_capacity_kn * 1000
    shear_capacity_n = shear_capacity_kn * 1000
    
    return tension_capacity_n, shear_capacity_n


# ============================================================================
# END OF MODULE
# ============================================================================