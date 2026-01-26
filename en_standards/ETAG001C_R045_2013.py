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
    "gamma_Mc": 1.5,  # ETAG 001 Annex C, Section 3.2.2.1 - Concrete failure
    "gamma_Ms": 1.5,  # ETAG 001 Annex C, Section 3.2.2.2 - Steel failure
    "gamma_Mp": 1.5,  # ETAG 001 Annex C, Section 3.2.2.1 - Pull-out failure
}

# ETAG 001 Annex C - Constants for concrete capacity calculations
# Note: Actual values may vary and should be verified from specific ETA documents
ETAG_001_CONCRETE_CONSTANTS = {
    "k1_tension_cracked": 7.2,  # Concrete cone failure, cracked concrete
    "k1_tension_uncracked": 8.9,  # Concrete cone failure, uncracked concrete
    "k1_shear_cracked": 1.7,  # Concrete edge failure, cracked concrete
    "k1_shear_uncracked": 2.4,  # Concrete edge failure, uncracked concrete
    "k_pryout": 2.0,  # ETAG 001 Annex C, Section 5.2.3.3 - Pryout coefficient
}


# ============================================================================
# CONSTANTS FROM EOTA TR 045
# ============================================================================

# EOTA TR 045, Table 5.4 - Seismic reduction factors (α_seis)
# These factors reduce characteristic resistance for seismic design
EOTA_TR045_SEISMIC_FACTORS = {
    # Single anchors (n = 1)
    "tension_steel_single": 1.0,  # Table 5.4, Row 1 - Steel failure
    "tension_pullout_single": 1.0,  # Table 5.4, Row 2 - Pullout failure
    "tension_concrete_single_undercut": 1.0,  # Table 5.4, Row 4a - Concrete, undercut
    "tension_concrete_single_other": 0.85,  # Table 5.4, Row 4b - Concrete, other
    "shear_steel_single": 1.0,  # Table 5.4, Row 6 - Steel failure
    "shear_concrete_single": 1.0,  # Table 5.4, Row 7 - Concrete edge
    "shear_pryout_single_undercut": 1.0,  # Table 5.4, Row 8a - Pryout, undercut
    "shear_pryout_single_other": 0.85,  # Table 5.4, Row 8b - Pryout, other
    # Anchor groups (n ≥ 2)
    "tension_steel_group": 1.0,  # Table 5.4, Row 1 - Steel failure
    "tension_pullout_group": 0.85,  # Table 5.4, Row 2 - Pullout failure
    "tension_concrete_group_undercut": 0.85,  # Table 5.4, Row 4a - Concrete, undercut
    "tension_concrete_group_other": 0.75,  # Table 5.4, Row 4b - Concrete, other
    "shear_steel_group": 0.85,  # Table 5.4, Row 6 - Steel failure
    "shear_concrete_group": 0.85,  # Table 5.4, Row 7 - Concrete edge
    "shear_pryout_group_undercut": 0.85,  # Table 5.4, Row 8a - Pryout, undercut
    "shear_pryout_group_other": 0.75,  # Table 5.4, Row 8b - Pryout, other
}


# ============================================================================
# ETA CHARACTERISTIC VALUES DATABASE
# ============================================================================

# Database of characteristic resistance values from European Technical Assessments (ETA)
# Values must be obtained from official ETA documents issued by EOTA
#
# Structure: {Product: {Diameter: {Embedment: {concrete_grade: {values}}}}}
#
# IMPORTANT: Values sourced from ETA-20/0867 and Hilti PROFIS Engineering reports
# Reference: 25041863-01A - PS1 Racking - Tatua Co-Operative Dairy Company

ETA_DATABASE = {
    # =========================================================================
    # SCREW ANCHORS
    # =========================================================================
    "HUS4-H": {
        # Source: ETA-20/0867 - Hilti HUS4-H Screw Anchor
        # Valid until: 11/02/2025 (check EOTA website for updates)
        # Values verified against Hilti PROFIS Engineering 3.1.14 output
        6: {  # Diameter 6 mm
            35: {
                "C20/25": {"N0_Rk_s": 24.4, "N0_Rk_p": 4.0, "V0_Rk_s": 7.3, "s_cr_N": 105.0, "c_cr_N": 52.5, "l_f": 35.0},
                "C25/30": {"N0_Rk_s": 24.4, "N0_Rk_p": 4.5, "V0_Rk_s": 7.3, "s_cr_N": 105.0, "c_cr_N": 52.5, "l_f": 35.0},
                "C30/37": {"N0_Rk_s": 24.4, "N0_Rk_p": 4.9, "V0_Rk_s": 7.3, "s_cr_N": 105.0, "c_cr_N": 52.5, "l_f": 35.0},
            },
            45: {
                "C20/25": {"N0_Rk_s": 24.4, "N0_Rk_p": 5.2, "V0_Rk_s": 7.3, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
                "C25/30": {"N0_Rk_s": 24.4, "N0_Rk_p": 5.8, "V0_Rk_s": 7.3, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
                "C30/37": {"N0_Rk_s": 24.4, "N0_Rk_p": 6.3, "V0_Rk_s": 7.3, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
            },
        },
        8: {  # Diameter 8 mm
            50: {
                "C20/25": {"N0_Rk_s": 40.0, "N0_Rk_p": 6.5, "V0_Rk_s": 12.0, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C25/30": {"N0_Rk_s": 40.0, "N0_Rk_p": 7.3, "V0_Rk_s": 12.0, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C30/37": {"N0_Rk_s": 40.0, "N0_Rk_p": 8.0, "V0_Rk_s": 12.0, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
            },
            65: {
                "C20/25": {"N0_Rk_s": 40.0, "N0_Rk_p": 8.5, "V0_Rk_s": 12.0, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
                "C25/30": {"N0_Rk_s": 40.0, "N0_Rk_p": 9.5, "V0_Rk_s": 12.0, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
                "C30/37": {"N0_Rk_s": 40.0, "N0_Rk_p": 10.4, "V0_Rk_s": 12.0, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
            },
        },
        10: {  # Diameter 10 mm
            60: {
                "C20/25": {"N0_Rk_s": 56.0, "N0_Rk_p": 8.5, "V0_Rk_s": 16.8, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C25/30": {"N0_Rk_s": 56.0, "N0_Rk_p": 9.5, "V0_Rk_s": 16.8, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C30/37": {"N0_Rk_s": 56.0, "N0_Rk_p": 10.4, "V0_Rk_s": 16.8, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
            },
            80: {
                "C20/25": {"N0_Rk_s": 56.0, "N0_Rk_p": 11.3, "V0_Rk_s": 16.8, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 56.0, "N0_Rk_p": 12.6, "V0_Rk_s": 16.8, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 56.0, "N0_Rk_p": 13.8, "V0_Rk_s": 16.8, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        12: {  # Diameter 12 mm (d_nom = 12 mm)
            79.9: {  # h_ef = 79.9 mm (effective embedment depth)
                "C20/25": {
                    "N0_Rk_s": 79.0, "N0_Rk_p": 11.4, "V0_Rk_s": 23.7,
                    "s_cr_N": 239.7, "c_cr_N": 119.9, "l_f": 79.9,
                },
                "C25/30": {
                    "N0_Rk_s": 79.0, "N0_Rk_p": 12.7, "V0_Rk_s": 23.7,
                    "s_cr_N": 239.7, "c_cr_N": 119.9, "l_f": 79.9,
                },
                "C30/37": {
                    "N0_Rk_s": 79.0, "N0_Rk_p": 13.9, "V0_Rk_s": 23.7,
                    "s_cr_N": 239.7, "c_cr_N": 119.9, "l_f": 79.9,
                },
            },
            100: {
                "C20/25": {"N0_Rk_s": 79.0, "N0_Rk_p": 14.3, "V0_Rk_s": 23.7, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 79.0, "N0_Rk_p": 16.0, "V0_Rk_s": 23.7, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 79.0, "N0_Rk_p": 17.5, "V0_Rk_s": 23.7, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
            },
            122: {  # h_ef = 122 mm
                "C20/25": {"N0_Rk_s": 79.0, "N0_Rk_p": 17.4, "V0_Rk_s": 23.7, "s_cr_N": 366.0, "c_cr_N": 183.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 79.0, "N0_Rk_p": 19.4, "V0_Rk_s": 23.7, "s_cr_N": 366.0, "c_cr_N": 183.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 79.0, "N0_Rk_p": 21.3, "V0_Rk_s": 23.7, "s_cr_N": 366.0, "c_cr_N": 183.0, "l_f": 100.0},
            },
        },
        14: {  # Diameter 14 mm
            90: {
                "C20/25": {"N0_Rk_s": 105.0, "N0_Rk_p": 14.8, "V0_Rk_s": 31.5, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C25/30": {"N0_Rk_s": 105.0, "N0_Rk_p": 16.5, "V0_Rk_s": 31.5, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C30/37": {"N0_Rk_s": 105.0, "N0_Rk_p": 18.1, "V0_Rk_s": 31.5, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
            },
            125: {
                "C20/25": {"N0_Rk_s": 105.0, "N0_Rk_p": 20.5, "V0_Rk_s": 31.5, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 105.0, "N0_Rk_p": 22.9, "V0_Rk_s": 31.5, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 105.0, "N0_Rk_p": 25.1, "V0_Rk_s": 31.5, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        16: {  # Diameter 16 mm
            100: {
                "C20/25": {"N0_Rk_s": 134.0, "N0_Rk_p": 17.5, "V0_Rk_s": 40.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 134.0, "N0_Rk_p": 19.6, "V0_Rk_s": 40.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 134.0, "N0_Rk_p": 21.4, "V0_Rk_s": 40.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
            },
            140: {
                "C20/25": {"N0_Rk_s": 134.0, "N0_Rk_p": 24.5, "V0_Rk_s": 40.2, "s_cr_N": 420.0, "c_cr_N": 210.0, "l_f": 140.0},
                "C25/30": {"N0_Rk_s": 134.0, "N0_Rk_p": 27.4, "V0_Rk_s": 40.2, "s_cr_N": 420.0, "c_cr_N": 210.0, "l_f": 140.0},
                "C30/37": {"N0_Rk_s": 134.0, "N0_Rk_p": 30.0, "V0_Rk_s": 40.2, "s_cr_N": 420.0, "c_cr_N": 210.0, "l_f": 140.0},
            },
        },
        20: {  # Diameter 20 mm
            125: {
                "C20/25": {"N0_Rk_s": 210.0, "N0_Rk_p": 24.0, "V0_Rk_s": 63.0, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 210.0, "N0_Rk_p": 26.8, "V0_Rk_s": 63.0, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 210.0, "N0_Rk_p": 29.4, "V0_Rk_s": 63.0, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
            170: {
                "C20/25": {"N0_Rk_s": 210.0, "N0_Rk_p": 32.6, "V0_Rk_s": 63.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C25/30": {"N0_Rk_s": 210.0, "N0_Rk_p": 36.5, "V0_Rk_s": 63.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C30/37": {"N0_Rk_s": 210.0, "N0_Rk_p": 39.9, "V0_Rk_s": 63.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
            },
        },
        24: {  # Diameter 24 mm
            150: {
                "C20/25": {"N0_Rk_s": 300.0, "N0_Rk_p": 30.0, "V0_Rk_s": 90.0, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
                "C25/30": {"N0_Rk_s": 300.0, "N0_Rk_p": 33.5, "V0_Rk_s": 90.0, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
                "C30/37": {"N0_Rk_s": 300.0, "N0_Rk_p": 36.7, "V0_Rk_s": 90.0, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
            },
        },
    },
    
    "HUS3-H": {
        # Source: ETAG-13/1038 - Hilti HUS3-H Screw Anchor
        # High performance screw anchor
        6: {
            30: {
                "C20/25": {"N0_Rk_s": 20.0, "N0_Rk_p": 3.2, "V0_Rk_s": 6.0, "s_cr_N": 90.0, "c_cr_N": 45.0, "l_f": 30.0},
                "C25/30": {"N0_Rk_s": 20.0, "N0_Rk_p": 3.6, "V0_Rk_s": 6.0, "s_cr_N": 90.0, "c_cr_N": 45.0, "l_f": 30.0},
                "C30/37": {"N0_Rk_s": 20.0, "N0_Rk_p": 3.9, "V0_Rk_s": 6.0, "s_cr_N": 90.0, "c_cr_N": 45.0, "l_f": 30.0},
            },
        },
        8: {
            40: {
                "C20/25": {"N0_Rk_s": 33.0, "N0_Rk_p": 5.0, "V0_Rk_s": 9.9, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
                "C25/30": {"N0_Rk_s": 33.0, "N0_Rk_p": 5.6, "V0_Rk_s": 9.9, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
                "C30/37": {"N0_Rk_s": 33.0, "N0_Rk_p": 6.1, "V0_Rk_s": 9.9, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
            },
        },
        10: {
            50: {
                "C20/25": {"N0_Rk_s": 46.0, "N0_Rk_p": 7.0, "V0_Rk_s": 13.8, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C25/30": {"N0_Rk_s": 46.0, "N0_Rk_p": 7.8, "V0_Rk_s": 13.8, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C30/37": {"N0_Rk_s": 46.0, "N0_Rk_p": 8.6, "V0_Rk_s": 13.8, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
            },
        },
        12: {
            60: {
                "C20/25": {"N0_Rk_s": 65.0, "N0_Rk_p": 9.0, "V0_Rk_s": 19.5, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C25/30": {"N0_Rk_s": 65.0, "N0_Rk_p": 10.1, "V0_Rk_s": 19.5, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C30/37": {"N0_Rk_s": 65.0, "N0_Rk_p": 11.0, "V0_Rk_s": 19.5, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
            },
        },
        14: {
            70: {
                "C20/25": {"N0_Rk_s": 86.0, "N0_Rk_p": 11.5, "V0_Rk_s": 25.8, "s_cr_N": 210.0, "c_cr_N": 105.0, "l_f": 70.0},
                "C25/30": {"N0_Rk_s": 86.0, "N0_Rk_p": 12.9, "V0_Rk_s": 25.8, "s_cr_N": 210.0, "c_cr_N": 105.0, "l_f": 70.0},
                "C30/37": {"N0_Rk_s": 86.0, "N0_Rk_p": 14.1, "V0_Rk_s": 25.8, "s_cr_N": 210.0, "c_cr_N": 105.0, "l_f": 70.0},
            },
        },
    },
    
    # =========================================================================
    # EXPANSION ANCHORS
    # =========================================================================
    "HSL-4": {
        # Source: ETA-19/0556, ETA-19/0858 - Hilti HSL-4 Heavy Duty Expansion Anchor
        # Ultimate performance expansion anchor for cracked and uncracked concrete
        8: {
            50: {
                "C20/25": {"N0_Rk_s": 32.0, "N0_Rk_p": 16.0, "V0_Rk_s": 20.5, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C25/30": {"N0_Rk_s": 32.0, "N0_Rk_p": 17.9, "V0_Rk_s": 20.5, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C30/37": {"N0_Rk_s": 32.0, "N0_Rk_p": 19.6, "V0_Rk_s": 20.5, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
            },
        },
        10: {
            60: {
                "C20/25": {"N0_Rk_s": 48.0, "N0_Rk_p": 22.0, "V0_Rk_s": 30.7, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C25/30": {"N0_Rk_s": 48.0, "N0_Rk_p": 24.6, "V0_Rk_s": 30.7, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C30/37": {"N0_Rk_s": 48.0, "N0_Rk_p": 26.9, "V0_Rk_s": 30.7, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
            },
            80: {
                "C20/25": {"N0_Rk_s": 48.0, "N0_Rk_p": 29.3, "V0_Rk_s": 30.7, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 48.0, "N0_Rk_p": 32.8, "V0_Rk_s": 30.7, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 48.0, "N0_Rk_p": 35.9, "V0_Rk_s": 30.7, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        12: {
            80: {
                "C20/25": {"N0_Rk_s": 67.0, "N0_Rk_p": 32.0, "V0_Rk_s": 42.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 67.0, "N0_Rk_p": 35.8, "V0_Rk_s": 42.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 67.0, "N0_Rk_p": 39.2, "V0_Rk_s": 42.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
            100: {
                "C20/25": {"N0_Rk_s": 67.0, "N0_Rk_p": 40.0, "V0_Rk_s": 42.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 67.0, "N0_Rk_p": 44.7, "V0_Rk_s": 42.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 67.0, "N0_Rk_p": 49.0, "V0_Rk_s": 42.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
            },
        },
        16: {
            100: {
                "C20/25": {"N0_Rk_s": 116.0, "N0_Rk_p": 50.0, "V0_Rk_s": 74.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 116.0, "N0_Rk_p": 55.9, "V0_Rk_s": 74.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 116.0, "N0_Rk_p": 61.2, "V0_Rk_s": 74.2, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
            },
            125: {
                "C20/25": {"N0_Rk_s": 116.0, "N0_Rk_p": 62.5, "V0_Rk_s": 74.2, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 116.0, "N0_Rk_p": 69.9, "V0_Rk_s": 74.2, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 116.0, "N0_Rk_p": 76.5, "V0_Rk_s": 74.2, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        20: {
            125: {
                "C20/25": {"N0_Rk_s": 178.0, "N0_Rk_p": 70.0, "V0_Rk_s": 113.9, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 178.0, "N0_Rk_p": 78.3, "V0_Rk_s": 113.9, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 178.0, "N0_Rk_p": 85.7, "V0_Rk_s": 113.9, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        24: {
            150: {
                "C20/25": {"N0_Rk_s": 254.0, "N0_Rk_p": 90.0, "V0_Rk_s": 162.6, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
                "C25/30": {"N0_Rk_s": 254.0, "N0_Rk_p": 100.6, "V0_Rk_s": 162.6, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
                "C30/37": {"N0_Rk_s": 254.0, "N0_Rk_p": 110.1, "V0_Rk_s": 162.6, "s_cr_N": 450.0, "c_cr_N": 225.0, "l_f": 150.0},
            },
        },
    },
    
    "HST3": {
        # Source: ETA-98/0001 - Hilti HST3 Expansion Anchor
        # Seismic restraints, anchoring into ComFlor, diamond cored holes
        8: {
            45: {
                "C20/25": {"N0_Rk_s": 25.0, "N0_Rk_p": 11.0, "V0_Rk_s": 16.0, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
                "C25/30": {"N0_Rk_s": 25.0, "N0_Rk_p": 12.3, "V0_Rk_s": 16.0, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
                "C30/37": {"N0_Rk_s": 25.0, "N0_Rk_p": 13.5, "V0_Rk_s": 16.0, "s_cr_N": 135.0, "c_cr_N": 67.5, "l_f": 45.0},
            },
        },
        10: {
            55: {
                "C20/25": {"N0_Rk_s": 38.0, "N0_Rk_p": 16.0, "V0_Rk_s": 24.3, "s_cr_N": 165.0, "c_cr_N": 82.5, "l_f": 55.0},
                "C25/30": {"N0_Rk_s": 38.0, "N0_Rk_p": 17.9, "V0_Rk_s": 24.3, "s_cr_N": 165.0, "c_cr_N": 82.5, "l_f": 55.0},
                "C30/37": {"N0_Rk_s": 38.0, "N0_Rk_p": 19.6, "V0_Rk_s": 24.3, "s_cr_N": 165.0, "c_cr_N": 82.5, "l_f": 55.0},
            },
        },
        12: {
            65: {
                "C20/25": {"N0_Rk_s": 53.0, "N0_Rk_p": 22.0, "V0_Rk_s": 33.9, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
                "C25/30": {"N0_Rk_s": 53.0, "N0_Rk_p": 24.6, "V0_Rk_s": 33.9, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
                "C30/37": {"N0_Rk_s": 53.0, "N0_Rk_p": 26.9, "V0_Rk_s": 33.9, "s_cr_N": 195.0, "c_cr_N": 97.5, "l_f": 65.0},
            },
        },
        16: {
            85: {
                "C20/25": {"N0_Rk_s": 92.0, "N0_Rk_p": 35.0, "V0_Rk_s": 58.9, "s_cr_N": 255.0, "c_cr_N": 127.5, "l_f": 85.0},
                "C25/30": {"N0_Rk_s": 92.0, "N0_Rk_p": 39.1, "V0_Rk_s": 58.9, "s_cr_N": 255.0, "c_cr_N": 127.5, "l_f": 85.0},
                "C30/37": {"N0_Rk_s": 92.0, "N0_Rk_p": 42.8, "V0_Rk_s": 58.9, "s_cr_N": 255.0, "c_cr_N": 127.5, "l_f": 85.0},
            },
        },
        20: {
            105: {
                "C20/25": {"N0_Rk_s": 142.0, "N0_Rk_p": 50.0, "V0_Rk_s": 90.9, "s_cr_N": 315.0, "c_cr_N": 157.5, "l_f": 105.0},
                "C25/30": {"N0_Rk_s": 142.0, "N0_Rk_p": 55.9, "V0_Rk_s": 90.9, "s_cr_N": 315.0, "c_cr_N": 157.5, "l_f": 105.0},
                "C30/37": {"N0_Rk_s": 142.0, "N0_Rk_p": 61.2, "V0_Rk_s": 90.9, "s_cr_N": 315.0, "c_cr_N": 157.5, "l_f": 105.0},
            },
        },
    },
    
    # =========================================================================
    # UNDERCUT ANCHORS
    # =========================================================================
    "HDA": {
        # Source: ETA-99/0009 - Hilti HDA Self-Undercut Anchor
        # Dynamic loads, shock loads, high seismic loads, safety relevant applications
        10: {
            50: {
                "C20/25": {"N0_Rk_s": 48.0, "N0_Rk_p": 25.0, "V0_Rk_s": 30.7, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C25/30": {"N0_Rk_s": 48.0, "N0_Rk_p": 28.0, "V0_Rk_s": 30.7, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C30/37": {"N0_Rk_s": 48.0, "N0_Rk_p": 30.6, "V0_Rk_s": 30.7, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
            },
        },
        12: {
            60: {
                "C20/25": {"N0_Rk_s": 67.0, "N0_Rk_p": 35.0, "V0_Rk_s": 42.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C25/30": {"N0_Rk_s": 67.0, "N0_Rk_p": 39.1, "V0_Rk_s": 42.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C30/37": {"N0_Rk_s": 67.0, "N0_Rk_p": 42.8, "V0_Rk_s": 42.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
            },
        },
        16: {
            80: {
                "C20/25": {"N0_Rk_s": 116.0, "N0_Rk_p": 55.0, "V0_Rk_s": 74.2, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 116.0, "N0_Rk_p": 61.5, "V0_Rk_s": 74.2, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 116.0, "N0_Rk_p": 67.3, "V0_Rk_s": 74.2, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        20: {
            100: {
                "C20/25": {"N0_Rk_s": 178.0, "N0_Rk_p": 75.0, "V0_Rk_s": 113.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C25/30": {"N0_Rk_s": 178.0, "N0_Rk_p": 83.9, "V0_Rk_s": 113.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
                "C30/37": {"N0_Rk_s": 178.0, "N0_Rk_p": 91.8, "V0_Rk_s": 113.9, "s_cr_N": 300.0, "c_cr_N": 150.0, "l_f": 100.0},
            },
        },
    },
    
    "HSC": {
        # Source: ETA-13/1038 - Hilti HSC Self-Undercut Anchor
        # Shallow base material, safety relevant applications
        8: {
            40: {
                "C20/25": {"N0_Rk_s": 25.0, "N0_Rk_p": 15.0, "V0_Rk_s": 16.0, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
                "C25/30": {"N0_Rk_s": 25.0, "N0_Rk_p": 16.8, "V0_Rk_s": 16.0, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
                "C30/37": {"N0_Rk_s": 25.0, "N0_Rk_p": 18.4, "V0_Rk_s": 16.0, "s_cr_N": 120.0, "c_cr_N": 60.0, "l_f": 40.0},
            },
        },
        10: {
            50: {
                "C20/25": {"N0_Rk_s": 38.0, "N0_Rk_p": 20.0, "V0_Rk_s": 24.3, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C25/30": {"N0_Rk_s": 38.0, "N0_Rk_p": 22.4, "V0_Rk_s": 24.3, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
                "C30/37": {"N0_Rk_s": 38.0, "N0_Rk_p": 24.5, "V0_Rk_s": 24.3, "s_cr_N": 150.0, "c_cr_N": 75.0, "l_f": 50.0},
            },
        },
        12: {
            60: {
                "C20/25": {"N0_Rk_s": 53.0, "N0_Rk_p": 28.0, "V0_Rk_s": 33.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C25/30": {"N0_Rk_s": 53.0, "N0_Rk_p": 31.3, "V0_Rk_s": 33.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
                "C30/37": {"N0_Rk_s": 53.0, "N0_Rk_p": 34.3, "V0_Rk_s": 33.9, "s_cr_N": 180.0, "c_cr_N": 90.0, "l_f": 60.0},
            },
        },
        16: {
            80: {
                "C20/25": {"N0_Rk_s": 92.0, "N0_Rk_p": 42.0, "V0_Rk_s": 58.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 92.0, "N0_Rk_p": 47.0, "V0_Rk_s": 58.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 92.0, "N0_Rk_p": 51.4, "V0_Rk_s": 58.9, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
    },
    
    # =========================================================================
    # CHEMICAL/BONDED ANCHORS (Injection Mortar Systems)
    # =========================================================================
    "HIT-HY 200": {
        # Source: ETA-19/0601 - Hilti HIT-HY 200-R V3 + HAS-U
        # Hybrid mortar - Ultimate bond strength, fast cure, high seismic loads
        8: {
            80: {
                "C20/25": {"N0_Rk_s": 35.0, "N0_Rk_p": 30.0, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 35.0, "N0_Rk_p": 33.5, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 35.0, "N0_Rk_p": 36.7, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        10: {
            90: {
                "C20/25": {"N0_Rk_s": 54.0, "N0_Rk_p": 40.0, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C25/30": {"N0_Rk_s": 54.0, "N0_Rk_p": 44.7, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C30/37": {"N0_Rk_s": 54.0, "N0_Rk_p": 49.0, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
            },
        },
        12: {
            110: {
                "C20/25": {"N0_Rk_s": 78.0, "N0_Rk_p": 55.0, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C25/30": {"N0_Rk_s": 78.0, "N0_Rk_p": 61.5, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C30/37": {"N0_Rk_s": 78.0, "N0_Rk_p": 67.3, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
            },
        },
        16: {
            125: {
                "C20/25": {"N0_Rk_s": 137.0, "N0_Rk_p": 75.0, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 137.0, "N0_Rk_p": 83.9, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 137.0, "N0_Rk_p": 91.8, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        20: {
            170: {
                "C20/25": {"N0_Rk_s": 214.0, "N0_Rk_p": 105.0, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C25/30": {"N0_Rk_s": 214.0, "N0_Rk_p": 117.4, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C30/37": {"N0_Rk_s": 214.0, "N0_Rk_p": 128.5, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
            },
        },
        24: {
            210: {
                "C20/25": {"N0_Rk_s": 309.0, "N0_Rk_p": 140.0, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
                "C25/30": {"N0_Rk_s": 309.0, "N0_Rk_p": 156.5, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
                "C30/37": {"N0_Rk_s": 309.0, "N0_Rk_p": 171.3, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
            },
        },
        27: {
            240: {
                "C20/25": {"N0_Rk_s": 393.0, "N0_Rk_p": 170.0, "V0_Rk_s": 251.5, "s_cr_N": 720.0, "c_cr_N": 360.0, "l_f": 240.0},
                "C25/30": {"N0_Rk_s": 393.0, "N0_Rk_p": 190.1, "V0_Rk_s": 251.5, "s_cr_N": 720.0, "c_cr_N": 360.0, "l_f": 240.0},
                "C30/37": {"N0_Rk_s": 393.0, "N0_Rk_p": 208.1, "V0_Rk_s": 251.5, "s_cr_N": 720.0, "c_cr_N": 360.0, "l_f": 240.0},
            },
        },
        30: {
            270: {
                "C20/25": {"N0_Rk_s": 488.0, "N0_Rk_p": 200.0, "V0_Rk_s": 312.3, "s_cr_N": 810.0, "c_cr_N": 405.0, "l_f": 270.0},
                "C25/30": {"N0_Rk_s": 488.0, "N0_Rk_p": 223.7, "V0_Rk_s": 312.3, "s_cr_N": 810.0, "c_cr_N": 405.0, "l_f": 270.0},
                "C30/37": {"N0_Rk_s": 488.0, "N0_Rk_p": 244.8, "V0_Rk_s": 312.3, "s_cr_N": 810.0, "c_cr_N": 405.0, "l_f": 270.0},
            },
        },
    },
    
    "HIT-RE 500": {
        # Source: ETA-16/0143 - Hilti HIT-RE 500 V4 + HAS-U
        # Epoxy mortar - Ultimate bond strength, slow cure, saturated & flooded holes
        8: {
            80: {
                "C20/25": {"N0_Rk_s": 35.0, "N0_Rk_p": 28.0, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 35.0, "N0_Rk_p": 31.3, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 35.0, "N0_Rk_p": 34.3, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        10: {
            90: {
                "C20/25": {"N0_Rk_s": 54.0, "N0_Rk_p": 38.0, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C25/30": {"N0_Rk_s": 54.0, "N0_Rk_p": 42.5, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C30/37": {"N0_Rk_s": 54.0, "N0_Rk_p": 46.5, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
            },
        },
        12: {
            110: {
                "C20/25": {"N0_Rk_s": 78.0, "N0_Rk_p": 52.0, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C25/30": {"N0_Rk_s": 78.0, "N0_Rk_p": 58.2, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C30/37": {"N0_Rk_s": 78.0, "N0_Rk_p": 63.6, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
            },
        },
        16: {
            125: {
                "C20/25": {"N0_Rk_s": 137.0, "N0_Rk_p": 70.0, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 137.0, "N0_Rk_p": 78.3, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 137.0, "N0_Rk_p": 85.7, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        20: {
            170: {
                "C20/25": {"N0_Rk_s": 214.0, "N0_Rk_p": 100.0, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C25/30": {"N0_Rk_s": 214.0, "N0_Rk_p": 111.8, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C30/37": {"N0_Rk_s": 214.0, "N0_Rk_p": 122.4, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
            },
        },
        24: {
            210: {
                "C20/25": {"N0_Rk_s": 309.0, "N0_Rk_p": 130.0, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
                "C25/30": {"N0_Rk_s": 309.0, "N0_Rk_p": 145.4, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
                "C30/37": {"N0_Rk_s": 309.0, "N0_Rk_p": 159.1, "V0_Rk_s": 197.8, "s_cr_N": 630.0, "c_cr_N": 315.0, "l_f": 210.0},
            },
        },
    },
    
    "HIT-HY 170": {
        # Source: ETA-19/0465 - Hilti HIT-HY 170 + HAS-U
        # Hybrid mortar - High bond strength, fast cure, low seismic loads
        8: {
            80: {
                "C20/25": {"N0_Rk_s": 35.0, "N0_Rk_p": 25.0, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 35.0, "N0_Rk_p": 28.0, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 35.0, "N0_Rk_p": 30.6, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        10: {
            90: {
                "C20/25": {"N0_Rk_s": 54.0, "N0_Rk_p": 35.0, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C25/30": {"N0_Rk_s": 54.0, "N0_Rk_p": 39.1, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C30/37": {"N0_Rk_s": 54.0, "N0_Rk_p": 42.8, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
            },
        },
        12: {
            110: {
                "C20/25": {"N0_Rk_s": 78.0, "N0_Rk_p": 48.0, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C25/30": {"N0_Rk_s": 78.0, "N0_Rk_p": 53.7, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C30/37": {"N0_Rk_s": 78.0, "N0_Rk_p": 58.7, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
            },
        },
        16: {
            125: {
                "C20/25": {"N0_Rk_s": 137.0, "N0_Rk_p": 65.0, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C25/30": {"N0_Rk_s": 137.0, "N0_Rk_p": 72.7, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
                "C30/37": {"N0_Rk_s": 137.0, "N0_Rk_p": 79.5, "V0_Rk_s": 87.7, "s_cr_N": 375.0, "c_cr_N": 187.5, "l_f": 125.0},
            },
        },
        20: {
            170: {
                "C20/25": {"N0_Rk_s": 214.0, "N0_Rk_p": 90.0, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C25/30": {"N0_Rk_s": 214.0, "N0_Rk_p": 100.6, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C30/37": {"N0_Rk_s": 214.0, "N0_Rk_p": 110.1, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
            },
        },
    },
    
    "HVU2": {
        # Source: ETA-16/0515 - Hilti HVU2 + HAS-U
        # Foil capsule - Ultimate bond strength, instant cure, high seismic loads
        8: {
            80: {
                "C20/25": {"N0_Rk_s": 35.0, "N0_Rk_p": 26.0, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C25/30": {"N0_Rk_s": 35.0, "N0_Rk_p": 29.1, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
                "C30/37": {"N0_Rk_s": 35.0, "N0_Rk_p": 31.8, "V0_Rk_s": 22.4, "s_cr_N": 240.0, "c_cr_N": 120.0, "l_f": 80.0},
            },
        },
        10: {
            90: {
                "C20/25": {"N0_Rk_s": 54.0, "N0_Rk_p": 36.0, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C25/30": {"N0_Rk_s": 54.0, "N0_Rk_p": 40.3, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
                "C30/37": {"N0_Rk_s": 54.0, "N0_Rk_p": 44.1, "V0_Rk_s": 34.6, "s_cr_N": 270.0, "c_cr_N": 135.0, "l_f": 90.0},
            },
        },
        12: {
            110: {
                "C20/25": {"N0_Rk_s": 78.0, "N0_Rk_p": 50.0, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C25/30": {"N0_Rk_s": 78.0, "N0_Rk_p": 55.9, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
                "C30/37": {"N0_Rk_s": 78.0, "N0_Rk_p": 61.2, "V0_Rk_s": 49.9, "s_cr_N": 330.0, "c_cr_N": 165.0, "l_f": 110.0},
            },
        },
        16: {
            130: {
                "C20/25": {"N0_Rk_s": 137.0, "N0_Rk_p": 70.0, "V0_Rk_s": 87.7, "s_cr_N": 390.0, "c_cr_N": 195.0, "l_f": 130.0},
                "C25/30": {"N0_Rk_s": 137.0, "N0_Rk_p": 78.3, "V0_Rk_s": 87.7, "s_cr_N": 390.0, "c_cr_N": 195.0, "l_f": 130.0},
                "C30/37": {"N0_Rk_s": 137.0, "N0_Rk_p": 85.7, "V0_Rk_s": 87.7, "s_cr_N": 390.0, "c_cr_N": 195.0, "l_f": 130.0},
            },
        },
        20: {
            170: {
                "C20/25": {"N0_Rk_s": 214.0, "N0_Rk_p": 95.0, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C25/30": {"N0_Rk_s": 214.0, "N0_Rk_p": 106.3, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
                "C30/37": {"N0_Rk_s": 214.0, "N0_Rk_p": 116.3, "V0_Rk_s": 137.0, "s_cr_N": 510.0, "c_cr_N": 255.0, "l_f": 170.0},
            },
        },
    },
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
        edge_distance_c2: Perpendicular edge distance c2 [mm] (for edge shear)

    Concrete Properties (ETAG 001 Annex C, Section 1.2):
        concrete_strength: Characteristic cylinder compressive strength f_ck [MPa]
        concrete_strength_cube: Characteristic cube compressive strength f_ck,cube [MPa]
        concrete_thickness: Member thickness h [mm]
        is_cracked_concrete: True if concrete is cracked per ETAG 001 Section 4.1

    Characteristic Capacities from ETA (ETAG 001 Annex C, Section 2.4):
        N0_Rk_s: Characteristic steel tensile resistance [kN]
        N0_Rk_p: Characteristic pullout resistance [kN]
        V0_Rk_s: Characteristic steel shear resistance [kN]

    Critical Distances from ETA (ETAG 001 Annex C):
        s_cr_N: Critical spacing for concrete cone s_cr,N [mm]
        c_cr_N: Critical edge distance for concrete cone c_cr,N [mm]
        l_f: Effective length for shear l_f [mm]

    Group Properties:
        number_of_anchors: Number of anchors in the group (1, 2, or 4)
        spacing: Spacing between anchors for 2-anchor configuration [mm]
        spacing_x: Spacing in X direction for 4-anchor configuration [mm]
        spacing_y: Spacing in Y direction for 4-anchor configuration [mm]

    Load Eccentricity (ETAG 001 Annex C, Eq 5.2e):
        e_N_x: Eccentricity of tension load in X direction [mm]
        e_N_y: Eccentricity of tension load in Y direction [mm]
        e_V: Eccentricity of shear load [mm]

    Gap/Clearance Factor (EOTA TR 045, Eq 5.8):
        alpha_gap: Gap factor for hole clearance (1.0 for no gap, 0.5 for clearance hole)

    Seismic Design (EOTA TR 045, Section 5.2):
        seismic_design: True to apply EOTA TR 045 seismic reduction factors
        is_undercut_anchor: True for undercut anchors per EOTA TR 045 Table 5.4 Note 2

    Reinforcement (ETAG 001 Annex C):
        has_edge_reinforcement: True if edge reinforcement present per Section 5.2.3.4
    """

    # Geometric Properties (ETAG 001 Annex C, Section 2.3)
    diameter: float  # d_nom [mm]
    embedment_depth: float  # h_ef [mm]
    edge_distance: float  # c1 [mm] - edge distance in shear direction

    # Concrete Properties (ETAG 001 Annex C, Section 1.2)
    concrete_strength: float  # f_ck [MPa] - cylinder strength
    concrete_thickness: float  # h [mm]
    is_cracked_concrete: bool  # Per ETAG 001 Annex C, Section 4.1

    # Characteristic Capacities from ETA (ETAG 001 Annex C, Section 2.4)
    N0_Rk_s: float  # Steel tensile capacity [kN]
    N0_Rk_p: float  # Pullout capacity [kN]
    V0_Rk_s: float  # Steel shear capacity [kN]

    # Critical distances from ETA (ETAG 001 Annex C)
    s_cr_N: float  # Critical spacing for tension s_cr,N [mm]
    c_cr_N: float  # Critical edge distance for tension c_cr,N [mm]
    l_f: float  # Effective length for shear l_f [mm]

    # Group properties - FLEXIBLE for 1, 2, or 4 anchors
    number_of_anchors: int = 1  # n: Number of anchors
    spacing: Optional[float] = None  # s: Spacing for 2 anchors [mm]
    spacing_x: Optional[float] = None  # s_x: Spacing in X for 4 anchors [mm]
    spacing_y: Optional[float] = None  # s_y: Spacing in Y for 4 anchors [mm]

    # Additional edge distance (for perpendicular direction)
    edge_distance_c2: Optional[float] = None  # c2 [mm] - perpendicular edge distance

    # Concrete cube strength (if not provided, calculated from f_ck)
    concrete_strength_cube: Optional[float] = None  # f_ck,cube [MPa]

    # Load eccentricity (ETAG 001 Annex C, Eq 5.2e)
    e_N_x: float = 0.0  # Eccentricity of tension in X [mm]
    e_N_y: float = 0.0  # Eccentricity of tension in Y [mm]
    e_V: float = 0.0  # Eccentricity of shear load [mm]

    # Gap/Clearance factor (EOTA TR 045, Eq 5.8)
    # 1.0 = anchor installed without clearance (ideal)
    # 0.5 = anchor with standard clearance hole (per ETAG 001 Table 4.1)
    alpha_gap: float = 1.0  # Gap factor α_gap

    # Seismic design (per EOTA TR 045, Section 5.2)
    seismic_design: bool = False  # True for seismic per EOTA TR 045
    is_undercut_anchor: bool = False  # Per EOTA TR 045, Table 5.4, Note 2

    # Reinforcement condition (ETAG 001 Annex C, Section 5.2.3.4)
    has_edge_reinforcement: bool = False  # True if edge reinforcement present

    def __post_init__(self):
        """Calculate derived values after initialization."""
        # Calculate cube strength from cylinder strength if not provided
        # f_ck,cube ≈ f_ck / 0.8 (approximate relationship)
        if self.concrete_strength_cube is None:
            self.concrete_strength_cube = self.concrete_strength / 0.8


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
            print(
                f"Warning: ETAG 001 Annex C requires C20/25 to C50/60 concrete. "
                f"Given: f_ck = {self.anchor.concrete_strength} MPa"
            )

        # Basic geometric validations
        if self.anchor.diameter <= 0 or self.anchor.embedment_depth <= 0:
            raise ValueError("ETAG 001: Geometric properties must be positive")

        # Validate group properties
        if self.anchor.number_of_anchors == 2 and self.anchor.spacing is None:
            raise ValueError(
                "ETAG 001: spacing must be provided for 2-anchor configuration"
            )

        if self.anchor.number_of_anchors == 4:
            if self.anchor.spacing_x is None or self.anchor.spacing_y is None:
                raise ValueError(
                    "ETAG 001: spacing_x and spacing_y must be provided for 4-anchor configuration"
                )

    def _initialize_factors(self):
        """
        Initialize partial safety factors and seismic reduction factors from standards

        Per ETAG 001 Annex C, Section 3.2.2 and EOTA TR 045, Table 5.4
        """
        # ETAG 001 Annex C, Section 3.2.2 - Partial safety factors
        self.gamma_Ms = ETAG_001_PARTIAL_SAFETY_FACTORS["gamma_Ms"]
        self.gamma_Mc = ETAG_001_PARTIAL_SAFETY_FACTORS["gamma_Mc"]
        self.gamma_Mp = ETAG_001_PARTIAL_SAFETY_FACTORS["gamma_Mp"]

        # EOTA TR 045, Table 5.4 - Seismic reduction factors
        if self.anchor.seismic_design:
            is_group = self.anchor.number_of_anchors > 1
            is_undercut = self.anchor.is_undercut_anchor

            # Select appropriate seismic reduction factors
            if is_group:
                self.alpha_seis_steel_tension = EOTA_TR045_SEISMIC_FACTORS[
                    "tension_steel_group"
                ]
                self.alpha_seis_pullout = EOTA_TR045_SEISMIC_FACTORS[
                    "tension_pullout_group"
                ]
                self.alpha_seis_concrete = (
                    EOTA_TR045_SEISMIC_FACTORS["tension_concrete_group_undercut"]
                    if is_undercut
                    else EOTA_TR045_SEISMIC_FACTORS["tension_concrete_group_other"]
                )
                self.alpha_seis_steel_shear = EOTA_TR045_SEISMIC_FACTORS[
                    "shear_steel_group"
                ]
                self.alpha_seis_concrete_shear = EOTA_TR045_SEISMIC_FACTORS[
                    "shear_concrete_group"
                ]
                self.alpha_seis_pryout = (
                    EOTA_TR045_SEISMIC_FACTORS["shear_pryout_group_undercut"]
                    if is_undercut
                    else EOTA_TR045_SEISMIC_FACTORS["shear_pryout_group_other"]
                )
            else:
                self.alpha_seis_steel_tension = EOTA_TR045_SEISMIC_FACTORS[
                    "tension_steel_single"
                ]
                self.alpha_seis_pullout = EOTA_TR045_SEISMIC_FACTORS[
                    "tension_pullout_single"
                ]
                self.alpha_seis_concrete = (
                    EOTA_TR045_SEISMIC_FACTORS["tension_concrete_single_undercut"]
                    if is_undercut
                    else EOTA_TR045_SEISMIC_FACTORS["tension_concrete_single_other"]
                )
                self.alpha_seis_steel_shear = EOTA_TR045_SEISMIC_FACTORS[
                    "shear_steel_single"
                ]
                self.alpha_seis_concrete_shear = EOTA_TR045_SEISMIC_FACTORS[
                    "shear_concrete_single"
                ]
                self.alpha_seis_pryout = (
                    EOTA_TR045_SEISMIC_FACTORS["shear_pryout_single_undercut"]
                    if is_undercut
                    else EOTA_TR045_SEISMIC_FACTORS["shear_pryout_single_other"]
                )
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
        Standard: EOTA TR 045, Eq (5.8)

        Equation: N_Rk,s,seis = α_gap · α_seis · N⁰_Rk,s
                  N_Rd,s = N_Rk,s,seis / γ_Ms

        For seismic: EOTA TR 045, Table 5.4, Row 1, apply α_seis

        Returns:
            Dict with:
                N0_Rk_s: Basic characteristic resistance [kN]
                N_Rk_s_seis: Characteristic resistance with factors [kN]
                N_Rd_s: Design resistance [kN]
                alpha_gap: Gap/clearance factor
                alpha_seis: Seismic reduction factor
                gamma_Ms: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.2.2 - Basic characteristic resistance from ETA
        N0_Rk_s = self.anchor.N0_Rk_s

        # EOTA TR 045, Eq (5.8) - Apply gap and seismic factors
        # N_Rk,s,seis = α_gap · α_seis · N⁰_Rk,s
        alpha_gap = self.anchor.alpha_gap
        N_Rk_s_seis = alpha_gap * self.alpha_seis_steel_tension * N0_Rk_s

        # ETAG 001 Annex C, Section 3.2.2.2 - Apply partial safety factor
        N_Rd_s = N_Rk_s_seis / self.gamma_Ms

        return {
            "N0_Rk_s": N0_Rk_s,
            "N_Rk_s_seis": N_Rk_s_seis,
            "N_Rd_s": N_Rd_s,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_steel_tension,
            "gamma_Ms": self.gamma_Ms,
            "standard_reference": "ETAG 001 Annex C Section 5.2.2.2, EOTA TR 045 Eq (5.8)",
        }

    def pullout_failure_ETAG_5_2_2_3(self) -> Dict:
        """
        Pull-out Failure

        Standard: ETAG 001 Annex C, Section 5.2.2.3
        Standard: EOTA TR 045, Eq (5.8)

        Equation: N_Rk,p,seis = α_gap · α_seis · N⁰_Rk,p
                  N_Rd,p = ψ_c,seis · N_Rk,p,seis / γ_Mp

        where ψ_c,seis = 1.0 (cracked concrete factor for seismic, per ETA)

        For seismic: EOTA TR 045, Table 5.4, Row 2, apply α_seis

        Returns:
            Dict with:
                N0_Rk_p: Basic characteristic resistance [kN]
                N_Rk_p_seis: Characteristic resistance with factors [kN]
                N_Rd_p: Design resistance [kN]
                alpha_gap: Gap/clearance factor
                alpha_seis: Seismic reduction factor
                psi_c_seis: Cracked concrete factor
                gamma_Mp: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.2.3 - Basic characteristic resistance from ETA
        N0_Rk_p = self.anchor.N0_Rk_p

        # EOTA TR 045, Eq (5.8) - Apply gap and seismic factors
        # N_Rk,p,seis = α_gap · α_seis · N⁰_Rk,p
        alpha_gap = self.anchor.alpha_gap
        N_Rk_p_seis = alpha_gap * self.alpha_seis_pullout * N0_Rk_p

        # ψ_c,seis factor (typically 1.0 for cracked concrete per ETA)
        psi_c_seis = 1.0

        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        # N_Rd,p = ψ_c,seis · N_Rk,p,seis / γ_Mp
        N_Rd_p = psi_c_seis * N_Rk_p_seis / self.gamma_Mp

        return {
            "N0_Rk_p": N0_Rk_p,
            "N_Rk_p_seis": N_Rk_p_seis,
            "N_Rd_p": N_Rd_p,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_pullout,
            "psi_c_seis": psi_c_seis,
            "gamma_Mp": self.gamma_Mp,
            "standard_reference": "ETAG 001 Annex C Section 5.2.2.3, EOTA TR 045 Eq (5.8)",
        }

    def concrete_cone_failure_ETAG_5_2_2_4(self) -> Dict:
        """
        Concrete Cone Failure

        Standard: ETAG 001 Annex C, Section 5.2.2.4, Equation (5.2)
        Standard: EOTA TR 045, Eq (5.8)

        Full equation per ETAG 001 Annex C, Eq (5.2):
        N_Rk,c,seis = α_gap · α_seis · N⁰_Rk,c · (A_c,N/A⁰_c,N) · ψ_s,N · ψ_re,N · ψ_ec1,N · ψ_ec2,N

        Where:
        - N⁰_Rk,c = k1 · √f_ck,cube · h_ef^1.5  [Eq 5.2a] (Note: uses cube strength!)
        - A⁰_c,N = s_cr,N · s_cr,N = (3·h_ef)²  [Eq 5.2b]
        - ψ_s,N = 0.7 + 0.3·(c/c_cr,N) ≤ 1.0   [Eq 5.2c]
        - ψ_re,N = 0.5 + h_ef/200 ≤ 1.0         [Eq 5.2d]
        - ψ_ec,N = 1/(1 + 2·e_N/s_cr,N) ≤ 1.0   [Eq 5.2e]

        For seismic: EOTA TR 045, Table 5.4, Row 4, apply α_seis

        Returns:
            Dict with all intermediate values and final design resistance
        """
        # ETAG 001 Annex C, Section 5.2.2.4 - Select coefficient based on cracking
        k1 = (
            ETAG_001_CONCRETE_CONSTANTS["k1_tension_cracked"]
            if self.anchor.is_cracked_concrete
            else ETAG_001_CONCRETE_CONSTANTS["k1_tension_uncracked"]
        )

        # Material and geometry parameters
        f_ck_cube = self.anchor.concrete_strength_cube  # Use cube strength per Eq 5.2a
        h_ef = self.anchor.embedment_depth
        c = self.anchor.edge_distance
        s_cr_N = self.anchor.s_cr_N
        c_cr_N = self.anchor.c_cr_N
        alpha_gap = self.anchor.alpha_gap

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.2a) - Basic characteristic resistance
        # N⁰_Rk,c = k1 · √f_ck,cube · h_ef^1.5
        # ====================================================================
        N0_Rk_c = k1 * np.sqrt(f_ck_cube) * (h_ef**1.5) / 1000  # kN

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.2b) - Reference projected area
        # A⁰_c,N = s_cr,N · s_cr,N
        # ====================================================================
        A0_c_N = s_cr_N * s_cr_N  # mm²

        # ====================================================================
        # Calculate actual projected area A_c,N
        # For single anchor far from edges: A_c,N = A⁰_c,N
        # For anchor near edge: A_c,N is reduced
        # For anchor groups: A_c,N considers overlap
        # ====================================================================
        A_c_N = self._calculate_projected_area_tension()

        # Area ratio
        area_ratio = A_c_N / A0_c_N

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.2c) - Edge distance factor
        # ψ_s,N = 0.7 + 0.3 · (c / c_cr,N) ≤ 1.0
        # ====================================================================
        c_min = min(c, c_cr_N)  # Use minimum edge distance
        psi_s_N = min(0.7 + 0.3 * (c_min / c_cr_N), 1.0)

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.2d) - Shell spalling factor
        # ψ_re,N = 0.5 + h_ef/200 ≤ 1.0
        # ====================================================================
        psi_re_N = min(0.5 + h_ef / 200.0, 1.0)

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.2e) - Eccentricity factors
        # ψ_ec1,N = 1 / (1 + 2·e_N,x / s_cr,N) ≤ 1.0
        # ψ_ec2,N = 1 / (1 + 2·e_N,y / s_cr,N) ≤ 1.0
        # ====================================================================
        e_N_x = self.anchor.e_N_x
        e_N_y = self.anchor.e_N_y

        psi_ec1_N = (
            min(1.0 / (1.0 + 2.0 * abs(e_N_x) / s_cr_N), 1.0) if s_cr_N > 0 else 1.0
        )
        psi_ec2_N = (
            min(1.0 / (1.0 + 2.0 * abs(e_N_y) / s_cr_N), 1.0) if s_cr_N > 0 else 1.0
        )

        # ====================================================================
        # EOTA TR 045, Eq (5.8) - Apply gap and seismic factors
        # N_Rk,c,seis = α_gap · α_seis · N⁰_Rk,c · (A_c,N/A⁰_c,N) · ψ_s,N · ψ_re,N · ψ_ec1,N · ψ_ec2,N
        # ====================================================================
        N_Rk_c_seis = (
            alpha_gap
            * self.alpha_seis_concrete
            * N0_Rk_c
            * area_ratio
            * psi_s_N
            * psi_re_N
            * psi_ec1_N
            * psi_ec2_N
        )

        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        N_Rd_c = N_Rk_c_seis / self.gamma_Mc

        return {
            "N0_Rk_c": N0_Rk_c,
            "N_Rk_c_seis": N_Rk_c_seis,
            "N_Rd_c": N_Rd_c,
            "k1": k1,
            "f_ck_cube": f_ck_cube,
            "h_ef": h_ef,
            "A_c_N": A_c_N,
            "A0_c_N": A0_c_N,
            "area_ratio": area_ratio,
            "s_cr_N": s_cr_N,
            "c_cr_N": c_cr_N,
            "psi_s_N": psi_s_N,
            "psi_re_N": psi_re_N,
            "psi_ec1_N": psi_ec1_N,
            "psi_ec2_N": psi_ec2_N,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_concrete,
            "gamma_Mc": self.gamma_Mc,
            "standard_reference": "ETAG 001 Annex C Eq (5.2), EOTA TR 045 Eq (5.8)",
        }

    def _calculate_projected_area_tension(self) -> float:
        """
        Calculate actual projected area A_c,N for concrete cone failure

        Per ETAG 001 Annex C, the projected area depends on:
        - Number of anchors
        - Spacing between anchors
        - Edge distances
        - Member thickness

        Returns:
            A_c_N: Actual projected area [mm²]
        """
        h_ef = self.anchor.embedment_depth
        s_cr_N = self.anchor.s_cr_N
        c_cr_N = self.anchor.c_cr_N
        c1 = self.anchor.edge_distance
        h = self.anchor.concrete_thickness
        n = self.anchor.number_of_anchors

        # Limit edge distance to critical value
        c1_eff = min(c1, c_cr_N)

        # Calculate effective member thickness influence
        # h_eff = min(h, c_cr_N) for single anchor
        h_eff = min(h, 1.5 * h_ef) if h > 0 else 1.5 * h_ef

        if n == 1:
            # Single anchor - rectangular projected area
            # Width = min(c1, c_cr,N) + s_cr,N/2 for edge, or s_cr,N for internal
            if c1 >= c_cr_N:
                # Internal anchor (far from edge)
                A_c_N = s_cr_N * s_cr_N
            else:
                # Near edge - reduced area
                width = c1_eff + s_cr_N / 2.0
                height = s_cr_N
                A_c_N = width * height

        elif n == 2:
            # Two-anchor group
            s = self.anchor.spacing if self.anchor.spacing is not None else s_cr_N
            s_eff = min(s, s_cr_N)

            if c1 >= c_cr_N:
                # Internal (far from edge)
                width = s_cr_N + s_eff
                height = s_cr_N
            else:
                # Near edge
                width = c1_eff + s_cr_N / 2.0
                height = s_cr_N + s_eff
            A_c_N = width * height

        elif n == 4:
            # Four-anchor group (rectangular pattern)
            s_x = self.anchor.spacing_x if self.anchor.spacing_x is not None else s_cr_N
            s_y = self.anchor.spacing_y if self.anchor.spacing_y is not None else s_cr_N
            s_x_eff = min(s_x, s_cr_N)
            s_y_eff = min(s_y, s_cr_N)

            if c1 >= c_cr_N:
                width = s_cr_N + s_x_eff
                height = s_cr_N + s_y_eff
            else:
                width = c1_eff + s_cr_N / 2.0 + s_x_eff
                height = s_cr_N + s_y_eff
            A_c_N = width * height
        else:
            # Default to reference area
            A_c_N = s_cr_N * s_cr_N

        return A_c_N

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

        N_Rd_steel = steel_result["N_Rd_s"]
        N_Rd_pullout = pullout_result["N_Rd_p"]
        N_Rd_concrete = concrete_result["N_Rd_c"]

        # Governing capacity is the minimum
        N_Rd = min(N_Rd_steel, N_Rd_pullout, N_Rd_concrete)

        # Determine which mode governs
        if N_Rd == N_Rd_steel:
            governing_mode = "Steel Failure (ETAG 5.2.2.2)"
        elif N_Rd == N_Rd_pullout:
            governing_mode = "Pull-out Failure (ETAG 5.2.2.3)"
        else:
            governing_mode = "Concrete Cone Failure (ETAG 5.2.2.4)"

        return {
            "N_Rd": N_Rd,
            "N_Rd_steel": N_Rd_steel,
            "N_Rd_pullout": N_Rd_pullout,
            "N_Rd_concrete": N_Rd_concrete,
            "governing_mode": governing_mode,
            "steel_details": steel_result,
            "pullout_details": pullout_result,
            "concrete_details": concrete_result,
            "standard_reference": "ETAG 001 Annex C, Section 5.2.2",
        }

    # ========================================================================
    # ETAG 001 ANNEX C, SECTION 5.2.3 - RESISTANCE TO SHEAR LOADS
    # ========================================================================

    def steel_failure_shear_ETAG_5_2_3_2(self) -> Dict:
        """
        Steel Failure in Shear (without lever arm)

        Standard: ETAG 001 Annex C, Section 5.2.3.2
        Standard: EOTA TR 045, Eq (5.8)

        Equation: V_Rk,s,seis = α_gap · α_seis · V⁰_Rk,s
                  V_Rd,s = V_Rk,s,seis / γ_Ms

        IMPORTANT: α_gap has significant effect for shear!
        - α_gap = 1.0 for anchors without hole clearance
        - α_gap = 0.5 for anchors with standard clearance hole (per ETAG Table 4.1)

        For seismic: EOTA TR 045, Table 5.4, Row 6, apply α_seis

        Returns:
            Dict with:
                V0_Rk_s: Basic characteristic resistance [kN]
                V_Rk_s_seis: Characteristic resistance with factors [kN]
                V_Rd_s: Design resistance [kN]
                alpha_gap: Gap/clearance factor
                alpha_seis: Seismic reduction factor
                gamma_Ms: Partial safety factor
        """
        # ETAG 001 Annex C, Section 5.2.3.2 - Basic characteristic resistance from ETA
        V0_Rk_s = self.anchor.V0_Rk_s

        # EOTA TR 045, Eq (5.8) - Apply gap and seismic factors
        # V_Rk,s,seis = α_gap · α_seis · V⁰_Rk,s
        # NOTE: α_gap is CRITICAL for shear - reduces capacity by 50% for clearance holes!
        alpha_gap = self.anchor.alpha_gap
        V_Rk_s_seis = alpha_gap * self.alpha_seis_steel_shear * V0_Rk_s

        # ETAG 001 Annex C, Section 3.2.2.2 - Apply partial safety factor
        V_Rd_s = V_Rk_s_seis / self.gamma_Ms

        return {
            "V0_Rk_s": V0_Rk_s,
            "V_Rk_s_seis": V_Rk_s_seis,
            "V_Rd_s": V_Rd_s,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_steel_shear,
            "gamma_Ms": self.gamma_Ms,
            "standard_reference": "ETAG 001 Annex C Section 5.2.3.2, EOTA TR 045 Eq (5.8)",
        }

    def concrete_pryout_failure_ETAG_5_2_3_3(self) -> Dict:
        """
        Concrete Pry-out Failure

        Standard: ETAG 001 Annex C, Section 5.2.3.3, Equation (5.6)
        Standard: EOTA TR 045, Eq (5.8)

        Equation: V_Rk,cp,seis = k · α_gap · α_seis · N⁰_Rk,c · (A_c,N/A⁰_c,N) · ψ_s,N · ψ_re,N · ψ_ec1,N · ψ_ec2,N
                  V_Rd,cp = V_Rk,cp,seis / γ_Mc

        where k = 2.0 for h_ef ≥ 60mm, k = 1.0 for h_ef < 60mm

        For seismic: EOTA TR 045, Table 5.4, Row 8, apply α_seis for pryout

        Returns:
            Dict with all intermediate values and final design resistance
        """
        # ETAG 001 Annex C, Section 5.2.3.3 - Pryout coefficient
        # k = 2.0 for h_ef ≥ 60mm, k = 1.0 for h_ef < 60mm
        h_ef = self.anchor.embedment_depth
        k_pryout = 2.0 if h_ef >= 60.0 else 1.0

        # Get concrete cone calculation results (includes all ψ factors)
        concrete_tension = self.concrete_cone_failure_ETAG_5_2_2_4()

        # ETAG 001 Annex C, Section 5.2.3.3 - Pryout uses N_Rk,c with pryout-specific seismic factor
        # V_Rk,cp,seis = k · α_gap · α_seis,pryout · N⁰_Rk,c · (A_c,N/A⁰_c,N) · ψ factors
        # Note: We need to recalculate with pryout α_seis instead of concrete α_seis

        alpha_gap = self.anchor.alpha_gap
        N0_Rk_c = concrete_tension["N0_Rk_c"]
        area_ratio = concrete_tension["area_ratio"]
        psi_s_N = concrete_tension["psi_s_N"]
        psi_re_N = concrete_tension["psi_re_N"]
        psi_ec1_N = concrete_tension["psi_ec1_N"]
        psi_ec2_N = concrete_tension["psi_ec2_N"]

        # Calculate pryout with correct seismic factor
        V_Rk_cp_seis = (
            k_pryout
            * alpha_gap
            * self.alpha_seis_pryout
            * N0_Rk_c
            * area_ratio
            * psi_s_N
            * psi_re_N
            * psi_ec1_N
            * psi_ec2_N
        )

        # ETAG 001 Annex C - Apply partial safety factor
        V_Rd_cp = V_Rk_cp_seis / self.gamma_Mc

        return {
            "V0_Rk_cp": k_pryout * N0_Rk_c,
            "V_Rk_cp_seis": V_Rk_cp_seis,
            "V_Rd_cp": V_Rd_cp,
            "k_pryout": k_pryout,
            "N0_Rk_c": N0_Rk_c,
            "area_ratio": area_ratio,
            "psi_s_N": psi_s_N,
            "psi_re_N": psi_re_N,
            "psi_ec1_N": psi_ec1_N,
            "psi_ec2_N": psi_ec2_N,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_pryout,
            "gamma_Mc": self.gamma_Mc,
            "standard_reference": "ETAG 001 Annex C Eq (5.6), EOTA TR 045 Eq (5.8)",
        }

    def concrete_edge_failure_ETAG_5_2_3_4(self) -> Dict:
        """
        Concrete Edge Failure

        Standard: ETAG 001 Annex C, Section 5.2.3.4, Equation (5.7)
        Standard: EOTA TR 045, Eq (5.8)

        Full equation per ETAG 001 Annex C, Eq (5.7):
        V_Rk,c,seis = α_gap · α_seis · V⁰_Rk,c · (A_c,V/A⁰_c,V) · ψ_s,V · ψ_h,V · ψ_α,V · ψ_ec,V · ψ_re,V

        Where:
        - V⁰_Rk,c = k1 · d_nom^β · l_f^α · √f_ck,cube · c1^1.5  [Eq 5.7a]
        - α = 0.1 · (l_f/c1)^0.5                                 [Eq 5.7b]
        - β = 0.1 · (d_nom/c1)^0.2                               [Eq 5.7c]
        - A⁰_c,V = 4.5 · c1²                                     [Eq 5.7d]
        - ψ_s,V = 0.7 + 0.3 · (c2/(1.5·c1)) ≤ 1.0               [Eq 5.7e]
        - ψ_h,V = (1.5·c1/h)^0.5 ≥ 1.0                          [Eq 5.7f]
        - ψ_α,V = √(1/(cos²α_v + (sin α_v/2.5)²)) ≥ 1.0         [Eq 5.7g]
        - ψ_ec,V = 1/(1 + 2·e_V/(3·c1)) ≤ 1.0                   [Eq 5.7h]
        - ψ_re,V = reinforcement factor (1.0 or 1.4)

        For seismic: EOTA TR 045, Table 5.4, Row 7, apply α_seis

        Returns:
            Dict with all intermediate values and final design resistance
        """
        # ETAG 001 Annex C, Section 5.2.3.4 - Select coefficient based on cracking
        k1 = (
            ETAG_001_CONCRETE_CONSTANTS["k1_shear_cracked"]
            if self.anchor.is_cracked_concrete
            else ETAG_001_CONCRETE_CONSTANTS["k1_shear_uncracked"]
        )

        # Parameters
        c1 = self.anchor.edge_distance  # Edge distance in load direction
        c2 = (
            self.anchor.edge_distance_c2
            if self.anchor.edge_distance_c2 is not None
            else c1
        )  # Perpendicular edge
        d_nom = self.anchor.diameter
        l_f = self.anchor.l_f  # Effective length for shear
        f_ck_cube = self.anchor.concrete_strength_cube
        h = self.anchor.concrete_thickness
        e_V = self.anchor.e_V  # Eccentricity of shear load
        alpha_gap = self.anchor.alpha_gap
        alpha_v_deg = 0.0  # Load angle from perpendicular to edge (0° = perpendicular)

        # Check for edge failure applicability
        # If edge distance is very large (e.g., 9999), skip edge failure calculation
        if c1 >= 9999:
            return {
                "V0_Rk_c": float("inf"),
                "V_Rk_c_seis": float("inf"),
                "V_Rd_c": float("inf"),
                "skip_reason": "Edge distance set to infinity - edge failure not applicable",
                "standard_reference": "ETAG 001 Annex C Section 5.2.3.4",
            }

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7b) and (5.7c) - Exponents
        # α = 0.1 · (l_f/c1)^0.5
        # β = 0.1 · (d_nom/c1)^0.2
        # ====================================================================
        alpha_exp = 0.1 * np.power(l_f / c1, 0.5)
        beta_exp = 0.1 * np.power(d_nom / c1, 0.2)

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7a) - Basic characteristic resistance
        # V⁰_Rk,c = k1 · d_nom^β · l_f^α · √f_ck,cube · c1^1.5
        # ====================================================================
        V0_Rk_c = (
            k1
            * np.power(d_nom, beta_exp)
            * np.power(l_f, alpha_exp)
            * np.sqrt(f_ck_cube)
            * np.power(c1, 1.5)
        ) / 1000  # kN

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7d) - Reference projected area
        # A⁰_c,V = 4.5 · c1²
        # ====================================================================
        A0_c_V = 4.5 * c1 * c1  # mm²

        # ====================================================================
        # Calculate actual projected area A_c,V
        # ====================================================================
        A_c_V = self._calculate_projected_area_shear()
        area_ratio = A_c_V / A0_c_V if A0_c_V > 0 else 1.0

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7e) - Perpendicular edge factor
        # ψ_s,V = 0.7 + 0.3 · (c2/(1.5·c1)) ≤ 1.0
        # ====================================================================
        psi_s_V = min(0.7 + 0.3 * (c2 / (1.5 * c1)), 1.0)

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7f) - Member thickness factor
        # ψ_h,V = (1.5·c1/h)^0.5 ≥ 1.0
        # ====================================================================
        if h > 0 and h < 1.5 * c1:
            psi_h_V = np.sqrt(1.5 * c1 / h)
        else:
            psi_h_V = 1.0

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7g) - Load angle factor
        # ψ_α,V = √(1/(cos²α_v + (sin α_v/2.5)²)) ≥ 1.0
        # For α_v = 0° (load perpendicular to edge): ψ_α,V = 1.0
        # ====================================================================
        alpha_v_rad = np.radians(alpha_v_deg)
        cos_alpha = np.cos(alpha_v_rad)
        sin_alpha = np.sin(alpha_v_rad)
        psi_alpha_V = max(np.sqrt(1.0 / (cos_alpha**2 + (sin_alpha / 2.5) ** 2)), 1.0)

        # ====================================================================
        # ETAG 001 Annex C, Eq (5.7h) - Eccentricity factor
        # ψ_ec,V = 1/(1 + 2·e_V/(3·c1)) ≤ 1.0
        # ====================================================================
        psi_ec_V = (
            min(1.0 / (1.0 + 2.0 * abs(e_V) / (3.0 * c1)), 1.0) if c1 > 0 else 1.0
        )

        # ====================================================================
        # ETAG 001 Annex C - Reinforcement factor
        # ψ_re,V = 1.0 (no edge reinforcement)
        # ψ_re,V = 1.4 (with edge reinforcement per Section 5.2.3.4)
        # ====================================================================
        psi_re_V = 1.4 if self.anchor.has_edge_reinforcement else 1.0

        # ====================================================================
        # EOTA TR 045, Eq (5.8) - Apply gap and seismic factors
        # V_Rk,c,seis = α_gap · α_seis · V⁰_Rk,c · (A_c,V/A⁰_c,V) · ψ_s,V · ψ_h,V · ψ_α,V · ψ_ec,V · ψ_re,V
        # ====================================================================
        V_Rk_c_seis = (
            alpha_gap
            * self.alpha_seis_concrete_shear
            * V0_Rk_c
            * area_ratio
            * psi_s_V
            * psi_h_V
            * psi_alpha_V
            * psi_ec_V
            * psi_re_V
        )

        # ETAG 001 Annex C, Section 3.2.2.1 - Apply partial safety factor
        V_Rd_c = V_Rk_c_seis / self.gamma_Mc

        return {
            "V0_Rk_c": V0_Rk_c,
            "V_Rk_c_seis": V_Rk_c_seis,
            "V_Rd_c": V_Rd_c,
            "k1": k1,
            "alpha_exp": alpha_exp,
            "beta_exp": beta_exp,
            "d_nom": d_nom,
            "l_f": l_f,
            "c1": c1,
            "c2": c2,
            "f_ck_cube": f_ck_cube,
            "A_c_V": A_c_V,
            "A0_c_V": A0_c_V,
            "area_ratio": area_ratio,
            "psi_s_V": psi_s_V,
            "psi_h_V": psi_h_V,
            "psi_alpha_V": psi_alpha_V,
            "psi_ec_V": psi_ec_V,
            "psi_re_V": psi_re_V,
            "alpha_gap": alpha_gap,
            "alpha_seis": self.alpha_seis_concrete_shear,
            "gamma_Mc": self.gamma_Mc,
            "standard_reference": "ETAG 001 Annex C Eq (5.7), EOTA TR 045 Eq (5.8)",
        }

    def _calculate_projected_area_shear(self) -> float:
        """
        Calculate actual projected area A_c,V for concrete edge failure

        Per ETAG 001 Annex C, the projected area for shear depends on:
        - Number of anchors
        - Spacing between anchors
        - Edge distances
        - Member thickness

        Returns:
            A_c_V: Actual projected area [mm²]
        """
        c1 = self.anchor.edge_distance
        c2 = (
            self.anchor.edge_distance_c2
            if self.anchor.edge_distance_c2 is not None
            else 1.5 * c1
        )
        h = self.anchor.concrete_thickness
        n = self.anchor.number_of_anchors

        # Calculate limiting dimensions
        h_eff = min(h, 1.5 * c1) if h > 0 else 1.5 * c1
        c2_eff = min(c2, 1.5 * c1)

        if n == 1:
            # Single anchor
            # A_c,V = (2·c2,eff + s) · h_eff, but for single anchor s=0
            width = 2 * c2_eff
            height = h_eff
            A_c_V = width * height

        elif n == 2:
            # Two-anchor group
            s = self.anchor.spacing if self.anchor.spacing is not None else 0
            s_eff = min(s, 3 * c1)  # Limit spacing effect

            width = 2 * c2_eff + s_eff
            height = h_eff
            A_c_V = width * height

        elif n == 4:
            # Four-anchor group
            s_x = self.anchor.spacing_x if self.anchor.spacing_x is not None else 0
            s_y = self.anchor.spacing_y if self.anchor.spacing_y is not None else 0
            s_y_eff = min(s_y, 3 * c1)

            width = 2 * c2_eff + s_y_eff
            height = h_eff
            A_c_V = width * height
        else:
            # Default to reference area
            A_c_V = 4.5 * c1 * c1

        return A_c_V

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

        V_Rd_steel = steel_result["V_Rd_s"]
        V_Rd_pryout = pryout_result["V_Rd_cp"]
        V_Rd_concrete = concrete_edge_result["V_Rd_c"]

        # Governing capacity is the minimum
        V_Rd = min(V_Rd_steel, V_Rd_pryout, V_Rd_concrete)

        # Determine which mode governs
        if V_Rd == V_Rd_steel:
            governing_mode = "Steel Failure (ETAG 5.2.3.2)"
        elif V_Rd == V_Rd_pryout:
            governing_mode = "Concrete Pry-out (ETAG 5.2.3.3)"
        else:
            governing_mode = "Concrete Edge Failure (ETAG 5.2.3.4)"

        return {
            "V_Rd": V_Rd,
            "V_Rd_steel": V_Rd_steel,
            "V_Rd_pryout": V_Rd_pryout,
            "V_Rd_concrete": V_Rd_concrete,
            "governing_mode": governing_mode,
            "steel_details": steel_result,
            "pryout_details": pryout_result,
            "concrete_edge_details": concrete_edge_result,
            "standard_reference": "ETAG 001 Annex C, Section 5.2.3",
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

        N_Rd = tension_result["N_Rd"]
        V_Rd = shear_result["V_Rd"]

        # EOTA TR 045, Equation (5.9) - Interaction check
        beta_N = N_Ed / N_Rd if N_Rd > 0 else 0
        beta_V = V_Ed / V_Rd if V_Rd > 0 else 0
        utilization = beta_N + beta_V

        # Check compliance per ETAG 001 Annex C, Section 5.2.4
        compliance = utilization <= 1.0
        status = "PASS" if compliance else "FAIL"

        return {
            "utilization": utilization,
            "compliance": compliance,
            "status": status,
            "beta_N": beta_N,
            "beta_V": beta_V,
            "N_Ed": N_Ed,
            "V_Ed": V_Ed,
            "N_Rd": N_Rd,
            "V_Rd": V_Rd,
            "tension_result": tension_result,
            "shear_result": shear_result,
            "standard_reference": "ETAG 001 Annex C Section 5.2.4, EOTA TR 045 Eq. (5.9)",
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def list_available_anchors() -> Dict:
    """
    List all available anchor products in the ETA database.
    
    Returns:
        Dict with product names as keys and their available configurations
        (diameters and embedment depths) as values.
    
    Example:
        >>> anchors = list_available_anchors()
        >>> print(anchors.keys())
        dict_keys(['HUS4-H', 'HUS3-H', 'HSL-4', 'HST3', 'HDA', 'HSC', 'HIT-HY 200', ...])
    """
    result = {}
    for product, diameters in ETA_DATABASE.items():
        result[product] = {
            "diameters": list(diameters.keys()),
            "configurations": {}
        }
        for diameter, embedments in diameters.items():
            result[product]["configurations"][diameter] = {
                "embedment_depths": list(embedments.keys()),
                "concrete_grades": list(list(embedments.values())[0].keys()) if embedments else []
            }
    return result


def print_available_anchors():
    """
    Print a formatted summary of all available anchor products.
    
    Example output:
        Available Anchor Products in ETA Database:
        ==========================================
        
        HUS4-H (Screw Anchor - ETA-20/0867)
          Diameters: 6, 8, 10, 12, 14, 16, 20, 24 mm
          
        HSL-4 (Expansion Anchor - ETA-19/0556)
          Diameters: 8, 10, 12, 16, 20, 24 mm
        ...
    """
    print("=" * 60)
    print("AVAILABLE ANCHOR PRODUCTS IN ETA DATABASE")
    print("=" * 60)
    
    # Product descriptions
    descriptions = {
        "HUS4-H": "Screw Anchor (ETA-20/0867)",
        "HUS3-H": "Screw Anchor (ETAG-13/1038)",
        "HSL-4": "Heavy Duty Expansion Anchor (ETA-19/0556)",
        "HST3": "Expansion Anchor (ETA-98/0001)",
        "HDA": "Self-Undercut Anchor (ETA-99/0009)",
        "HSC": "Self-Undercut Anchor (ETA-13/1038)",
        "HIT-HY 200": "Hybrid Mortar + HAS-U (ETA-19/0601)",
        "HIT-RE 500": "Epoxy Mortar + HAS-U (ETA-16/0143)",
        "HIT-HY 170": "Hybrid Mortar + HAS-U (ETA-19/0465)",
        "HVU2": "Foil Capsule + HAS-U (ETA-16/0515)",
    }
    
    for product, diameters in ETA_DATABASE.items():
        desc = descriptions.get(product, "")
        print(f"\n{product} - {desc}")
        print(f"  Diameters: {', '.join(str(d) for d in sorted(diameters.keys()))} mm")
        
        for diameter in sorted(diameters.keys()):
            embedments = diameters[diameter]
            emb_str = ', '.join(str(e) for e in sorted(embedments.keys()))
            print(f"    d={diameter}mm: h_ef = {emb_str} mm")
    
    print("\n" + "=" * 60)


def parse_anchor_product_string(anchor_string: str) -> Tuple[str, int, float]:
    """
    Parse an anchor product string like "HUS4-H-12M-80" into its components.
    
    The format is: "{ProductName}-{Diameter}M-{EmbedmentDepth}"
    
    Args:
        anchor_string: Anchor product string (e.g., "HUS4-H-12M-80", "HIT-HY200-16M-125")
    
    Returns:
        Tuple of (product_name, diameter, embedment_depth)
        - product_name: str - The base product name for ETA database lookup
        - diameter: int - Anchor diameter in mm
        - embedment_depth: float - Effective embedment depth in mm
    
    Examples:
        >>> parse_anchor_product_string("HUS4-H-12M-80")
        ('HUS4-H', 12, 80.0)
        
        >>> parse_anchor_product_string("HIT-HY200-16M-125")
        ('HIT-HY 200', 16, 125.0)
        
        >>> parse_anchor_product_string("HSL-4-10M-60")
        ('HSL-4', 10, 60.0)
    
    Raises:
        ValueError: If the string format is invalid
    """
    # Split from the right to handle product names with hyphens
    parts = anchor_string.rsplit("-", 2)
    
    if len(parts) != 3:
        raise ValueError(
            f"Invalid anchor string format: '{anchor_string}'. "
            f"Expected format: 'ProductName-DiameterM-EmbedmentDepth' "
            f"(e.g., 'HUS4-H-12M-80')"
        )
    
    product_name_raw = parts[0]
    diameter_str = parts[1]
    embedment_str = parts[2]
    
    # Parse diameter (remove 'M' suffix)
    if not diameter_str.endswith("M"):
        raise ValueError(
            f"Invalid diameter format: '{diameter_str}'. "
            f"Expected format like '12M'"
        )
    
    try:
        diameter = int(diameter_str.replace("M", ""))
    except ValueError:
        raise ValueError(f"Invalid diameter value: '{diameter_str}'")
    
    # Parse embedment depth
    try:
        embedment_depth = float(embedment_str)
    except ValueError:
        raise ValueError(f"Invalid embedment depth value: '{embedment_str}'")
    
    # Convert product name to match ETA_DATABASE keys
    # Handle special cases where spaces were removed in the dropdown
    product_name_mapping = {
        "HIT-HY200": "HIT-HY 200",
        "HIT-RE500": "HIT-RE 500",
        "HIT-HY170": "HIT-HY 170",
    }
    
    product_name = product_name_mapping.get(product_name_raw, product_name_raw)
    
    # Validate product exists in database
    if product_name not in ETA_DATABASE:
        available = list(ETA_DATABASE.keys())
        raise ValueError(
            f"Unknown anchor product: '{product_name}'. "
            f"Available products: {available}"
        )
    
    return product_name, diameter, embedment_depth


def get_anchor_properties_from_string(
    anchor_string: str,
    concrete_grade: str = "C20/25"
) -> Dict:
    """
    Get all anchor properties from a product string like "HUS4-H-12M-80".
    
    This is a convenience function that parses the string and retrieves
    the ETA database values in one call.
    
    Args:
        anchor_string: Anchor product string (e.g., "HUS4-H-12M-80")
        concrete_grade: Concrete grade (e.g., 'C20/25', 'C25/30', 'C30/37')
    
    Returns:
        Dict containing:
        - product_name: Base product name
        - diameter: Anchor diameter [mm]
        - embedment_depth: Effective embedment depth [mm]
        - N0_Rk_s: Steel tension capacity [kN]
        - N0_Rk_p: Pullout capacity [kN]
        - V0_Rk_s: Steel shear capacity [kN]
        - s_cr_N: Critical spacing [mm]
        - c_cr_N: Critical edge distance [mm]
        - l_f: Effective length for shear [mm]
    
    Example:
        >>> props = get_anchor_properties_from_string("HUS4-H-12M-80", "C20/25")
        >>> print(f"Product: {props['product_name']}, d={props['diameter']}mm")
        Product: HUS4-H, d=12mm
        >>> print(f"N0_Rk_s = {props['N0_Rk_s']} kN")
        N0_Rk_s = 79.0 kN
    """
    # Parse the string
    product_name, diameter, embedment_depth = parse_anchor_product_string(anchor_string)
    
    # Get ETA values
    eta_values = get_eta_values(product_name, diameter, embedment_depth, concrete_grade)
    
    # Combine all properties
    result = {
        "product_name": product_name,
        "diameter": diameter,
        "embedment_depth": embedment_depth,
        "concrete_grade": concrete_grade,
        **eta_values
    }
    
    return result


def _try_parse_anchor_string(anchor_product: str) -> Tuple[str, Optional[int], Optional[float]]:
    """
    Try to parse an anchor product string to extract product name, diameter, and embedment.
    
    If the string is in format "HUS4-H-12M-80", returns (product_name, diameter, embedment).
    If the string is just a product name like "HUS4-H", returns (product_name, None, None).
    
    Args:
        anchor_product: Anchor product string (e.g., "HUS4-H-12M-80" or "HUS4-H")
    
    Returns:
        Tuple of (product_name, diameter_or_None, embedment_or_None)
    """
    # Product name mapping for dropdown format -> database format
    product_name_mapping = {
        "HIT-HY200": "HIT-HY 200",
        "HIT-RE500": "HIT-RE 500",
        "HIT-HY170": "HIT-HY 170",
    }
    
    # Check if it's already a valid product name (no parsing needed)
    if anchor_product in ETA_DATABASE:
        return anchor_product, None, None
    
    # Check mapped names
    if anchor_product in product_name_mapping:
        return product_name_mapping[anchor_product], None, None
    
    # Try to parse the string format "Product-DiameterM-Embedment"
    parts = anchor_product.rsplit("-", 2)
    
    if len(parts) == 3:
        product_name_raw = parts[0]
        diameter_str = parts[1]
        embedment_str = parts[2]
        
        # Check if diameter part ends with 'M'
        if diameter_str.endswith("M"):
            try:
                diameter = int(diameter_str.replace("M", ""))
                embedment = float(embedment_str)
                
                # Map product name if needed
                product_name = product_name_mapping.get(product_name_raw, product_name_raw)
                
                return product_name, diameter, embedment
            except (ValueError, TypeError):
                pass
    
    # If parsing fails, return the original string (will fail validation later)
    return anchor_product, None, None


def get_eta_values(
    anchor_product: str,
    diameter: float,
    embedment_depth: float,
    concrete_grade: str = "C20/25",
) -> Dict:
    """
    Retrieve characteristic values from ETA database

    Args:
        anchor_product: Product name (e.g., 'HUS4-H') or full string (e.g., 'HUS4-H-12M-80')
                       If full string is provided, diameter and embedment from the string
                       will be used instead of the parameters.
        diameter: Anchor diameter [mm] (ignored if anchor_product contains diameter)
        embedment_depth: Effective embedment depth [mm] (ignored if anchor_product contains embedment)
        concrete_grade: Concrete grade (e.g., 'C20/25', 'C25/30')

    Returns:
        Dict with N0_Rk_s, N0_Rk_p, V0_Rk_s [kN]

    Raises:
        ValueError: If product/diameter/embedment not found in database
    """
    # Try to parse the anchor_product string (handles both "HUS4-H" and "HUS4-H-12M-80" formats)
    parsed_product, parsed_diameter, parsed_embedment = _try_parse_anchor_string(anchor_product)
    
    # Use parsed values if available, otherwise use parameters
    anchor_product = parsed_product
    if parsed_diameter is not None:
        diameter = parsed_diameter
    if parsed_embedment is not None:
        embedment_depth = parsed_embedment
    
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

    # Find nearest available embedment depth if exact match not found
    if embedment_depth not in diameter_data:
        available_embedments = list(diameter_data.keys())

        # Find the closest available embedment
        closest_embedment = min(
            available_embedments, key=lambda x: abs(x - embedment_depth)
        )
        difference = abs(closest_embedment - embedment_depth)
        tolerance_percent = difference / closest_embedment * 100

        # If within 15% tolerance, use the closest value with a warning
        if tolerance_percent <= 15:
            print(
                f"NOTE: Embedment depth {embedment_depth} mm adjusted to {closest_embedment} mm "
                f"(difference: {difference:.1f} mm, {tolerance_percent:.1f}%)"
            )
            print(
                f"      Consider adjusting anchor length to match available embedment: {available_embedments} mm"
            )
            embedment_depth = closest_embedment
        else:
            raise ValueError(
                f"Embedment depth {embedment_depth} mm not available for "
                f"{anchor_product} {diameter}mm. "
                f"Available embedments: {available_embedments} mm. "
                f"Adjust anchor length to achieve one of these embedment depths."
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
    if "/" in concrete_grade:
        parts = concrete_grade.replace("C", "").split("/")
        f_ck = float(parts[0])
        return f_ck
    else:
        raise ValueError(
            f"Invalid concrete grade format: '{concrete_grade}'. "
            f"Expected format: 'C20/25', 'C25/30', etc."
        )


def convert_fck_to_concrete_grade(fck: float) -> str:
    """
    Convert f_ck value to European concrete grade notation

    European notation: C{cylinder}/{cube}
    Common grades: C20/25, C25/30, C30/37, C35/45, C40/50

    Args:
        fck: Characteristic cylinder compressive strength [MPa]

    Returns:
        concrete_grade: Grade notation (e.g., 'C20/25', 'C25/30', 'C30/37')

    Examples:
        >>> convert_fck_to_concrete_grade(20)
        'C20/25'
        >>> convert_fck_to_concrete_grade(25)
        'C25/30'
        >>> convert_fck_to_concrete_grade(30)
        'C30/37'
    """
    # Mapping of cylinder strength to cube strength (European standard)
    fck_to_grade = {
        20: "C20/25",
        25: "C25/30",
        30: "C30/37",
        35: "C35/45",
        40: "C40/50",
        45: "C45/55",
        50: "C50/60",
    }

    # Round to nearest available grade
    fck_int = int(round(fck))

    if fck_int in fck_to_grade:
        return fck_to_grade[fck_int]
    else:
        # Find closest available grade
        available = sorted(fck_to_grade.keys())
        closest = min(available, key=lambda x: abs(x - fck_int))
        return fck_to_grade[closest]


# ============================================================================
# MAIN INTEGRATION FUNCTION - Matches existing NZS 3101 pattern
# ============================================================================


def calculate_anchor_capacity_etag(
    thickness_concrete: float,
    diameter: float,
    embedment_depth: float,
    edge_distance: float,
    number_of_anchors: int,
    concrete_grade: str,
    is_cracked_concrete: bool,
    anchor_product: str = "HUS4-H",
    seismic_design: bool = False,
    spacing: Optional[float] = None,
    spacing_x: Optional[float] = None,
    spacing_y: Optional[float] = None,
    alpha_gap: float = 1.0,
    edge_distance_c2: Optional[float] = None,
    e_N_x: float = 0.0,
    e_N_y: float = 0.0,
    e_V: float = 0.0,
    has_edge_reinforcement: bool = False,
) -> Tuple[float, float]:
    """
    Calculate metal anchor capacity per ETAG 001 Annex C and EOTA TR 045

    This function matches the pattern of calculate_anchor_capacity() from NZS 3101,
    providing a consistent interface for anchor capacity calculations.

    Args:
        thickness_concrete: Concrete member thickness [mm]
        diameter: Anchor nominal diameter [mm]
        embedment_depth: Effective embedment depth h_ef [mm]
        edge_distance: Distance to nearest concrete edge c1 [mm]
                      Use 9999 for infinite/very large slabs (skips edge failure)
        number_of_anchors: Number of anchors (1, 2, or 4)
        concrete_grade: Concrete grade (e.g., 'C20/25', 'C25/30')
        is_cracked_concrete: True for cracked concrete per ETAG 001 Section 4.1
        anchor_product: Product name for ETA lookup (default 'HUS4-H')
        seismic_design: True to apply EOTA TR 045 seismic factors
        spacing: Spacing between anchors for 2-anchor config [mm] (required if n=2)
        spacing_x: Spacing in X direction for 4-anchor config [mm] (required if n=4)
        spacing_y: Spacing in Y direction for 4-anchor config [mm] (required if n=4)
        alpha_gap: Gap/clearance factor (1.0 = no gap, 0.5 = clearance hole)
                  IMPORTANT: Use 0.5 for anchors with standard clearance holes!
        edge_distance_c2: Perpendicular edge distance [mm] (defaults to c1)
        e_N_x: Eccentricity of tension load in X direction [mm]
        e_N_y: Eccentricity of tension load in Y direction [mm]
        e_V: Eccentricity of shear load [mm]
        has_edge_reinforcement: True if edge reinforcement is present

    Returns:
        Tuple[float, float]: (tension_capacity [N], shear_capacity [N])

    Standard References:
        - ETAG 001 Annex C: Design Methods for Anchorages
        - EOTA TR 045: Design Under Seismic Actions

    Example:
        >>> # Non-seismic, no clearance hole
        >>> Nt, Vs = calculate_anchor_capacity_etag(
        ...     thickness_concrete=165, diameter=12, embedment_depth=79.9,
        ...     edge_distance=250, spacing=165, number_of_anchors=2,
        ...     concrete_grade='C20/25', is_cracked_concrete=True,
        ...     seismic_design=False, alpha_gap=1.0)

        >>> # Seismic design with clearance hole (α_gap = 0.5)
        >>> Nt_seis, Vs_seis = calculate_anchor_capacity_etag(
        ...     thickness_concrete=165, diameter=12, embedment_depth=79.9,
        ...     edge_distance=250, spacing=165, number_of_anchors=2,
        ...     concrete_grade='C20/25', is_cracked_concrete=True,
        ...     seismic_design=True, alpha_gap=0.5)
    """
    # Auto-parse anchor_product string if in format "HUS4-H-12M-80"
    parsed_product, parsed_diameter, parsed_embedment = _try_parse_anchor_string(anchor_product)
    anchor_product = parsed_product
    if parsed_diameter is not None:
        diameter = parsed_diameter
    if parsed_embedment is not None:
        embedment_depth = parsed_embedment
    
    # Convert concrete grade to f_ck
    f_ck = convert_concrete_grade_to_fck(concrete_grade)

    # Get characteristic values from ETA database
    eta_values = get_eta_values(
        anchor_product, diameter, embedment_depth, concrete_grade
    )

    # Extract critical distances from ETA database
    s_cr_N = eta_values.get("s_cr_N", 3.0 * embedment_depth)  # Default: 3.0 * h_ef
    c_cr_N = eta_values.get("c_cr_N", 1.5 * embedment_depth)  # Default: 1.5 * h_ef
    l_f = eta_values.get("l_f", embedment_depth)  # Default: h_ef

    # Create anchor properties with all parameters
    anchor_props = AnchorProperties_ETAG(
        # Geometry
        diameter=diameter,
        embedment_depth=embedment_depth,
        edge_distance=edge_distance,
        edge_distance_c2=edge_distance_c2,
        # Concrete
        concrete_strength=f_ck,
        concrete_thickness=thickness_concrete,
        is_cracked_concrete=is_cracked_concrete,
        # Characteristic values from ETA
        N0_Rk_s=eta_values["N0_Rk_s"],
        N0_Rk_p=eta_values["N0_Rk_p"],
        V0_Rk_s=eta_values["V0_Rk_s"],
        # Critical distances from ETA
        s_cr_N=s_cr_N,
        c_cr_N=c_cr_N,
        l_f=l_f,
        # Group configuration
        number_of_anchors=number_of_anchors,
        spacing=spacing,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        # Load eccentricity
        e_N_x=e_N_x,
        e_N_y=e_N_y,
        e_V=e_V,
        # Gap factor (CRITICAL for shear with clearance holes!)
        alpha_gap=alpha_gap,
        # Seismic design
        seismic_design=seismic_design,
        is_undercut_anchor=False,  # HUS4-H is not undercut type
        # Reinforcement
        has_edge_reinforcement=has_edge_reinforcement,
    )

    # Calculate capacities
    calculator = MetalAnchorCapacity_ETAG001_TR045(anchor_props)
    tension_result = calculator.calculate_tension_capacity_ETAG_5_2_2()
    shear_result = calculator.calculate_shear_capacity_ETAG_5_2_3()

    # Extract design capacities in kN
    tension_capacity_kn = tension_result["N_Rd"]
    shear_capacity_kn = shear_result["V_Rd"]

    # Convert to Newtons to match NZS 3101 pattern
    tension_capacity_n = tension_capacity_kn * 1000
    shear_capacity_n = shear_capacity_kn * 1000

    return tension_capacity_n, shear_capacity_n


def calculate_anchor_capacity_etag_detailed(
    thickness_concrete: float,
    diameter: float,
    embedment_depth: float,
    edge_distance: float,
    number_of_anchors: int,
    concrete_grade: str,
    is_cracked_concrete: bool,
    anchor_product: str = "HUS4-H",
    seismic_design: bool = False,
    spacing: Optional[float] = None,
    spacing_x: Optional[float] = None,
    spacing_y: Optional[float] = None,
    alpha_gap: float = 1.0,
    edge_distance_c2: Optional[float] = None,
    e_N_x: float = 0.0,
    e_N_y: float = 0.0,
    e_V: float = 0.0,
    has_edge_reinforcement: bool = False,
) -> Dict:
    """
    Calculate detailed anchor capacity per ETAG 001 Annex C and EOTA TR 045.

    Returns ALL failure mode capacities like Hilti PROFIS report:

    TENSION (ETAG Annex C, Section 5.2.2):
    - Steel Strength (per anchor)
    - Pullout Strength (per anchor)
    - Concrete Breakout (group)

    SHEAR (ETAG Annex C, Section 5.2.3):
    - Steel Strength (per anchor)
    - Pryout Strength (group)
    - Concrete Edge Failure (group)

    Returns:
        Dict with detailed results for each failure mode in Newtons
    """
    # Auto-parse anchor_product string if in format "HUS4-H-12M-80"
    parsed_product, parsed_diameter, parsed_embedment = _try_parse_anchor_string(anchor_product)
    anchor_product = parsed_product
    if parsed_diameter is not None:
        diameter = parsed_diameter
    if parsed_embedment is not None:
        embedment_depth = parsed_embedment
    
    f_ck = convert_concrete_grade_to_fck(concrete_grade)

    # Get characteristic values from ETA database
    eta_values = get_eta_values(
        anchor_product, diameter, embedment_depth, concrete_grade
    )

    # Extract critical distances from ETA database
    s_cr_N = eta_values.get("s_cr_N", 3.0 * embedment_depth)
    c_cr_N = eta_values.get("c_cr_N", 1.5 * embedment_depth)
    l_f = eta_values.get("l_f", embedment_depth)

    # Create anchor properties
    anchor_props = AnchorProperties_ETAG(
        # Geometric properties
        diameter=diameter,
        embedment_depth=embedment_depth,
        edge_distance=edge_distance,
        edge_distance_c2=(
            edge_distance_c2 if edge_distance_c2 is not None else edge_distance
        ),
        # Concrete properties
        concrete_strength=f_ck,
        concrete_thickness=thickness_concrete,
        is_cracked_concrete=is_cracked_concrete,
        # Characteristic capacities from ETA
        N0_Rk_s=eta_values["N0_Rk_s"],
        N0_Rk_p=eta_values["N0_Rk_p"],
        V0_Rk_s=eta_values["V0_Rk_s"],
        # Critical distances from ETA
        s_cr_N=s_cr_N,
        c_cr_N=c_cr_N,
        l_f=l_f,
        # Gap factor (clearance hole effect)
        alpha_gap=alpha_gap,
        # Group configuration
        number_of_anchors=number_of_anchors,
        spacing=spacing,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        # Load eccentricity
        e_N_x=e_N_x,
        e_N_y=e_N_y,
        e_V=e_V,
        # Seismic design
        seismic_design=seismic_design,
        is_undercut_anchor=False,
        # Reinforcement
        has_edge_reinforcement=has_edge_reinforcement,
    )

    # Calculate capacities using the class
    calculator = MetalAnchorCapacity_ETAG001_TR045(anchor_props)
    tension_result = calculator.calculate_tension_capacity_ETAG_5_2_2()
    shear_result = calculator.calculate_shear_capacity_ETAG_5_2_3()

    # Build detailed results dictionary (convert kN to N)
    detailed_results = {
        # ===== TENSION CAPACITIES =====
        "tension": {
            "governing_capacity_N": tension_result["N_Rd"] * 1000,
            "governing_mode": tension_result["governing_mode"],
            # Steel Failure (per anchor) - ETAG 5.2.2.2
            "steel_failure": {
                "N_Rd_s_N": tension_result["N_Rd_steel"] * 1000,
                "N_Rd_s_kN": tension_result["N_Rd_steel"],
                "details": tension_result["steel_details"],
                "note": "Per anchor capacity",
            },
            # Pullout Failure (per anchor) - ETAG 5.2.2.3
            "pullout_failure": {
                "N_Rd_p_N": tension_result["N_Rd_pullout"] * 1000,
                "N_Rd_p_kN": tension_result["N_Rd_pullout"],
                "details": tension_result["pullout_details"],
                "note": "Per anchor capacity",
            },
            # Concrete Cone Failure (group) - ETAG 5.2.2.4
            "concrete_breakout": {
                "N_Rd_c_N": tension_result["N_Rd_concrete"] * 1000,
                "N_Rd_c_kN": tension_result["N_Rd_concrete"],
                "details": tension_result["concrete_details"],
                "note": "Anchor group capacity",
            },
        },
        # ===== SHEAR CAPACITIES =====
        "shear": {
            "governing_capacity_N": shear_result["V_Rd"] * 1000,
            "governing_mode": shear_result["governing_mode"],
            # Steel Failure (per anchor) - ETAG 5.2.3.2
            "steel_failure": {
                "V_Rd_s_N": shear_result["V_Rd_steel"] * 1000,
                "V_Rd_s_kN": shear_result["V_Rd_steel"],
                "details": shear_result["steel_details"],
                "note": "Per anchor capacity (without lever arm)",
            },
            # Pryout Failure (group) - ETAG 5.2.3.3
            "pryout_failure": {
                "V_Rd_cp_N": shear_result["V_Rd_pryout"] * 1000,
                "V_Rd_cp_kN": shear_result["V_Rd_pryout"],
                "details": shear_result["pryout_details"],
                "note": "Anchor group capacity",
            },
            # Concrete Edge Failure (group) - ETAG 5.2.3.4
            "concrete_edge": {
                "V_Rd_c_N": shear_result["V_Rd_concrete"] * 1000,
                "V_Rd_c_kN": shear_result["V_Rd_concrete"],
                "details": shear_result["concrete_edge_details"],
                "note": "Anchor group capacity",
            },
        },
        # ===== SUMMARY =====
        "summary": {
            "N_Rd_N": tension_result["N_Rd"] * 1000,
            "V_Rd_N": shear_result["V_Rd"] * 1000,
            "N_Rd_kN": tension_result["N_Rd"],
            "V_Rd_kN": shear_result["V_Rd"],
            "tension_governing": tension_result["governing_mode"],
            "shear_governing": shear_result["governing_mode"],
            "number_of_anchors": number_of_anchors,
            "seismic_design": seismic_design,
        },
        # ===== INPUT PARAMETERS =====
        "input_parameters": {
            "diameter": diameter,
            "embedment_depth": embedment_depth,
            "edge_distance": edge_distance,
            "concrete_grade": concrete_grade,
            "f_ck": f_ck,
            "is_cracked": is_cracked_concrete,
            "number_of_anchors": number_of_anchors,
            "spacing": spacing,
            "alpha_gap": alpha_gap,
            "anchor_product": anchor_product,
        },
    }

    return detailed_results


# ============================================================================
# END OF MODULE
# ============================================================================
