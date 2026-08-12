"""NZS 4219:2009 Method — non-specific design of engineering systems restraints.

This module implements the non-specific design route of NZS 4219:2009.
It does not implement TS 1170.5:2025. Loading methodologies must not be mixed.
Verification of the supporting primary structure is outside NZS 4219 scope;
report reactions for separate structural verification (NZS 3101 slab checks
may be used at the interface when inputs are supplied).

This module references the following standard:
NZS 4219:2009 — Seismic performance of engineering systems in buildings.

Method developed: August 2026
(c) Constratum Ltd

Developed - NSh
Reviewed  - —

Binding rule: every engineering number traces to a named function in a standards
module. This module owns NZS 4219 demand and NZS 4219 tabulated capacities only.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

GRAVITY_M_S2 = 9.80665
COEFFICIENT_FACTOR_2_7 = 2.7
COEFFICIENT_CAP = 3.6
SHARED_HANGER_MULTIPLIER = 1.4
CP_ANCHOR = 0.85
MODULE_VERSION = "0.3.0"
RESILIENT_EQUATIONS_PDF_VERIFIED = False  # gated — integrity register pages 39–43


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def _require_finite(name: str, value: float) -> float:
    """Require a finite numeric value; raise ValueError otherwise."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; got {value!r}") from exc
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return v


def _require_positive(name: str, value: float) -> float:
    """Require a strictly positive finite value."""
    v = _require_finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0; got {value!r}")
    return v


def _require_nonnegative(name: str, value: float) -> float:
    """Require a non-negative finite value."""
    v = _require_finite(name, value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")
    return v


def _normalise_key(value: str) -> str:
    """Case-insensitive key normalisation: strip, collapse whitespace, lower."""
    if not isinstance(value, str):
        raise ValueError(f"key must be a string; got {type(value).__name__}")
    return " ".join(value.strip().split()).lower()


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool; got {type(value).__name__}")
    return value


def _require_int(name: str, value: Any, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Reject bool; allow int-like only when exactly int
        try:
            if isinstance(value, float) and value.is_integer():
                iv = int(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                iv = value
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    else:
        iv = value
    if minimum is not None and iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {iv}")
    return iv


# ------------------------------------------------------------------
# Section 1 — Scope
# ------------------------------------------------------------------

class Section1:
    """Scope, exclusions, notation helpers (NZS 4219 cl. 1.1.2)."""

    @staticmethod
    def validate_within_scope_cl_1_1_2(inputs: dict) -> None:
        """
        Raise ValueError with clause reference for out-of-scope cases (cl. 1.1.2).

        Expected keys (checked when present):
          importance_level, mass_ratio, period_s,
          ground_supported_independent_of_building, lifts, escalators,
          building_contents, portable, fire_sprinkler_pipework,
          suspended_ceilings, outside_1m_footprint_boundary
        """
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dict")

        if "importance_level" in inputs:
            il = _require_int("importance_level", inputs["importance_level"], minimum=1)
            if il == 5:
                raise ValueError(
                    "NZS 4219 cl. 1.1.2(a): importance level 5 is outside the scope "
                    "of this Standard (see AS/NZS 1170.0)."
                )
            if il not in (1, 2, 3, 4):
                raise ValueError(
                    f"importance_level must be 1–4 for NZS 4219 non-specific design; got {il}"
                )

        mass_ratio = inputs.get("mass_ratio")
        period_s = inputs.get("period_s")
        if mass_ratio is not None and period_s is not None:
            mr = _require_nonnegative("mass_ratio", mass_ratio)
            t = _require_nonnegative("period_s", period_s)
            if mr > 0.20 and t > 0.2:
                raise ValueError(
                    "NZS 4219 cl. 1.1.2(b): component mass exceeding 20% of the "
                    "combined mass with period > 0.2 s is outside scope."
                )

        if inputs.get("ground_supported_independent_of_building"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2(c): items supported on the ground independently "
                "of the building, and external to the building, are outside scope."
            )
        if inputs.get("lifts") or inputs.get("escalators"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2(d): lifts (including guide rails) and escalators "
                "are outside scope (see NZS 4332)."
            )
        if inputs.get("building_contents") or inputs.get("portable"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2(e): contents of buildings including portable "
                "appliances, and items not attached to the building structure, "
                "are outside scope."
            )
        if inputs.get("fire_sprinkler_pipework"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2(f): fire sprinkler system pipework is outside "
                "scope; design to NZS 4541."
            )
        if inputs.get("suspended_ceilings"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2(g): suspended ceilings are outside scope; "
                "design to AS/NZS 2785."
            )
        if inputs.get("outside_1m_footprint_boundary"):
            raise ValueError(
                "NZS 4219 cl. 1.1.2 / C5.11: services outside the 1 m footprint "
                "boundary from the building are outside the applicable scope for "
                "this module path."
            )


# ------------------------------------------------------------------
# Section 2 — Performance criteria helpers
# ------------------------------------------------------------------

class Section2:
    """Performance criteria and load-path requirements."""

    TABLE_2_LIMIT_STATE = {
        "p1": "ULS",
        "p2": "ULS",
        "p3": "ULS",
        "p4": "ULS",
        "p5": "SLS2",
        "p6": "SLS1",
        "p7": "SLS1",
    }

    @staticmethod
    def get_limit_state_table_2(category: str) -> str:
        """Return 'ULS' | 'SLS1' | 'SLS2'. NZS 4219 Table 2."""
        key = _normalise_key(category)
        if key not in Section2.TABLE_2_LIMIT_STATE:
            raise ValueError(
                f"Unknown category '{category}'. Valid: P1–P7 (NZS 4219 Table 2)."
            )
        return Section2.TABLE_2_LIMIT_STATE[key]

    @staticmethod
    def validate_category_for_importance_level_table_2(
        category: str, importance_level: int
    ) -> None:
        """Raise if P5 used with IL other than 4 (Table 2 NOTE)."""
        key = _normalise_key(category)
        if key not in Section2.TABLE_2_LIMIT_STATE:
            raise ValueError(
                f"Unknown category '{category}'. Valid: P1–P7 (NZS 4219 Table 2)."
            )
        il = _require_int("importance_level", importance_level, minimum=1)
        if il not in (1, 2, 3, 4):
            raise ValueError(f"importance_level must be 1–4; got {il}")
        if key == "p5" and il != 4:
            raise ValueError(
                "NZS 4219 Table 2 NOTE: category P5 only applies to importance "
                f"level 4 buildings; got IL{il}."
            )


# ------------------------------------------------------------------
# Section 3 — Non-specific design
# ------------------------------------------------------------------

class Section3:
    """Non-specific design: coefficients, demand, subtype actions, fixings."""

    # Table 3 — Zone factors (NZS 4219:2009, north to south)
    ZONE_FACTORS = {
        "kaitaia": 0.13,
        "paihia/russell": 0.13,
        "kaikohe": 0.13,
        "whangarei": 0.13,
        "dargaville": 0.13,
        "warkworth": 0.13,
        "auckland": 0.13,
        "manakau city": 0.13,
        "waiuku": 0.13,
        "pukekohe": 0.13,
        "thames": 0.16,
        "paeroa": 0.18,
        "waihi": 0.18,
        "huntly": 0.15,
        "ngaruawahia": 0.15,
        "morrinsville": 0.18,
        "te aroha": 0.18,
        "tauranga": 0.20,
        "mount maunganui": 0.20,
        "hamilton": 0.16,
        "cambridge": 0.18,
        "te awamutu": 0.17,
        "matamata": 0.19,
        "te puke": 0.22,
        "putaruru": 0.21,
        "tokoroa": 0.21,
        "otorohanga": 0.17,
        "te kuiti": 0.18,
        "mangakino": 0.21,
        "rotorua": 0.24,
        "kawerau": 0.29,
        "whakatane": 0.30,
        "opotiki": 0.30,
        "ruatoria": 0.33,
        "murupara": 0.30,
        "taupo": 0.28,
        "taumarunui": 0.21,
        "turangi": 0.27,
        "gisborne": 0.36,
        "wairoa": 0.37,
        "waitara": 0.18,
        "new plymouth": 0.18,
        "inglewood": 0.18,
        "stratford": 0.18,
        "opunake": 0.18,
        "hawera": 0.18,
        "patea": 0.19,
        "raetihi": 0.26,
        "ohakune": 0.27,
        "waiouru": 0.29,
        "napier": 0.38,
        "hastings": 0.39,
        "wanganui": 0.25,
        "waipawa": 0.41,
        "waipukurau": 0.41,
        "taihape": 0.33,
        "marton": 0.30,
        "bulls": 0.31,
        "feilding": 0.37,
        "palmerston north": 0.38,
        "dannevirke": 0.42,
        "woodville": 0.41,
        "pahiatua": 0.42,
        "foxton/foxton beach": 0.36,
        "levin": 0.40,
        "otaki": 0.40,
        "waikanae": 0.40,
        "paraparaumu": 0.40,
        "masterton": 0.42,
        "porirua": 0.40,
        "wellington cbd": 0.40,
        "wellington cbd (north of basin reserve)": 0.40,
        "wellington": 0.40,
        "hutt valley - south of taita gorge": 0.40,
        "upper hutt": 0.42,
        "eastbourne - point howard": 0.40,
        "wainuiomata": 0.40,
        "takaka": 0.23,
        "motueka": 0.26,
        "nelson": 0.27,
        "picton": 0.30,
        "blenheim": 0.33,
        "st arnaud": 0.36,
        "westport": 0.30,
        "reefton": 0.37,
        "murchison": 0.34,
        "springs junction": 0.45,
        "hanmer springs": 0.55,
        "seddon": 0.40,
        "ward": 0.40,
        "cheviot": 0.40,
        "greymouth": 0.37,
        "kaikoura": 0.42,
        "harihari": 0.46,
        "hokitika": 0.45,
        "fox glacier": 0.44,
        "franz josef": 0.44,
        "otira": 0.60,
        "arthur's pass": 0.60,
        "arthurs pass": 0.60,
        "rangiora": 0.33,
        "darfield": 0.30,
        "akaroa": 0.16,
        "christchurch": 0.22,
        "geraldine": 0.19,
        "ashburton": 0.20,
        "fairlie": 0.24,
        "temuka": 0.17,
        "timaru": 0.15,
        "mt cook": 0.38,
        "twizel": 0.27,
        "waimate": 0.14,
        "cromwell": 0.24,
        "wanaka": 0.30,
        "arrowtown": 0.30,
        "alexandra": 0.21,
        "queenstown": 0.32,
        "milford sound": 0.54,
        "palmerston": 0.13,
        "oamaru": 0.13,
        "dunedin": 0.13,
        "mosgiel": 0.13,
        "riverton": 0.20,
        "te anau": 0.36,
        "gore": 0.18,
        "winton": 0.20,
        "balclutha": 0.13,
        "mataura": 0.17,
        "bluff": 0.15,
        "invercargill": 0.17,
        "oban": 0.14,
    }

    LOCATION_ALIASES = {
        "auckland city": "auckland",
        "manukau city": "manakau city",
        "manukau": "manakau city",
        "wellington cbd north of basin reserve": "wellington cbd (north of basin reserve)",
        "hutt valley": "hutt valley - south of taita gorge",
        "eastbourne": "eastbourne - point howard",
        "foxton": "foxton/foxton beach",
        "foxton beach": "foxton/foxton beach",
        "paihia": "paihia/russell",
        "russell": "paihia/russell",
        "mt maunganui": "mount maunganui",
        "mount cook": "mt cook",
    }

    # Table 5 — provisional Rc matrix (PUBLISHED_VALUE_PENDING_PDF)
    # Keys: category -> {il_bucket: Rc} where il_bucket in (12, 3, 4)
    TABLE_5_RC = {
        "p1": {12: 1.0, 3: 1.3, 4: 1.8},
        "p2": {12: 1.0, 3: 1.3, 4: 1.8},
        "p4": {12: 1.0, 3: 1.3, 4: 1.8},
        "p3": {12: 0.9, 3: 1.2, 4: 1.6},
        "p5": {4: 1.0},
        "p6": {12: 0.5, 3: 0.5, 4: 0.5},
        "p7": {12: 0.25, 3: 0.25, 4: 0.25},
    }

    # Table 6/7 pipe restraint — provisional (integrity register)
    # row: diameter -> {min_wall_mm, C: (spacing_m, force_kn)}
    TABLE_6_STEEL_TRANSVERSE = {
        50: {1.0: (7.7, 0.45), 2.0: (6.1, 0.71), 3.6: (5.0, 1.05), "min_wall_mm": 2.90},
        65: {1.0: (8.9, 0.84), 2.0: (7.0, 1.33), 3.6: (5.7, 1.96), "min_wall_mm": 3.60},
        80: {1.0: (9.6, 1.28), 2.0: (7.6, 2.04), 3.6: (6.0, 2.88), "min_wall_mm": 4.00},
        100: {1.0: (11.4, 2.22), 2.0: (9.0, 3.52), 3.6: (6.8, 4.78), "min_wall_mm": 4.50},
        150: {1.0: (12.0, 4.36), 2.0: (10.2, 7.43), 3.6: (7.6, 9.97), "min_wall_mm": 4.88},
        200: {1.0: (12.0, 6.68), 2.0: (10.6, 11.90), 3.6: (7.9, 15.97), "min_wall_mm": 4.80},
    }
    TABLE_6_COPPER_TRANSVERSE = {
        50: {1.0: (2.6, 0.09), 2.0: (2.1, 0.15), 3.6: (1.7, 0.22), "min_wall_mm": 1.22},
        65: {1.0: (3.0, 0.16), 2.0: (2.4, 0.26), 3.6: (1.9, 0.38), "min_wall_mm": 1.22},
        80: {1.0: (3.4, 0.27), 2.0: (2.7, 0.43), 3.6: (2.2, 0.64), "min_wall_mm": 1.42},
        100: {1.0: (3.9, 0.47), 2.0: (3.1, 0.74), 3.6: (2.5, 1.10), "min_wall_mm": 1.63},
    }
    TABLE_7_STEEL_LONGITUDINAL = {
        50: {1.0: (23.0, 1.32), 2.0: (18.0, 2.07), 3.6: (15.0, 3.10), "min_wall_mm": 2.90},
        65: {1.0: (26.0, 2.45), 2.0: (21.0, 3.95), 3.6: (17.0, 5.76), "min_wall_mm": 3.60},
        80: {1.0: (28.0, 3.73), 2.0: (22.0, 5.86), 3.6: (18.0, 8.63), "min_wall_mm": 4.00},
        100: {1.0: (34.0, 6.61), 2.0: (27.0, 10.50), 3.6: (20.0, 14.00), "min_wall_mm": 4.50},
        150: {1.0: (36.0, 13.07), 2.0: (30.0, 21.79), 3.6: (22.0, 28.76), "min_wall_mm": 4.88},
        200: {1.0: (36.0, 20.04), 2.0: (32.0, 35.62), 3.6: (23.0, 46.09), "min_wall_mm": 4.80},
    }
    TABLE_7_COPPER_LONGITUDINAL = {
        50: {1.0: (8.0, 0.28), 2.0: (6.0, 0.42), 3.6: (5.0, 0.63), "min_wall_mm": 1.22},
        65: {1.0: (9.0, 0.48), 2.0: (7.0, 0.74), 3.6: (5.0, 0.96), "min_wall_mm": 1.22},
        80: {1.0: (10.0, 0.79), 2.0: (8.0, 1.26), 3.6: (6.0, 1.70), "min_wall_mm": 1.42},
        100: {1.0: (11.0, 1.31), 2.0: (9.0, 2.15), 3.6: (7.0, 3.01), "min_wall_mm": 1.63},
    }
    TABLE_7_MAX_OFFSET_M = {1.0: 0.9, 2.0: 0.6, 3.6: 0.4}
    PIPE_C_COLUMNS = (1.0, 2.0, 3.6)

    # Tables 8–11 fixing capacities
    TABLE_8_WOODSCREWS = {
        8: {
            "diameter_mm": 4.17,
            "minimum_penetration_mm": 30,
            "tension_kn": 1.10,
            "shear_kn": 1.10,
            "minimum_edge_distance_mm": 20,
            "minimum_end_distance_and_spacing_mm": 45,
        },
        9: {
            "diameter_mm": 4.52,
            "minimum_penetration_mm": 32,
            "tension_kn": 1.28,
            "shear_kn": 1.25,
            "minimum_edge_distance_mm": 23,
            "minimum_end_distance_and_spacing_mm": 45,
        },
        10: {
            "diameter_mm": 4.88,
            "minimum_penetration_mm": 35,
            "tension_kn": 1.51,
            "shear_kn": 1.45,
            "minimum_edge_distance_mm": 25,
            "minimum_end_distance_and_spacing_mm": 50,
        },
        12: {
            "diameter_mm": 5.59,
            "minimum_penetration_mm": 40,
            "tension_kn": 1.98,
            "shear_kn": 1.87,
            "minimum_edge_distance_mm": 28,
            "minimum_end_distance_and_spacing_mm": 55,
        },
        14: {
            "diameter_mm": 6.30,
            "minimum_penetration_mm": 45,
            "tension_kn": 2.50,
            "shear_kn": 2.33,
            "minimum_edge_distance_mm": 32,
            "minimum_end_distance_and_spacing_mm": 65,
        },
    }
    TABLE_9_COACH_SCREWS = {
        8.0: {
            "minimum_penetration_mm": 80,
            "tension_kn": 5.38,
            "shear_kn": 3.54,
            "minimum_edge_distance_mm": 40,
            "minimum_end_distance_and_spacing_mm": 80,
        },
        10.0: {
            "minimum_penetration_mm": 100,
            "tension_kn": 7.49,
            "shear_kn": 4.42,
            "minimum_edge_distance_mm": 50,
            "minimum_end_distance_and_spacing_mm": 100,
        },
        12.0: {
            "minimum_penetration_mm": 120,
            "tension_kn": 9.91,
            "shear_kn": 7.28,
            "minimum_edge_distance_mm": 60,
            "minimum_end_distance_and_spacing_mm": 120,
        },
    }
    TABLE_10_BOLTS_IN_SHEAR = {
        "m8": {"minimum_end_distance_mm": 16, "capacity_single_shear_kn": 6.1},
        "m10": {"minimum_end_distance_mm": 20, "capacity_single_shear_kn": 10.1},
        "m12": {"minimum_end_distance_mm": 24, "capacity_single_shear_kn": 15.1},
        "m16": {"minimum_end_distance_mm": 32, "capacity_single_shear_kn": 28.6},
        "m20": {"minimum_end_distance_mm": 40, "capacity_single_shear_kn": 45.0},
    }
    TABLE_11_MASONRY_BOLTS = {
        12.0: {"minimum_embedment_mm": 100, "tension_and_shear_kn": 10.0},
        16.0: {"minimum_embedment_mm": 125, "tension_and_shear_kn": 15.0},
        20.0: {"minimum_embedment_mm": 150, "tension_and_shear_kn": 25.0},
        24.0: {"minimum_embedment_mm": 175, "tension_and_shear_kn": 35.0},
    }

    # ------------------------------------------------------------------
    # Table 12 — brace / fastener material Standards (reference only)
    # ------------------------------------------------------------------
    TABLE_12_BRACE_MATERIALS = {
        "angles": "AS/NZS 3679.1 – 300 (Grade 300)",
        "flats": "AS/NZS 3679.1 – 300",
        "shs": "AS 1163, Grade C350LO",
        "bolts": "AS 1111.1, Property class 4.6",
        "threaded_rods": "AS 1111.1, Property class 4.6",
        "nuts": "AS 1112.3",
    }

    # Transcription status for Tables 13/14 (integrity register).
    # TRANSCRIBED from NZS4219_2009.md OCR — pending second-person PDF verification.
    BRACE_TABLE_TRANSCRIPTION_STATUS = "TRANSCRIBED"

    # Table 13 lengths are not applicable (tension); connection configs for angles/flats.
    # Keys: section_size -> bolt_size -> {1_bolt, 2_bolts, welded, fillet_weld_size_mm, fillet_weld_length_mm}
    TABLE_13_ANGLE_TENSION = {
        "25x25x3ea": {
            "m8": {
                "1_bolt": 6.0,
                "2_bolts": 12.0,
                "welded": 32.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 80.0,
            },
            "m10": {
                "1_bolt": 10.0,
                "2_bolts": 20.0,
                "welded": 39.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 100.0,
            },
        },
        "40x40x3ea": {
            "m12": {
                "1_bolt": 15.0,
                "2_bolts": 30.0,
                "welded": 55.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 140.0,
            },
        },
        "50x50x3ea": {
            "m16": {
                "1_bolt": 28.0,
                "2_bolts": 57.0,
                "welded": 69.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 170.0,
            },
        },
        "50x50x5ea": {
            "m16": {
                "1_bolt": 28.0,
                "2_bolts": 57.0,
                "welded": 101.0,
                "fillet_weld_size_mm": 5.0,
                "fillet_weld_length_mm": 150.0,
            },
        },
        "50x50x8ea": {
            "m16": {
                "1_bolt": 28.0,
                "2_bolts": 57.0,
                "welded": 166.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 200.0,
            },
        },
        "65x65x8ea": {
            "m16": {
                "1_bolt": 28.0,
                "2_bolts": 57.0,
                "welded": 233.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 280.0,
            },
        },
        "75x75x8ea": {
            "m20": {
                "1_bolt": 62.0,
                "2_bolts": 124.0,
                "welded": 267.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 330.0,
            },
        },
        "75x75x10ea": {
            "m20": {
                "1_bolt": 62.0,
                "2_bolts": 124.0,
                "welded": 377.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 460.0,
            },
        },
        "90x90x10ea": {
            "m20": {
                "1_bolt": 62.0,
                "2_bolts": 124.0,
                "welded": 457.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 550.0,
            },
        },
        "100x100x8ea": {
            "m20": {
                "1_bolt": 62.0,
                "2_bolts": 124.0,
                "welded": 429.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 520.0,
            },
        },
    }
    TABLE_13_FLAT_TENSION = {
        "20x3": {
            "m8": {
                "1_bolt": 6.0,
                "2_bolts": 12.0,
                "welded": 17.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 50.0,
            },
        },
        "20x5": {
            "m8": {
                "1_bolt": 6.0,
                "2_bolts": 12.0,
                "welded": 29.0,
                "fillet_weld_size_mm": 5.0,
                "fillet_weld_length_mm": 50.0,
            },
        },
        "20x6": {
            "m8": {
                "1_bolt": 6.0,
                "2_bolts": 12.0,
                "welded": 35.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 50.0,
            },
        },
        "20x10": {
            "m8": {
                "1_bolt": 6.0,
                "2_bolts": 12.0,
                "welded": 58.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 70.0,
            },
        },
        "25x3": {
            "m10": {
                "1_bolt": 10.0,
                "2_bolts": 20.0,
                "welded": 22.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 60.0,
            },
        },
        "25x5": {
            "m10": {
                "1_bolt": 10.0,
                "2_bolts": 20.0,
                "welded": 36.0,
                "fillet_weld_size_mm": 5.0,
                "fillet_weld_length_mm": 60.0,
            },
        },
        "25x10": {
            "m10": {
                "1_bolt": 10.0,
                "2_bolts": 20.0,
                "welded": 72.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 90.0,
            },
        },
        "25x12": {
            "m10": {
                "1_bolt": 10.0,
                "2_bolts": 20.0,
                "welded": 86.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 110.0,
            },
        },
        "40x3": {
            "m16": {
                "1_bolt": 22.0,
                "2_bolts": 35.0,
                "welded": 35.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 90.0,
            },
        },
        "40x6": {
            "m16": {
                "1_bolt": 29.0,
                "2_bolts": 57.0,
                "welded": 69.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 90.0,
            },
        },
        "40x10": {
            "m16": {
                "1_bolt": 29.0,
                "2_bolts": 57.0,
                "welded": 115.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 140.0,
            },
        },
        "40x12": {
            "m16": {
                "1_bolt": 29.0,
                "2_bolts": 57.0,
                "welded": 138.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 170.0,
            },
        },
        "50x3": {
            "m20": {
                "1_bolt": 28.0,
                "2_bolts": 43.0,
                "welded": 43.0,
                "fillet_weld_size_mm": 3.0,
                "fillet_weld_length_mm": 110.0,
            },
        },
        "50x6": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 86.0,
                "welded": 86.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 110.0,
            },
        },
        "50x10": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 144.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 180.0,
            },
        },
        "50x12": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 173.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 210.0,
            },
        },
        "75x6": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 130.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 160.0,
            },
        },
        "75x10": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 216.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 260.0,
            },
        },
        "75x12": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 259.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 320.0,
            },
        },
        "100x6": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 173.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 210.0,
            },
        },
        "100x10": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 288.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 350.0,
            },
        },
        "100x12": {
            "m20": {
                "1_bolt": 45.0,
                "2_bolts": 89.0,
                "welded": 346.0,
                "fillet_weld_size_mm": 6.0,
                "fillet_weld_length_mm": 420.0,
            },
        },
    }
    TABLE_13_THREADED_ROD_TENSION = {
        "m10": 18.0,
        "m12": 27.0,
        "m16": 50.0,
        "m20": 78.0,
        "m24": 113.0,
    }
    TABLE_13_WIRE_TENSION = {
        "3.2mm": 1.5,
    }

    # Table 14 compression — lengths 0.5–3.0 m.
    # Angle values: (capacity_kn, bolts_required) or (capacity_kn, "welded_base") for *.
    # None = not permitted / no capacity at that length.
    TABLE_14_LENGTHS_M = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    TABLE_14_ANGLE_COMPRESSION = {
        # section -> {bolt_size, capacities: {length_m: (kn, bolts_or_"welded_base")}}
        "25x25x3ea": {
            "bolt_size": "m8",
            "capacities": {
                0.5: (7.4, 2),
                1.0: (3.5, 1),
                1.5: (1.4, 1),
                2.0: (0.8, 1),
                2.5: (0.5, 1),
                3.0: None,
            },
        },
        "30x30x3ea": {
            "bolt_size": "m10",
            "capacities": {
                0.5: (11.2, 2),
                1.0: (4.8, 1),
                1.5: (2.4, 1),
                2.0: (1.4, 1),
                2.5: (0.9, 1),
                3.0: (0.6, 1),
            },
        },
        "40x40x3ea": {
            "bolt_size": "m12",
            "capacities": {
                0.5: (17.5, 2),
                1.0: (10.0, 1),
                1.5: (5.7, 1),
                2.0: (3.4, 1),
                2.5: (2.4, 1),
                3.0: (1.6, 1),
            },
        },
        "50x50x3ea": {
            "bolt_size": "m16",
            "capacities": {
                0.5: (23.7, 1),
                1.0: (15.4, 1),
                1.5: (11.5, 1),
                2.0: (6.5, 1),
                2.5: (4.4, 1),
                3.0: (3.1, 1),
            },
        },
        "50x50x5ea": {
            "bolt_size": "m16",
            "capacities": {
                0.5: (37.7, 2),
                1.0: (25.6, 1),
                1.5: (15.4, 1),
                2.0: (9.5, 1),
                2.5: (6.3, 1),
                3.0: (4.6, 1),
            },
        },
        "50x50x8ea": {
            "bolt_size": "m16",
            "capacities": {
                0.5: (61.9, 3),
                1.0: (41.0, 2),
                1.5: (25.2, 1),
                2.0: (15.6, 1),
                2.5: (9.6, 1),
                3.0: (7.2, 1),
            },
        },
        "65x65x8ea": {
            "bolt_size": "m16",
            "capacities": {
                0.5: (93.0, "welded_base"),
                1.0: (69.5, 3),
                1.5: (47.2, 2),
                2.0: (33.7, 2),
                2.5: (23.6, 1),
                3.0: (16.9, 1),
            },
        },
        "75x75x8ea": {
            "bolt_size": "m20",
            "capacities": {
                0.5: (100.0, 3),
                1.0: (72.6, 2),
                1.5: (49.1, 2),
                2.0: (34.9, 1),
                2.5: (23.2, 1),
                3.0: (16.8, 1),
            },
        },
        "75x75x10ea": {
            "bolt_size": "m20",
            "capacities": {
                0.5: (134.9, "welded_base"),
                1.0: (100.8, 3),
                1.5: (79.9, 2),
                2.0: (57.4, 2),
                2.5: (42.9, 1),
                3.0: (30.1, 1),
            },
        },
        "90x90x10ea": {
            "bolt_size": "m20",
            "capacities": {
                0.5: (173.3, "welded_base"),
                1.0: (161.5, "welded_base"),
                1.5: (132.1, 3),
                2.0: (106.9, 3),
                2.5: (87.3, 2),
                3.0: (70.8, 2),
            },
        },
        "100x100x8ea": {
            "bolt_size": "m20",
            "capacities": {
                0.5: (145.2, "welded_base"),
                1.0: (137.3, "welded_base"),
                1.5: (116.4, 3),
                2.0: (98.8, 3),
                2.5: (80.3, 2),
                3.0: (64.1, 2),
            },
        },
    }
    # Flat compression: capacity_kn or None
    TABLE_14_FLAT_COMPRESSION = {
        "20x3": {0.5: None, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "20x5": {0.5: 0.6, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "20x6": {0.5: 1.1, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "20x10": {0.5: 4.6, 1.0: 1.2, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "25x3": {0.5: None, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "25x5": {0.5: 0.8, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "25x10": {0.5: 6.1, 1.0: 1.6, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "25x12": {0.5: 9.7, 1.0: 2.8, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "40x3": {0.5: None, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "40x6": {0.5: 2.4, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "40x10": {0.5: 10.0, 1.0: 2.7, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "40x12": {0.5: 16.4, 1.0: 4.8, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "50x3": {0.5: None, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "50x6": {0.5: 3.0, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "50x10": {0.5: 12.8, 1.0: 3.5, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "50x12": {0.5: 20.9, 1.0: 6.1, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "75x6": {0.5: 5.8, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "75x10": {0.5: 24.2, 1.0: 6.6, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "75x12": {0.5: 39.6, 1.0: 11.5, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "100x6": {0.5: 8.5, 1.0: None, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "100x10": {0.5: 35.6, 1.0: 9.7, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
        "100x12": {0.5: 58.2, 1.0: 17.0, 1.5: None, 2.0: None, 2.5: None, 3.0: None},
    }
    TABLE_14_SHS_COMPRESSION = {
        "25x25x3.0shs": {
            0.5: 35.0,
            1.0: 17.0,
            1.5: 7.7,
            2.0: 4.1,
            2.5: 2.6,
            3.0: 1.8,
        },
        "40x40x3.0shs": {
            0.5: 116.0,
            1.0: 95.0,
            1.5: 61.0,
            2.0: 36.0,
            2.5: 21.0,
            3.0: 17.0,
        },
        "50x50x3.0shs": {
            0.5: 152.0,
            1.0: 138.0,
            1.5: 107.0,
            2.0: 72.0,
            2.5: 49.0,
            3.0: 34.0,
        },
        "50x50x6.0shs": {
            0.5: 259.0,
            1.0: 227.0,
            1.5: 164.0,
            2.0: 103.0,
            2.5: 69.0,
            3.0: 48.0,
        },
        "65x65x3.0shs": {
            0.5: 202.0,
            1.0: 194.0,
            1.5: 176.0,
            2.0: 143.0,
            2.5: 108.0,
            3.0: 78.0,
        },
        "75x75x3.0shs": {
            0.5: 238.0,
            1.0: 231.0,
            1.5: 215.0,
            2.0: 191.0,
            2.5: 157.0,
            3.0: 119.0,
        },
    }
    # Families that NZS 4219 Tables 13/14 support under the non-specific route.
    BRACE_TABLE_FAMILIES = frozenset(
        {"angle", "flat", "shs", "threaded_rod", "wire"}
    )

    @staticmethod
    def _il_bucket(importance_level: int) -> int:
        il = _require_int("importance_level", importance_level, minimum=1)
        if il in (1, 2):
            return 12
        if il == 3:
            return 3
        if il == 4:
            return 4
        raise ValueError(f"importance_level must be 1–4; got {il}")

    @staticmethod
    def _resolve_location_key(location: str) -> str:
        key = _normalise_key(location)
        key = Section3.LOCATION_ALIASES.get(key, key)
        if key not in Section3.ZONE_FACTORS:
            raise ValueError(
                f"Location '{location}' not found in NZS 4219 Table 3. "
                "Supply zone_factor_override for Figure 2 / unlisted sites, "
                "or use an exact Table 3 location name."
            )
        return key

    @staticmethod
    def get_zone_factor_table_3(
        location: str,
        zone_factor_override: Optional[float],
        allow_nzs4219_1170_mismatch: bool,
    ) -> dict:
        """
        Return NZS 4219 Table 3 zone factor (or override) with NZS 1170.5 cross-check.

        If override is None: look up Table 3.
        Cross-check NZS_1170_5_2004.hazard_factor(location) when importable.
        If Z differs and allow_nzs4219_1170_mismatch is False: raise ValueError.
        """
        _require_bool("allow_nzs4219_1170_mismatch", allow_nzs4219_1170_mismatch)

        notes: list[str] = []
        if zone_factor_override is not None:
            z = _require_positive("zone_factor_override", zone_factor_override)
            source = "override"
            table_z = None
            try:
                loc_key = Section3._resolve_location_key(location)
                table_z = Section3.ZONE_FACTORS[loc_key]
            except ValueError:
                notes.append(
                    "override used; location not in Table 3 (Figure 2 / unlisted site)."
                )
        else:
            loc_key = Section3._resolve_location_key(location)
            z = Section3.ZONE_FACTORS[loc_key]
            source = "table_3"
            table_z = z

        nzs1170_5_z = None
        mismatch = False
        try:
            from nz_standards import NZS_1170_5_2004 as _nzs1170

            hz = _nzs1170.hazard_factor(location)
            if isinstance(hz, tuple):
                nzs1170_5_z = float(hz[0])
            else:
                nzs1170_5_z = float(hz)
        except Exception as exc:
            # Try bare module name (Services Engineering Method / site-packages)
            try:
                import NZS_1170_5_2004 as _nzs1170_bare

                hz = _nzs1170_bare.hazard_factor(location)
                if isinstance(hz, tuple):
                    nzs1170_5_z = float(hz[0])
                else:
                    nzs1170_5_z = float(hz)
            except Exception as exc2:
                notes.append(
                    "NZS_1170_5_2004.hazard_factor cross-check skipped "
                    f"(import/lookup failed: {exc2})."
                )

        if nzs1170_5_z is not None:
            # Compare the NZS 4219 Table 3 value (not override) when available
            compare_z = table_z if table_z is not None else z
            if abs(compare_z - nzs1170_5_z) > 1e-9:
                mismatch = True
                if not allow_nzs4219_1170_mismatch and zone_factor_override is None:
                    raise ValueError(
                        f"NZS 4219 Table 3 Z={compare_z} for '{location}' differs from "
                        f"NZS 1170.5 hazard_factor Z={nzs1170_5_z}. "
                        "Set allow_nzs4219_1170_mismatch=True after recording the "
                        "decision, or supply zone_factor_override explicitly "
                        "(decisions register D1: Christchurch 0.22 vs 0.30; "
                        "Akaroa 0.16 vs 0.30)."
                    )
                if not allow_nzs4219_1170_mismatch and zone_factor_override is not None:
                    # Override path is the explicit decision; mismatch noted only
                    notes.append(
                        f"NZS 4219/1170.5 Z mismatch noted "
                        f"(table/compare={compare_z}, 1170.5={nzs1170_5_z}); "
                        f"override Z={z} used."
                    )
                else:
                    notes.append(
                        f"NZS 4219/1170.5 Z mismatch allowed "
                        f"(4219={compare_z}, 1170.5={nzs1170_5_z})."
                    )

        return {
            "zone_factor": z,
            "source": source,
            "nzs1170_5_z": nzs1170_5_z,
            "mismatch": mismatch,
            "clause_reference": "NZS 4219 Table 3",
            "notes": notes,
        }

    @staticmethod
    def get_performance_factor_table_4(
        element: str,
        importance_level: int,
        category: str,
        appendix_c_component_key: Optional[str],
    ) -> dict:
        """
        Anchors/fixings/fasteners: always Cp = 0.85.
        Braces/supports: Table 4; optional Appendix C lower value when key supplied
        and valid for ULS P1–P4 rows (Table 4 footnote*).
        """
        el = _normalise_key(element)
        cat = _normalise_key(category)
        Section2.validate_category_for_importance_level_table_2(cat, importance_level)
        il = _require_int("importance_level", importance_level, minimum=1)

        if el in ("anchor", "fixing", "fastener", "anchors", "fixings", "fasteners"):
            return {
                "performance_factor": CP_ANCHOR,
                "source": "table_4",
                "element": el,
                "clause_reference": "NZS 4219 Table 4",
            }

        if el in ("component",):
            # Component Cp via Appendix C when provided; else Table 4 default 0.85
            if appendix_c_component_key is not None:
                cp = AppendixC.get_cp(appendix_c_component_key)
                return {
                    "performance_factor": cp,
                    "source": "appendix_c",
                    "element": el,
                    "appendix_c_component_key": appendix_c_component_key,
                    "clause_reference": "NZS 4219 Appendix C",
                }
            return {
                "performance_factor": CP_ANCHOR,
                "source": "table_4_default",
                "element": el,
                "clause_reference": "NZS 4219 Table 4",
            }

        if el not in ("brace", "support", "braces", "supports", "brace_support"):
            raise ValueError(
                f"element must be one of component|brace|support|anchor|fixing|fastener; "
                f"got {element!r}"
            )

        # Braces and supports
        limit_state = Section2.get_limit_state_table_2(cat)
        cp = CP_ANCHOR
        source = "table_4"
        if (
            appendix_c_component_key is not None
            and limit_state == "ULS"
            and cat in ("p1", "p2", "p3", "p4")
            and il in (1, 2, 3, 4)
        ):
            cp_app = AppendixC.get_cp(appendix_c_component_key)
            cp = min(CP_ANCHOR, cp_app)
            source = "appendix_c" if cp < CP_ANCHOR else "table_4"
        elif appendix_c_component_key is not None:
            raise ValueError(
                "Appendix C lower Cp applies only to braces/supports for "
                "P1–P4 (ULS) per NZS 4219 Table 4 footnote*; "
                f"got category={category!r}, IL={il}."
            )

        return {
            "performance_factor": cp,
            "source": source,
            "element": el,
            "importance_level": il,
            "category": cat.upper(),
            "clause_reference": "NZS 4219 Table 4",
        }

    @staticmethod
    def get_component_risk_factor_table_5(importance_level: int, category: str) -> float:
        """Exact IL/category lookup. Raise on invalid combinations. Table 5 provisional."""
        cat = _normalise_key(category)
        Section2.validate_category_for_importance_level_table_2(cat, importance_level)
        bucket = Section3._il_bucket(importance_level)
        row = Section3.TABLE_5_RC.get(cat)
        if row is None or bucket not in row:
            raise ValueError(
                f"No Table 5 Rc for category={category!r}, IL={importance_level}. "
                "P5 is IL4 only."
            )
        return float(row[bucket])

    @staticmethod
    def lateral_force_coefficient_eq_3_2(
        zone_factor: float,
        performance_factor: float,
        component_risk_factor: float,
        above_ground_floor: bool,
    ) -> dict:
        """
        C = 2.7 * C_H * Z * C_p * R_c, but not greater than 3.6.
        C_H = 3.0 above ground floor; 1.0 at or below ground floor.
        """
        z = _require_positive("zone_factor", zone_factor)
        cp = _require_positive("performance_factor", performance_factor)
        rc = _require_positive("component_risk_factor", component_risk_factor)
        above = _require_bool("above_ground_floor", above_ground_floor)
        ch = 3.0 if above else 1.0
        unclipped = COEFFICIENT_FACTOR_2_7 * ch * z * cp * rc
        c = min(unclipped, COEFFICIENT_CAP)
        return {
            "height_coefficient": ch,
            "coefficient_unclipped": unclipped,
            "coefficient": c,
            "capped": unclipped > COEFFICIENT_CAP,
            "clause_reference": "NZS 4219 Eq. 3.2",
        }

    @staticmethod
    def earthquake_load_demand_eq_3_1(coefficient: float, operating_weight_kn: float) -> dict:
        """F = C W. Raise if weight ≤ 0. NZS 4219 Eq. 3.1."""
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        f = c * w
        return {
            "force_kn": f,
            "coefficient": c,
            "operating_weight_kn": w,
            "clause_reference": "NZS 4219 Eq. 3.1",
        }

    @staticmethod
    def mass_kg_to_weight_kn(mass_kg: float) -> float:
        """W = mass_kg * GRAVITY_M_S2 / 1000. Raise if mass_kg ≤ 0."""
        m = _require_positive("mass_kg", mass_kg)
        return m * GRAVITY_M_S2 / 1000.0

    @staticmethod
    def relative_seismic_displacement_eq_3_3(
        risk_factor: float,
        height_between_fixings_mm: float,
        known_design_displacement_mm: Optional[float],
    ) -> dict:
        """
        If known displacement supplied, use it.
        Else D = 0.025 * min(Rc, 1.0) * Hz. NZS 4219 Eq. 3.3.
        """
        rc = _require_positive("risk_factor", risk_factor)
        hz = _require_positive("height_between_fixings_mm", height_between_fixings_mm)
        if known_design_displacement_mm is not None:
            d = _require_nonnegative(
                "known_design_displacement_mm", known_design_displacement_mm
            )
            return {
                "displacement_mm": d,
                "source": "known_design_displacement",
                "risk_factor_used": None,
                "clause_reference": "NZS 4219 cl. 3.5",
            }
        rc_used = min(rc, 1.0)
        d = 0.025 * rc_used * hz
        return {
            "displacement_mm": d,
            "source": "equation_3_3",
            "risk_factor_used": rc_used,
            "risk_factor_capped": rc > 1.0,
            "clause_reference": "NZS 4219 Eq. 3.3",
        }

    @staticmethod
    def brace_axial_force_eq_3_4(horizontal_force_kn: float, brace_angle_deg: float) -> dict:
        """P = F / cos(θ). θ from horizontal. Raise if θ < 0 or θ > 60. Eq. 3.4."""
        f = _require_positive("horizontal_force_kn", horizontal_force_kn)
        theta = _require_finite("brace_angle_deg", brace_angle_deg)
        if theta < 0.0 or theta > 60.0:
            raise ValueError(
                f"brace_angle_deg must be in [0, 60] (NZS 4219 Eq. 3.4); got {theta}"
            )
        p = f / np.cos(np.radians(theta))
        return {
            "brace_axial_force_kn": p,
            "horizontal_force_kn": f,
            "brace_angle_deg": theta,
            "additional_vertical_reaction_required": theta > 30.0,
            "clause_reference": "NZS 4219 Eq. 3.4",
        }

    @staticmethod
    def rigid_floor_mounted_actions_eq_3_5_3_6(
        coefficient: float,
        operating_weight_kn: float,
        total_supports: int,
        tension_supports: int,
        support_spacing_mm: float,
        cg_height_mm: float,
        cg_offset_mm: float,
    ) -> dict:
        """Eq 3.5 / 3.6. Raise if not 0.45B < b < 0.55B."""
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        n = _require_int("total_supports", total_supports, minimum=1)
        n_t = _require_int("tension_supports", tension_supports, minimum=1)
        if n_t > n:
            raise ValueError(
                f"tension_supports ({n_t}) cannot exceed total_supports ({n})"
            )
        b_spacing = _require_positive("support_spacing_mm", support_spacing_mm)
        h = _require_positive("cg_height_mm", cg_height_mm)
        b = _require_nonnegative("cg_offset_mm", cg_offset_mm)
        if not (0.45 * b_spacing < b < 0.55 * b_spacing):
            raise ValueError(
                "NZS 4219 Eq. 3.5/3.6 NOTE: requires 0.45B < b < 0.55B; "
                f"got b={b} mm, B={b_spacing} mm "
                f"(bounds {0.45 * b_spacing:.3f} … {0.55 * b_spacing:.3f})."
            )
        # Convert mm geometry consistently; B and h appear as ratio h/B so units cancel
        rh = c * w / n
        overturning = c * w * h / (n_t * b_spacing)
        gravity = w / n
        rvc = overturning + gravity
        rvt = -overturning + gravity
        return {
            "horizontal_force_per_support_kn": rh,
            "vertical_compression_kn": rvc,
            "vertical_tension_kn": rvt,
            "clause_reference": "NZS 4219 Eq. 3.5 / 3.6",
        }

    @staticmethod
    def braced_floor_mounted_actions_eq_3_7_3_8(
        coefficient: float,
        operating_weight_kn: float,
        total_supports: int,
        cg_height_mm: float,
        brace_height_mm: float,
        brace_angle_deg: float,
    ) -> dict:
        """Eq 3.7 / 3.8."""
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        n = _require_int("total_supports", total_supports, minimum=1)
        h = _require_positive("cg_height_mm", cg_height_mm)
        h_brace = _require_positive("brace_height_mm", brace_height_mm)
        theta = _require_finite("brace_angle_deg", brace_angle_deg)
        if theta < 0.0 or theta > 60.0:
            raise ValueError(
                f"brace_angle_deg must be in [0, 60] (NZS 4219 Eq. 3.7); got {theta}"
            )
        cos_t = np.cos(np.radians(theta))
        p = c * w * h / (h_brace * cos_t)
        rh = (c * w - p * cos_t) / n
        return {
            "brace_axial_force_kn": p,
            "horizontal_force_per_support_kn": rh,
            "additional_vertical_reaction_required": theta > 30.0,
            "clause_reference": "NZS 4219 Eq. 3.7 / 3.8",
        }

    @staticmethod
    def _require_resilient_ack(acknowledge_unverified_resilient_equations: bool) -> None:
        ack = _require_bool(
            "acknowledge_unverified_resilient_equations",
            acknowledge_unverified_resilient_equations,
        )
        if RESILIENT_EQUATIONS_PDF_VERIFIED:
            return
        if not ack:
            raise ValueError(
                "Resilient mount Eq. 3.9–3.12 are provisional "
                "(RESILIENT_EQUATIONS_PDF_VERIFIED=False). "
                "NZS4219_Standard_Data_Integrity_Register.md marks pages 39–43 "
                "BLOCKED until PDF verification. Pass "
                "acknowledge_unverified_resilient_equations=True to proceed with "
                "provisional formulas."
            )

    @staticmethod
    def type1_impact_factor_unused() -> None:
        """Placeholder — Type 1 has no impact factor i."""
        return None

    @staticmethod
    def type2_impact_factor(snubber_clearance_mm: float, resilient_pads_present: bool) -> float:
        """Impact factor i from clearance / pad presence (provisional)."""
        c = _require_nonnegative("snubber_clearance_mm", snubber_clearance_mm)
        pads = _require_bool("resilient_pads_present", resilient_pads_present)
        if c <= 6.0 and pads:
            return 1.0
        if c <= 6.0 and not pads:
            return 1.5
        if c > 6.0 and pads:
            return 2.0
        return 3.0

    @staticmethod
    def type1_resilient_mount_actions_eq_3_9_3_10(
        coefficient: float,
        operating_weight_kn: float,
        total_supports: int,
        tension_supports: int,
        support_spacing_mm: float,
        cg_height_mm: float,
        cg_offset_mm: float,
        acknowledge_unverified_resilient_equations: bool,
    ) -> dict:
        """
        Provisional Eq 3.9 / 3.10.
        Horizontal: Rh = CW/N.
        Vertical: ±1.3 * CWh/(nB) + W/N (1.3 on overturning term).
        """
        Section3._require_resilient_ack(acknowledge_unverified_resilient_equations)
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        n = _require_int("total_supports", total_supports, minimum=1)
        n_t = _require_int("tension_supports", tension_supports, minimum=1)
        if n_t > n:
            raise ValueError(
                f"tension_supports ({n_t}) cannot exceed total_supports ({n})"
            )
        b_spacing = _require_positive("support_spacing_mm", support_spacing_mm)
        h = _require_positive("cg_height_mm", cg_height_mm)
        b = _require_nonnegative("cg_offset_mm", cg_offset_mm)
        if not (0.45 * b_spacing < b < 0.55 * b_spacing):
            raise ValueError(
                "Type 1 resilient provisional formulas require 0.45B < b < 0.55B; "
                f"got b={b} mm, B={b_spacing} mm."
            )
        rh = c * w / n
        overturning = 1.3 * c * w * h / (n_t * b_spacing)
        gravity = w / n
        return {
            "horizontal_force_per_support_kn": rh,
            "vertical_compression_kn": overturning + gravity,
            "vertical_tension_kn": -overturning + gravity,
            "overturning_multiplier": 1.3,
            "provisional": True,
            "clause_reference": "NZS 4219 Eq. 3.9 / 3.10 (provisional)",
        }

    @staticmethod
    def type2_resilient_snubber_actions_eq_3_11_3_12(
        coefficient: float,
        operating_weight_kn: float,
        total_supports: int,
        tension_supports: int,
        support_spacing_mm: float,
        cg_height_mm: float,
        cg_offset_mm: float,
        snubber_clearance_mm: float,
        resilient_pads_present: bool,
        acknowledge_unverified_resilient_equations: bool,
    ) -> dict:
        """
        Provisional Eq 3.11 / 3.12.
        Rh = (CW/N)*i; vertical excludes W/N.
        """
        Section3._require_resilient_ack(acknowledge_unverified_resilient_equations)
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        n = _require_int("total_supports", total_supports, minimum=1)
        n_t = _require_int("tension_supports", tension_supports, minimum=1)
        if n_t > n:
            raise ValueError(
                f"tension_supports ({n_t}) cannot exceed total_supports ({n})"
            )
        b_spacing = _require_positive("support_spacing_mm", support_spacing_mm)
        h = _require_positive("cg_height_mm", cg_height_mm)
        b = _require_nonnegative("cg_offset_mm", cg_offset_mm)
        if not (0.45 * b_spacing < b < 0.55 * b_spacing):
            raise ValueError(
                "Type 2 resilient provisional formulas require 0.45B < b < 0.55B; "
                f"got b={b} mm, B={b_spacing} mm."
            )
        i = Section3.type2_impact_factor(snubber_clearance_mm, resilient_pads_present)
        rh = (c * w / n) * i
        # Vertical excludes W/N (C3.7.2.2.2)
        rr = i * c * w * h / (n_t * b_spacing)
        return {
            "horizontal_force_per_support_kn": rh,
            "vertical_uplift_kn": rr,
            "impact_factor": i,
            "weight_excluded_from_vertical": True,
            "provisional": True,
            "clause_reference": "NZS 4219 Eq. 3.11 / 3.12 (provisional)",
        }

    @staticmethod
    def independent_support_anchor_bolt_forces_cl_3_7_3_1(
        rh_kn: float,
        rr_kn: float,
        height_to_restraint_mm: float,
        snubber_width_mm: float,
        number_of_bolts: int,
    ) -> dict:
        """cl. 3.7.3.1 / Figure 9: T_bolt and V_bolt."""
        rh = _require_finite("rh_kn", rh_kn)
        rr = _require_finite("rr_kn", rr_kn)
        h = _require_positive("height_to_restraint_mm", height_to_restraint_mm)
        b_s = _require_positive("snubber_width_mm", snubber_width_mm)
        n_b = _require_int("number_of_bolts", number_of_bolts, minimum=1)
        t_bolt = (rh * h + rr * b_s) / (n_b * (b_s / 2.0))
        v_bolt = rh / n_b
        return {
            "tension_per_bolt_kn": t_bolt,
            "shear_per_bolt_kn": v_bolt,
            "clause_reference": "NZS 4219 cl. 3.7.3.1 / Figure 9",
        }

    @staticmethod
    def _pack_suspended_brace_result(
        p_total: float,
        brace_angle_deg: float,
        number_of_active_braces: int,
        tension_only_pair: bool,
        shared_hanger_orthogonal_pair: bool,
        geometry: str,
        clause_reference: str,
    ) -> dict:
        n_br = _require_int("number_of_active_braces", number_of_active_braces, minimum=1)
        t_pair = _require_bool("tension_only_pair", tension_only_pair)
        shared = _require_bool(
            "shared_hanger_orthogonal_pair", shared_hanger_orthogonal_pair
        )
        notes: list[str] = []
        if t_pair:
            notes.append(
                "Tension-only braces require opposing pairs; gravity hangers are separate."
            )
        multiplier = SHARED_HANGER_MULTIPLIER if shared else 1.0
        if shared:
            notes.append(
                f"Shared hanger orthogonal pair: applied multiplier "
                f"{SHARED_HANGER_MULTIPLIER} (Appendix D practice)."
            )
        p_scaled = p_total * multiplier
        return {
            "brace_axial_force_total_kn": p_scaled,
            "brace_axial_force_per_active_brace_kn": p_scaled / n_br,
            "number_of_active_braces": n_br,
            "shared_hanger_multiplier_applied": multiplier,
            "geometry": geometry,
            "additional_vertical_reaction_required": brace_angle_deg > 30.0,
            "notes": notes,
            "clause_reference": clause_reference,
        }

    @staticmethod
    def suspended_component_actions_eq_3_13(
        coefficient: float,
        operating_weight_kn: float,
        brace_angle_deg: float,
        number_of_active_braces: int,
        tension_only_pair: bool,
        shared_hanger_orthogonal_pair: bool,
    ) -> dict:
        """
        Concentrated mass — NZS 4219 Eq. 3.13 / figure 10 (a):
        P = C W / cos(θ).
        """
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        theta = _require_finite("brace_angle_deg", brace_angle_deg)
        if theta < 0.0 or theta > 60.0:
            raise ValueError(
                f"brace_angle_deg must be in [0, 60] (NZS 4219 cl. 3.8); got {theta}"
            )
        p_total = c * w / np.cos(np.radians(theta))
        return Section3._pack_suspended_brace_result(
            p_total,
            theta,
            number_of_active_braces,
            tension_only_pair,
            shared_hanger_orthogonal_pair,
            "concentrated",
            "NZS 4219 Eq. 3.13",
        )

    @staticmethod
    def suspended_component_actions_eq_3_14(
        coefficient: float,
        operating_weight_kn: float,
        brace_angle_deg: float,
        number_of_active_braces: int,
        tension_only_pair: bool,
        cg_to_upper_support_mm: float,
        lower_to_upper_support_mm: float,
        shared_hanger_orthogonal_pair: bool,
    ) -> dict:
        """
        Distributed mass — NZS 4219 Eq. 3.14 / figure 10 (b):
        P = h C W / (H cos(θ)).
        """
        c = _require_positive("coefficient", coefficient)
        w = _require_positive("operating_weight_kn", operating_weight_kn)
        theta = _require_finite("brace_angle_deg", brace_angle_deg)
        if theta < 0.0 or theta > 60.0:
            raise ValueError(
                f"brace_angle_deg must be in [0, 60] (NZS 4219 cl. 3.8); got {theta}"
            )
        h = _require_positive("cg_to_upper_support_mm", cg_to_upper_support_mm)
        h_span = _require_positive(
            "lower_to_upper_support_mm", lower_to_upper_support_mm
        )
        p_total = h * c * w / (h_span * np.cos(np.radians(theta)))
        return Section3._pack_suspended_brace_result(
            p_total,
            theta,
            number_of_active_braces,
            tension_only_pair,
            shared_hanger_orthogonal_pair,
            "distributed",
            "NZS 4219 Eq. 3.14",
        )

    @staticmethod
    def lateral_coefficients_for_load_path_eq_3_2(
        zone_factor: float,
        importance_level: int,
        category: str,
        above_ground_floor: bool,
        appendix_c_component_key: Optional[str],
    ) -> dict:
        """
        Return C_component, C_brace_support, C_fixing_anchor (each via Eq 3.2
        with its own Cp). Cap each at 3.6.
        """
        z = _require_positive("zone_factor", zone_factor)
        rc = Section3.get_component_risk_factor_table_5(importance_level, category)
        cp_comp = Section3.get_performance_factor_table_4(
            "component", importance_level, category, appendix_c_component_key
        )["performance_factor"]
        cp_brace = Section3.get_performance_factor_table_4(
            "brace", importance_level, category, appendix_c_component_key
        )["performance_factor"]
        cp_anchor = Section3.get_performance_factor_table_4(
            "anchor", importance_level, category, None
        )["performance_factor"]
        c_comp = Section3.lateral_force_coefficient_eq_3_2(
            z, cp_comp, rc, above_ground_floor
        )
        c_brace = Section3.lateral_force_coefficient_eq_3_2(
            z, cp_brace, rc, above_ground_floor
        )
        c_anc = Section3.lateral_force_coefficient_eq_3_2(
            z, cp_anchor, rc, above_ground_floor
        )
        return {
            "component_risk_factor": rc,
            "cp_component": cp_comp,
            "cp_brace_support": cp_brace,
            "cp_fixing_anchor": cp_anchor,
            "c_component": c_comp["coefficient"],
            "c_brace_support": c_brace["coefficient"],
            "c_fixing_anchor": c_anc["coefficient"],
            "details": {
                "component": c_comp,
                "brace_support": c_brace,
                "fixing_anchor": c_anc,
            },
            "clause_reference": "NZS 4219 Eq. 3.2 / Table 4",
        }

    @staticmethod
    def scale_anchor_demand_from_brace_actions(
        brace_horizontal_kn: float,
        brace_axial_kn: float,
        cp_brace: float,
        cp_anchor: float,
    ) -> dict:
        """
        Appendix D pattern: scale demands by cp_anchor/cp_brace.
        """
        fh = _require_finite("brace_horizontal_kn", brace_horizontal_kn)
        fa = _require_finite("brace_axial_kn", brace_axial_kn)
        cpb = _require_positive("cp_brace", cp_brace)
        cpa = _require_positive("cp_anchor", cp_anchor)
        ratio = cpa / cpb
        return {
            "scale_ratio": ratio,
            "anchor_horizontal_kn": fh * ratio,
            "anchor_axial_kn": fa * ratio,
            "cp_brace": cpb,
            "cp_anchor": cpa,
            "clause_reference": "NZS 4219 Appendix D (Cp scaling)",
        }

    @staticmethod
    def _pipe_table(material: str, direction: str) -> dict:
        mat = _normalise_key(material)
        direction_key = _normalise_key(direction)
        if mat == "steel" and direction_key == "transverse":
            return Section3.TABLE_6_STEEL_TRANSVERSE
        if mat == "copper" and direction_key == "transverse":
            return Section3.TABLE_6_COPPER_TRANSVERSE
        if mat == "steel" and direction_key == "longitudinal":
            return Section3.TABLE_7_STEEL_LONGITUDINAL
        if mat == "copper" and direction_key == "longitudinal":
            return Section3.TABLE_7_COPPER_LONGITUDINAL
        raise ValueError(
            f"Unsupported material/direction: {material!r} / {direction!r}. "
            "material in ('steel','copper'); direction in ('transverse','longitudinal')."
        )

    @staticmethod
    def _select_c_column(
        coefficient: float, interpolation_policy: str
    ) -> tuple[float, str, Optional[tuple[float, float, float]]]:
        """
        Return (c_used, policy_note, interp_triplet_or_None).
        interp_triplet = (c_lo, c_hi, t) for linear interpolation.
        """
        c = _require_positive("coefficient", coefficient)
        policy = _normalise_key(interpolation_policy)
        valid = (
            "conservative_next_higher",
            "linear_interpolation",
            "exact_only",
        )
        if policy not in valid:
            raise ValueError(
                f"interpolation_policy must be one of {valid}; got {interpolation_policy!r}"
            )
        cols = Section3.PIPE_C_COLUMNS
        for col in cols:
            if abs(c - col) < 1e-9:
                return col, "exact_column", None
        if policy == "exact_only":
            raise ValueError(
                f"coefficient {c} is not an exact Table 6/7 column "
                f"{list(cols)}; exact_only policy rejects interpolation."
            )
        if c < cols[0] or c > cols[-1]:
            raise ValueError(
                f"coefficient {c} outside Table 6/7 C range "
                f"[{cols[0]}, {cols[-1]}]."
            )
        # find bounding columns
        lo = cols[0]
        hi = cols[-1]
        for i in range(len(cols) - 1):
            if cols[i] < c < cols[i + 1]:
                lo, hi = cols[i], cols[i + 1]
                break
        if policy == "conservative_next_higher":
            return hi, "conservative_next_higher", None
        t = (c - lo) / (hi - lo)
        return c, "linear_interpolation", (lo, hi, t)

    @staticmethod
    def get_pipe_restraint_table_6_7(
        material: str,
        nominal_diameter_mm: float,
        coefficient: float,
        direction: str,
        interpolation_policy: str,
    ) -> dict:
        """Return max_spacing_m, table_force_kn, max_offset_m (longitudinal), table_basis."""
        diam = _require_positive("nominal_diameter_mm", nominal_diameter_mm)
        table = Section3._pipe_table(material, direction)
        # exact diameter rows only for now; diameter interpolation via policy on C
        diam_key = None
        for d in table:
            if abs(float(d) - diam) < 1e-9:
                diam_key = d
                break
        if diam_key is None:
            if _normalise_key(interpolation_policy) == "exact_only":
                raise ValueError(
                    f"nominal_diameter_mm={diam} not an exact table row for "
                    f"{material}/{direction}. Rows: {sorted(table.keys())}."
                )
            # conservative: next higher diameter
            larger = sorted(d for d in table if float(d) > diam)
            if not larger:
                raise ValueError(
                    f"nominal_diameter_mm={diam} exceeds table range for "
                    f"{material}/{direction}."
                )
            diam_key = larger[0]
        row = table[diam_key]
        c_used, policy_note, interp = Section3._select_c_column(
            coefficient, interpolation_policy
        )
        direction_key = _normalise_key(direction)
        if interp is None:
            spacing, force = row[c_used]
            c_report = c_used
        else:
            lo, hi, t = interp
            s_lo, f_lo = row[lo]
            s_hi, f_hi = row[hi]
            # conservative spacing = lower of interpolated; force interpolate
            spacing = s_lo + t * (s_hi - s_lo)
            force = f_lo + t * (f_hi - f_lo)
            c_report = c_used

        max_offset = None
        if direction_key == "longitudinal":
            if interp is None:
                max_offset = Section3.TABLE_7_MAX_OFFSET_M[c_used]
            else:
                lo, hi, t = interp
                max_offset = Section3.TABLE_7_MAX_OFFSET_M[lo] + t * (
                    Section3.TABLE_7_MAX_OFFSET_M[hi] - Section3.TABLE_7_MAX_OFFSET_M[lo]
                )

        return {
            "nominal_diameter_mm": float(diam_key),
            "min_wall_mm": row["min_wall_mm"],
            "coefficient_column": c_report,
            "max_spacing_m": float(spacing),
            "table_force_kn": float(force),
            "max_offset_m": max_offset,
            "direction": direction_key,
            "material": _normalise_key(material),
            "interpolation_policy": policy_note,
            "table_basis": "PUBLISHED_VALUE_PENDING_PDF",
            "clause_reference": (
                "NZS 4219 Table 6" if direction_key == "transverse" else "NZS 4219 Table 7"
            ),
        }

    @staticmethod
    def scale_force_for_actual_spacing(
        table_force_kn: float,
        actual_spacing_m: float,
        table_max_spacing_m: float,
    ) -> float:
        """Proportional reduction when actual < max. Raise if actual > max."""
        f = _require_positive("table_force_kn", table_force_kn)
        actual = _require_positive("actual_spacing_m", actual_spacing_m)
        max_s = _require_positive("table_max_spacing_m", table_max_spacing_m)
        if actual > max_s + 1e-12:
            raise ValueError(
                f"actual_spacing_m ({actual}) exceeds table_max_spacing_m ({max_s})."
            )
        return f * (actual / max_s)

    @staticmethod
    def linear_generic_actions_cl_3_6(
        line_weight_kn_per_m: float,
        actual_spacing_m: float,
        coefficient: float,
        brace_angle_deg: float,
        direction: str,
    ) -> dict:
        """
        Tributary F = C * w * spacing; brace via Eq 3.4.
        For ducts/cable trays: set specific_service_span_check_required=True.
        """
        w = _require_positive("line_weight_kn_per_m", line_weight_kn_per_m)
        s = _require_positive("actual_spacing_m", actual_spacing_m)
        c = _require_positive("coefficient", coefficient)
        direction_key = _normalise_key(direction)
        if direction_key not in ("transverse", "longitudinal"):
            raise ValueError(
                f"direction must be 'transverse' or 'longitudinal'; got {direction!r}"
            )
        f = c * w * s
        brace = Section3.brace_axial_force_eq_3_4(f, brace_angle_deg)
        return {
            "tributary_force_kn": f,
            "brace_axial_force_kn": brace["brace_axial_force_kn"],
            "direction": direction_key,
            "specific_service_span_check_required": True,
            "additional_vertical_reaction_required": brace[
                "additional_vertical_reaction_required"
            ],
            "clause_reference": "NZS 4219 cl. 3.6 / Eq. 3.4",
        }

    @staticmethod
    def get_woodscrew_capacities_table_8(gauge: int) -> dict:
        g = _require_int("gauge", gauge, minimum=1)
        if g not in Section3.TABLE_8_WOODSCREWS:
            raise ValueError(
                f"Gauge {g} not in NZS 4219 Table 8. Valid: "
                f"{sorted(Section3.TABLE_8_WOODSCREWS)}"
            )
        out = dict(Section3.TABLE_8_WOODSCREWS[g])
        out["gauge"] = g
        out["clause_reference"] = "NZS 4219 Table 8"
        return out

    @staticmethod
    def get_coach_screw_capacities_table_9(diameter_mm: float) -> dict:
        d = _require_positive("diameter_mm", diameter_mm)
        key = None
        for k in Section3.TABLE_9_COACH_SCREWS:
            if abs(k - d) < 1e-9:
                key = k
                break
        if key is None:
            raise ValueError(
                f"diameter_mm={d} not in NZS 4219 Table 9. Valid: "
                f"{sorted(Section3.TABLE_9_COACH_SCREWS)}"
            )
        out = dict(Section3.TABLE_9_COACH_SCREWS[key])
        out["diameter_mm"] = key
        out["clause_reference"] = "NZS 4219 Table 9"
        return out

    @staticmethod
    def get_bolt_shear_capacity_table_10(bolt_size: str) -> dict:
        key = _normalise_key(bolt_size).replace(" ", "")
        if key not in Section3.TABLE_10_BOLTS_IN_SHEAR:
            raise ValueError(
                f"bolt_size={bolt_size!r} not in NZS 4219 Table 10. Valid: "
                f"{sorted(Section3.TABLE_10_BOLTS_IN_SHEAR)}"
            )
        out = dict(Section3.TABLE_10_BOLTS_IN_SHEAR[key])
        out["bolt_size"] = key.upper()
        out["clause_reference"] = "NZS 4219 Table 10"
        return out

    @staticmethod
    def get_masonry_bolt_capacity_table_11(diameter_mm: float) -> dict:
        d = _require_positive("diameter_mm", diameter_mm)
        key = None
        for k in Section3.TABLE_11_MASONRY_BOLTS:
            if abs(k - d) < 1e-9:
                key = k
                break
        if key is None:
            raise ValueError(
                f"diameter_mm={d} not in NZS 4219 Table 11. Valid: "
                f"{sorted(Section3.TABLE_11_MASONRY_BOLTS)}"
            )
        out = dict(Section3.TABLE_11_MASONRY_BOLTS[key])
        out["diameter_mm"] = key
        out["clause_reference"] = "NZS 4219 Table 11"
        return out

    @staticmethod
    def _normalise_brace_section_key(section_size: str) -> str:
        if not isinstance(section_size, str):
            raise ValueError(
                f"section_size must be a string; got {type(section_size).__name__}"
            )
        key = (
            section_size.strip()
            .lower()
            .replace(" ", "")
            .replace("×", "x")
            .replace("*", "x")
        )
        return key

    @staticmethod
    def _normalise_connection_type(connection_type: str) -> str:
        key = _normalise_key(connection_type).replace(" ", "_").replace("-", "_")
        aliases = {
            "1_bolt": "1_bolt",
            "one_bolt": "1_bolt",
            "single_bolt": "1_bolt",
            "2_bolts": "2_bolts",
            "two_bolts": "2_bolts",
            "welded": "welded",
            "weld": "welded",
        }
        if key not in aliases:
            raise ValueError(
                f"connection_type must be one of "
                f"{{'1_bolt','2_bolts','welded'}}; got {connection_type!r}"
            )
        return aliases[key]

    @staticmethod
    def _normalise_brace_family(brace_family: str) -> str:
        key = _normalise_key(brace_family).replace(" ", "_").replace("-", "_")
        aliases = {
            "angle": "angle",
            "angles": "angle",
            "ea": "angle",
            "flat": "flat",
            "flats": "flat",
            "shs": "shs",
            "hollow_section": "shs",
            "rhs": "shs",
            "threaded_rod": "threaded_rod",
            "threaded_rods": "threaded_rod",
            "rod": "threaded_rod",
            "wire": "wire",
            "galvanised_steel_wire": "wire",
            "galvanized_steel_wire": "wire",
            "specific_design": "specific_design",
            "channel": "specific_design",
            "strut": "specific_design",
            "strut_channel": "specific_design",
        }
        if key not in aliases:
            raise ValueError(
                f"brace_family must be one of "
                f"{{'angle','flat','shs','threaded_rod','wire','specific_design'}}; "
                f"got {brace_family!r}"
            )
        return aliases[key]

    @staticmethod
    def get_brace_tension_capacity_table_13(
        brace_family: str,
        section_size: str,
        connection_type: str,
        bolt_size: Optional[str] = None,
    ) -> dict:
        """
        NZS 4219 Table 13 — axial tension capacity of braces (kN).

        Status: TRANSCRIBED from OCR markdown; pending second-person PDF check.
        Unsupported proprietary profiles (e.g. strut channel) must use
        specific_design / Section 4 — this function raises for those families.
        """
        family = Section3._normalise_brace_family(brace_family)
        if family == "specific_design":
            raise ValueError(
                "SPECIFIC_DESIGN_REQUIRED_SECTION_4: brace_family='specific_design' "
                "is outside NZS 4219 Tables 13/14. Supply verified tension capacity "
                "via specific design (NZS 3404 / manufacturer data) or select a "
                "listed angle/flat/SHS/threaded_rod/wire."
            )
        if family not in Section3.BRACE_TABLE_FAMILIES:
            raise ValueError(
                f"brace_family {brace_family!r} is not covered by NZS 4219 Table 13."
            )

        section_key = Section3._normalise_brace_section_key(section_size)
        status = Section3.BRACE_TABLE_TRANSCRIPTION_STATUS

        if family == "threaded_rod":
            rod_key = section_key if section_key.startswith("m") else f"m{section_key}"
            if bolt_size is not None:
                rod_key = _normalise_key(bolt_size).replace(" ", "")
            if rod_key not in Section3.TABLE_13_THREADED_ROD_TENSION:
                raise ValueError(
                    f"Threaded rod {section_size!r} not in NZS 4219 Table 13. "
                    f"Valid: {sorted(Section3.TABLE_13_THREADED_ROD_TENSION)}"
                )
            return {
                "brace_family": family,
                "section_size": rod_key.upper(),
                "connection_type": None,
                "tension_capacity_kn": float(
                    Section3.TABLE_13_THREADED_ROD_TENSION[rod_key]
                ),
                "clause_reference": "NZS 4219 Table 13",
                "transcription_status": status,
            }

        if family == "wire":
            wire_key = section_key if "mm" in section_key else f"{section_key}mm"
            if wire_key not in Section3.TABLE_13_WIRE_TENSION:
                raise ValueError(
                    f"Wire size {section_size!r} not in NZS 4219 Table 13. "
                    f"Valid: {sorted(Section3.TABLE_13_WIRE_TENSION)}"
                )
            return {
                "brace_family": family,
                "section_size": wire_key,
                "connection_type": None,
                "tension_capacity_kn": float(Section3.TABLE_13_WIRE_TENSION[wire_key]),
                "clause_reference": "NZS 4219 Table 13",
                "transcription_status": status,
            }

        conn = Section3._normalise_connection_type(connection_type)
        table = (
            Section3.TABLE_13_ANGLE_TENSION
            if family == "angle"
            else Section3.TABLE_13_FLAT_TENSION
        )
        if family == "shs":
            raise ValueError(
                "NZS 4219 Table 13 does not list SHS tension capacities; "
                "SHS braces are covered for compression in Table 14 with "
                "fully welded end fixings. Use specific design for SHS tension "
                "or select an angle/flat/threaded_rod from Table 13."
            )
        if section_key not in table:
            raise ValueError(
                f"section_size {section_size!r} not in NZS 4219 Table 13 "
                f"({family}). Valid: {sorted(table)}"
            )
        if bolt_size is None:
            raise ValueError(
                "bolt_size is required for angle/flat tension lookups in Table 13"
            )
        bolt_key = _normalise_key(bolt_size).replace(" ", "")
        row = table[section_key]
        if bolt_key not in row:
            raise ValueError(
                f"bolt_size {bolt_size!r} not listed for {section_size} in "
                f"Table 13. Valid: {sorted(row)}"
            )
        entry = row[bolt_key]
        if conn not in entry:
            raise ValueError(
                f"connection_type {connection_type!r} missing for "
                f"{section_size}/{bolt_size} in Table 13"
            )
        return {
            "brace_family": family,
            "section_size": section_key,
            "bolt_size": bolt_key.upper(),
            "connection_type": conn,
            "tension_capacity_kn": float(entry[conn]),
            "fillet_weld_size_mm": entry.get("fillet_weld_size_mm"),
            "fillet_weld_length_mm": entry.get("fillet_weld_length_mm"),
            "clause_reference": "NZS 4219 Table 13",
            "transcription_status": status,
        }

    @staticmethod
    def get_brace_compression_capacity_table_14(
        brace_family: str,
        section_size: str,
        brace_length_m: float,
        interpolation_policy: str,
        bolt_size: Optional[str] = None,
    ) -> dict:
        """
        NZS 4219 Table 14 — axial compression capacity of braces (kN).

        interpolation_policy:
          - 'exact_only': length must match a tabulated column
          - 'linear_interpolation': interpolate between columns (Appendix D practice)
          - 'conservative_next_higher': use capacity at next-higher tabulated length

        Status: TRANSCRIBED from OCR markdown; pending second-person PDF check.
        """
        family = Section3._normalise_brace_family(brace_family)
        if family == "specific_design":
            raise ValueError(
                "SPECIFIC_DESIGN_REQUIRED_SECTION_4: brace_family='specific_design' "
                "is outside NZS 4219 Tables 13/14."
            )
        if family in ("threaded_rod", "wire"):
            raise ValueError(
                f"brace_family {family!r} has no NZS 4219 Table 14 compression "
                "capacities (tension-only members)."
            )
        policy = _normalise_key(interpolation_policy).replace(" ", "_")
        allowed_policies = (
            "exact_only",
            "linear_interpolation",
            "conservative_next_higher",
        )
        if policy not in allowed_policies:
            raise ValueError(
                f"interpolation_policy must be one of {allowed_policies}; "
                f"got {interpolation_policy!r}"
            )

        section_key = Section3._normalise_brace_section_key(section_size)
        status = Section3.BRACE_TABLE_TRANSCRIPTION_STATUS
        L = _require_positive("brace_length_m", brace_length_m)

        if family == "angle":
            if section_key not in Section3.TABLE_14_ANGLE_COMPRESSION:
                raise ValueError(
                    f"section_size {section_size!r} not in NZS 4219 Table 14 "
                    f"(angles). Valid: {sorted(Section3.TABLE_14_ANGLE_COMPRESSION)}"
                )
            row = Section3.TABLE_14_ANGLE_COMPRESSION[section_key]
            if bolt_size is not None:
                expected = row["bolt_size"]
                given = _normalise_key(bolt_size).replace(" ", "")
                if given != expected:
                    raise ValueError(
                        f"bolt_size {bolt_size!r} does not match Table 14 row "
                        f"for {section_size} (expected {expected.upper()})"
                    )
            capacities = row["capacities"]
            result_core = Section3._resolve_table14_cell(
                capacities, L, policy, is_angle=True
            )
            return {
                "brace_family": family,
                "section_size": section_key,
                "bolt_size": row["bolt_size"].upper(),
                "brace_length_m": L,
                "clause_reference": "NZS 4219 Table 14(a)",
                "transcription_status": status,
                "shs_requires_welded_ends": False,
                **result_core,
            }

        if family == "flat":
            if section_key not in Section3.TABLE_14_FLAT_COMPRESSION:
                raise ValueError(
                    f"section_size {section_size!r} not in NZS 4219 Table 14 "
                    f"(flats). Valid: {sorted(Section3.TABLE_14_FLAT_COMPRESSION)}"
                )
            capacities = Section3.TABLE_14_FLAT_COMPRESSION[section_key]
            result_core = Section3._resolve_table14_cell(
                capacities, L, policy, is_angle=False
            )
            return {
                "brace_family": family,
                "section_size": section_key,
                "bolt_size": None,
                "brace_length_m": L,
                "clause_reference": "NZS 4219 Table 14(b)",
                "transcription_status": status,
                "shs_requires_welded_ends": False,
                "requires_welded_base_plate": False,
                "bolts_required": None,
                **result_core,
            }

        # SHS
        shs_key = section_key
        if not shs_key.endswith("shs"):
            shs_key = f"{shs_key}shs"
        # Normalise 25x25x3 -> 25x25x3.0shs
        if shs_key not in Section3.TABLE_14_SHS_COMPRESSION:
            # try inserting .0 before shs when wall thickness is integer-like
            alt = section_key
            if alt.count("x") == 2 and "shs" not in alt:
                parts = alt.split("x")
                if len(parts) == 3 and "." not in parts[2]:
                    alt = f"{parts[0]}x{parts[1]}x{parts[2]}.0shs"
                else:
                    alt = f"{alt}shs"
            else:
                alt = shs_key
            if alt not in Section3.TABLE_14_SHS_COMPRESSION:
                raise ValueError(
                    f"section_size {section_size!r} not in NZS 4219 Table 14 "
                    f"(SHS). Valid: {sorted(Section3.TABLE_14_SHS_COMPRESSION)}"
                )
            shs_key = alt
        capacities = Section3.TABLE_14_SHS_COMPRESSION[shs_key]
        result_core = Section3._resolve_table14_cell(
            capacities, L, policy, is_angle=False
        )
        return {
            "brace_family": family,
            "section_size": shs_key,
            "bolt_size": None,
            "brace_length_m": L,
            "clause_reference": "NZS 4219 Table 14(c)",
            "transcription_status": status,
            "shs_requires_welded_ends": True,
            "requires_welded_base_plate": False,
            "bolts_required": None,
            **result_core,
        }

    @staticmethod
    def _resolve_table14_cell(
        capacities: dict,
        length_m: float,
        policy: str,
        *,
        is_angle: bool,
    ) -> dict:
        L = length_m
        lengths = list(Section3.TABLE_14_LENGTHS_M)

        def _unpack(cell, length_used, interpolation, bounds=None):
            if cell is None:
                raise ValueError(
                    f"No Table 14 compression capacity at length {length_used} m "
                    "(tabulated as not applicable)."
                )
            out = {
                "length_m_used": length_used,
                "interpolation": interpolation,
            }
            if bounds is not None:
                out["bounding_lengths_m"] = bounds
            if is_angle:
                cap, fix = cell
                out["compression_capacity_kn"] = float(cap)
                out["requires_welded_base_plate"] = fix == "welded_base"
                out["bolts_required"] = (
                    None if fix == "welded_base" else int(fix)
                )
            else:
                out["compression_capacity_kn"] = float(cell)
            return out

        if policy == "exact_only":
            if L not in capacities:
                raise ValueError(
                    f"brace_length_m={L} is not an exact Table 14 column "
                    f"{list(lengths)}. Use linear_interpolation or "
                    "conservative_next_higher, or SPECIFIC_DESIGN_REQUIRED_SECTION_4."
                )
            return _unpack(capacities[L], L, "exact")

        if policy == "conservative_next_higher":
            candidates = [x for x in lengths if x >= L and capacities.get(x) is not None]
            if not candidates:
                raise ValueError(
                    f"No Table 14 capacity at or above length {L} m. "
                    "SPECIFIC_DESIGN_REQUIRED_SECTION_4."
                )
            use_L = candidates[0]
            return _unpack(capacities[use_L], use_L, "conservative_next_higher")

        # linear_interpolation
        if L < lengths[0] or L > lengths[-1]:
            raise ValueError(
                f"brace_length_m={L} is outside NZS 4219 Table 14 range "
                f"[{lengths[0]}, {lengths[-1]}] m. "
                "SPECIFIC_DESIGN_REQUIRED_SECTION_4."
            )
        if L in capacities and capacities[L] is not None:
            return _unpack(capacities[L], L, "exact")

        lo = None
        hi = None
        for tab_L in lengths:
            if tab_L <= L and capacities.get(tab_L) is not None:
                lo = tab_L
            if tab_L >= L and capacities.get(tab_L) is not None and hi is None:
                hi = tab_L
        if lo is None or hi is None:
            raise ValueError(
                f"Cannot interpolate Table 14 at length {L} m: missing "
                "tabulated capacities on one or both sides (no extrapolation)."
            )
        if lo == hi:
            return _unpack(capacities[lo], lo, "exact")
        t = (L - lo) / (hi - lo)
        if is_angle:
            cap_lo, fix_lo = capacities[lo]
            cap_hi, fix_hi = capacities[hi]
            cap = float(cap_lo) + t * (float(cap_hi) - float(cap_lo))
            welded = fix_lo == "welded_base" or fix_hi == "welded_base"
            bolts = None
            if not welded:
                bolts = max(int(fix_lo), int(fix_hi))
            return {
                "compression_capacity_kn": cap,
                "length_m_used": L,
                "interpolation": "linear",
                "bounding_lengths_m": (lo, hi),
                "bolts_required": bolts,
                "requires_welded_base_plate": welded,
            }
        cap = float(capacities[lo]) + t * (
            float(capacities[hi]) - float(capacities[lo])
        )
        return {
            "compression_capacity_kn": cap,
            "length_m_used": L,
            "interpolation": "linear",
            "bounding_lengths_m": (lo, hi),
        }


# ------------------------------------------------------------------
# Section 5 — Numeric triggers
# ------------------------------------------------------------------

class Section5:
    """Numeric triggers and clearances (NZS 4219 Section 5)."""

    TABLE_15_CLEARANCES = {
        "unrestrained_to_unrestrained": {"horizontal_mm": 250, "vertical_mm": 50},
        "unrestrained_to_restrained": {"horizontal_mm": 150, "vertical_mm": 50},
        "restrained_to_restrained": {"horizontal_mm": 50, "vertical_mm": 50},
        "penetration_through_structure": {"horizontal_mm": 50, "vertical_mm": 50},
    }

    @staticmethod
    def pipe_restraint_applicability_cl_5_8_1(
        nominal_diameter_mm: float, hanger_length_mm: float
    ) -> dict:
        """
        < 50 mm OR hanger ≤ 150 mm → restraint not required (cl. 5.8.1);
        then clearance 150 mm applies. > 200 mm → Section 4.
        """
        d = _require_positive("nominal_diameter_mm", nominal_diameter_mm)
        h = _require_nonnegative("hanger_length_mm", hanger_length_mm)
        if d > 200.0:
            return {
                "restraint_required": True,
                "status": "SPECIFIC_DESIGN_REQUIRED_SECTION_4",
                "clause_reference": "NZS 4219 cl. 5.8.1",
                "basis": "NZS4219_MANDATORY",
                "message": "Pipes > 200 mm diameter require Section 4 specific design.",
                "clearance_mm": None,
            }
        if d < 50.0 or h <= 150.0:
            return {
                "restraint_required": False,
                "status": "NOT_APPLICABLE",
                "clause_reference": "NZS 4219 cl. 5.8.1",
                "basis": "NZS4219_MANDATORY",
                "message": (
                    "Restraint not required; install with 150 mm clearance from "
                    "hangers/braces for suspended ceilings or adjacent components."
                ),
                "clearance_mm": 150.0,
            }
        return {
            "restraint_required": True,
            "status": "PASS",
            "clause_reference": "NZS 4219 cl. 5.8.1",
            "basis": "NZS4219_MANDATORY",
            "message": "Seismic restraint required under non-specific design.",
            "clearance_mm": None,
        }

    @staticmethod
    def min_restraints_per_straight_run(
        transverse_count: int, longitudinal_count: int
    ) -> dict:
        """cl. 5.8.4.1: ≥ 2 transverse and ≥ 1 longitudinal."""
        t = _require_int("transverse_count", transverse_count, minimum=0)
        lng = _require_int("longitudinal_count", longitudinal_count, minimum=0)
        ok = t >= 2 and lng >= 1
        return {
            "transverse_count": t,
            "longitudinal_count": lng,
            "minimum_transverse": 2,
            "minimum_longitudinal": 1,
            "status": "PASS" if ok else "FAIL",
            "clause_reference": "NZS 4219 cl. 5.8.4.1",
            "basis": "NZS4219_MANDATORY",
            "message": (
                "Meets minimum restraint counts."
                if ok
                else "Requires ≥ 2 transverse and ≥ 1 longitudinal restraints."
            ),
        }

    @staticmethod
    def seismic_gap_allowance_mm(separation_height_m: float) -> float:
        """160 mm per 4 m of height (cl. 5.8.3)."""
        h = _require_positive("separation_height_m", separation_height_m)
        return 160.0 * (h / 4.0)

    @staticmethod
    def duct_restraint_applicability(
        duct_type: str,
        hanger_length_mm: float,
        flexible_length_m: float,
        inline_equipment_mass_kg: float,
    ) -> dict:
        """cl. 5.9 triggers."""
        dtype = _normalise_key(duct_type)
        if dtype not in ("rigid", "flexible"):
            raise ValueError(f"duct_type must be 'rigid' or 'flexible'; got {duct_type!r}")
        h = _require_nonnegative("hanger_length_mm", hanger_length_mm)
        fl = _require_nonnegative("flexible_length_m", flexible_length_m)
        m = _require_nonnegative("inline_equipment_mass_kg", inline_equipment_mass_kg)
        messages: list[str] = []
        restraint_required = False
        if dtype == "rigid" and h > 200.0:
            restraint_required = True
            messages.append("Rigid duct hanger length > 200 mm → restrain (cl. 5.9).")
        if dtype == "flexible" and fl > 1.5:
            restraint_required = True
            messages.append("Flexible duct > 1.5 m → restrain (cl. 5.9).")
        independent_inline = m > 10.0
        if independent_inline:
            messages.append(
                "Inline equipment > 10 kg must be supported/braced independently "
                "of the duct (cl. 5.9 / 5.15)."
            )
        return {
            "restraint_required": restraint_required,
            "independent_inline_equipment_required": independent_inline,
            "status": "PASS" if restraint_required or dtype == "flexible" else "REVIEW",
            "clause_reference": "NZS 4219 cl. 5.9",
            "basis": "NZS4219_MANDATORY",
            "message": " ".join(messages) if messages else "No cl. 5.9 restraint trigger.",
            "clearance_mm": None if restraint_required else 150.0,
        }

    @staticmethod
    def cable_tray_restraint_applicability(distance_below_support_mm: float) -> dict:
        """cl. 5.11: > 400 mm below support → restrained."""
        d = _require_nonnegative("distance_below_support_mm", distance_below_support_mm)
        required = d > 400.0
        return {
            "restraint_required": required,
            "status": "PASS" if required else "NOT_APPLICABLE",
            "clause_reference": "NZS 4219 cl. 5.11",
            "basis": "NZS4219_MANDATORY",
            "message": (
                "Cable tray > 400 mm below support requires restraint."
                if required
                else "Restraint not required; provide 150 mm clearance if unrestrained."
            ),
            "clearance_mm": None if required else 150.0,
        }

    @staticmethod
    def ceiling_void_equipment_gate(mass_kg: float) -> dict:
        """> 10 kg independent; > 25 kg → Section 4 (cl. 5.13)."""
        m = _require_positive("mass_kg", mass_kg)
        if m > 25.0:
            return {
                "independent_fixing_required": True,
                "status": "SPECIFIC_DESIGN_REQUIRED_SECTION_4",
                "clause_reference": "NZS 4219 cl. 5.13",
                "basis": "NZS4219_MANDATORY",
                "message": "Individual components > 25 kg require Section 4 design.",
            }
        if m > 10.0:
            return {
                "independent_fixing_required": True,
                "status": "PASS",
                "clause_reference": "NZS 4219 cl. 5.13",
                "basis": "NZS4219_MANDATORY",
                "message": (
                    "Equipment > 10 kg in ceiling void shall be independently fixed "
                    "to the structure; 25 mm clearance to ceiling."
                ),
                "clearance_mm": 25.0,
            }
        return {
            "independent_fixing_required": False,
            "status": "REVIEW",
            "clause_reference": "NZS 4219 cl. 5.13",
            "basis": "NZS4219_MANDATORY",
            "message": (
                "≤ 10 kg may be fixed to ceiling suspension system (not panels/tiles) "
                "with positive fixings."
            ),
        }

    @staticmethod
    def expansion_anchor_permitted(
        vibration_isolated: bool, mechanical_rating_kw: float
    ) -> dict:
        """cl. 3.10.5: not for non-isolated mechanical > 8 kW."""
        vi = _require_bool("vibration_isolated", vibration_isolated)
        kw = _require_nonnegative("mechanical_rating_kw", mechanical_rating_kw)
        permitted = True
        message = "Expansion anchors permitted subject to ACI 355.2 seismic qualification."
        if (not vi) and kw > 8.0:
            permitted = False
            message = (
                "NZS 4219 cl. 3.10.5: expansion anchors shall not be used for "
                "non-vibration-isolated mechanical equipment rated over 8 kW."
            )
        return {
            "permitted": permitted,
            "status": "PASS" if permitted else "FAIL",
            "clause_reference": "NZS 4219 cl. 3.10.5",
            "basis": "NZS4219_MANDATORY",
            "message": message,
        }

    @staticmethod
    def get_clearance_mm(adjacency: str) -> dict:
        """Table 15 lookup; REVIEW if geometry insufficient."""
        key = _normalise_key(adjacency).replace(" ", "_").replace("-", "_")
        aliases = {
            "unrestrained_component_to_unrestrained_component": "unrestrained_to_unrestrained",
            "unrestrained_component_to_restrained_component": "unrestrained_to_restrained",
            "restrained_component_to_restrained_component": "restrained_to_restrained",
            "penetration": "penetration_through_structure",
            "penetration_through_walls_and_floor": "penetration_through_structure",
        }
        key = aliases.get(key, key)
        if key not in Section5.TABLE_15_CLEARANCES:
            raise ValueError(
                f"Unknown adjacency '{adjacency}'. Valid keys: "
                f"{sorted(Section5.TABLE_15_CLEARANCES)}"
            )
        vals = Section5.TABLE_15_CLEARANCES[key]
        return {
            "adjacency": key,
            "horizontal_mm": vals["horizontal_mm"],
            "vertical_mm": vals["vertical_mm"],
            "status": "REVIEW",
            "clause_reference": "NZS 4219 Table 15",
            "basis": "NZS4219_MANDATORY",
            "message": (
                "Apply Table 15 clearances; where geometry is insufficient, "
                "specific assessment is required (REVIEW)."
            ),
        }


# ------------------------------------------------------------------
# Appendix B — Classifications
# ------------------------------------------------------------------

class AppendixB:
    """Component classifications (normative Appendix B)."""

    CLASSIFICATIONS = {
        "air conditioning systems (distributed)": {
            "categories": ["P7"],
            "comment": "—",
        },
        "air conditioning systems (self-contained)": {
            "categories": ["P7"],
            "comment": (
                "If able to fall > 3 m category is P3. If the unit is also over a "
                "publicly accessible open space, category is P1"
            ),
        },
        "boiler": {"categories": ["P3"], "comment": "See 5.10"},
        "building maintenance unit": {"categories": ["P1"], "comment": "—"},
        "communication equipment (phone, data, security, control systems)": {
            "categories": ["P7"],
            "comment": "—",
        },
        "computer equipment": {"categories": ["P7"], "comment": "—"},
        "electrical distribution": {"categories": ["P7"], "comment": "—"},
        "electrical supply": {"categories": ["P7"], "comment": "See 5.11"},
        "emergency lighting": {"categories": ["P4"], "comment": "See 5.12"},
        "emergency power supply": {"categories": ["P4"], "comment": "See 5.12"},
        "fire door": {"categories": ["P4"], "comment": "—"},
        "fire fighting system other than sprinklers (including smoke extraction)": {
            "categories": ["P4"],
            "comment": "—",
        },
        "hazardous materials systems (including gas, steam and so forth)": {
            "categories": ["P3"],
            "comment": "See 5.8.5, 5.8.6, and 5.8.7",
        },
        "lighting systems (non-emergency)": {"categories": ["P7"], "comment": "—"},
        "solid fuel heater": {"categories": ["P3"], "comment": "—"},
        "suspended ceilings": {
            "categories": ["P2", "P3", "P4", "P5"],
            "comment": (
                "Any one or more of these apply subject to the building importance "
                "level and occupancy type. Suspended ceilings are outside NZS 4219 "
                "scope (cl. 1.1.2(g)); listed for classification reference only."
            ),
        },
        "ventilation systems (including extractor fans)": {
            "categories": ["P7"],
            "comment": "—",
        },
        "waste disposal system": {"categories": ["P7"], "comment": "—"},
        "water heater (low/mains pressure)": {"categories": ["P3"], "comment": "—"},
        "water storage tank": {
            "categories": ["P7"],
            "comment": "Roof mounted tanks adjacent to public open spaces, category is P1",
        },
        "water supply system (non-fire suppression)": {
            "categories": ["P7"],
            "comment": "Where a leak could affect critical contents below, category is P6",
        },
    }

    # Severity for resolving governing category (higher = more severe demand intent)
    _CATEGORY_SEVERITY = {
        "p1": 70,
        "p2": 60,
        "p3": 50,
        "p4": 40,
        "p5": 30,
        "p6": 20,
        "p7": 10,
    }

    @staticmethod
    def get_appendix_b_categories(component_or_system: str) -> dict:
        """Return listed categories and comments."""
        key = _normalise_key(component_or_system)
        if key not in AppendixB.CLASSIFICATIONS:
            raise ValueError(
                f"Component/system '{component_or_system}' not listed in Appendix B. "
                f"Valid keys include: {sorted(AppendixB.CLASSIFICATIONS)[:8]} ..."
            )
        row = AppendixB.CLASSIFICATIONS[key]
        return {
            "component_or_system": key,
            "categories": list(row["categories"]),
            "comment": row["comment"],
            "clause_reference": "NZS 4219 Appendix B",
        }

    @staticmethod
    def resolve_governing_category(
        candidate_categories: list,
        importance_level: int,
        service_flags: dict,
    ) -> dict:
        """
        Apply Section 5 overrides; higher severity governs.
        Water pipework ≥ P5 (cl. 5.8.8) overrides Appendix B P7 for water supply.
        """
        if not isinstance(candidate_categories, list) or not candidate_categories:
            raise ValueError("candidate_categories must be a non-empty list of strings")
        if not isinstance(service_flags, dict):
            raise ValueError("service_flags must be a dict")
        il = _require_int("importance_level", importance_level, minimum=1)
        cats = [_normalise_key(c) for c in candidate_categories]
        notes: list[str] = []

        if service_flags.get("water_supply") or service_flags.get("water_pipework"):
            if il != 4:
                raise ValueError(
                    "NZS 4219 cl. 5.8.8 requires water supply piping as at least P5, "
                    "and Table 2 NOTE restricts P5 to importance level 4. "
                    f"Got IL{il}."
                )
            if "p5" not in cats:
                cats.append("p5")
                notes.append(
                    "cl. 5.8.8 water pipework override: enforced ≥ P5 (overrides App B P7)."
                )
        if service_flags.get("hazardous"):
            if not any(c in ("p1", "p2", "p3") for c in cats):
                cats.append("p3")
                notes.append("cl. 5.10 hazardous ≥ P3 applied.")
        if service_flags.get("emergency_electrical"):
            if "p4" not in cats:
                cats.append("p4")
                notes.append("cl. 5.12 emergency electrical P4 applied.")
        if service_flags.get("non_essential_electrical"):
            if "p7" not in cats:
                cats.append("p7")
                notes.append("cl. 5.11 non-essential electrical P7 applied.")
        if service_flags.get("control_panels"):
            if "p6" not in cats:
                cats.append("p6")
                notes.append("cl. 5.16 control panels P6 applied.")
        if service_flags.get("il4_operational_continuity") and il == 4:
            if "p5" not in cats:
                cats.append("p5")
                notes.append("IL4 operational continuity → P5 applied.")

        # Validate each and score by Rc then severity
        scored = []
        for c in cats:
            Section2.validate_category_for_importance_level_table_2(c, il)
            rc = Section3.get_component_risk_factor_table_5(il, c)
            scored.append((rc, AppendixB._CATEGORY_SEVERITY[c], c))
        scored.sort(reverse=True)
        governing = scored[0][2]
        return {
            "governing_category": governing.upper(),
            "component_risk_factor": scored[0][0],
            "limit_state": Section2.get_limit_state_table_2(governing),
            "candidates_considered": [c.upper() for _, _, c in scored],
            "notes": notes,
            "clause_reference": "NZS 4219 Appendix B / Section 5 overrides",
        }


# ------------------------------------------------------------------
# Appendix C — Performance factors
# ------------------------------------------------------------------

class AppendixC:
    """Performance factors by component type (normative Appendix C)."""

    # Keys are stable snake_case identifiers for API use
    CP_BY_KEY = {
        "piping_steel_flanged_suspended_braced": 0.45,
        "piping_steel_welded_or_grooved_suspended_braced": 0.65,
        "piping_steel_screwed_suspended_braced": 0.55,
        "piping_copper_brazed_suspended_braced": 0.55,
        "piping_polypropylene_suspended_braced": 0.25,
        "rigid_ducting_suspended_braced": 0.45,
        "rigid_metal_exhaust_flue_braced": 0.45,
        "rigid_metal_exhaust_flue_cantilevered": 0.55,
        "cable_tray_suspended_braced": 0.45,
        "tank_non_pressure_floor_ductile_base": 0.55,
        "tank_non_pressure_floor_limited_ductile_base": 0.85,
        "tank_non_pressure_braced": 0.55,
        "tank_non_pressure_wall_timber_or_steel": 0.55,
        "tank_non_pressure_wall_concrete_or_masonry": 0.85,
        "tank_non_pressure_stand_moment_resisting": 0.45,
        "tank_non_pressure_stand_braced": 0.55,
        "pressure_tank_floor_mounted_cradle": 0.85,
        "compact_component_floor_ductile_base": 0.55,
        "compact_component_floor_limited_ductile_base": 0.85,
        "compact_component_vibration_isolated": 0.75,
        "compact_component_braced": 0.55,
        "compact_component_suspended_braced": 0.55,
        "non_compact_component_floor_mounted": 0.45,
        "metal_cabinet_floor_mounted": 0.45,
        "metal_cabinet_braced": 0.55,
        "light_fitting_direct_fixed": 0.85,
        "light_fitting_suspended_braced": 0.55,
    }

    @staticmethod
    def get_cp(appendix_c_component_key: str) -> float:
        key = _normalise_key(appendix_c_component_key).replace(" ", "_").replace("-", "_")
        if key not in AppendixC.CP_BY_KEY:
            raise ValueError(
                f"Unknown appendix_c_component_key '{appendix_c_component_key}'. "
                f"Valid examples: {sorted(AppendixC.CP_BY_KEY)[:6]} ..."
            )
        return float(AppendixC.CP_BY_KEY[key])


# ------------------------------------------------------------------
# Module-level re-exports (public API)
# ------------------------------------------------------------------

validate_within_scope_cl_1_1_2 = Section1.validate_within_scope_cl_1_1_2
get_limit_state_table_2 = Section2.get_limit_state_table_2
validate_category_for_importance_level_table_2 = Section2.validate_category_for_importance_level_table_2
get_zone_factor_table_3 = Section3.get_zone_factor_table_3
get_performance_factor_table_4 = Section3.get_performance_factor_table_4
get_component_risk_factor_table_5 = Section3.get_component_risk_factor_table_5
lateral_force_coefficient_eq_3_2 = Section3.lateral_force_coefficient_eq_3_2
earthquake_load_demand_eq_3_1 = Section3.earthquake_load_demand_eq_3_1
mass_kg_to_weight_kn = Section3.mass_kg_to_weight_kn
relative_seismic_displacement_eq_3_3 = Section3.relative_seismic_displacement_eq_3_3
brace_axial_force_eq_3_4 = Section3.brace_axial_force_eq_3_4
rigid_floor_mounted_actions_eq_3_5_3_6 = Section3.rigid_floor_mounted_actions_eq_3_5_3_6
braced_floor_mounted_actions_eq_3_7_3_8 = Section3.braced_floor_mounted_actions_eq_3_7_3_8
type1_resilient_mount_actions_eq_3_9_3_10 = Section3.type1_resilient_mount_actions_eq_3_9_3_10
type2_resilient_snubber_actions_eq_3_11_3_12 = Section3.type2_resilient_snubber_actions_eq_3_11_3_12
type2_impact_factor = Section3.type2_impact_factor
independent_support_anchor_bolt_forces_cl_3_7_3_1 = Section3.independent_support_anchor_bolt_forces_cl_3_7_3_1
suspended_component_actions_eq_3_13 = Section3.suspended_component_actions_eq_3_13
suspended_component_actions_eq_3_14 = Section3.suspended_component_actions_eq_3_14
lateral_coefficients_for_load_path_eq_3_2 = Section3.lateral_coefficients_for_load_path_eq_3_2
scale_anchor_demand_from_brace_actions = Section3.scale_anchor_demand_from_brace_actions
get_pipe_restraint_table_6_7 = Section3.get_pipe_restraint_table_6_7
scale_force_for_actual_spacing = Section3.scale_force_for_actual_spacing
linear_generic_actions_cl_3_6 = Section3.linear_generic_actions_cl_3_6
get_woodscrew_capacities_table_8 = Section3.get_woodscrew_capacities_table_8
get_coach_screw_capacities_table_9 = Section3.get_coach_screw_capacities_table_9
get_bolt_shear_capacity_table_10 = Section3.get_bolt_shear_capacity_table_10
get_masonry_bolt_capacity_table_11 = Section3.get_masonry_bolt_capacity_table_11
get_brace_tension_capacity_table_13 = Section3.get_brace_tension_capacity_table_13
get_brace_compression_capacity_table_14 = Section3.get_brace_compression_capacity_table_14
pipe_restraint_applicability_cl_5_8_1 = Section5.pipe_restraint_applicability_cl_5_8_1
min_restraints_per_straight_run = Section5.min_restraints_per_straight_run
seismic_gap_allowance_mm = Section5.seismic_gap_allowance_mm
duct_restraint_applicability = Section5.duct_restraint_applicability
cable_tray_restraint_applicability = Section5.cable_tray_restraint_applicability
ceiling_void_equipment_gate = Section5.ceiling_void_equipment_gate
expansion_anchor_permitted = Section5.expansion_anchor_permitted
get_clearance_mm = Section5.get_clearance_mm
get_appendix_b_categories = AppendixB.get_appendix_b_categories
resolve_governing_category = AppendixB.resolve_governing_category

ZONE_FACTORS = Section3.ZONE_FACTORS
