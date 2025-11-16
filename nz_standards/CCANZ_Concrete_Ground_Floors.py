"""
CCANZ Concrete Ground Floors & Pavements Design Module

Based on "CCANZ - Concrete Ground Floors & Pavements for Commercial
and Industrial Use - Part 2 - Specific Design [2001]"

This module implements Westergaard analysis for concrete slabs on grade,
including load-bearing capacity and uplift checks.
"""

import numpy as np
from typing import Dict, Tuple


def calculate_modulus_of_rupture_ccanz(f_c_MPa: float, k1: float, k2: float) -> float:
    """
    Calculates the Modulus of Rupture (flexural tensile strength) of concrete
    using CCANZ method.

    Ref: CCANZ Equation 3.1, page 26

    Args:
        f_c_MPa: Specified 28-day cylinder compressive strength (MPa)
        k1: Time factor (e.g., 1.0 for 28 days, 1.1 for >90 days)
        k2: Load repetition factor (e.g., 1.0 for <8,000 reps)

    Returns:
        Modulus of Rupture (fr) in MPa
    """
    return 0.456 * k1 * k2 * (f_c_MPa**0.66)


def calculate_radius_of_relative_stiffness(
    E_c_MPa: float, h_mm: float, k_MN_m3: float, mu: float
) -> float:
    """
    Calculates the radius of relative stiffness (l) for Westergaard analysis.

    Ref: CCANZ Equation 3.3, page 27

    Args:
        E_c_MPa: Modulus of elasticity of concrete (MPa)
        h_mm: Slab thickness (mm)
        k_MN_m3: Modulus of sub-grade reaction (MN/m³ or N/mm³)
        mu: Poisson ratio of the slab (typically 0.15)

    Returns:
        Radius of relative stiffness (l) in mm
    """
    # Note: E_c in MPa (N/mm²), h in mm, k in MN/m³ (which is N/mm³)
    numerator = E_c_MPa * (h_mm**3)
    denominator = 12 * (1 - mu**2) * k_MN_m3
    return (numerator / denominator) ** 0.25


def calculate_equivalent_radius(r_mm: float, h_mm: float) -> float:
    """
    Calculates the equivalent radius of the loaded area (b).

    Ref: CCANZ Equation 3.4, page 27

    Args:
        r_mm: Radius of the loaded area (mm)
        h_mm: Slab thickness (mm)

    Returns:
        Equivalent radius (b) in mm
    """
    if r_mm < 1.72 * h_mm:
        return np.sqrt(1.6 * r_mm**2 + h_mm**2) - 0.675 * h_mm
    else:
        return r_mm


def calculate_load_bearing_capacity(
    f_c_MPa: float,
) -> float:
    # 1. Calculate Allowable Stress (Capacity)
    k1 = 1.1  # Time to first application of max load
    k2 = 1.0  # Load repetition factor
    fr_MPa = calculate_modulus_of_rupture_ccanz(f_c_MPa, k1, k2)
    return fr_MPa


def calculate_load_bearing(
    P_tonnes: float,
    load_factor: float,
    h_mm: float,
    f_c_MPa: float,
    r_mm: float,
) -> Dict[str, float]:
    """
    Performs a load-bearing (flexural stress) check for an interior load
    using Westergaard analysis.

    Compares the induced stress (sigma_i) with the allowable stress (fr).

    Ref: CCANZ Equation 3.2, page 27

    Args:
        P_tonnes: Unfactored applied load (tonnes)
        load_factor: Load factor (e.g., 1.5)
        h_mm: Slab thickness (mm)
        f_c_MPa: Concrete compressive strength (MPa)
        E_c_MPa: Modulus of Elasticity (MPa)
        k_MN_m3: Modulus of sub-grade reaction (MN/m³)
        mu: Poisson ratio (typically 0.15)
        r_mm: Radius of loaded area (mm)
        k1: Time factor for fr
        k2: Repetition factor for fr

    Returns:
        Dictionary containing:
        - induced_stress_MPa: Calculated stress from factored load
        - allowable_stress_MPa: Allowable flexural stress
        - status: "OK" or "NOT OK"
        - utilization: Ratio of induced/allowable stress
    """

    mu = 0.15  # Poisson's ratio (typically 0.15)
    k_MN_m3 = 70  # Modulus of sub-grade reaction (MN/m³)
    E_c_MPa = 4700 * np.sqrt(f_c_MPa)

    # 2. Calculate Induced Stress (Factored Load Effect)
    l_mm = calculate_radius_of_relative_stiffness(E_c_MPa, h_mm, k_MN_m3, mu)
    b_mm = calculate_equivalent_radius(r_mm, h_mm)

    # Using Equation 3.2, which gives result in kN/m² (kPa)
    log_term = (4 * np.log10(l_mm / b_mm)) + 1.069
    constant_term = 2.70 * (1 + mu) * (10**6)  # for unit conversion in formula

    # Calculate stress in kPa (kN/m²) as per Eq 3.2
    sigma_i_kPa = (constant_term * (P_tonnes / (h_mm**2))) * log_term
    sigma_i_MPa = sigma_i_kPa / 1000.0

    return load_factor * sigma_i_MPa


def check_uplift(
    max_factored_uplift_N: float, critical_uplift_from_table_N: float
) -> Dict[str, float]:
    """
    Compares the factored uplift load against a critical (no-lift) capacity.

    The critical capacity must be found from CCANZ tables (e.g., Table 3.4-3.6)
    or a separate rational analysis.

    Args:
        max_factored_uplift_N: The maximum uplift force from the rack leg (N)
        critical_uplift_from_table_N: The force required to cause lift-off (N)

    Returns:
        Dictionary containing:
        - factored_uplift_N: Applied uplift force
        - uplift_capacity_N: Capacity against uplift
        - utilization: Ratio of uplift/capacity
        - status: "OK" or "NOT OK"
    """
    utilization = (
        max_factored_uplift_N / critical_uplift_from_table_N
        if critical_uplift_from_table_N > 0
        else float("inf")
    )

    if max_factored_uplift_N <= critical_uplift_from_table_N:
        status = "OK - Westergaard analysis is valid."
    else:
        status = "NOT OK - Uplift occurs. Westergaard analysis is invalid."

    return {
        "factored_uplift_N": max_factored_uplift_N,
        "uplift_capacity_N": critical_uplift_from_table_N,
        "utilization": utilization,
        "status": status,
    }


def calculate_modulus_of_elasticity(f_c_MPa: float) -> float:
    """
    Estimates the modulus of elasticity of concrete.

    Ref: CCANZ Section 3.1.1, page 25

    Args:
        f_c_MPa: Specified 28-day cylinder compressive strength (MPa)

    Returns:
        Modulus of Elasticity (Ec) in MPa
    """
    # For normal density concrete
    return 3320 * np.sqrt(f_c_MPa) + 6900


def check_min_reinforcement(f_sy_MPa: float) -> float:
    """
    Calculates the minimum reinforcement percentage for shrinkage control.

    Ref: CCANZ Section 5.2, page 68 (recommends 0.1% for high-yield steel)

    Args:
        f_sy_MPa: Reinforcement yield strength (MPa)

    Returns:
        Minimum reinforcement percentage (e.g., 0.001 for 0.1%)
    """
    # Per section 5.2, a simple minimum is recommended.
    # For high-yield steel (e.g., 400-500 MPa), 0.1% is the minimum.
    # For mild-steel, 0.2% is the minimum.
    if f_sy_MPa >= 400:
        return 0.001  # 0.1%
    else:
        return 0.002  # 0.2%
