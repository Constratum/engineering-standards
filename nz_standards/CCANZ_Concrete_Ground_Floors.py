"""CCANZ TM38 concrete ground-floor calculation helpers.



Important limits
----------------
This is NOT a complete CCANZ floor-design program. TM38 also requires, as
applicable, superposition of stresses from multiple wheels/posts, radial and
tangential influence diagrams, corner loading, punching shear, bearing,
uniformly distributed loads, settlement, joint/load-transfer assessment,
edge thickening, durability, abrasion and reinforcement/crack control.

All loads called ``tonnes`` below are metric tonne-force units as used in the
published CCANZ equations (1 tonne-force ~= 9.80665 kN), not mass inputs that
are converted internally. Project-specific engineering verification remains
required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, log10, pi, sqrt
from typing import Any, Dict, Optional, Union
import warnings

Number = Union[int, float]

DEFAULT_POISSON_RATIO = 0.15
DEFAULT_LOAD_FACTOR = 1.5


class LoadPosition(str, Enum):
    """CCANZ point-load location classification."""

    INTERIOR = "interior"
    TRANSITION = "transition"
    EDGE = "edge"


@dataclass(frozen=True)
class PointLoadCheckResult:
    """Transparent output from a CCANZ single-point flexural check."""

    position: str
    load_tonnes_force: float
    load_factor: float
    slab_thickness_mm: float
    concrete_strength_MPa: float
    concrete_modulus_MPa: float
    subgrade_modulus_MN_m3: float
    poisson_ratio: float
    loaded_radius_mm: float
    equivalent_radius_mm: float
    relative_stiffness_radius_mm: float
    unfactored_stress_MPa: float
    factored_stress_MPa: float
    modulus_of_rupture_MPa: float
    utilization: float
    status: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        result = asdict(self)
        result["notes"] = list(self.notes)
        return result


def _positive(name: str, value: Number) -> float:
    value_f = float(value)
    if not isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than zero; got {value!r}.")
    return value_f


def _nonnegative(name: str, value: Number) -> float:
    value_f = float(value)
    if not isfinite(value_f) or value_f < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value; got {value!r}.")
    return value_f


def _poisson(value: Number) -> float:
    value_f = float(value)
    if not isfinite(value_f) or not (0.0 <= value_f < 0.5):
        raise ValueError(
            f"poisson_ratio must satisfy 0 <= nu < 0.5; got {value!r}."
        )
    return value_f


def ccanz_time_factor(first_max_load_age_days: Number) -> float:
    """Return CCANZ k1 for the two ages explicitly defined by TM38.

    TM38 gives k1 = 1.0 at 28 days and k1 = 1.1 for first application of the
    maximum load at greater than 90 days. It does not prescribe interpolation
    for intermediate ages, so this function deliberately does not invent one.

    For an intermediate age, provide a project-justified ``k1`` directly to
    :func:`point_load_check`.
    """

    age = _positive("first_max_load_age_days", first_max_load_age_days)
    if abs(age - 28.0) <= 1e-9:
        return 1.0
    if age > 90.0:
        return 1.1
    raise ValueError(
        "CCANZ TM38 explicitly defines k1 only at 28 days (1.0) and for "
        "greater than 90 days (1.1). Supply a justified k1 directly for this age."
    )


def ccanz_load_repetition_factor(repetitions: Optional[Number]) -> float:
    """Calculate CCANZ load-repetition factor k2.

    Parameters
    ----------
    repetitions:
        Expected repetitions ``N``. ``None`` or 0 denotes a non-repeated load
        and returns 1.0. ``float('inf')`` denotes unlimited repetitions and
        returns 0.75.

    Notes
    -----
    For 8,000 <= N <= 400,000, TM38 gives::

        k2 = 1.5 * (0.73 - 0.0846 * (log10(N) - 3))

    bounded to 0.75 <= k2 <= 1.0. Below 8,000, k2 = 1.0; above 400,000,
    the lower bound 0.75 is used.
    """

    if repetitions is None:
        return 1.0

    n = float(repetitions)
    if n == float("inf"):
        return 0.75
    if not isfinite(n) or n < 0.0:
        raise ValueError("repetitions must be non-negative, finite, or +infinity.")
    if n < 8_000.0:
        return 1.0
    if n > 400_000.0:
        return 0.75

    calculated = 1.5 * (0.73 - 0.0846 * (log10(n) - 3.0))
    return min(1.0, max(0.75, calculated))


def calculate_modulus_of_rupture_ccanz(
    f_c_MPa: Number, k1: Number = 1.0, k2: Number = 1.0
) -> float:
    """Calculate CCANZ modulus of rupture ``fr`` in MPa (Equation 3.1)."""

    fc = _positive("f_c_MPa", f_c_MPa)
    time_factor = _positive("k1", k1)
    repetition_factor = _positive("k2", k2)
    if not (0.75 <= repetition_factor <= 1.0):
        raise ValueError("CCANZ k2 must be between 0.75 and 1.0 inclusive.")
    return 0.456 * time_factor * repetition_factor * (fc**0.66)


def calculate_modulus_of_elasticity(
    f_c_MPa: Number, density_kg_m3: Number = 2300.0
) -> float:
    """Estimate normal-density concrete ``Ec`` in MPa.

    This is the legacy NZS 3101 expression used by the original script::

        Ec = (3320*sqrt(f'c) + 6900) * (density/2300)^1.5

    Direct project test data or the governing edition of NZS 3101 should be
    used when available.
    """

    fc = _positive("f_c_MPa", f_c_MPa)
    density = _positive("density_kg_m3", density_kg_m3)
    return (3320.0 * sqrt(fc) + 6900.0) * ((density / 2300.0) ** 1.5)


def calculate_radius_of_relative_stiffness(
    E_c_MPa: Number,
    h_mm: Number,
    k_MN_m3: Number,
    mu: Number = DEFAULT_POISSON_RATIO,
) -> float:
    """Calculate Westergaard radius of relative stiffness ``l`` in mm.

    CCANZ Equation 3.3, with E in MPa, h in mm and k in MN/m^3::

        l = [E*h^3*10^3 / (12*(1-mu^2)*k)]^0.25

    The factor 10^3 is essential. Equivalently, 1 MN/m^3 = 0.001 N/mm^3.
    """

    ec = _positive("E_c_MPa", E_c_MPa)
    h = _positive("h_mm", h_mm)
    k = _positive("k_MN_m3", k_MN_m3)
    nu = _poisson(mu)

    numerator = ec * (h**3) * 1_000.0
    denominator = 12.0 * (1.0 - nu**2) * k
    return (numerator / denominator) ** 0.25


def loaded_radius_from_rectangular_area(width_mm: Number, length_mm: Number) -> float:
    """Return equivalent circular radius for one rectangular contact area."""

    width = _positive("width_mm", width_mm)
    length = _positive("length_mm", length_mm)
    return sqrt((width * length) / pi)


def calculate_equivalent_radius(r_mm: Number, h_mm: Number) -> float:
    """Calculate CCANZ equivalent loaded radius ``b`` in mm (Equation 3.4)."""

    radius = _positive("r_mm", r_mm)
    h = _positive("h_mm", h_mm)
    if radius < 1.72 * h:
        b = sqrt(1.6 * radius**2 + h**2) - 0.675 * h
    else:
        b = radius
    if b <= 0.0:
        raise ValueError("Calculated equivalent radius is non-positive; check inputs.")
    return b


def calculate_interior_point_stress_MPa(
    P_tonnes: Number,
    h_mm: Number,
    l_mm: Number,
    b_mm: Number,
    mu: Number = DEFAULT_POISSON_RATIO,
) -> float:
    """Calculate unfactored CCANZ interior point-load stress (Equation 3.2).

    Returns MPa. ``P_tonnes`` is tonne-force as used in TM38.
    """

    load = _nonnegative("P_tonnes", P_tonnes)
    h = _positive("h_mm", h_mm)
    relative_radius = _positive("l_mm", l_mm)
    equivalent_radius = _positive("b_mm", b_mm)
    nu = _poisson(mu)

    bracket = 4.0 * log10(relative_radius / equivalent_radius) + 1.069
    if bracket <= 0.0:
        raise ValueError(
            "CCANZ interior-stress logarithmic term is non-positive. "
            "Check units and whether the point-load equation is applicable."
        )

    # Published result is kN/m^2 after multiplication by 10^6.
    # 1 kN/m^2 = 0.001 MPa, hence the net factor of 10^3 below.
    return 2.70 * (1.0 + nu) * (load / h**2) * bracket * 1_000.0


def calculate_edge_point_stress_MPa(
    P_tonnes: Number,
    h_mm: Number,
    l_mm: Number,
    b_mm: Number,
    mu: Number = DEFAULT_POISSON_RATIO,
    load_transfer_factor: Number = 1.0,
) -> float:
    """Calculate unfactored CCANZ/Kelly edge stress (Equation 3.5).

    ``load_transfer_factor`` is normally 1.0 for a free edge. TM38 permits
    multiplication by 0.85 where qualifying load transfer exists along the
    shared edge; its detailing and joint-opening conditions must be checked
    independently.
    """

    load = _nonnegative("P_tonnes", P_tonnes)
    h = _positive("h_mm", h_mm)
    relative_radius = _positive("l_mm", l_mm)
    equivalent_radius = _positive("b_mm", b_mm)
    nu = _poisson(mu)
    transfer = _positive("load_transfer_factor", load_transfer_factor)
    if transfer > 1.0:
        raise ValueError("load_transfer_factor cannot exceed 1.0.")

    bracket = (
        4.0 * log10(relative_radius / equivalent_radius)
        + log10(equivalent_radius / 25.4)
    )
    if bracket <= 0.0:
        raise ValueError(
            "CCANZ edge-stress logarithmic term is non-positive. "
            "Check units and equation applicability."
        )

    stress = (
        5.19
        * (1.0 + 0.54 * nu)
        * (load / h**2)
        * bracket
        * 1_000.0
    )
    return transfer * stress


def classify_point_load_position(
    distance_to_edge_mm: Number, relative_stiffness_radius_mm: Number
) -> LoadPosition:
    """Classify a point load using the CCANZ 0.5l and l edge-distance rule."""

    distance = _nonnegative("distance_to_edge_mm", distance_to_edge_mm)
    relative_radius = _positive(
        "relative_stiffness_radius_mm", relative_stiffness_radius_mm
    )
    if distance <= 0.5 * relative_radius:
        return LoadPosition.EDGE
    if distance >= relative_radius:
        return LoadPosition.INTERIOR
    return LoadPosition.TRANSITION


def calculate_point_stress_by_edge_distance_MPa(
    P_tonnes: Number,
    h_mm: Number,
    l_mm: Number,
    b_mm: Number,
    distance_to_edge_mm: Number,
    mu: Number = DEFAULT_POISSON_RATIO,
    edge_load_transfer_factor: Number = 1.0,
) -> tuple[float, LoadPosition]:
    """Calculate stress using CCANZ's edge/interior linear transition rule."""

    position = classify_point_load_position(distance_to_edge_mm, l_mm)
    interior = calculate_interior_point_stress_MPa(P_tonnes, h_mm, l_mm, b_mm, mu)
    edge = calculate_edge_point_stress_MPa(
        P_tonnes,
        h_mm,
        l_mm,
        b_mm,
        mu,
        load_transfer_factor=edge_load_transfer_factor,
    )

    if position is LoadPosition.INTERIOR:
        return interior, position
    if position is LoadPosition.EDGE:
        return edge, position

    distance = float(distance_to_edge_mm)
    # Weight is 0 at 0.5l and 1 at l.
    interior_weight = (distance - 0.5 * l_mm) / (0.5 * l_mm)
    stress = edge + interior_weight * (interior - edge)
    return stress, position


def point_load_check(
    *,
    P_tonnes: Number,
    h_mm: Number,
    f_c_MPa: Number,
    r_mm: Number,
    k_MN_m3: Number,
    mu: Number = DEFAULT_POISSON_RATIO,
    load_factor: Number = DEFAULT_LOAD_FACTOR,
    k1: Number = 1.0,
    k2: Optional[Number] = None,
    repetitions: Optional[Number] = None,
    E_c_MPa: Optional[Number] = None,
    distance_to_edge_mm: Optional[Number] = None,
    edge_load_transfer_factor: Number = 1.0,
) -> PointLoadCheckResult:
    """Run a transparent single-point CCANZ flexural check.

    Either provide ``k2`` directly or provide ``repetitions``. When neither is
    provided, k2 = 1.0 is used. A load is treated as interior when
    ``distance_to_edge_mm`` is omitted.
    """

    load = _nonnegative("P_tonnes", P_tonnes)
    h = _positive("h_mm", h_mm)
    fc = _positive("f_c_MPa", f_c_MPa)
    radius = _positive("r_mm", r_mm)
    subgrade = _positive("k_MN_m3", k_MN_m3)
    nu = _poisson(mu)
    gamma = _positive("load_factor", load_factor)
    time_factor = _positive("k1", k1)

    if k2 is not None and repetitions is not None:
        raise ValueError("Provide either k2 or repetitions, not both.")
    repetition_factor = (
        ccanz_load_repetition_factor(repetitions)
        if k2 is None
        else float(k2)
    )
    if not (0.75 <= repetition_factor <= 1.0):
        raise ValueError("CCANZ k2 must be between 0.75 and 1.0 inclusive.")

    ec = (
        calculate_modulus_of_elasticity(fc)
        if E_c_MPa is None
        else _positive("E_c_MPa", E_c_MPa)
    )
    relative_radius = calculate_radius_of_relative_stiffness(
        ec, h, subgrade, nu
    )
    equivalent_radius = calculate_equivalent_radius(radius, h)

    notes: list[str] = [
        "Single point-load equation only; no multi-load stress superposition is included."
    ]
    if distance_to_edge_mm is None:
        unfactored_stress = calculate_interior_point_stress_MPa(
            load, h, relative_radius, equivalent_radius, nu
        )
        position = LoadPosition.INTERIOR
        notes.append("No edge distance supplied; load treated as interior.")
    else:
        unfactored_stress, position = calculate_point_stress_by_edge_distance_MPa(
            load,
            h,
            relative_radius,
            equivalent_radius,
            distance_to_edge_mm,
            nu,
            edge_load_transfer_factor,
        )

    factored_stress = gamma * unfactored_stress
    capacity = calculate_modulus_of_rupture_ccanz(
        fc, time_factor, repetition_factor
    )
    utilization = factored_stress / capacity
    status = "OK" if utilization <= 1.0 else "NOT OK"

    return PointLoadCheckResult(
        position=position.value,
        load_tonnes_force=load,
        load_factor=gamma,
        slab_thickness_mm=h,
        concrete_strength_MPa=fc,
        concrete_modulus_MPa=ec,
        subgrade_modulus_MN_m3=subgrade,
        poisson_ratio=nu,
        loaded_radius_mm=radius,
        equivalent_radius_mm=equivalent_radius,
        relative_stiffness_radius_mm=relative_radius,
        unfactored_stress_MPa=unfactored_stress,
        factored_stress_MPa=factored_stress,
        modulus_of_rupture_MPa=capacity,
        utilization=utilization,
        status=status,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Backwards-compatible public wrappers for the uploaded module
# ---------------------------------------------------------------------------


def calculate_load_bearing_capacity(
    f_c_MPa: Number, k1: Number = 1.0, k2: Number = 1.0
) -> float:
    """Return CCANZ flexural capacity; factors are no longer hard-coded."""

    return calculate_modulus_of_rupture_ccanz(f_c_MPa, k1, k2)


def calculate_load_bearing(
    P_tonnes: Number,
    load_factor: Number,
    h_mm: Number,
    f_c_MPa: Number,
    r_mm: Number,
    *,
    E_c_MPa: Optional[Number] = None,
    k_MN_m3: Number = 70.0,
    mu: Number = DEFAULT_POISSON_RATIO,
    k1: Number = 1.0,
    k2: Number = 1.0,
    distance_to_edge_mm: Optional[Number] = None,
    edge_load_transfer_factor: Number = 1.0,
) -> Dict[str, Any]:
    """Backward-compatible wrapper that now returns the documented dictionary."""

    return point_load_check(
        P_tonnes=P_tonnes,
        load_factor=load_factor,
        h_mm=h_mm,
        f_c_MPa=f_c_MPa,
        r_mm=r_mm,
        E_c_MPa=E_c_MPa,
        k_MN_m3=k_MN_m3,
        mu=mu,
        k1=k1,
        k2=k2,
        distance_to_edge_mm=distance_to_edge_mm,
        edge_load_transfer_factor=edge_load_transfer_factor,
    ).to_dict()


def check_uplift_kN(
    max_factored_uplift_kN: Number, critical_uplift_from_table_kN: Number
) -> Dict[str, Any]:
    """Compare uplift with an independently selected CCANZ table capacity.

    Tables 3.4--3.6 are limited to a 70 mm loaded radius and their listed
    concrete strengths, thicknesses and subgrade moduli. Do not interpolate or
    extrapolate without an independently justified method.
    """

    demand = _nonnegative("max_factored_uplift_kN", max_factored_uplift_kN)
    capacity = _positive(
        "critical_uplift_from_table_kN", critical_uplift_from_table_kN
    )
    utilization = demand / capacity
    is_ok = utilization <= 1.0
    return {
        "factored_uplift_kN": demand,
        "uplift_capacity_kN": capacity,
        "utilization": utilization,
        "status": (
            "OK - slab remains in contact for the selected table case."
            if is_ok
            else "NOT OK - lift-off occurs; Westergaard analysis is not applicable."
        ),
    }


def check_uplift(
    max_factored_uplift_N: Number, critical_uplift_from_table_N: Number
) -> Dict[str, Any]:
    """Legacy N-unit wrapper retained for compatibility with the uploaded file."""

    demand_n = _nonnegative("max_factored_uplift_N", max_factored_uplift_N)
    capacity_n = _positive(
        "critical_uplift_from_table_N", critical_uplift_from_table_N
    )
    result = check_uplift_kN(demand_n / 1_000.0, capacity_n / 1_000.0)
    return {
        "factored_uplift_N": demand_n,
        "uplift_capacity_N": capacity_n,
        "utilization": result["utilization"],
        "status": result["status"],
    }


def check_min_reinforcement(f_sy_MPa: Number) -> float:
    """Legacy heuristic from the uploaded script; not a CCANZ design check.

    Retained only to avoid silently breaking an existing caller. Shrinkage and
    crack-control reinforcement in TM38 depends on the selected slab/jointing
    system and detailed Chapter 5 provisions; it cannot be selected from yield
    strength alone.
    """

    fy = _positive("f_sy_MPa", f_sy_MPa)
    warnings.warn(
        "check_min_reinforcement() is a legacy heuristic, not a complete CCANZ "
        "reinforcement design. Verify Chapter 5 and the governing NZ standards.",
        category=RuntimeWarning,
        stacklevel=2,
    )
    return 0.001 if fy >= 400.0 else 0.002


if __name__ == "__main__":
    # Example only: one interior point load. Replace all project inputs.
    example = point_load_check(
        P_tonnes=6.0,
        h_mm=225.0,
        f_c_MPa=35.0,
        r_mm=118.0,
        k_MN_m3=54.0,
        load_factor=1.5,
        k1=1.1,
        k2=1.0,
    )
    for key, value in example.to_dict().items():
        print(f"{key}: {value}")
