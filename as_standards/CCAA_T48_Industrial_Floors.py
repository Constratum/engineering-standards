"""Calculation helpers for CCAA T48 industrial floors and pavements.

Based on CCAA T48, *Guide to Industrial Floors and Pavements – design,
construction and specification*, third edition (October 2009).

What this module automates
--------------------------
* Table 1.13 simplified minimum thickness selection.
* Equations 1--8 used by the rigorous thickness-design workflow.
* Table 1.16 material-factor ranges and Table 1.17 repetition factor.
* Equation 2 flexural tensile strength and Table 1.22 wheel-load factor k4.
* Equation 3 equivalent uniform soil modulus.
* Equations 9--10 combined wheel/post interaction factors.
* Table 1.23 edge-thickening distance.
* Equation 11 punching shear and the T48 bearing check.
* Published Appendix D example calculations for regression checking.

What this module deliberately does NOT automate
-----------------------------------------------
The final base thickness in the rigorous method is read from Charts 1.1--1.4
(or calculated using CCAA's associated worksheet / a separately validated
analysis). The FE, FH and FS/FW chart factors are therefore explicit inputs.
They are not guessed or silently digitised here. This avoids presenting an
approximate chart trace as an exact T48 calculation.

T48 is industry guidance, not a replacement for project-specific engineering
or current Australian Standards. Its own publisher warns that standards and
industry practice may change. Verify all references, load combinations,
material resistance factors, durability, joints, reinforcement, settlement,
and construction requirements against the governing project documents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, log10, sqrt
from typing import Any, Dict, Iterable, Mapping, Optional, Union

Number = Union[int, float]


class LoadType(str, Enum):
    WHEEL = "wheel"
    POST = "post"
    DISTRIBUTED = "distributed"


class LoadLocation(str, Enum):
    INTERIOR = "interior"
    EDGE = "edge"


class SubgradeRating(str, Enum):
    POOR = "poor"
    MEDIUM_TO_GOOD = "medium_to_good"


class SimplifiedApplication(str, Enum):
    """Application classes in T48 Table 1.13."""

    LIGHT = "shops_private_car_garages_light_industrial_up_to_5_kPa"
    COMMERCIAL = "commercial_vehicle_garages_industrial_warehouses_5_to_20_kPa"


class SupportingSoil(str, Enum):
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MEDIUM = "medium"
    STIFF = "stiff"
    VERY_STIFF = "very_stiff"


MATERIAL_FACTOR_RANGES: Mapping[LoadType, tuple[float, float]] = {
    LoadType.WHEEL: (0.85, 0.95),
    LoadType.POST: (0.75, 0.85),
    LoadType.DISTRIBUTED: (0.75, 0.85),
}

# Table 1.22. For f'c > 50 MPa, T48 specifies k4 = 1.0.
_K4_POINTS: tuple[tuple[float, float], ...] = (
    (20.0, 1.03),
    (25.0, 1.07),
    (32.0, 1.11),
    (40.0, 1.16),
    (50.0, 1.20),
)

_EDGE_THICKENING_MULTIPLIER: Mapping[SupportingSoil, float] = {
    SupportingSoil.VERY_WEAK: 20.0,
    SupportingSoil.WEAK: 15.0,
    SupportingSoil.MEDIUM: 10.0,
    SupportingSoil.STIFF: 8.0,
    SupportingSoil.VERY_STIFF: 6.0,
}

_TYPICAL_SOIL_MODULUS_MPA: Mapping[SupportingSoil, float] = {
    SupportingSoil.VERY_WEAK: 2.0,
    SupportingSoil.WEAK: 5.0,
    SupportingSoil.MEDIUM: 15.0,
    SupportingSoil.STIFF: 30.0,
    SupportingSoil.VERY_STIFF: 80.0,
}

_SOIL_CORRELATION_B: Mapping[str, float] = {
    "gravel": 0.9,
    "sand": 0.8,
    "silt_or_silty_clay": 0.7,
    "stiff_clay": 0.6,
    "soft_clay": 0.4,
}

_CPT_CORRELATION_A: Mapping[str, float] = {
    "loose_sand": 5.0,
    "medium_dense_sand": 8.0,
    "dense_sand": 10.0,
    "silt": 12.0,
    "silty_clay": 15.0,
    "highly_plastic_clay": 20.0,
}


@dataclass(frozen=True)
class SoilLayer:
    """A layer already assigned its T48 Figure 1.23 weighting factor."""

    thickness_m: float
    youngs_modulus_MPa: float
    weighting_factor: float
    label: str = ""


@dataclass(frozen=True)
class StressFactorResult:
    """Result used to enter a T48 thickness chart."""

    load_type: str
    location: str
    design_tensile_strength_MPa: float
    FE: float
    FH: float
    FS_or_FW: float
    calibration_k3: Optional[float]
    concrete_correction_k4: Optional[float]
    load_magnitude: Optional[float]
    stress_factor: float
    chart: str
    notes: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True)
class CapacityCheckResult:
    demand: float
    capacity: float
    utilization: float
    status: str
    demand_unit: str
    capacity_unit: str
    details: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["details"] = dict(self.details)
        return result


def _positive(name: str, value: Number) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero; got {value!r}.")
    return result


def _nonnegative(name: str, value: Number) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return result


def _factor(name: str, value: Number) -> float:
    return _positive(name, value)


def _coerce_enum(enum_type: type[Enum], value: Union[Enum, str], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        options = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {options}.") from exc


def simplified_minimum_thickness_mm(
    application: Union[SimplifiedApplication, str],
    subgrade_rating: Union[SubgradeRating, str],
) -> float:
    """Return T48 Table 1.13 minimum base thickness for light applications.

    The simplified table is only for the application and live-load ranges
    explicitly covered by T48. It does not replace rigorous design where wheel,
    post, edge, fatigue, settlement or other project conditions govern.
    """

    app = _coerce_enum(SimplifiedApplication, application, "application")
    rating = _coerce_enum(SubgradeRating, subgrade_rating, "subgrade_rating")

    table: Mapping[tuple[SimplifiedApplication, SubgradeRating], float] = {
        (SimplifiedApplication.LIGHT, SubgradeRating.POOR): 150.0,
        (SimplifiedApplication.LIGHT, SubgradeRating.MEDIUM_TO_GOOD): 130.0,
        (SimplifiedApplication.COMMERCIAL, SubgradeRating.POOR): 200.0,
        (SimplifiedApplication.COMMERCIAL, SubgradeRating.MEDIUM_TO_GOOD): 180.0,
    }
    return table[(app, rating)]


def total_load_repetitions(
    repetitions_per_operating_day: Number,
    design_life_years: Number,
    operating_days_per_year: Number = 260.0,
) -> float:
    """Calculate repetitions; the T48 Table 1.15 convention uses 260 days/year."""

    daily = _nonnegative("repetitions_per_operating_day", repetitions_per_operating_day)
    years = _positive("design_life_years", design_life_years)
    days = _positive("operating_days_per_year", operating_days_per_year)
    return daily * years * days


def load_repetition_factor_k2(
    repetitions: Optional[Number] = None,
    *,
    permanent_load: bool = False,
) -> float:
    """Calculate T48 Table 1.17 load-repetition factor ``k2``.

    * Permanent/non-repeated load: k2 = 1.0.
    * 50 <= N <= 400,000: k2 = 0.73 - 0.0846(log10(N) - 3).
    * Above 400,000: the conservative unlimited-repetition value 0.50.
    * ``+infinity``: 0.50.

    T48 does not prescribe the equation for 0 < N < 50, so no value is
    invented for that range.
    """

    if permanent_load:
        if repetitions not in (None, 0, 0.0):
            raise ValueError("Do not supply repetitions when permanent_load=True.")
        return 1.0
    if repetitions is None:
        raise ValueError(
            "Provide repetitions or set permanent_load=True. This avoids confusing "
            "a permanent load (k2=1.0) with unlimited repetitions (k2=0.50)."
        )

    n = float(repetitions)
    if n == float("inf"):
        return 0.50
    if not isfinite(n) or n < 0.0:
        raise ValueError("repetitions must be non-negative, finite, or +infinity.")
    if n == 0.0:
        return 1.0
    if n < 50.0:
        raise ValueError(
            "T48 states its k2 equation for N=50 to 400,000. Supply a justified "
            "project value for fewer than 50 repetitions."
        )
    if n > 400_000.0:
        return 0.50
    return max(0.50, 0.73 - 0.0846 * (log10(n) - 3.0))


def characteristic_flexural_tensile_strength_MPa(
    f_c_MPa: Number,
    *,
    strength_age_factor: Number = 1.0,
) -> float:
    """Calculate T48 characteristic flexural strength, Eq. 2.

    ``strength_age_factor=1.10`` represents T48's frequently assumed 10%
    increase from 28 to 90 days when no mix-specific relationship exists.
    Use 1.0 (or a lower verified factor) where substantial loading occurs early.
    """

    fc = _positive("f_c_MPa", f_c_MPa)
    age_factor = _positive("strength_age_factor", strength_age_factor)
    return age_factor * 0.7 * sqrt(fc)


def material_factor_range(load_type: Union[LoadType, str]) -> tuple[float, float]:
    """Return T48 Table 1.16 recommended range for material factor k1."""

    kind = _coerce_enum(LoadType, load_type, "load_type")
    return MATERIAL_FACTOR_RANGES[kind]


def validate_material_factor_k1(
    load_type: Union[LoadType, str], k1: Number
) -> float:
    """Validate k1 against T48 Table 1.16 for the selected loading type."""

    kind = _coerce_enum(LoadType, load_type, "load_type")
    value = _positive("k1", k1)
    lower, upper = MATERIAL_FACTOR_RANGES[kind]
    if not (lower <= value <= upper):
        raise ValueError(
            f"For {kind.value} loading, T48 Table 1.16 gives {lower:.2f} <= k1 <= "
            f"{upper:.2f}; got {value:.3f}."
        )
    return value


def design_tensile_strength_MPa(
    f_c_MPa: Number,
    *,
    load_type: Union[LoadType, str],
    k1: Number,
    k2: Number,
    strength_age_factor: Number = 1.0,
    validate_k1: bool = True,
) -> float:
    """Calculate T48 design tensile strength ``fall`` (Equation 1)."""

    kind = _coerce_enum(LoadType, load_type, "load_type")
    material = (
        validate_material_factor_k1(kind, k1)
        if validate_k1
        else _positive("k1", k1)
    )
    repetition = _positive("k2", k2)
    if repetition > 1.0:
        raise ValueError("k2 cannot exceed 1.0.")
    fcf = characteristic_flexural_tensile_strength_MPa(
        f_c_MPa, strength_age_factor=strength_age_factor
    )
    return material * repetition * fcf


def wheel_concrete_correction_k4(
    f_c_MPa: Number, *, allow_interpolation: bool = False
) -> float:
    """Return T48 Table 1.22 wheel-load correction factor ``k4``.

    T48 lists k4 only for the standard strengths 20, 25, 32, 40 and 50 MPa,
    and specifies k4 = 1.0 above 50 MPa. By default this function therefore
    requires an exact listed strength rather than silently inventing a value.

    Set ``allow_interpolation=True`` only where linear interpolation between
    adjacent table entries has been accepted as a project-specific engineering
    assumption. Strengths below 20 MPa are outside the table and rejected.
    """

    fc = _positive("f_c_MPa", f_c_MPa)
    if fc > 50.0:
        return 1.0
    if fc < 20.0:
        raise ValueError("T48 Table 1.22 starts at f'c = 20 MPa.")

    for table_fc, table_k4 in _K4_POINTS:
        if abs(fc - table_fc) <= 1e-9:
            return table_k4

    if not allow_interpolation:
        listed = ", ".join(f"{x:g}" for x, _ in _K4_POINTS)
        raise ValueError(
            "T48 Table 1.22 does not list this concrete strength. Use one of "
            f"{listed} MPa, use f'c > 50 MPa (k4=1.0), or explicitly set "
            "allow_interpolation=True with project-specific justification."
        )

    for (x0, y0), (x1, y1) in zip(_K4_POINTS[:-1], _K4_POINTS[1:]):
        if x0 < fc < x1:
            fraction = (fc - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)

    raise RuntimeError("Internal k4 interpolation error.")


def equivalent_uniform_soil_modulus_MPa(layers: Iterable[SoilLayer]) -> float:
    """Calculate T48 equivalent uniform soil modulus, Equation 3.

    The weighting factor for each layer must first be obtained from T48 Figure
    1.23 for the applicable wheel, post or distributed loading geometry.
    """

    layer_list = list(layers)
    if not layer_list:
        raise ValueError("At least one soil layer is required.")

    numerator = 0.0
    denominator = 0.0
    for index, layer in enumerate(layer_list, start=1):
        h = _positive(f"layers[{index}].thickness_m", layer.thickness_m)
        es = _positive(
            f"layers[{index}].youngs_modulus_MPa", layer.youngs_modulus_MPa
        )
        wf = _positive(f"layers[{index}].weighting_factor", layer.weighting_factor)
        numerator += wf * h
        denominator += wf * h / es
    return numerator / denominator


def short_term_soil_modulus_MPa(
    long_term_modulus_MPa: Number,
    *,
    soil_type: Optional[str] = None,
    correlation_b: Optional[Number] = None,
) -> float:
    """Convert long-term to short-term soil modulus using T48 Equation 4.

    ``Ess = Esl / b``. Supply either a key from :data:`_SOIL_CORRELATION_B`
    through ``soil_type`` or a project-specific ``correlation_b``.
    """

    esl = _positive("long_term_modulus_MPa", long_term_modulus_MPa)
    if (soil_type is None) == (correlation_b is None):
        raise ValueError("Supply exactly one of soil_type or correlation_b.")
    if soil_type is not None:
        try:
            b = _SOIL_CORRELATION_B[soil_type]
        except KeyError as exc:
            options = ", ".join(_SOIL_CORRELATION_B)
            raise ValueError(f"soil_type must be one of: {options}.") from exc
    else:
        b = _positive("correlation_b", correlation_b)  # type: ignore[arg-type]
    if b > 1.0:
        raise ValueError("T48 correlation b should not exceed 1.0.")
    return esl / b


def cpt_short_term_soil_modulus_MPa(
    cone_resistance_qc_MPa: Number,
    *,
    soil_type: Optional[str] = None,
    correlation_a: Optional[Number] = None,
) -> float:
    """Estimate short-term soil modulus from CPT resistance, T48 Equation 5."""

    qc = _positive("cone_resistance_qc_MPa", cone_resistance_qc_MPa)
    if (soil_type is None) == (correlation_a is None):
        raise ValueError("Supply exactly one of soil_type or correlation_a.")
    if soil_type is not None:
        try:
            a = _CPT_CORRELATION_A[soil_type]
        except KeyError as exc:
            options = ", ".join(_CPT_CORRELATION_A)
            raise ValueError(f"soil_type must be one of: {options}.") from exc
    else:
        a = _positive("correlation_a", correlation_a)  # type: ignore[arg-type]
    return a * qc


def wheel_stress_factor(
    *,
    f_all_MPa: Number,
    f_c_MPa: Number,
    FE: Number,
    FH: Number,
    FS: Number,
    location: Union[LoadLocation, str],
    k4_override: Optional[Number] = None,
    allow_k4_interpolation: bool = False,
) -> StressFactorResult:
    """Calculate T48 wheel stress factor F1/F2, Equation 6.

    FE, FH and FS must be read from Chart 1.1 (interior) or Chart 1.2 (edge).
    The resulting factor is then taken back to the same chart with axle load to
    read the required thickness. For a non-tabulated concrete strength, either
    provide an independently justified ``k4_override`` or explicitly permit
    linear interpolation with ``allow_k4_interpolation=True``.
    """

    loc = _coerce_enum(LoadLocation, location, "location")
    fall = _positive("f_all_MPa", f_all_MPa)
    fe = _factor("FE", FE)
    fh = _factor("FH", FH)
    fs = _factor("FS", FS)
    k3 = 1.2 if loc is LoadLocation.INTERIOR else 1.05
    if k4_override is None:
        k4 = wheel_concrete_correction_k4(
            f_c_MPa, allow_interpolation=allow_k4_interpolation
        )
    else:
        k4 = _positive("k4_override", k4_override)
    stress_factor = fall * fe * fh * fs * k3 * k4
    chart = "Chart 1.1" if loc is LoadLocation.INTERIOR else "Chart 1.2"
    label = "F1" if loc is LoadLocation.INTERIOR else "F2"
    return StressFactorResult(
        load_type=LoadType.WHEEL.value,
        location=loc.value,
        design_tensile_strength_MPa=fall,
        FE=fe,
        FH=fh,
        FS_or_FW=fs,
        calibration_k3=k3,
        concrete_correction_k4=k4,
        load_magnitude=None,
        stress_factor=stress_factor,
        chart=chart,
        notes=(
            f"Stress factor is {label}; read final thickness from {chart} using axle load.",
            "Chart factors are explicit inputs and have not been approximated by this module.",
        ),
    )


def post_stress_factor(
    *,
    f_all_MPa: Number,
    post_load_kN: Number,
    FE: Number,
    FH: Number,
    FS: Number,
    location: Union[LoadLocation, str],
) -> StressFactorResult:
    """Calculate T48 post stress factor F3, Equation 7."""

    loc = _coerce_enum(LoadLocation, location, "location")
    fall = _positive("f_all_MPa", f_all_MPa)
    load = _positive("post_load_kN", post_load_kN)
    fe = _factor("FE", FE)
    fh = _factor("FH", FH)
    fs = _factor("FS", FS)
    stress_factor = 1_000.0 * (fall / load) * fe * fh * fs
    return StressFactorResult(
        load_type=LoadType.POST.value,
        location=loc.value,
        design_tensile_strength_MPa=fall,
        FE=fe,
        FH=fh,
        FS_or_FW=fs,
        calibration_k3=None,
        concrete_correction_k4=None,
        load_magnitude=load,
        stress_factor=stress_factor,
        chart="Chart 1.3",
        notes=(
            "Read final interior/edge thickness from Chart 1.3.",
            "Chart 1.3 is based on a 25,000 mm^2 base plate; verify sensitivity for other areas.",
        ),
    )


def distributed_stress_factor(
    *,
    f_all_MPa: Number,
    distributed_load_kPa: Number,
    FE: Number,
    FH: Number,
    FW: Number,
) -> StressFactorResult:
    """Calculate T48 distributed-load stress factor F4, Equation 8."""

    fall = _positive("f_all_MPa", f_all_MPa)
    load = _positive("distributed_load_kPa", distributed_load_kPa)
    fe = _factor("FE", FE)
    fh = _factor("FH", FH)
    fw = _factor("FW", FW)
    stress_factor = 1_000.0 * (fall / load) * fe * fh * fw
    return StressFactorResult(
        load_type=LoadType.DISTRIBUTED.value,
        location=LoadLocation.INTERIOR.value,
        design_tensile_strength_MPa=fall,
        FE=fe,
        FH=fh,
        FS_or_FW=fw,
        calibration_k3=None,
        concrete_correction_k4=None,
        load_magnitude=load,
        stress_factor=stress_factor,
        chart="Chart 1.4",
        notes=(
            "T48 identifies the interior case as critical for distributed loading.",
            "Read final thickness from Chart 1.4; no edge thickening is required for this load type.",
        ),
    )


def combined_factor_post_on_wheel(
    *, Q: Number, post_load_kN: Number, axle_load_kN: Number
) -> float:
    """Return FC1/FC2 for the effect of a post at the wheel location, Eq. 9."""

    q = _nonnegative("Q", Q)
    post = _positive("post_load_kN", post_load_kN)
    axle = _positive("axle_load_kN", axle_load_kN)
    return 1.0 / (1.0 + q * (post / axle))


def combined_factor_wheel_on_post(
    *, Q3: Number, axle_load_kN: Number, post_load_kN: Number
) -> float:
    """Return FC3 for the effect of a wheel at the post location, Eq. 10."""

    q = _nonnegative("Q3", Q3)
    axle = _positive("axle_load_kN", axle_load_kN)
    post = _positive("post_load_kN", post_load_kN)
    return 1.0 / (1.0 + q * (axle / post))


def apply_combined_factor(stress_factor: Number, interaction_factor: Number) -> float:
    """Apply FC to F, ready for re-entry to the applicable T48 chart."""

    factor = _positive("stress_factor", stress_factor)
    fc = _positive("interaction_factor", interaction_factor)
    if fc > 1.0:
        raise ValueError("A T48 combined-load interaction factor FC cannot exceed 1.0.")
    return factor * fc


def edge_thickening_distance_mm(
    interior_thickness_mm: Number,
    supporting_soil: Union[SupportingSoil, str],
) -> Dict[str, float]:
    """Return T48 Table 1.23 distance from edge where thickening commences."""

    thickness = _positive("interior_thickness_mm", interior_thickness_mm)
    soil = _coerce_enum(SupportingSoil, supporting_soil, "supporting_soil")
    multiplier = _EDGE_THICKENING_MULTIPLIER[soil]
    return {
        "interior_thickness_mm": thickness,
        "typical_soil_modulus_MPa": _TYPICAL_SOIL_MODULUS_MPA[soil],
        "edge_distance_multiplier": multiplier,
        "edge_thickening_distance_mm": multiplier * thickness,
    }


def punching_shear_capacity_kN(
    *,
    f_c_MPa: Number,
    slab_thickness_mm: Number,
    base_plate_width_mm: Number,
    base_plate_length_mm: Optional[Number] = None,
    phi: Number = 0.8,
) -> Dict[str, float]:
    """Calculate T48 Equation 11 interior punching capacity.

    The first-pass T48 model assumes the full factored post load is resisted by
    the slab. This function is for an interior rectangular base plate and does
    not deduct direct transfer to the subgrade.
    """

    fc = _positive("f_c_MPa", f_c_MPa)
    t = _positive("slab_thickness_mm", slab_thickness_mm)
    width = _positive("base_plate_width_mm", base_plate_width_mm)
    length = width if base_plate_length_mm is None else _positive(
        "base_plate_length_mm", base_plate_length_mm
    )
    strength_factor = _positive("phi", phi)
    if strength_factor > 1.0:
        raise ValueError("phi cannot exceed 1.0.")

    d = 0.9 * t
    beta_h = max(width, length) / min(width, length)
    fcv_uncapped = 0.17 * (1.0 + 2.0 / beta_h) * sqrt(fc)
    fcv_cap = 0.34 * sqrt(fc)
    fcv = min(fcv_uncapped, fcv_cap)
    critical_perimeter = 2.0 * ((width + d) + (length + d))
    nominal_capacity = fcv * critical_perimeter * d / 1_000.0
    design_capacity = strength_factor * nominal_capacity
    return {
        "effective_depth_mm": d,
        "beta_h": beta_h,
        "fcv_uncapped_MPa": fcv_uncapped,
        "fcv_cap_MPa": fcv_cap,
        "fcv_used_MPa": fcv,
        "critical_perimeter_mm": critical_perimeter,
        "nominal_capacity_kN": nominal_capacity,
        "phi": strength_factor,
        "design_capacity_kN": design_capacity,
    }


def check_punching_shear(
    *,
    factored_post_load_kN: Number,
    f_c_MPa: Number,
    slab_thickness_mm: Number,
    base_plate_width_mm: Number,
    base_plate_length_mm: Optional[Number] = None,
    phi: Number = 0.8,
) -> CapacityCheckResult:
    """Check factored post load against T48 Equation 11 design capacity."""

    demand = _nonnegative("factored_post_load_kN", factored_post_load_kN)
    details = punching_shear_capacity_kN(
        f_c_MPa=f_c_MPa,
        slab_thickness_mm=slab_thickness_mm,
        base_plate_width_mm=base_plate_width_mm,
        base_plate_length_mm=base_plate_length_mm,
        phi=phi,
    )
    capacity = details["design_capacity_kN"]
    utilization = demand / capacity
    return CapacityCheckResult(
        demand=demand,
        capacity=capacity,
        utilization=utilization,
        status="OK" if utilization <= 1.0 else "NOT OK",
        demand_unit="kN",
        capacity_unit="kN",
        details=details,
    )


def concrete_bearing_design_stress_MPa(
    *,
    f_c_MPa: Number,
    loaded_area_A1_mm2: Number,
    supporting_area_A2_mm2: Optional[Number] = None,
    phi: Number = 0.6,
) -> Dict[str, float]:
    """Calculate T48/AS 3600 bearing design stress under a post plate.

    Design stress = phi * min(0.85 f'c sqrt(A2/A1), 2 f'c).
    A2 defaults to A1 for the conservative first trial used in T48 examples.
    """

    fc = _positive("f_c_MPa", f_c_MPa)
    a1 = _positive("loaded_area_A1_mm2", loaded_area_A1_mm2)
    a2 = a1 if supporting_area_A2_mm2 is None else _positive(
        "supporting_area_A2_mm2", supporting_area_A2_mm2
    )
    if a2 < a1:
        raise ValueError("supporting_area_A2_mm2 cannot be less than loaded area A1.")
    strength_factor = _positive("phi", phi)
    if strength_factor > 1.0:
        raise ValueError("phi cannot exceed 1.0.")

    limit_1 = 0.85 * fc * sqrt(a2 / a1)
    limit_2 = 2.0 * fc
    nominal = min(limit_1, limit_2)
    return {
        "bearing_limit_1_MPa": limit_1,
        "bearing_limit_2_MPa": limit_2,
        "nominal_bearing_stress_MPa": nominal,
        "phi": strength_factor,
        "design_bearing_stress_MPa": strength_factor * nominal,
    }


def check_concrete_bearing(
    *,
    factored_post_load_kN: Number,
    f_c_MPa: Number,
    loaded_area_A1_mm2: Number,
    supporting_area_A2_mm2: Optional[Number] = None,
    phi: Number = 0.6,
) -> CapacityCheckResult:
    """Check factored bearing stress under a post base plate."""

    load = _nonnegative("factored_post_load_kN", factored_post_load_kN)
    a1 = _positive("loaded_area_A1_mm2", loaded_area_A1_mm2)
    details = concrete_bearing_design_stress_MPa(
        f_c_MPa=f_c_MPa,
        loaded_area_A1_mm2=a1,
        supporting_area_A2_mm2=supporting_area_A2_mm2,
        phi=phi,
    )
    demand_stress = load * 1_000.0 / a1
    capacity_stress = details["design_bearing_stress_MPa"]
    utilization = demand_stress / capacity_stress
    return CapacityCheckResult(
        demand=demand_stress,
        capacity=capacity_stress,
        utilization=utilization,
        status="OK" if utilization <= 1.0 else "NOT OK",
        demand_unit="MPa",
        capacity_unit="MPa",
        details=details,
    )


def check_adopted_chart_thickness(
    *, adopted_thickness_mm: Number, required_from_chart_mm: Number
) -> CapacityCheckResult:
    """Record/check a thickness read independently from a T48 chart."""

    adopted = _positive("adopted_thickness_mm", adopted_thickness_mm)
    required = _positive("required_from_chart_mm", required_from_chart_mm)
    utilization = required / adopted
    return CapacityCheckResult(
        demand=required,
        capacity=adopted,
        utilization=utilization,
        status="OK" if adopted >= required else "NOT OK",
        demand_unit="mm required",
        capacity_unit="mm adopted",
        details={},
    )


def published_example_regression() -> Dict[str, float]:
    """Reproduce selected numerical results printed in T48 Appendix D.

    Two wheel-load paths are reported deliberately:

    * ``*_equation`` uses the unrounded k2 equation at N = 146,000.
    * ``*_guide`` follows Appendix D exactly, where k2 is adopted as 0.54
      from Table 1.17 and f_all is rounded to 2.41 MPa before F1/F2 are
      calculated.

    The distinction is important because a regression test should not hide
    differences caused solely by the guide's tabulation and intermediate
    rounding. The final thickness still has to be read from Charts 1.1--1.4.
    """

    # Combined wheel/post example, Appendix D3.
    wheel_k2_equation = load_repetition_factor_k2(146_000)
    wheel_fall_equation = design_tensile_strength_MPa(
        50.0,
        load_type=LoadType.WHEEL,
        k1=0.90,
        k2=wheel_k2_equation,
    )
    wheel_interior_equation = wheel_stress_factor(
        f_all_MPa=wheel_fall_equation,
        f_c_MPa=50.0,
        FE=1.33,
        FH=1.00,
        FS=1.00,
        location=LoadLocation.INTERIOR,
    ).stress_factor
    wheel_edge_equation = wheel_stress_factor(
        f_all_MPa=wheel_fall_equation,
        f_c_MPa=50.0,
        FE=1.41,
        FH=0.995,
        FS=1.00,
        location=LoadLocation.EDGE,
    ).stress_factor

    # Values explicitly adopted/rounded in the printed Appendix D example.
    wheel_k2_guide = 0.54
    wheel_fall_guide = 2.41
    wheel_interior_guide = wheel_stress_factor(
        f_all_MPa=wheel_fall_guide,
        f_c_MPa=50.0,
        FE=1.33,
        FH=1.00,
        FS=1.00,
        location=LoadLocation.INTERIOR,
    ).stress_factor
    wheel_edge_guide = wheel_stress_factor(
        f_all_MPa=wheel_fall_guide,
        f_c_MPa=50.0,
        FE=1.41,
        FH=0.995,
        FS=1.00,
        location=LoadLocation.EDGE,
    ).stress_factor

    post_fall = design_tensile_strength_MPa(
        50.0,
        load_type=LoadType.POST,
        k1=0.80,
        k2=1.0,
    )
    post_interior = post_stress_factor(
        f_all_MPa=post_fall,
        post_load_kN=70.0,
        FE=1.25,
        FH=1.01,
        FS=1.14,
        location=LoadLocation.INTERIOR,
    ).stress_factor
    post_edge = post_stress_factor(
        f_all_MPa=post_fall,
        post_load_kN=70.0,
        FE=1.18,
        FH=1.01,
        FS=1.14,
        location=LoadLocation.EDGE,
    ).stress_factor

    # Distributed-load example, Appendix D2.
    distributed_fall = design_tensile_strength_MPa(
        40.0,
        load_type=LoadType.DISTRIBUTED,
        k1=0.80,
        k2=load_repetition_factor_k2(1_000),
    )
    distributed_stack = distributed_stress_factor(
        f_all_MPa=distributed_fall,
        distributed_load_kPa=30.0,
        FE=1.22,
        FH=0.96,
        FW=0.88,
    ).stress_factor
    distributed_aisle = distributed_stress_factor(
        f_all_MPa=distributed_fall,
        distributed_load_kPa=30.0,
        FE=1.22,
        FH=0.96,
        FW=1.00,
    ).stress_factor

    punching = punching_shear_capacity_kN(
        f_c_MPa=50.0,
        slab_thickness_mm=150.0,
        base_plate_width_mm=165.0,
        phi=0.8,
    )
    bearing = concrete_bearing_design_stress_MPa(
        f_c_MPa=50.0,
        loaded_area_A1_mm2=165.0 * 165.0,
        phi=0.6,
    )

    fc1 = combined_factor_post_on_wheel(
        Q=0.85, post_load_kN=70.0, axle_load_kN=100.0
    )
    fc2 = combined_factor_post_on_wheel(
        Q=1.05, post_load_kN=70.0, axle_load_kN=100.0
    )
    fc3_interior = combined_factor_wheel_on_post(
        Q3=0.10, axle_load_kN=100.0, post_load_kN=70.0
    )
    fc3_edge = combined_factor_wheel_on_post(
        Q3=0.175, axle_load_kN=100.0, post_load_kN=70.0
    )

    return {
        "wheel_k2_equation": wheel_k2_equation,
        "wheel_k2_guide": wheel_k2_guide,
        "wheel_fall_equation_MPa": wheel_fall_equation,
        "wheel_fall_guide_MPa": wheel_fall_guide,
        "wheel_F1_equation": wheel_interior_equation,
        "wheel_F1_guide": wheel_interior_guide,
        "wheel_F2_equation": wheel_edge_equation,
        "wheel_F2_guide": wheel_edge_guide,
        "post_fall_MPa": post_fall,
        "post_F3_interior": post_interior,
        "post_F3_edge": post_edge,
        "distributed_fall_MPa": distributed_fall,
        "distributed_F4_stack": distributed_stack,
        "distributed_F4_aisle": distributed_aisle,
        "combined_FC1": fc1,
        "combined_FC2": fc2,
        "combined_FC3_interior": fc3_interior,
        "combined_FC3_edge": fc3_edge,
        "punching_design_capacity_kN": punching["design_capacity_kN"],
        "bearing_design_stress_MPa": bearing["design_bearing_stress_MPa"],
    }


if __name__ == "__main__":
    print("CCAA T48 published-example regression values")
    for name, value in published_example_regression().items():
        print(f"{name}: {value:.4f}")
