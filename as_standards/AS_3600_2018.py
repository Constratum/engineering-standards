"""
AS 3600:2018 concrete slab punching-shear and bearing functions.

Target documents
----------------
- AS 3600:2018, including Amendments 1:2018 and 2:2021
- AS 3600:2018 Supplement 1:2022 commentary

Scope
-----
This module provides AS 3600 equivalents for the slab-related calculation
style used in ``NZS_3101_1_22006.py``:

1. Punching shear of slabs around a support or concentrated load
   (AS 3600 Clause 9.3).
2. Concrete bearing surfaces under a plate or other loaded area
   (AS 3600 Clause 12.6).
3. Slab uplift resistance using Clause 3.1.1.3 flexural tensile strength
   ``f'ct.f = 0.6 * sqrt(f'c)``.

The module intentionally does not convert the NZS 3101 Chapter 17 anchor
functions. Anchor/fixing design in Australia is generally covered by
AS 5216 rather than AS 3600 Clause 12.6.

Units
-----
- Length and area: mm and mm^2
- Stress and concrete strength: MPa (= N/mm^2)
- Force: N
- Moment: N.mm

Important engineering limitations
---------------------------------
- The automatic critical-perimeter geometry assumes a rectangular loaded
  area with no nearby openings and, for edge/corner cases, the loaded area
  directly adjoins the free edge(s). Supply ``critical_perimeter`` when the
  actual AS 3600 Figure 9.3(A) perimeter differs.
- For non-zero moment transfer, Clause 9.3.4 depends on the actual torsion
  strip/spandrel geometry and reinforcement. This implementation includes
  Clause 9.3.4(1), i.e. no closed fitments in the torsion strip or spandrel
  beam. Other Clause 9.3.4 cases require project-specific inputs.
- Clause 12.6 does not apply where special confinement reinforcement is being
  relied upon or to nodes within a strut-and-tie model.
- This is a transparent calculation library, not a substitute for checking
  the current Standard, amendments, drawings, load combinations and detailing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np


LoadPosition = Literal["interior", "edge", "corner"]
PlateAxis = Literal["length", "width"]


# ============================================================================
# PUNCHING SHEAR FOR SLABS - AS 3600:2018 Clause 9.3
# ============================================================================


class SlabPunchingShearCapacity_9_3:
    """Calculate punching shear capacity in accordance with AS 3600 Clause 9.3.

    The constructor deliberately follows the format of
    ``SlabPunchingShearCapacity_12_7`` in the supplied NZS 3101 module.

    Parameters
    ----------
    thickness_slab:
        Mean effective depth around the critical shear perimeter, ``d_om``
        in mm. The legacy parameter name is retained for compatibility; this
        is not necessarily the total slab thickness.
    concrete_strength:
        Characteristic concrete cylinder strength, ``f'c`` in MPa.
    length_base_plate, width_base_plate:
        Overall dimensions of the effective rectangular loaded area in mm.
    position_load_area:
        ``"interior"``, ``"edge"`` or ``"corner"``.
    sigma_cp:
        Average effective prestress compression at the critical perimeter,
        ``sigma_cp`` in MPa. Use a positive value for compression.
    has_shear_head:
        Selects AS 3600 Equation 9.3.3(2). Otherwise Equation 9.3.3(1) is used.
    class_n_fitments:
        Used only to select the default capacity reduction factor. The default
        is phi = 0.75 when qualifying Class N fitments are present, otherwise
        phi = 0.70, in accordance with Table 2.2.2(e).
    closed_fitments_in_torsion_strip:
        Set to True where the torsion strip or spandrel beam contains closed
        fitments. For non-zero moment transfer, the present implementation then
        stops because Equations 9.3.4(2) to 9.3.4(4) require additional geometry
        and reinforcement inputs.
    strength_reduction_factor:
        Optional explicit phi override. Use only where the design basis has
        established the applicable Table 2.2.2 value.
    critical_perimeter:
        Optional explicit critical shear perimeter ``u`` in mm. This should be
        used for openings, irregular supports, spandrels, drop panels or other
        geometries not represented by the simple automatic model.
    edge_parallel_to:
        For ``position_load_area="edge"``, identifies which loaded-area
        dimension is parallel to the free edge.
    effective_loaded_length, effective_loaded_width:
        Optional dimensions used only to calculate ``beta_h``. When omitted,
        the base-plate/loaded-area dimensions are used.
    """

    def __init__(
        self,
        thickness_slab: float,
        concrete_strength: float,
        length_base_plate: float,
        width_base_plate: float,
        position_load_area: LoadPosition,
        sigma_cp: float = 0.0,
        has_shear_head: bool = False,
        class_n_fitments: bool = False,
        closed_fitments_in_torsion_strip: bool = False,
        strength_reduction_factor: Optional[float] = None,
        critical_perimeter: Optional[float] = None,
        edge_parallel_to: PlateAxis = "length",
        effective_loaded_length: Optional[float] = None,
        effective_loaded_width: Optional[float] = None,
    ) -> None:
        self.thickness_slab = float(thickness_slab)
        self.concrete_strength = float(concrete_strength)
        self.length_base_plate = float(length_base_plate)
        self.width_base_plate = float(width_base_plate)
        self.position_load_area = position_load_area
        self.sigma_cp = float(sigma_cp)
        self.has_shear_head = bool(has_shear_head)
        self.class_n_fitments = bool(class_n_fitments)
        self.closed_fitments_in_torsion_strip = bool(
            closed_fitments_in_torsion_strip
        )
        self.critical_perimeter = (
            None if critical_perimeter is None else float(critical_perimeter)
        )
        self.edge_parallel_to = edge_parallel_to
        self.effective_loaded_length = float(
            effective_loaded_length
            if effective_loaded_length is not None
            else length_base_plate
        )
        self.effective_loaded_width = float(
            effective_loaded_width
            if effective_loaded_width is not None
            else width_base_plate
        )

        if strength_reduction_factor is None:
            self.strength_reduction_factor = 0.75 if class_n_fitments else 0.70
        else:
            self.strength_reduction_factor = float(strength_reduction_factor)

        self._validate_inputs()

    @property
    def effective_depth(self) -> float:
        """Mean effective depth ``d_om`` in mm."""

        return self.thickness_slab

    def _validate_inputs(self) -> None:
        """Validate geometry, material and reduction-factor inputs."""

        valid_positions = {"interior", "edge", "corner"}
        if self.position_load_area not in valid_positions:
            raise ValueError(
                f"Invalid position_load_area: {self.position_load_area!r}. "
                f"Use one of {sorted(valid_positions)}."
            )

        if self.edge_parallel_to not in {"length", "width"}:
            raise ValueError("edge_parallel_to must be 'length' or 'width'.")

        positive_values = {
            "mean effective depth d_om": self.effective_depth,
            "concrete strength f'c": self.concrete_strength,
            "loaded-area length": self.length_base_plate,
            "loaded-area width": self.width_base_plate,
            "effective loaded length": self.effective_loaded_length,
            "effective loaded width": self.effective_loaded_width,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value.")

        if not np.isfinite(self.sigma_cp) or self.sigma_cp < 0:
            raise ValueError(
                "sigma_cp must be a finite non-negative compression stress in MPa."
            )

        if self.critical_perimeter is not None and (
            not np.isfinite(self.critical_perimeter)
            or self.critical_perimeter <= 0
        ):
            raise ValueError("critical_perimeter must be positive when supplied.")

        if not (0 < self.strength_reduction_factor <= 1.0):
            raise ValueError("strength_reduction_factor must be in the range (0, 1].")

    def _calculate_beta_h(self) -> float:
        """Calculate ``beta_h`` from the effective loaded-area dimensions.

        AS 3600 Clause 9.3.1 defines beta_h as the longest overall dimension
        divided by the perpendicular overall dimension.
        """

        long_side = max(self.effective_loaded_length, self.effective_loaded_width)
        short_side = min(self.effective_loaded_length, self.effective_loaded_width)
        return long_side / short_side

    def _calculate_perimeter_critical_section(self) -> float:
        """Calculate the critical shear perimeter ``u`` in mm.

        The perimeter is located ``d_om/2`` from the loaded-area faces. The
        automatic geometry is for a rectangular loaded area with no openings.
        An explicit ``critical_perimeter`` overrides this calculation.
        """

        if self.critical_perimeter is not None:
            return self.critical_perimeter

        d = self.effective_depth
        length = self.length_base_plate
        width = self.width_base_plate

        if self.position_load_area == "interior":
            # Closed rectangle at d/2 from each face:
            # dimensions = (length + d) by (width + d)
            return 2.0 * ((length + d) + (width + d))

        if self.position_load_area == "edge":
            # Three-sided perimeter for a loaded area adjoining one free edge.
            if self.edge_parallel_to == "length":
                parallel = length
                perpendicular = width
            else:
                parallel = width
                perpendicular = length
            return (parallel + d) + 2.0 * (perpendicular + d / 2.0)

        # Corner: two-sided perimeter for a loaded area adjoining two free edges.
        return (length + d / 2.0) + (width + d / 2.0)

    def calculate_fcv(self) -> float:
        """Calculate concrete punching-shear stress ``f_cv`` in MPa.

        AS 3600 Equation 9.3.3(1):

        ``f_cv = min[0.17(1 + 2/beta_h)sqrt(f'c), 0.34sqrt(f'c)]``

        This value is used when there is no shear head.
        """

        beta_h = self._calculate_beta_h()
        sqrt_fc = np.sqrt(self.concrete_strength)
        return float(
            min(0.17 * (1.0 + 2.0 / beta_h) * sqrt_fc, 0.34 * sqrt_fc)
        )

    def calculate_vc(self) -> float:
        """Calculate the average nominal punching-shear stress in MPa.

        This compatibility method corresponds to the stress term multiplying
        ``u*d_om`` in AS 3600 Equation 9.3.3(1) or 9.3.3(2).
        """

        fc = self.concrete_strength
        if self.has_shear_head:
            stress_from_equation = 0.5 * np.sqrt(fc) + 0.3 * self.sigma_cp
            web_crushing_limit = 0.2 * fc
            return float(min(stress_from_equation, web_crushing_limit))

        return float(self.calculate_fcv() + 0.3 * self.sigma_cp)

    def calculate_vc_max(self) -> float:
        """Return the governing stress cap used by Clause 9.3.3 in MPa.

        For a shear head this is the explicit ``0.2 f'c`` web-crushing limit.
        Without a shear head it is the ``0.34 sqrt(f'c)`` cap on ``f_cv``, plus
        the permitted prestress term.
        """

        if self.has_shear_head:
            return float(0.2 * self.concrete_strength)
        return float(0.34 * np.sqrt(self.concrete_strength) + 0.3 * self.sigma_cp)

    def calculate_Vuo(self) -> float:
        """Calculate ultimate punching-shear strength ``V_uo`` in N.

        Implements AS 3600 Equation 9.3.3(1) or 9.3.3(2), as selected by
        ``has_shear_head``.
        """

        u = self._calculate_perimeter_critical_section()
        d_om = self.effective_depth
        return float(u * d_om * self.calculate_vc())

    def calculate_Vc(self) -> float:
        """Compatibility alias for ``calculate_Vuo()``; result is in N."""

        return self.calculate_Vuo()

    def calculate_Vu(
        self,
        moment_transfer: float = 0.0,
        design_shear: Optional[float] = None,
        dimension_a: Optional[float] = None,
    ) -> float:
        """Calculate ultimate punching-shear strength ``V_u`` in N.

        When ``moment_transfer`` is zero, ``V_u = V_uo``.

        For non-zero moment transfer, this method implements AS 3600
        Equation 9.3.4(1), applicable where there are no closed fitments in
        the torsion strip or spandrel beam:

        ``V_u = V_uo / [1 + u*M_v*/(8*V* * a * d_om)]``

        Parameters
        ----------
        moment_transfer:
            Magnitude of ``M_v*`` in N.mm.
        design_shear:
            Magnitude of ``V*`` in N.
        dimension_a:
            Dimension of the critical shear perimeter measured parallel to
            the direction of ``M_v*``, in mm.
        """

        Vuo = self.calculate_Vuo()
        Mv = abs(float(moment_transfer))
        if np.isclose(Mv, 0.0):
            return Vuo

        if self.closed_fitments_in_torsion_strip:
            raise NotImplementedError(
                "Non-zero moment transfer with closed fitments requires the "
                "additional inputs in AS 3600 Equations 9.3.4(2) to 9.3.4(4)."
            )

        if design_shear is None or dimension_a is None:
            raise ValueError(
                "design_shear and dimension_a are required when moment_transfer is non-zero."
            )

        V_star = abs(float(design_shear))
        a = float(dimension_a)
        if not np.isfinite(V_star) or V_star <= 0:
            raise ValueError("design_shear must be a positive finite force in N.")
        if not np.isfinite(a) or a <= 0:
            raise ValueError("dimension_a must be a positive finite length in mm.")

        u = self._calculate_perimeter_critical_section()
        d_om = self.effective_depth
        denominator = 1.0 + u * Mv / (8.0 * V_star * a * d_om)
        return float(Vuo / denominator)

    def calculate_design_Vu(
        self,
        moment_transfer: float = 0.0,
        design_shear: Optional[float] = None,
        dimension_a: Optional[float] = None,
    ) -> float:
        """Calculate design punching-shear strength ``phi*V_u`` in N."""

        return float(
            self.strength_reduction_factor
            * self.calculate_Vu(moment_transfer, design_shear, dimension_a)
        )

    def calculate_design_Vc(self) -> float:
        """Compatibility alias for design strength with zero moment transfer."""

        return self.calculate_design_Vu()

    def check_capacity(
        self,
        applied_shear: float,
        moment_transfer: float = 0.0,
        dimension_a: Optional[float] = None,
    ) -> Dict[str, float | bool | str]:
        """Check an applied ultimate shear force against ``phi*V_u``.

        ``applied_shear`` is also used as ``V*`` in Equation 9.3.4(1) where
        ``moment_transfer`` is non-zero.
        """

        V_star = abs(float(applied_shear))
        if not np.isfinite(V_star) or V_star < 0:
            raise ValueError("applied_shear must be a finite force in N.")

        design_capacity = self.calculate_design_Vu(
            moment_transfer=moment_transfer,
            design_shear=V_star if not np.isclose(moment_transfer, 0.0) else None,
            dimension_a=dimension_a,
        )
        utilization = V_star / design_capacity if design_capacity > 0 else np.inf
        compliant = bool(utilization <= 1.0)

        return {
            "critical_perimeter_mm": self._calculate_perimeter_critical_section(),
            "beta_h": self._calculate_beta_h(),
            "fcv_MPa": self.calculate_fcv(),
            "nominal_shear_stress_MPa": self.calculate_vc(),
            "Vuo_N": self.calculate_Vuo(),
            "Vu_N": self.calculate_Vu(
                moment_transfer=moment_transfer,
                design_shear=V_star if not np.isclose(moment_transfer, 0.0) else None,
                dimension_a=dimension_a,
            ),
            "phi": self.strength_reduction_factor,
            "design_capacity_N": design_capacity,
            "applied_shear_N": V_star,
            "utilization": float(utilization),
            "compliant": compliant,
            "status": "PASS" if compliant else "FAIL",
        }


# ============================================================================
# CONCRETE BEARING SURFACES - AS 3600:2018 Clause 12.6
# ============================================================================


@dataclass(frozen=True)
class BearingSurfaceProperties:
    """Properties required for the AS 3600 Clause 12.6 bearing check."""

    concrete_strength: float  # f'c, MPa
    bearing_area_A1: float  # loaded bearing area, mm^2
    supporting_area_A2: float  # similar and concentric supporting area, mm^2
    strength_reduction_factor: float = 0.60  # Table 2.2.2(f)
    special_confinement_reinforcement: bool = False


class ConcreteBearingCapacity_12_6:
    """Calculate concrete bearing capacity under AS 3600 Clause 12.6.

    Unless special confinement reinforcement is provided, the design bearing
    stress is the lesser of:

    ``phi * 0.9 f'c sqrt(A2/A1)`` and ``phi * 1.8 f'c``.

    The area ratio contributing to the enhancement is therefore effectively
    limited to ``A2/A1 = 4``.
    """

    def __init__(
        self,
        concrete_strength: float,
        bearing_area_A1: float,
        supporting_area_A2: float,
        strength_reduction_factor: float = 0.60,
        special_confinement_reinforcement: bool = False,
    ) -> None:
        self.props = BearingSurfaceProperties(
            concrete_strength=float(concrete_strength),
            bearing_area_A1=float(bearing_area_A1),
            supporting_area_A2=float(supporting_area_A2),
            strength_reduction_factor=float(strength_reduction_factor),
            special_confinement_reinforcement=bool(
                special_confinement_reinforcement
            ),
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        p = self.props
        for name, value in {
            "concrete_strength": p.concrete_strength,
            "bearing_area_A1": p.bearing_area_A1,
            "supporting_area_A2": p.supporting_area_A2,
        }.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value.")

        if p.supporting_area_A2 < p.bearing_area_A1:
            raise ValueError(
                "supporting_area_A2 must not be smaller than bearing_area_A1."
            )

        if not (0 < p.strength_reduction_factor <= 1.0):
            raise ValueError("strength_reduction_factor must be in the range (0, 1].")

    def _check_clause_scope(self) -> None:
        if self.props.special_confinement_reinforcement:
            raise NotImplementedError(
                "AS 3600 Clause 12.6's unconfined bearing expression is not "
                "sufficient where special confinement reinforcement is relied upon."
            )

    def calculate_area_ratio(self) -> float:
        """Return the actual geometric ratio ``A2/A1``."""

        return self.props.supporting_area_A2 / self.props.bearing_area_A1

    def calculate_effective_area_ratio(self) -> float:
        """Return ``A2/A1`` limited to 4 for the Clause 12.6 enhancement."""

        return float(min(self.calculate_area_ratio(), 4.0))

    def calculate_confinement_factor(self) -> float:
        """Return ``sqrt(A2/A1)`` limited to 2.0."""

        return float(np.sqrt(self.calculate_effective_area_ratio()))

    def calculate_nominal_bearing_stress(self) -> float:
        """Calculate nominal bearing stress before ``phi`` in MPa."""

        self._check_clause_scope()
        fc = self.props.concrete_strength
        enhanced = 0.9 * fc * self.calculate_confinement_factor()
        upper_limit = 1.8 * fc
        return float(min(enhanced, upper_limit))

    def calculate_design_bearing_stress(self) -> float:
        """Calculate design bearing stress in MPa."""

        return float(
            self.props.strength_reduction_factor
            * self.calculate_nominal_bearing_stress()
        )

    def calculate_nominal_bearing_capacity(self) -> float:
        """Calculate nominal bearing capacity in N."""

        return float(
            self.calculate_nominal_bearing_stress() * self.props.bearing_area_A1
        )

    def calculate_design_bearing_capacity(self) -> float:
        """Calculate design bearing capacity ``phi*R_u`` in N."""

        return float(
            self.calculate_design_bearing_stress() * self.props.bearing_area_A1
        )

    def check_capacity(self, applied_bearing_force: float) -> Dict[str, float | bool | str]:
        """Check an applied ultimate compressive force against design capacity."""

        force = abs(float(applied_bearing_force))
        if not np.isfinite(force):
            raise ValueError("applied_bearing_force must be finite.")

        capacity = self.calculate_design_bearing_capacity()
        applied_stress = force / self.props.bearing_area_A1
        utilization = force / capacity if capacity > 0 else np.inf
        compliant = bool(utilization <= 1.0)

        return {
            "A1_mm2": self.props.bearing_area_A1,
            "A2_mm2": self.props.supporting_area_A2,
            "A2_A1_actual": self.calculate_area_ratio(),
            "A2_A1_effective": self.calculate_effective_area_ratio(),
            "confinement_factor": self.calculate_confinement_factor(),
            "nominal_bearing_stress_MPa": self.calculate_nominal_bearing_stress(),
            "phi": self.props.strength_reduction_factor,
            "design_bearing_stress_MPa": self.calculate_design_bearing_stress(),
            "design_capacity_N": capacity,
            "applied_force_N": force,
            "applied_bearing_stress_MPa": applied_stress,
            "utilization": float(utilization),
            "compliant": compliant,
            "status": "PASS" if compliant else "FAIL",
        }

    @classmethod
    def from_rectangular_geometry(
        cls,
        concrete_strength: float,
        bearing_length: float,
        bearing_width: float,
        supporting_length: float,
        supporting_width: float,
        support_depth: Optional[float] = None,
        strength_reduction_factor: float = 0.60,
        special_confinement_reinforcement: bool = False,
    ) -> "ConcreteBearingCapacity_12_6":
        """Construct a bearing check from concentric rectangular geometry.

        ``A2`` is calculated as the largest rectangle geometrically similar to
        and concentric with ``A1`` that fits inside the supporting surface.

        When ``support_depth`` is supplied, the additional Clause 12.6 frustum
        limit is included using side slopes of 1 longitudinal to 2 transverse;
        this corresponds to a maximum horizontal spread of ``2*depth`` on each
        side. The method assumes the loaded area is centred.
        """

        values = {
            "bearing_length": bearing_length,
            "bearing_width": bearing_width,
            "supporting_length": supporting_length,
            "supporting_width": supporting_width,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value.")

        if supporting_length < bearing_length or supporting_width < bearing_width:
            raise ValueError(
                "The supporting surface must be at least as large as the bearing area."
            )

        scale_from_support = min(
            supporting_length / bearing_length,
            supporting_width / bearing_width,
        )
        scale = scale_from_support

        if support_depth is not None:
            if not np.isfinite(support_depth) or support_depth < 0:
                raise ValueError("support_depth must be a non-negative finite value.")
            projected_length = bearing_length + 4.0 * support_depth
            projected_width = bearing_width + 4.0 * support_depth
            scale_from_frustum = min(
                projected_length / bearing_length,
                projected_width / bearing_width,
            )
            scale = min(scale, scale_from_frustum)

        A1 = bearing_length * bearing_width
        A2 = A1 * scale**2

        return cls(
            concrete_strength=concrete_strength,
            bearing_area_A1=A1,
            supporting_area_A2=A2,
            strength_reduction_factor=strength_reduction_factor,
            special_confinement_reinforcement=special_confinement_reinforcement,
        )


class SlabUpliftResistance:
    """Resisting slab weight and cracking length for a concentrated uplift.

    Flexural tensile strength uses AS 3600:2018 Clause 3.1.1.3:

    ``f'ct.f = 0.6 * sqrt(f'c)``
    """

    def __init__(
        self,
        thickness_slab: float,
        concrete_strength: float,
        concrete_density: float,
        edge_distance_top_m: float,
        edge_distance_bot_m: float,
        edge_distance_lhs_m: float,
        edge_distance_rhs_m: float,
        strength_reduction_factor: float,
    ) -> None:
        if strength_reduction_factor is None:
            raise ValueError("strength_reduction_factor is required.")
        self.thickness_slab = thickness_slab
        self.concrete_strength = concrete_strength
        self.concrete_density = concrete_density
        self.edge_distance_top_m = edge_distance_top_m
        self.edge_distance_bot_m = edge_distance_bot_m
        self.edge_distance_lhs_m = edge_distance_lhs_m
        self.edge_distance_rhs_m = edge_distance_rhs_m
        self.strength_reduction_factor = float(strength_reduction_factor)
        self.g = 9.81

    def calculate_weight_of_slab_per_width(self) -> float:
        thickness_m = self.thickness_slab / 1000
        return thickness_m * self.concrete_density * self.g

    def calculate_moment_of_inertia(self) -> float:
        b = 1000
        return (b * self.thickness_slab**3) / 12

    def calculate_modulus_of_rupture(self) -> float:
        """AS 3600 Clause 3.1.1.3 characteristic flexural tensile strength, MPa."""
        return 0.6 * np.sqrt(self.concrete_strength)

    def calculate_cracking_moment(self) -> float:
        fr = self.calculate_modulus_of_rupture()
        I = self.calculate_moment_of_inertia()
        y = self.thickness_slab / 2
        Mcr = (fr * I) / y
        return Mcr / 1000

    def calculate_cracking_length(self) -> float:
        Mcr = self.calculate_cracking_moment()
        w = self.calculate_weight_of_slab_per_width()
        if w <= 0:
            raise ValueError("Weight per unit width must be positive.")
        return np.sqrt(2 * Mcr / w) * self.strength_reduction_factor

    def calculate_resisting_slab_weight(self) -> float:
        thickness_m = self.thickness_slab / 1000
        return (
            self.concrete_density
            * self.g
            * thickness_m
            * (self.edge_distance_top_m + self.edge_distance_bot_m)
            * (self.edge_distance_lhs_m + self.edge_distance_rhs_m)
        )


# Backward-friendly aliases for application wiring.
SlabPunchingShearCapacity_AS3600 = SlabPunchingShearCapacity_9_3
BearingCapacity_12_6 = ConcreteBearingCapacity_12_6


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def calculate_slab_punching_capacity(
    thickness_slab: float,
    concrete_strength: float,
    length_base_plate: float,
    width_base_plate: float,
    position_load_area: LoadPosition,
    sigma_cp: float = 0.0,
    has_shear_head: bool = False,
    class_n_fitments: bool = False,
    closed_fitments_in_torsion_strip: bool = False,
    strength_reduction_factor: Optional[float] = None,
    critical_perimeter: Optional[float] = None,
    edge_parallel_to: PlateAxis = "length",
    moment_transfer: float = 0.0,
    design_shear: Optional[float] = None,
    dimension_a: Optional[float] = None,
) -> float:
    """Return the AS 3600 design punching-shear capacity ``phi*V_u`` in N."""

    calculator = SlabPunchingShearCapacity_9_3(
        thickness_slab=thickness_slab,
        concrete_strength=concrete_strength,
        length_base_plate=length_base_plate,
        width_base_plate=width_base_plate,
        position_load_area=position_load_area,
        sigma_cp=sigma_cp,
        has_shear_head=has_shear_head,
        class_n_fitments=class_n_fitments,
        closed_fitments_in_torsion_strip=closed_fitments_in_torsion_strip,
        strength_reduction_factor=strength_reduction_factor,
        critical_perimeter=critical_perimeter,
        edge_parallel_to=edge_parallel_to,
    )
    return calculator.calculate_design_Vu(
        moment_transfer=moment_transfer,
        design_shear=design_shear,
        dimension_a=dimension_a,
    )


def calculate_concrete_bearing_capacity(
    concrete_strength: float,
    bearing_area_A1: float,
    supporting_area_A2: float,
    strength_reduction_factor: float = 0.60,
) -> float:
    """Return the AS 3600 Clause 12.6 design bearing capacity in N."""

    calculator = ConcreteBearingCapacity_12_6(
        concrete_strength=concrete_strength,
        bearing_area_A1=bearing_area_A1,
        supporting_area_A2=supporting_area_A2,
        strength_reduction_factor=strength_reduction_factor,
    )
    return calculator.calculate_design_bearing_capacity()
