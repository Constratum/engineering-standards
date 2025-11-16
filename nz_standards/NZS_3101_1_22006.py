from dataclasses import dataclass
from typing import Optional, List, Union
import numpy as np
import scipy.interpolate


@dataclass
class AnchorProperties:
    """Properties of an anchor required for capacity calculations"""

    thickness_concrete: float  # Thickness of concrete (mm)
    diameter: float  # Outside diameter of anchor shaft (mm)
    head_diameter: float  # Outside diameter of anchor head (mm)
    effective_embedment_depth: float  # hef (mm)
    edge_distance: float  # Distance to nearest edge (mm)
    eccentricity: float  # Eccentricity of anchor group (mm)
    edge_distance_perp: Optional[float]  # Distance to perpendicular edge (mm)
    spacing: Optional[float]  # Center-to-center spacing of anchors (mm)
    number_of_anchors: int  # Number of anchors in group
    ultimate_tensile_strength: float  # Ultimate tensile strength (MPa)
    concrete_strength: float  # Specified compressive strength of concrete (MPa)
    is_cracked_concrete: bool  # Whether concrete is cracked at service load levels
    length_hook: Optional[
        float
    ]  # Distance from inner surface to outer tip of hooked bolt (mm)
    anchor_type: str  # Type of anchor: "headed_stud", "headed_bolt", or "hooked_bolt"
    shear_prep: bool


class AnchorCapacity_Chapter_17:
    """Calculate anchor capacities according to NZS 3101:Part 1:2006"""

    def __init__(self, anchor: AnchorProperties):
        self.anchor = anchor

    def effective_area(self) -> float:
        """Calculate effective area of anchor"""
        return (np.pi * self.anchor.diameter**2) / 4

    def head_bearing_area(self) -> float:
        """Calculate head bearing area of anchor"""
        return (np.pi * (self.anchor.head_diameter**2 - self.anchor.diameter**2)) / 4

    def _calc_anchor_tension_strength_17_5_7(self) -> float:
        """Calculate anchor tension strength"""

        if (
            self.anchor.edge_distance >= 0.4 * self.anchor.effective_embedment_depth
            or self.anchor.anchor_type not in ["headed_stud", "headed_bolt"]
        ):
            Nn = min(
                self.steel_strength_tensio_17_5_7_1(),
                self.concrete_breakout_tension_17_5_7_2(),
                self.tension_pullout_strength_17_5_7_3(),
            )
        else:
            Nn = min(
                self.steel_strength_tensio_17_5_7_1(),
                self.concrete_breakout_tension_17_5_7_2(),
                self.tension_pullout_strength_17_5_7_3(),
                self.side_face_blowout_strength_17_5_7_4(),
            )

        return Nn

    def steel_strength_tensio_17_5_7_1(self) -> float:
        """Calculate steel strength of anchor in tension (N) - Section 17.5.7.1"""
        phi = 0.75  # Steel strength reduction factor
        Ns = phi * (
            self.anchor.number_of_anchors
            * self.effective_area()
            * self.anchor.ultimate_tensile_strength
        )
        return Ns

    def concrete_breakout_tension_17_5_7_2(self) -> float:
        """Calculate concrete breakout strength in tension (N) - Section 17.5.7.2"""
        # Basic concrete breakout strength
        phi = 0.65  # Concrete strength reduction factor
        k = 10  # For cast-in anchors
        fc = min(self.anchor.concrete_strength, 70)  # Limited to 70 MPa per standard
        hef = self.anchor.effective_embedment_depth
        if hef < 280:
            Nb = k * np.sqrt(fc) * hef**1.5
        else:
            Nb = min(k * np.sqrt(fc) * hef**1.5, 3.9 * np.sqrt(fc) * hef**1.67)

        # Modification factors
        psi1 = self._get_anchor_group_modification_factor()
        psi2 = self._get_edge_distance_modification_factor()
        psi3 = 1.25 if not self.anchor.is_cracked_concrete else 1.0

        # Calculate projected areas
        ANc = self._calculate_projected_concrete_area()
        ANco = 9 * hef**2  # Projected area for single anchor

        return phi * (ANc / ANco) * psi1 * psi2 * psi3 * Nb

    def _get_anchor_group_modification_factor(self) -> float:
        """Calculate modification factor for anchor groups (psi1)"""
        if self.anchor.number_of_anchors == 1:
            return 1.0

        if not self.anchor.spacing:
            raise ValueError("Spacing must be provided for anchor groups")

        e_N = 0  # Assume zero eccentricity for now
        return min(1.0, 1 / (1 + 2 * e_N / (3 * self.anchor.effective_embedment_depth)))

    def _get_edge_distance_modification_factor(self) -> float:
        """Calculate modification factor for edge distance (psi2)"""
        if not self.anchor.edge_distance:
            return 1.0

        cmin = self.anchor.edge_distance
        if cmin >= 1.5 * self.anchor.effective_embedment_depth:
            return 1.0
        else:
            return 0.7 + 0.3 * cmin / (1.5 * self.anchor.effective_embedment_depth)

    def _calculate_projected_concrete_area(self) -> float:
        """Calculate projected concrete failure area ANc according to Figure C17.2(b)

        Three cases are considered:
        1. Single anchor with one edge distance < 1.5hef
        2. Two anchors in a row with edge distance < 1.5hef and spacing < 3hef
        3. Four anchors in a group with both edge distances < 1.5hef and both spacings < 3hef
        """
        hef = self.anchor.effective_embedment_depth
        c1 = self.anchor.edge_distance
        c2 = (
            self.anchor.edge_distance_perp
            if self.anchor.edge_distance_perp
            else 1.5 * hef
        )
        s1 = self.anchor.spacing if self.anchor.spacing else 3 * hef
        s2 = s1  # Assuming equal spacing in both directions for 4-anchor group

        # Case 1: Single anchor with one edge distance < 1.5hef
        if self.anchor.number_of_anchors == 1:
            if c1 < 1.5 * hef:
                return (c1 + 1.5 * hef) * (2 * 1.5 * hef)
            else:
                return 9 * hef**2  # Standard case for single anchor

        # Case 2: Two anchors in a row
        elif self.anchor.number_of_anchors == 2:
            if c1 < 1.5 * hef and s1 < 3 * hef:
                return (c1 + s1 + 1.5 * hef) * (2 * 1.5 * hef)
            else:
                return 2 * 9 * hef**2  # Two independent anchors

        # Case 3: Four anchors in a group
        elif self.anchor.number_of_anchors == 4:
            if c1 < 1.5 * hef and c2 < 1.5 * hef and s1 < 3 * hef and s2 < 3 * hef:
                return (c1 + s1 + 1.5 * hef) * (c2 + s2 + 1.5 * hef)
            else:
                return 4 * 9 * hef**2  # Four independent anchors

        else:
            raise ValueError(
                "Number of anchors must be 1, 2, or 4 for this implementation"
            )

    def tension_pullout_strength_17_5_7_3(self) -> float:
        """Calculate lower characteristic tension pullout strength (N) - Section 17.5.7.3"""
        # Get modification factor psi4 based on cracking condition
        phi = 0.75  # Pullout strength reduction factor
        psi4 = 1.0 if self.anchor.is_cracked_concrete else 1.4

        # Calculate basic pullout strength Np based on anchor type
        if self.anchor.anchor_type in ["headed_stud", "headed_bolt"]:
            # Equation 17-11
            Np = phi * 8 * self.anchor.concrete_strength * self.head_bearing_area()
        elif self.anchor.anchor_type == "hooked_bolt":
            # Check if hook length is within valid range
            if not self.anchor.length_hook:
                raise ValueError("Hook length must be provided for hooked bolts")
            eh = self.anchor.length_hook
            do = self.anchor.diameter
            if not (3 * do <= eh <= 4.5 * do):
                raise ValueError("Hook length must be between 3do and 4.5do")
            # Equation 17-12
            Np = phi * 0.9 * self.anchor.concrete_strength * eh * do
        else:
            raise ValueError("Invalid anchor type")

        # Equation 17-10
        return psi4 * Np

    def side_face_blowout_strength_17_5_7_4(self) -> float:
        """Calculate lower characteristic concrete side face blowout strength (N) - Section 17.5.7.4"""
        phi = 0.65  # Side face blowout strength reduction factor
        # Check if anchor is close to edge

        # Equation 17-13
        k1 = self._get_edge_distance_multiplier()
        return (
            phi
            * 13.3
            * k1
            * np.sqrt(self.anchor.concrete_strength * self.head_bearing_area())
        )

    def _get_edge_distance_multiplier(self) -> float:
        """Calculate multiplier for edge distance (k1)"""
        if not self.anchor.edge_distance_perp:
            return 1.0

        c1 = self.anchor.edge_distance
        c2 = self.anchor.edge_distance_perp

        if c2 >= 3 * c1:
            return 1.0
        else:
            return (1 + c2 / c1) / 4

    def _calc_anchor_shear_strength_17_5_8(self) -> float:
        """Calculate anchor shear strength"""
        return min(
            self.steel_strength_shear_17_5_8_1(),
            self.concrete_breakout_shear_17_5_8_2(),
            self.concrete_pryout_strength_17_5_8_4(),
        )

    def steel_strength_shear_17_5_8_1(self) -> float:
        """Calculate steel strength of anchor in shear (N) - Section 17.5.8.1"""
        # For cast-in headed studs
        phi = 0.65  # Steel strength reduction factor
        if self.anchor.anchor_type == "headed_stud":
            Vs = (
                phi
                * self.anchor.number_of_anchors
                * self.effective_area()
                * self.anchor.ultimate_tensile_strength
            )
        elif self.anchor.anchor_type in ["headed_bolt", "hooked_bolt"]:
            Vs = (
                phi
                * 0.6
                * self.anchor.number_of_anchors
                * self.effective_area()
                * self.anchor.ultimate_tensile_strength
            )
        return Vs

    def concrete_breakout_shear_17_5_8_2(self) -> float:
        """Calculate concrete breakout strength in shear (N) - Section 17.5.8.2 and 17.5.8.3

        Implements equations 17-16 through 17-21 for both perpendicular and parallel to edge cases.
        For perpendicular to edge: Vcb = (Av/Avo) * psi5 * psi6 * psi7 * Vb
        For parallel to edge: Vcb = 2(Av/Avo) * psi5 * psi7 * Vb
        """
        phi = 0.65  # Concrete strength reduction factor
        # Basic concrete breakout strength calculation
        k2 = 0.6  # For cast-in headed studs, headed bolts, or hooked bolts
        le = (
            0.8 * self.anchor.effective_embedment_depth
        )  # Effective length for constant stiffness
        do = self.anchor.diameter
        c1 = self.anchor.edge_distance
        fc = self.anchor.concrete_strength

        # Calculate basic concrete breakout strength (Vb) - Equation 17-17
        Vb = 3.84 * np.sqrt(fc) * (le / do) ** 0.2 * np.sqrt(do) * c1**1.5

        # Calculate projected areas
        Av = self._calculate_projected_shear_area()
        Avo = 4.5 * c1**2
        # Calculate modification factor for anchor groups (psi5) - Equation 17-18
        ev = self.anchor.eccentricity
        if self.anchor.spacing:
            s = self.anchor.spacing
            if ev < s / 2:
                psi5 = 1.0 / (1 + 2 * ev / (3 * c1))
            else:
                psi5 = 1.0
        else:
            psi5 = 1.0

        # Calculate modification factor for edge effects (psi6) - Equations 17-19 and 17-20
        if self.anchor.edge_distance_perp:
            c2 = self.anchor.edge_distance_perp
            if c2 >= 1.5 * c1:
                psi6 = 1.0
            else:
                psi6 = 0.7 + 0.3 * c2 / (1.5 * c1)
        else:
            psi6 = 1.0

        # Calculate modification factor for cracked concrete (psi7)
        if not self.anchor.is_cracked_concrete:
            psi7 = 1.4
        else:
            # Assuming no supplementary reinforcement or < 12mm diameter
            psi7 = 1.0

        # Calculate final strength based on loading direction
        if self.anchor.shear_prep:  # Perpendicular to edge
            return phi * (Av / Avo) * psi5 * psi6 * psi7 * Vb
        else:  # Parallel to edge
            return phi * (
                2 * (Av / Avo) * psi5 * psi7 * Vb
            )  # Note: psi6 not used for parallel case

    def _calculate_projected_shear_area(self) -> float:
        """Calculate projected failure area for shear (Av) according to Figure 17.2

        The projected area depends on:
        - Edge distance ca1
        - Perpendicular edge distance ca2 (if applicable)
        - Member thickness h
        - Whether anchor is near one edge or two edges
        """
        ca1 = self.anchor.edge_distance
        ca2 = self.anchor.edge_distance_perp if self.anchor.edge_distance_perp else None
        h = self.anchor.thickness_concrete  # Using hef as member thickness

        # Base case - single anchor not near edge (1.5ca1 spacing)
        if self.anchor.number_of_anchors == 1:
            if h < 1.5 * ca1:
                return 2 * 1.5 * ca1 * h
            else:
                if ca2 < 1.5 * ca1:
                    return 1.5 * ca1 * (1.5 * ca1 + ca2)
                else:
                    return 2 * 1.5 * ca1 * h

        # For groups of anchors
        elif self.anchor.number_of_anchors > 1:
            if not self.anchor.spacing:
                raise ValueError("Spacing must be provided for anchor groups")

            s = self.anchor.spacing

            # For anchors in a row parallel to edge
            if self.anchor.shear_prep:
                if h < 1.5 * ca1 and s < 3 * ca1:
                    return (2 * (1.5 * ca1) + s) * h
                else:
                    return 2 * (1.5 * ca1) ** 2
            if not self.anchor.shear_prep:
                if h < 1.5 * ca1:
                    return 2 * 1.5 * ca1 * h
                else:
                    return 2 * (1.5 * ca1) ** 2
        else:
            raise ValueError("Number of anchors must be positive")

    def concrete_pryout_strength_17_5_8_4(self) -> float:
        """Calculate lower characteristic concrete pry-out strength (N) - Section 17.5.8.4

        Implements equation 17-22: Vcp = kcp * Ncb
        where:
        Vcp = lower characteristic concrete pry-out strength
        Ncb = nominal concrete breakout strength in tension
        kcp = coefficient of pry-out strength (1.0 for hef < 65mm, 2.0 for hef ≥ 65mm)
        """
        phi = 0.65  # Concrete strength reduction factor
        # Get nominal concrete breakout strength in tension
        Ncb = self.concrete_breakout_tension_17_5_7_2()

        # Determine kcp based on effective embedment depth
        hef = self.anchor.effective_embedment_depth
        if hef < 65:
            kcp = 1.0
        else:
            kcp = 2.0

        # Calculate pry-out strength (Equation 17-22)
        return phi * kcp * Ncb

    def calc_interaction_tension_shear_17_5_6(
        self, N: float, V: float
    ) -> tuple[bool, float]:
        """Calculate interaction between tension and shear - Section 17.5.6

        Args:
            N: Applied tension force (N)
            V: Applied shear force (N)

        Returns:
            tuple containing:
            - compliance: True if interaction equation ≤ 1.2, False otherwise
            - unity: Value of interaction equation (N/Nt + V/Ns)
        """
        # Calculate capacities
        Nt = self._calc_anchor_tension_strength_17_5_7()
        Ns = self._calc_anchor_shear_strength_17_5_8()

        # Check interaction equation
        unity = (N / Nt) + (V / Ns)
        compliance = unity <= 1.2

        return compliance, unity


# ============================================================================
# PUNCHING SHEAR FOR SLABS - Section 12.7 (Two-way action)
# ============================================================================


class SlabPunchingShearCapacity_12_7:
    """Calculate punching shear capacity for slabs according to NZS 3101:Part 1:2006 Section 12.7"""

    def __init__(
        self,
        thickness_slab: float,
        concrete_strength: float,
        length_base_plate: float,
        width_base_plate: float,
        position_load_area: str,
    ):
        """
        Initialize punching shear calculator with slab properties.

        Args:
            thickness_slab: Effective depth of slab (d) in mm
            concrete_strength: Specified compressive strength of concrete (f'c) in MPa
            length_base_plate: Length of loaded area in mm
            width_base_plate: Width of loaded area in mm
            position_load_area: Position of load: "interior", "edge", or "corner"
        """
        self.thickness_slab = thickness_slab
        self.concrete_strength = concrete_strength
        self.length_base_plate = length_base_plate
        self.width_base_plate = width_base_plate
        self.position_load_area = position_load_area
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input properties"""
        valid_positions = ["interior", "edge", "corner"]
        if self.position_load_area not in valid_positions:
            raise ValueError(
                f"Invalid position_load_area: {self.position_load_area}. Must be one of {valid_positions}."
            )

        if self.thickness_slab <= 0:
            raise ValueError("Slab thickness must be positive")

        if self.concrete_strength <= 0:
            raise ValueError("Concrete strength must be positive")

        if self.length_base_plate <= 0 or self.width_base_plate <= 0:
            raise ValueError("Base plate dimensions must be positive")

    def _calculate_loaded_radius(self) -> float:
        """Calculate equivalent loaded radius based on position - Section 12.7.1(b)"""
        area = self.length_base_plate * self.width_base_plate

        if self.position_load_area == "interior":
            # For interior column: circular area equal to loaded area
            loaded_radius = np.sqrt(area / np.pi)
        elif self.position_load_area == "edge":
            # For edge column: semicircular projection
            loaded_radius = np.sqrt((2 * area) / np.pi)
        elif self.position_load_area == "corner":
            # For corner column: quarter-circular projection
            loaded_radius = np.sqrt((4 * area) / np.pi)
        else:
            raise ValueError(f"Invalid position: {self.position_load_area}")

        return loaded_radius

    def _get_alpha_s(self) -> float:
        """Get alpha_s factor accounting for column position - Section 12.7.3.2"""
        alpha_s_values = {"interior": 20, "edge": 15, "corner": 10}
        return alpha_s_values[self.position_load_area]

    def _calculate_perimeter_critical_section(self) -> float:
        """Calculate perimeter of critical section bo at d/2 from face of support - Section 12.7.1(b)"""
        loaded_radius = self._calculate_loaded_radius()
        d = self.thickness_slab

        # Critical section is at d/2 from face of loaded area
        # For simplified calculation with equivalent radius
        if self.position_load_area == "interior":
            # Full perimeter around loaded area
            bo = 2 * np.pi * (loaded_radius + d / 2)
        elif self.position_load_area == "edge":
            # Half perimeter (semicircle) plus two radial lines
            bo = np.pi * (loaded_radius + d / 2) + 2 * d
        elif self.position_load_area == "corner":
            # Quarter perimeter (quarter circle) plus two radial lines
            bo = (np.pi / 2) * (loaded_radius + d / 2) + 2 * d
        else:
            raise ValueError(f"Invalid position: {self.position_load_area}")

        return bo

    def _get_k_ds(self) -> float:
        """Calculate size effect factor k_ds for punching shear - Section 12.7.3.2

        k_ds = sqrt(200/d) with limits 0.5 <= k_ds <= 1.0
        where d is the average effective depth in mm
        """
        d = self.thickness_slab
        k_ds = np.sqrt(200 / d)

        # Apply limits
        k_ds = max(0.5, min(1.0, k_ds))

        return k_ds

    def _calculate_vc_a(self) -> float:
        """Calculate nominal shear stress vc using Equation 12-6

        vc = (1/6) * k_ds * (1 + 2/beta_c) * sqrt(f'c)

        where beta_c is the ratio of long side to short side of loaded area
        """
        k_ds = self._get_k_ds()
        fc = self.concrete_strength

        # Calculate beta_c (ratio of long side to short side)
        long_side = max(self.length_base_plate, self.width_base_plate)
        short_side = min(self.length_base_plate, self.width_base_plate)
        beta_c = long_side / short_side if short_side > 0 else 1.0

        # Equation 12-6
        vc_a = (1 / 6) * k_ds * (1 + 2 / beta_c) * np.sqrt(fc)

        return vc_a  # MPa

    def _calculate_vc_b(self) -> float:
        """Calculate nominal shear stress vc using Equation 12-7

        vc = (1/6) * k_ds * (alpha_s * d / bo + 1) * sqrt(f'c)

        where alpha_s = 20 for interior, 15 for edge, 10 for corner
        """
        k_ds = self._get_k_ds()
        fc = self.concrete_strength
        alpha_s = self._get_alpha_s()
        d = self.thickness_slab
        bo = self._calculate_perimeter_critical_section()

        # Equation 12-7
        vc_b = (1 / 6) * k_ds * (alpha_s * d / bo + 1) * np.sqrt(fc)

        return vc_b  # MPa

    def _calculate_vc_c(self) -> float:
        """Calculate nominal shear stress vc using Equation 12-8

        vc = (1/3) * k_ds * sqrt(f'c)
        """
        k_ds = self._get_k_ds()
        fc = self.concrete_strength

        # Equation 12-8
        vc_c = (1 / 3) * k_ds * np.sqrt(fc)

        return vc_c  # MPa

    def calculate_vc(self) -> float:
        """Calculate nominal shear stress vc - Section 12.7.3.2

        vc is the smallest of three equations (12-6, 12-7, 12-8)

        Returns:
            vc in MPa
        """
        vc_a = self._calculate_vc_a()
        vc_b = self._calculate_vc_b()
        vc_c = self._calculate_vc_c()

        vc = min(vc_a, vc_b, vc_c)

        return vc  # MPa

    def calculate_vc_max(self) -> float:
        """Calculate maximum nominal shear stress - Section 12.7.3.4

        The maximum nominal shear stress shall not exceed 0.5 * sqrt(f'c)

        Returns:
            vc_max in MPa
        """
        fc = self.concrete_strength
        vc_max = 0.5 * np.sqrt(fc)

        return vc_max  # MPa

    def calculate_Vc(self) -> float:
        """Calculate nominal shear strength Vc in Newtons - Section 12.7.3.1

        Vc = vc * bo * d

        Returns:
            Vc in N
        """
        vc = self.calculate_vc()
        bo = self._calculate_perimeter_critical_section()
        d = self.thickness_slab

        Vc = vc * bo * d  # MPa * mm * mm = N

        return Vc  # N

    def calculate_design_Vc(self) -> float:
        """Calculate design shear strength phi*Vc in Newtons

        Returns:
            phi*Vc in N (with phi = 0.75 for shear)
        """
        phi = 0.75
        Vc = self.calculate_Vc()
        return phi * Vc  # N


# ============================================================================
# SLAB UPLIFT RESISTANCE - Cracking Moment and Edge Distance Check
# ============================================================================


class SlabUpliftResistance:
    """Calculate slab uplift resistance based on cracking moment and edge distances"""

    def __init__(
        self,
        thickness_slab: float,
        concrete_strength: float,
        concrete_density: float,
        edge_distance_top_m: float,
        edge_distance_bot_m: float,
        edge_distance_lhs_m: float,
        edge_distance_rhs_m: float,
        strength_reduction_factor: float = 0.75,
    ):
        """
        Initialize slab uplift resistance calculator.

        Args:
            thickness_slab: Thickness of slab in mm
            concrete_strength: Specified compressive strength of concrete (f'c) in MPa
            concrete_density: Concrete density in kg/m³
            edge_distance_top_m: Top edge distance in meters
            edge_distance_bot_m: Bottom edge distance in meters
            edge_distance_lhs_m: Left-hand side edge distance in meters
            edge_distance_rhs_m: Right-hand side edge distance in meters
            strength_reduction_factor: Strength reduction factor (default 0.75)
        """
        self.thickness_slab = thickness_slab  # mm
        self.concrete_strength = concrete_strength  # MPa
        self.concrete_density = concrete_density  # kg/m³
        self.edge_distance_top_m = edge_distance_top_m
        self.edge_distance_bot_m = edge_distance_bot_m
        self.edge_distance_lhs_m = edge_distance_lhs_m
        self.edge_distance_rhs_m = edge_distance_rhs_m
        self.strength_reduction_factor = strength_reduction_factor
        self.g = 9.81  # m/s²

    def calculate_weight_of_slab_per_width(self) -> float:
        """Calculate weight of slab per unit width (N/m)

        Returns:
            Weight per unit width in N/m
        """
        thickness_m = self.thickness_slab / 1000  # Convert mm to m
        weight_per_width = thickness_m * self.concrete_density * self.g
        return weight_per_width  # N/m

    def calculate_moment_of_inertia(self) -> float:
        """Calculate moment of inertia per unit width (mm⁴)

        I = b*t³/12 where b = 1000 mm (1 meter width)

        Returns:
            Moment of inertia in mm⁴
        """
        b = 1000  # Unit width: 1 meter = 1000 mm
        Ix = (b * self.thickness_slab**3) / 12
        return Ix  # mm⁴

    def calculate_modulus_of_rupture(self) -> float:
        """Calculate modulus of rupture (MPa)

        fr = 0.6 * λ * sqrt(f'c)
        where λ = 1.0 for normal density concrete (NZS3101, Cl 5.2.4, 5.2.5)

        Returns:
            Modulus of rupture in MPa
        """
        lambda_factor = 1.0  # Normal density concrete
        fr = 0.6 * lambda_factor * np.sqrt(self.concrete_strength)
        return fr  # MPa

    def calculate_cracking_moment(self) -> float:
        """Calculate cracking moment (N·m)

        Mcr = (fr * I) / y
        where y is distance from neutral axis to extreme tension fiber

        Returns:
            Cracking moment in N·m
        """
        fr = self.calculate_modulus_of_rupture()  # MPa
        I = self.calculate_moment_of_inertia()  # mm³
        y = self.thickness_slab / 2  # mm (distance to extreme fiber)

        # Mcr = fr * I / y
        # MPa * mm³ / mm = N·mm = N·m / 1000
        Mcr = (fr * I) / y  # N·mm
        Mcr_Nm = Mcr / 1000  # Convert N·mm to N·m

        return Mcr_Nm  # N·m

    def calculate_cracking_length(self) -> float:
        """Calculate cracking length (m)

        Lcr = sqrt(2 * Mcr / w) * φ
        where w is weight per unit width

        Returns:
            Cracking length in meters
        """
        Mcr = self.calculate_cracking_moment()  # N·m
        w = self.calculate_weight_of_slab_per_width()  # N/m

        if w <= 0:
            raise ValueError("Weight per unit width must be positive")

        Lcr = np.sqrt(2 * Mcr / w) * self.strength_reduction_factor
        return Lcr  # m

    def check_edge_distances(self) -> dict:
        """Check all edge distances against cracking length

        Returns:
            Dictionary with edge check results
        """
        Lcr = self.calculate_cracking_length()

        edges_dict = {
            "Top": self.edge_distance_top_m,
            "Bottom": self.edge_distance_bot_m,
            "LHS": self.edge_distance_lhs_m,
            "RHS": self.edge_distance_rhs_m,
        }

        edge_check_results = {}
        all_edges_ok = True

        for edge_name, edge_distance in edges_dict.items():
            ratio = edge_distance / Lcr if Lcr > 0 else float("inf")
            is_compliant = edge_distance <= Lcr

            edge_check_results[f"Edge_{edge_name}"] = {
                "distance_m": edge_distance,
                "cracking_length_m": Lcr,
                "ratio": ratio,
                "compliant": is_compliant,
                "status": "PASS" if is_compliant else "FAIL",
            }

            if not is_compliant:
                all_edges_ok = False

        # Add overall check
        edge_check_results["overall"] = {
            "all_edges_compliant": all_edges_ok,
            "message": (
                "All edge distances are within cracking length"
                if all_edges_ok
                else "Some edge distances exceed cracking length"
            ),
        }

        return edge_check_results

    def calculate_resisting_slab_weight(self) -> float:
        """Calculate resisting slab weight (N)

        W = ρ * g * t * (Top + Bot) * (LHS + RHS)

        Returns:
            Resisting slab weight in N
        """
        thickness_m = self.thickness_slab / 1000  # Convert mm to m

        resisting_weight = (
            self.concrete_density
            * self.g
            * thickness_m
            * (self.edge_distance_top_m + self.edge_distance_bot_m)
            * (self.edge_distance_lhs_m + self.edge_distance_rhs_m)
        )

        return resisting_weight  # N


# --- Data for Post-Installed Screws based on provided tables ---

# REMOVE OLD TABLE 2A and 2B
# TABLE_2A_PHI_N_UC = { ... }
# TABLE_2B_X_NC = { ... }

# ADD MAIN TABLE DATA (combining Nuc for different f'c and Vuc)
MAIN_TABLE_DATA = {
    # Anchor Size (db): { Effective Depth (h): { 'Nuc': {f'c: value}, 'Vuc': value } }
    5: {
        25: {"Nuc": {20: 2.1, 25: 2.3, 32: 2.5}, "Vuc": 0.9},
    },
    6: {
        30: {"Nuc": {20: 3.7, 25: 4.0, 32: 4.3}, "Vuc": 6.8},
        37: {"Nuc": {20: 4.7, 25: 5.0, 32: 5.5}, "Vuc": 7.5},
        45: {"Nuc": {20: 5.8, 25: 6.3, 32: 6.9}, "Vuc": 7.5},
    },
    8: {
        40: {
            "Nuc": {20: 5.5, 25: 5.9, 32: 6.4},
            "Vuc": 12.6,
        },  # Assuming Vuc 12.6 applies to h=40
        50: {"Nuc": {20: 7.3, 25: 7.9, 32: 8.6}, "Vuc": 13.3},
        60: {"Nuc": {20: 9.3, 25: 10.0, 32: 10.9}, "Vuc": 13.3},
    },
    10: {
        50: {
            "Nuc": {20: 7.9, 25: 8.5, 32: 9.3},
            "Vuc": 20.7,
        },  # Assuming Vuc 20.7 applies to h=50
        62: {"Nuc": {20: 10.6, 25: 11.5, 32: 12.5}, "Vuc": 20.7},
        75: {"Nuc": {20: 13.9, 25: 15.0, 32: 16.3}, "Vuc": 20.7},
    },
    12: {
        60: {
            "Nuc": {20: 11.2, 25: 12.1, 32: 13.2},
            "Vuc": 25.0,
        },  # Assuming Vuc 25 (#) applies to h=60
        75: {
            "Nuc": {20: 15.5, 25: 16.8, 32: 18.3},
            "Vuc": 28.4,
        },  # Assuming Vuc 28.4 (#) applies to h=75
        90: {"Nuc": {20: 20.3, 25: 21.9, 32: 23.9}, "Vuc": 29.8},
    },
    16: {
        90: {
            "Nuc": {20: 20.7, 25: 24.1, 32: 28.4},
            "Vuc": 53.0,
        },  # Assuming Vuc 53.0 applies to h=90
        105: {"Nuc": {20: 25.4, 25: 29.5, 32: 34.8}, "Vuc": 53.0},
        120: {"Nuc": {20: 30.3, 25: 35.2, 32: 41.5}, "Vuc": 53.0},
    },
}
# Note: Vuc values marked with # might have specific conditions not fully captured here.
# The assignment of Vuc to specific depths is an interpretation where multiple depths share a Vuc value.

# --- Existing Tables (Keep 2C, 2D, 2E, 3A) ---
TABLE_2C_X_NE = {
    # ... (keep existing data) ...
    6: {20: 0.53, 25: 0.59, 30: 0.65, 35: 0.71, 40: 0.77, 50: 0.88, 60: 1.00},
    8: {25: 0.52, 30: 0.56, 35: 0.61, 40: 0.65, 50: 0.74, 60: 0.83, 70: 0.91, 80: 1.00},
    10: {
        30: 0.51,
        35: 0.55,
        40: 0.58,
        50: 0.65,
        60: 0.72,
        70: 0.79,
        80: 0.86,
        90: 0.93,
        100: 1.00,
    },
    12: {
        35: 0.50,
        40: 0.53,
        50: 0.59,
        60: 0.65,
        70: 0.71,
        80: 0.77,
        90: 0.83,
        100: 0.88,
        110: 0.94,
        120: 1.00,
    },
    16: {
        50: 0.51,
        60: 0.55,
        70: 0.60,
        80: 0.64,
        90: 0.69,
        100: 0.73,
        110: 0.78,
        120: 0.82,
        145: 0.93,
        160: 1.00,
    },
}

TABLE_2D_X_NAE = {
    # ... (keep existing data) ...
    6: {20: 0.78, 25: 0.85, 30: 0.92, 35: 1.00},
    8: {25: 0.76, 30: 0.81, 35: 0.86, 40: 0.92, 45: 1.00},
    10: {30: 0.75, 35: 0.79, 40: 0.83, 45: 0.88, 50: 0.92, 55: 0.96, 60: 1.00},
    12: {35: 0.78, 40: 0.81, 45: 0.81, 50: 0.85, 55: 0.88, 60: 0.92, 70: 1.00},
    16: {50: 0.76, 55: 0.79, 60: 0.81, 70: 0.86, 80: 0.92, 90: 0.97, 100: 1.00},
}

TABLE_2E_X_NAI = {
    # ... (keep existing data) ...
    6: {20: 0.56, 25: 0.69, 30: 0.83, 35: 1.00},
    8: {25: 0.52, 30: 0.63, 35: 0.73, 40: 0.83, 45: 0.94, 50: 1.00},
    10: {30: 0.50, 35: 0.58, 40: 0.67, 45: 0.75, 50: 0.83, 55: 0.92, 60: 1.00},
    12: {35: 0.49, 40: 0.56, 45: 0.63, 50: 0.69, 55: 0.76, 60: 0.83, 70: 1.00},
    16: {50: 0.52, 55: 0.57, 60: 0.63, 70: 0.73, 80: 0.83, 90: 0.94, 100: 1.00},
}

TABLE_3A_PHI_N_US = {
    # ... (keep existing data) ...
    6: 14.6,
    8: 27.1,
    10: 44.4,
    12: 53.8,
    16: 119.2,
}

# --- ADD SHEAR TABLES ---

TABLE_4A_PHI_V_UC_BASE = {
    # Anchor Size (db): {Edge Distance (e): phi_V_uc (kN) @ f'c=32MPa}
    6: {
        20: 0.9,
        25: 1.3,
        30: 1.7,
        35: 2.1,
        50: 3.6,
        75: 6.6,
        100: 10.1,
        150: 18.6,
        200: 28.7,
    },
    8: {
        25: 1.5,
        30: 1.9,
        35: 2.4,
        50: 4.1,
        75: 7.6,
        100: 11.7,
        150: 21.5,
        200: 33.1,
        250: 46.3,
    },
    10: {
        30: 2.2,
        35: 2.7,
        50: 4.6,
        75: 8.5,
        100: 13.1,
        150: 24.1,
        200: 37.0,
        250: 51.8,
        300: 68.0,
    },
    12: {
        35: 3.0,
        50: 5.1,
        75: 9.3,
        100: 14.3,
        150: 26.4,
        200: 40.6,
        250: 56.7,
        300: 74.5,
        400: 114.8,
    },
    16: {
        50: 5.9,
        75: 10.8,
        100: 16.6,
        150: 30.4,
        200: 46.8,
        250: 65.5,
        300: 86.1,
        400: 132.5,
        500: 185.2,
    },
}

TABLE_4B_X_VC = {
    # f'c (MPa): X_vc
    20: 0.79,
    25: 0.88,
    32: 1.00,
    40: 1.12,
}

TABLE_4C_X_VD = {
    # Angle (alpha degrees): X_vd
    0: 1.00,
    10: 1.04,
    20: 1.16,
    30: 1.32,
    40: 1.50,
    50: 1.66,
    60: 1.80,
    70: 1.91,
    80: 1.98,
    90: 2.00,
    180: 2.00,  # Assuming 90-180 is flat at 2.00
}

TABLE_4D_X_VA = {
    # Edge Distance (e): {Anchor Spacing (a): X_va}
    # Note: Single anchor X_va = 1.0 (handled in calculation method)
    20: {20: 0.70, 25: 0.75, 30: 0.80, 35: 0.85, 40: 0.90, 50: 1.00},
    25: {20: 0.66, 25: 0.70, 30: 0.74, 35: 0.78, 40: 0.82, 50: 0.90, 65: 1.00},
    30: {
        20: 0.63,
        25: 0.67,
        30: 0.70,
        35: 0.73,
        40: 0.77,
        50: 0.83,
        65: 0.93,
        80: 1.00,
    },
    35: {
        20: 0.61,
        25: 0.64,
        30: 0.67,
        35: 0.70,
        40: 0.73,
        50: 0.79,
        65: 0.87,
        80: 0.96,
        100: 1.00,
    },
    50: {
        20: 0.58,
        25: 0.60,
        30: 0.62,
        35: 0.64,
        40: 0.66,
        50: 0.70,
        65: 0.76,
        80: 0.82,
        100: 0.90,
        125: 1.00,
    },
    75: {
        20: 0.55,
        25: 0.57,
        30: 0.58,
        35: 0.59,
        40: 0.61,
        50: 0.63,
        65: 0.67,
        80: 0.71,
        100: 0.77,
        125: 0.83,
        150: 0.90,
        200: 1.00,
    },
    100: {
        20: 0.54,
        25: 0.55,
        30: 0.56,
        35: 0.57,
        40: 0.58,
        50: 0.60,
        65: 0.63,
        80: 0.66,
        100: 0.70,
        125: 0.75,
        150: 0.80,
        200: 0.90,
        250: 1.00,
    },
    150: {
        20: 0.53,
        25: 0.53,
        30: 0.54,
        35: 0.54,
        40: 0.55,
        50: 0.57,
        65: 0.59,
        80: 0.61,
        100: 0.63,
        125: 0.67,
        150: 0.70,
        200: 0.77,
        250: 0.83,
        300: 0.90,
        450: 1.00,
    },
    200: {
        20: 0.52,
        25: 0.52,
        30: 0.52,
        35: 0.53,
        40: 0.54,
        50: 0.55,
        65: 0.57,
        80: 0.58,
        100: 0.60,
        125: 0.62,
        150: 0.65,
        200: 0.70,
        250: 0.75,
        300: 0.80,
        450: 0.95,
        600: 1.00,
    },
    250: {
        25: 0.52,
        30: 0.52,
        35: 0.52,
        40: 0.53,
        50: 0.53,
        65: 0.54,
        80: 0.56,
        100: 0.58,
        125: 0.60,
        150: 0.62,
        200: 0.67,
        250: 0.70,
        300: 0.74,
        450: 0.86,
        600: 0.98,
        1000: 1.00,
    },
    300: {
        30: 0.52,
        35: 0.52,
        40: 0.52,
        50: 0.53,
        65: 0.53,
        80: 0.55,
        100: 0.57,
        125: 0.58,
        150: 0.60,
        200: 0.63,
        250: 0.67,
        300: 0.70,
        450: 0.80,
        600: 0.90,
        1000: 1.00,
    },
    400: {
        35: 0.52,
        40: 0.52,
        50: 0.52,
        65: 0.53,
        80: 0.54,
        100: 0.55,
        125: 0.56,
        150: 0.58,
        200: 0.60,
        250: 0.63,
        300: 0.65,
        450: 0.73,
        600: 0.80,
        1000: 0.90,
        1250: 1.00,
    },
    500: {
        40: 0.52,
        50: 0.52,
        65: 0.53,
        80: 0.53,
        100: 0.54,
        125: 0.55,
        150: 0.56,
        200: 0.58,
        250: 0.60,
        300: 0.62,
        450: 0.68,
        600: 0.74,
        1000: 0.84,
        1250: 1.00,
    },
}

TABLE_4E_X_VN = {
    # Number of anchors (n): {Anchor spacing / Edge distance (a/e): X_vn}
    # Note: Single anchor X_vn = 1.0 (handled in calculation method)
    2: {
        0.20: 1.00,
        0.40: 1.00,
        0.60: 1.00,
        0.80: 1.00,
        1.00: 1.00,
        1.20: 1.00,
        1.40: 1.00,
        1.60: 1.00,
        1.80: 1.00,
        2.00: 1.00,
        2.25: 1.00,
        2.50: 1.00,
    },
    3: {
        0.20: 0.72,
        0.40: 0.76,
        0.60: 0.80,
        0.80: 0.83,
        1.00: 0.86,
        1.20: 0.88,
        1.40: 0.91,
        1.60: 0.93,
        1.80: 0.95,
        2.00: 0.96,
        2.25: 0.98,
        2.50: 1.00,
    },
    4: {
        0.20: 0.57,
        0.40: 0.64,
        0.60: 0.69,
        0.80: 0.74,
        1.00: 0.79,
        1.20: 0.82,
        1.40: 0.86,
        1.60: 0.89,
        1.80: 0.92,
        2.00: 0.94,
        2.25: 0.97,
        2.50: 1.00,
    },
    5: {
        0.20: 0.49,
        0.40: 0.57,
        0.60: 0.63,
        0.80: 0.69,
        1.00: 0.74,
        1.20: 0.79,
        1.40: 0.83,
        1.60: 0.87,
        1.80: 0.90,
        2.00: 0.93,
        2.25: 0.96,
        2.50: 1.00,
    },
    6: {
        0.20: 0.43,
        0.40: 0.52,
        0.60: 0.59,
        0.80: 0.66,
        1.00: 0.71,
        1.20: 0.77,
        1.40: 0.81,
        1.60: 0.85,
        1.80: 0.89,
        2.00: 0.93,
        2.25: 0.96,
        2.50: 1.00,
    },
    7: {
        0.20: 0.39,
        0.40: 0.48,
        0.60: 0.56,
        0.80: 0.63,
        1.00: 0.69,
        1.20: 0.75,
        1.40: 0.80,
        1.60: 0.84,
        1.80: 0.88,
        2.00: 0.92,
        2.25: 0.96,
        2.50: 1.00,
    },
    8: {
        0.20: 0.36,
        0.40: 0.46,
        0.60: 0.54,
        0.80: 0.61,
        1.00: 0.68,
        1.20: 0.74,
        1.40: 0.79,
        1.60: 0.84,
        1.80: 0.88,
        2.00: 0.92,
        2.25: 0.96,
        2.50: 1.00,
    },
    9: {
        0.20: 0.34,
        0.40: 0.44,
        0.60: 0.52,
        0.80: 0.60,
        1.00: 0.67,
        1.20: 0.73,
        1.40: 0.78,
        1.60: 0.83,
        1.80: 0.87,
        2.00: 0.91,
        2.25: 0.96,
        2.50: 1.00,
    },
    10: {
        0.20: 0.32,
        0.40: 0.42,
        0.60: 0.51,
        0.80: 0.59,
        1.00: 0.66,
        1.20: 0.72,
        1.40: 0.77,
        1.60: 0.82,
        1.80: 0.87,
        2.00: 0.91,
        2.25: 0.96,
        2.50: 1.00,
    },
    15: {
        0.20: 0.26,
        0.40: 0.37,
        0.60: 0.47,
        0.80: 0.55,
        1.00: 0.63,
        1.20: 0.70,
        1.40: 0.76,
        1.60: 0.81,
        1.80: 0.86,
        2.00: 0.90,
        2.25: 0.95,
        2.50: 1.00,
    },
    20: {
        0.20: 0.23,
        0.40: 0.35,
        0.60: 0.45,
        0.80: 0.54,
        1.00: 0.61,
        1.20: 0.68,
        1.40: 0.75,
        1.60: 0.80,
        1.80: 0.85,
        2.00: 0.89,
        2.25: 0.95,
        2.50: 1.00,
    },
}

TABLE_4F_X_VS = {
    # Edge distance e1 (mm): {Edge distance e2 (mm): X_vs}
    # Note: For e1/e2 > 1.25, X_vs = 1.0 (handled in calculation method)
    25: {
        25: 0.86,
        30: 0.97,
        35: 1.00,
        50: 1.00,
        60: 1.00,
        75: 1.00,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    30: {
        25: 0.77,
        30: 0.86,
        35: 0.95,
        50: 1.00,
        60: 1.00,
        75: 1.00,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    35: {
        25: 0.70,
        30: 0.78,
        35: 0.86,
        50: 1.00,
        60: 1.00,
        75: 1.00,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    50: {
        25: 0.58,
        30: 0.64,
        35: 0.69,
        50: 0.86,
        60: 0.97,
        75: 1.00,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    60: {
        25: 0.53,
        30: 0.58,
        35: 0.63,
        50: 0.77,
        60: 0.86,
        75: 1.00,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    75: {
        25: 0.49,
        30: 0.52,
        35: 0.56,
        50: 0.67,
        60: 0.75,
        75: 0.86,
        125: 1.00,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    125: {
        25: 0.41,
        30: 0.43,
        35: 0.46,
        50: 0.52,
        60: 0.57,
        75: 0.64,
        125: 0.86,
        200: 1.00,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    200: {
        25: 0.37,
        30: 0.38,
        35: 0.40,
        50: 0.44,
        60: 0.47,
        75: 0.51,
        125: 0.65,
        200: 0.86,
        300: 1.00,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    300: {
        25: 0.35,
        30: 0.36,
        35: 0.37,
        50: 0.39,
        60: 0.41,
        75: 0.44,
        125: 0.53,
        200: 0.67,
        300: 0.86,
        400: 1.00,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    400: {
        25: 0.34,
        30: 0.34,
        35: 0.35,
        50: 0.37,
        60: 0.38,
        75: 0.41,
        125: 0.48,
        200: 0.58,
        300: 0.72,
        400: 0.86,
        500: 1.00,
        600: 1.00,
        900: 1.00,
    },
    500: {
        25: 0.33,
        30: 0.33,
        35: 0.33,
        50: 0.35,
        60: 0.36,
        75: 0.37,
        125: 0.42,
        200: 0.49,
        300: 0.58,
        400: 0.67,
        500: 0.86,
        600: 1.00,
        900: 1.00,
    },
    600: {
        25: 0.32,
        30: 0.32,
        35: 0.33,
        50: 0.34,
        60: 0.34,
        75: 0.35,
        125: 0.38,
        200: 0.42,
        300: 0.49,
        400: 0.55,
        500: 0.77,
        600: 0.86,
        900: 1.00,
    },
    900: {
        25: 0.32,
        30: 0.32,
        35: 0.32,
        50: 0.32,
        60: 0.33,
        75: 0.34,
        125: 0.35,
        200: 0.38,
        300: 0.42,
        400: 0.49,
        500: 0.61,
        600: 0.67,
        900: 0.86,
    },
}

TABLE_5A_PHI_V_US = {
    # Anchor Size (db): { h / db ratio : phi_V_us (kN) }
    # Using h_ratio as key, e.g., 5 for h >= 5*db
    6: {5: 6.8, 6: 7.7, 7: 8.6, 7.5: 9.1},
    8: {5: 12.6, 6: 14.3, 7: 16.0, 7.5: 16.8},
    10: {5: 20.7, 6: 23.4, 7: 26.2, 7.5: 27.5},
    12: {5: 25.0, 6: 28.4, 7: 31.7, 7.5: 33.4},
    16: {5: 55.4, 6: 62.8, 7: 70.2, 7.5: 73.9},
}

# --- Helper Functions ---


# Keep _interpolate_table for 1D interpolation (used for Xne, Xnae, Xnai)
def _interpolate_table(table_data, anchor_size, x_value):
    """Interpolate value from a table dictionary based on anchor size and x_value (e.g., depth, distance, spacing)."""
    if anchor_size not in table_data:
        raise ValueError(f"Anchor size {anchor_size} not found in table.")

    data_for_size = table_data[anchor_size]
    x_points = sorted(data_for_size.keys())
    y_points = [data_for_size[x] for x in x_points]

    if len(x_points) < 2:
        if x_value == x_points[0]:
            return y_points[0]
        else:
            # Allow extrapolation by returning boundary value if only one point exists
            print(
                f"Warning: Only one data point for anchor size {anchor_size} in table. Returning value {y_points[0]}."
            )
            return y_points[0]
            # raise ValueError("Cannot interpolate with less than two data points.")

    # Use numpy for linear interpolation (allows extrapolation by default)
    return np.interp(x_value, x_points, y_points)


# REMOVE _interpolate_table_2b
# def _interpolate_table_2b(table_data, anchor_size, fc_value): ...


# ADD 2D interpolation for phi_N_uc
def _interpolate_phi_N_uc(anchor_size, effective_depth, concrete_strength):
    """Interpolate phi_N_uc from MAIN_TABLE_DATA based on depth and concrete strength."""
    if anchor_size not in MAIN_TABLE_DATA:
        raise ValueError(f"Anchor size {anchor_size} not found in main table.")

    size_data = MAIN_TABLE_DATA[anchor_size]
    depths = sorted(size_data.keys())

    if len(depths) < 1:
        raise ValueError(f"No depth data found for anchor size {anchor_size}.")

    # Check if exact depth exists
    if effective_depth in size_data:
        depth_data = size_data[effective_depth]["Nuc"]
        fcs = sorted(depth_data.keys())
        nucs = [depth_data[fc] for fc in fcs]
        # Interpolate/extrapolate based on concrete strength
        return np.interp(concrete_strength, fcs, nucs)
    else:
        # Need to interpolate between depths first, then by concrete strength
        if len(depths) < 2:
            print(
                f"Warning: Only one depth ({depths[0]}) available for anchor size {anchor_size}. Cannot interpolate depth. Using data for depth {depths[0]}."
            )
            depth_data = size_data[depths[0]]["Nuc"]
            fcs = sorted(depth_data.keys())
            nucs = [depth_data[fc] for fc in fcs]
            return np.interp(concrete_strength, fcs, nucs)

        # Find depths surrounding the target depth
        depth1, depth2 = None, None
        for i in range(len(depths) - 1):
            if depths[i] <= effective_depth < depths[i + 1]:
                depth1, depth2 = depths[i], depths[i + 1]
                break
        # Handle extrapolation cases
        if effective_depth < depths[0]:
            # print(
            #     f"Warning: Effective depth {effective_depth} is below minimum ({depths[0]}) for anchor size {anchor_size}. Extrapolating."
            # )
            depth1, depth2 = depths[0], depths[1]
        elif effective_depth >= depths[-1]:
            # print(
            #     f"Warning: Effective depth {effective_depth} is above maximum ({depths[-1]}) for anchor size {anchor_size}. Extrapolating."
            # )
            depth1, depth2 = depths[-2], depths[-1]

        if depth1 is None or depth2 is None:
            raise ValueError(
                f"Could not determine interpolation depths for effective depth {effective_depth}."
            )

        # Interpolate Nuc for each bounding depth based on concrete_strength
        nuc1_data = size_data[depth1]["Nuc"]
        fcs1 = sorted(nuc1_data.keys())
        nucs1 = [nuc1_data[fc] for fc in fcs1]
        interp_nuc1 = np.interp(concrete_strength, fcs1, nucs1)

        nuc2_data = size_data[depth2]["Nuc"]
        fcs2 = sorted(nuc2_data.keys())
        nucs2 = [nuc2_data[fc] for fc in fcs2]
        interp_nuc2 = np.interp(concrete_strength, fcs2, nucs2)

        # Interpolate between the two depth-interpolated Nuc values
        # Using inverse distance weighting logic or linear interp
        if (
            depth2 == depth1
        ):  # Avoid division by zero if depths are the same (e.g., single point case handled above)
            return interp_nuc1
        else:
            # Linear interpolation between the two depths
            return interp_nuc1 + (interp_nuc2 - interp_nuc1) * (
                effective_depth - depth1
            ) / (depth2 - depth1)


# ADD 1D interpolation for phi_V_uc
def _interpolate_phi_V_uc(anchor_size, effective_depth):
    """Interpolate phi_V_uc from MAIN_TABLE_DATA based on effective_depth."""
    if anchor_size not in MAIN_TABLE_DATA:
        raise ValueError(f"Anchor size {anchor_size} not found in main table.")

    size_data = MAIN_TABLE_DATA[anchor_size]
    depths = sorted(size_data.keys())
    # Extract unique Vuc values corresponding to sorted depths
    vucs = [size_data[d]["Vuc"] for d in depths]

    if len(depths) < 2:
        if len(depths) == 1:
            print(
                f"Warning: Only one depth data point for Vuc for anchor size {anchor_size}. Returning Vuc={vucs[0]}."
            )
            return vucs[0]
        else:
            raise ValueError(f"No Vuc data found for anchor size {anchor_size}.")

    # Use numpy for linear interpolation (allows extrapolation)
    return np.interp(effective_depth, depths, vucs)


# --- ADD Interpolation functions for Shear Tables ---


def _interpolate_1d(data, x_value):
    """Generic 1D interpolation from a simple {x: y} dictionary."""
    x_points = sorted(data.keys())
    y_points = [data[x] for x in x_points]
    if len(x_points) < 2:
        if len(x_points) == 1:
            # Return the single value if only one point exists (constant)
            # print(f"Warning: Only one data point available. Returning value {y_points[0]}.")
            return y_points[0]
        else:
            raise ValueError("Cannot interpolate with no data points.")
    return np.interp(x_value, x_points, y_points)


def _interpolate_2d(table_data, key1_value, key2_value):
    """Generic 2D interpolation from a nested dict {key1: {key2: value}}."""
    # Ensure primary keys are numeric for interpolation/extrapolation
    key1_points = sorted([k for k in table_data.keys() if isinstance(k, (int, float))])

    if not key1_points:
        raise ValueError(
            f"No numeric primary keys found in table data for key1: {key1_value}"
        )

    # Check if key1_value is directly present or needs interpolation
    if key1_value in table_data:
        inner_dict = table_data[key1_value]
        return _interpolate_1d(inner_dict, key2_value)
    else:
        # Need to interpolate/extrapolate between key1 values
        if len(key1_points) < 2:
            # print(f"Warning: Only one primary key ({key1_points[0]}) available. Cannot interpolate key1. Using data for {key1_points[0]}.")
            inner_dict = table_data[key1_points[0]]
            return _interpolate_1d(inner_dict, key2_value)

        # Find bounding keys for key1_value
        k1_lower, k1_upper = None, None
        if key1_value < key1_points[0]:  # Extrapolation below range
            # print(f"Warning: key1_value {key1_value} is below minimum ({key1_points[0]}). Extrapolating.")
            k1_lower, k1_upper = key1_points[0], key1_points[1]
        elif key1_value >= key1_points[-1]:  # Extrapolation above range
            # print(f"Warning: key1_value {key1_value} is above maximum ({key1_points[-1]}). Extrapolating.")
            k1_lower, k1_upper = key1_points[-2], key1_points[-1]
        else:  # Interpolation within range
            for i in range(len(key1_points) - 1):
                if key1_points[i] <= key1_value < key1_points[i + 1]:
                    k1_lower, k1_upper = key1_points[i], key1_points[i + 1]
                    break

        if k1_lower is None or k1_upper is None:
            raise ValueError(
                f"Could not determine bounding keys for key1_value {key1_value}."
            )

        # Interpolate key2_value at the lower and upper bounds of key1
        val_at_k1_lower = _interpolate_1d(table_data[k1_lower], key2_value)
        val_at_k1_upper = _interpolate_1d(table_data[k1_upper], key2_value)

        # Interpolate/extrapolate between the two key1-interpolated values
        if k1_upper == k1_lower:
            return val_at_k1_lower
        else:
            return val_at_k1_lower + (val_at_k1_upper - val_at_k1_lower) * (
                key1_value - k1_lower
            ) / (k1_upper - k1_lower)


# --- Dataclass and Class Definitions ---


@dataclass
class PostInstalledScrewProperties:
    """Properties of a post-installed screw required for capacity calculations."""

    anchor_size: int  # db (mm)
    effective_depth: float  # h (mm)
    concrete_strength: float  # f'c (MPa)
    edge_distance: float  # e (mm)
    anchor_spacing: float  # a (mm)
    is_end_of_row: bool  # True if anchor is at the end of a row (use X_nae), False if internal (use X_nai)
    load_angle: float = (
        0.0  # Angle alpha in degrees (default 0, perpendicular towards edge)
    )
    number_of_anchors: int = 1  # n (default 1)
    edge_distance_2: Optional[float] = (
        None  # e2 for corner effect (Xvs), None if not near a corner
    )


class PostInstalledScrewCapacity:
    """Calculate tensile capacity for post-installed screws based on provided tables."""

    def __init__(self, props: PostInstalledScrewProperties):
        self.props = props
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input properties against table limits."""
        valid_anchor_sizes = list(
            MAIN_TABLE_DATA.keys()
        )  # Use keys from the main table
        if self.props.anchor_size not in valid_anchor_sizes:
            raise ValueError(
                f"Invalid anchor_size: {self.props.anchor_size}. Must be one of {valid_anchor_sizes}."
            )

        # Validate concrete strength (only check range, interpolation handles specifics)
        valid_fc = [20, 25, 32]  # From main table columns
        if not (min(valid_fc) <= self.props.concrete_strength <= max(valid_fc)):
            print(
                f"Warning: Concrete strength {self.props.concrete_strength} MPa is outside the tabulated range {valid_fc}. Extrapolation will occur."
            )

        # --- ADD Validation for shear properties ---
        # Table 4a Note: h must be >= 3.5 * db
        min_depth = 3.5 * self.props.anchor_size
        if self.props.effective_depth < min_depth:
            print(
                f"Warning: Effective depth {self.props.effective_depth} mm is less than recommended minimum ({min_depth:.1f} mm = 3.5 * db) for Table 4a shear capacities."
            )

        # Load angle range
        if not (0 <= self.props.load_angle <= 180):
            print(
                f"Warning: Load angle {self.props.load_angle} degrees is outside the typical 0-180 range for Table 4c."
            )

        # Number of anchors
        if self.props.number_of_anchors < 1:
            raise ValueError("Number of anchors must be at least 1.")

        # Corner distance e2
        if self.props.edge_distance_2 is not None and self.props.edge_distance_2 <= 0:
            raise ValueError("Edge distance e2 must be positive if specified.")
        if self.props.edge_distance_2 is not None and self.props.number_of_anchors > 1:
            print(
                f"Warning: Corner effect factor (Xvs) using e1={self.props.edge_distance}, e2={self.props.edge_distance_2} applied to a group of {self.props.number_of_anchors} anchors. Ensure this is intended."
            )

    def _get_X_ne(self) -> float:
        """Interpolate Edge distance effect factor (X_ne) from Table 2c."""
        return _interpolate_table(
            TABLE_2C_X_NE, self.props.anchor_size, self.props.edge_distance
        )

    def _get_X_nae_or_X_nai(self) -> float:
        """Interpolate Anchor spacing effect factor (X_nae or X_nai) from Table 2d or 2e."""
        if self.props.is_end_of_row:
            return _interpolate_table(
                TABLE_2D_X_NAE, self.props.anchor_size, self.props.anchor_spacing
            )
        else:
            return _interpolate_table(
                TABLE_2E_X_NAI, self.props.anchor_size, self.props.anchor_spacing
            )

    def _get_phi_N_us(self) -> float:
        """Get Reduced characteristic ultimate steel tensile capacity (phi_N_us) from Table 3a."""
        if self.props.anchor_size not in TABLE_3A_PHI_N_US:
            raise ValueError(
                f"Anchor size {self.props.anchor_size} not found in Table 3a."
            )
        return TABLE_3A_PHI_N_US[self.props.anchor_size]

    def calculate_phi_N_urc(self) -> float:
        """Calculate Design reduced ultimate concrete tensile capacity (phi_N_urc)."""
        phi_N_uc = _interpolate_phi_N_uc(
            self.props.anchor_size,
            self.props.effective_depth,
            self.props.concrete_strength,
        )
        X_ne = self._get_X_ne()
        X_nae_or_X_nai = self._get_X_nae_or_X_nai()
        phi_N_urc = phi_N_uc * X_ne * X_nae_or_X_nai
        return phi_N_urc

    def calculate_phi_N_ur(self) -> float:
        """Calculate Design reduced ultimate tensile capacity (phi_N_ur)."""
        phi_N_urc = self.calculate_phi_N_urc()
        phi_N_us = self._get_phi_N_us()
        phi_N_ur = min(phi_N_urc, phi_N_us)
        return phi_N_ur

    # ADD calculate_phi_V_uc
    def calculate_phi_V_uc(self) -> float:
        """Calculate Design reduced ultimate concrete shear capacity (phi_V_uc)."""
        # Interpolate phi_V_uc using the new 1D interpolation function
        phi_V_uc = _interpolate_phi_V_uc(
            self.props.anchor_size, self.props.effective_depth
        )
        # Note: This doesn't include potential modification factors if needed
        # based on the full standard or specific conditions (like #).
        return phi_V_uc  # Result in kN

    def _get_X_vc(self) -> float:
        """Interpolate concrete compressive strength effect factor (X_vc) from Table 4b."""
        # Use _interpolate_1d as Table 4b is not dependent on anchor_size
        return _interpolate_1d(TABLE_4B_X_VC, self.props.concrete_strength)

    def _get_X_vd(self) -> float:
        """Interpolate load direction effect factor (X_vd) from Table 4c."""
        # Adjust angle if outside 0-90 range before interpolating
        angle = abs(self.props.load_angle)
        if angle > 90:
            angle = min(angle, 180)  # Cap at 180
            if (
                angle > 90
            ):  # Mirror angles > 90 back into the 0-90 range if needed, or use the 90-180 value
                # Table 4c goes up to 90-180 directly
                pass  # Keep angle as is for interpolation up to 180
        return _interpolate_1d(TABLE_4C_X_VD, angle)

    def _get_X_va(self) -> float:
        """Interpolate anchor spacing effect factor (X_va) from Table 4d."""
        if self.props.number_of_anchors == 1:
            return 1.0  # Note in Table 4d
        # Table 4d uses edge distance 'e' and anchor spacing 'a'
        return _interpolate_2d(
            TABLE_4D_X_VA, self.props.edge_distance, self.props.anchor_spacing
        )

    def _get_X_vn(self) -> float:
        """Interpolate multiple anchors effect factor (X_vn) from Table 4e."""
        if self.props.number_of_anchors == 1:
            return 1.0  # Note in Table 4e

        if self.props.edge_distance <= 0:
            print(
                "Warning: Cannot calculate a/e ratio for X_vn with edge_distance <= 0. Returning X_vn = 1.0."
            )
            return 1.0

        a_e_ratio = self.props.anchor_spacing / self.props.edge_distance
        # Table 4e uses number of anchors 'n' and ratio 'a/e'
        # Need to handle potential non-integer number of anchors if needed, but assuming int here
        num_anchors_key = self.props.number_of_anchors
        # Find closest or bounding key if exact number of anchors not in table
        valid_n = sorted(TABLE_4E_X_VN.keys())
        if num_anchors_key not in valid_n:
            # Simple approach: use closest n, or could interpolate between n values
            closest_n = min(valid_n, key=lambda x: abs(x - num_anchors_key))
            print(
                f"Warning: Number of anchors {num_anchors_key} not directly in Table 4e. Using data for n={closest_n}."
            )
            num_anchors_key = closest_n

        return _interpolate_1d(TABLE_4E_X_VN[num_anchors_key], a_e_ratio)

    def _get_X_vs(self) -> float:
        """Interpolate anchor at a corner effect factor (X_vs) from Table 4f."""
        if self.props.edge_distance_2 is None:
            # Not a corner anchor
            return 1.0

        e1 = self.props.edge_distance
        e2 = self.props.edge_distance_2

        if e2 <= 0:
            print(
                "Warning: edge_distance_2 must be positive for corner effect. Assuming X_vs = 1.0."
            )
            return 1.0

        # Check note: For e1/e2 > 1.25, Xvs = 1.0
        if e1 / e2 > 1.25:
            return 1.0
        # Check note: For e2/e1 > 1.25, Xvs = 1.0 (handle symmetrically)
        if e2 / e1 > 1.25:
            return 1.0

        # Table 4f uses e1 and e2 directly. Use the generic 2D interpolation.
        # Ensure keys are handled correctly if e1/e2 are not exact matches.
        try:
            # Try direct lookup/interpolation with e1 as primary key
            return _interpolate_2d(TABLE_4F_X_VS, e1, e2)
        except ValueError:
            try:
                # If e1 failed, try with e2 as primary key (table might be symmetric)
                print(
                    "Warning: Trying Table 4f interpolation with e2 as the primary key due to potential non-match with e1."
                )
                return _interpolate_2d(TABLE_4F_X_VS, e2, e1)
            except ValueError as e:
                print(
                    f"Error during X_vs interpolation with e1={e1}, e2={e2}: {e}. Returning X_vs = 1.0."
                )
                return 1.0

    def calculate_phi_V_urc(self) -> float:
        """Calculate Design reduced ultimate concrete edge shear capacity (phi_V_urc)."""
        phi_V_uc_base = self.calculate_phi_V_uc()  # Base capacity at f'c=32MPa
        X_vc = self._get_X_vc()  # Concrete strength effect
        X_vd = self._get_X_vd()  # Load direction effect
        X_va = self._get_X_va()  # Anchor spacing effect
        X_vn = self._get_X_vn()  # Multiple anchor effect
        X_vs = self._get_X_vs()  # Corner effect

        # Formula: phi_V_urc = phi_V_uc_base * X_vc * X_vd * X_va * X_vn * X_vs
        phi_V_urc = phi_V_uc_base * X_vc * X_vd * X_va * X_vn * X_vs
        return phi_V_urc  # Result in kN

    def _get_phi_V_us(self) -> float:
        """Interpolate Reduced characteristic ultimate steel shear capacity (phi_V_us) from Table 5a."""
        if self.props.anchor_size <= 0:
            raise ValueError("Anchor size must be positive to calculate h/db ratio.")

        h_db_ratio = self.props.effective_depth / self.props.anchor_size

        # Table 5a uses anchor size 'db' and ratio 'h/db'
        return _interpolate_2d(TABLE_5A_PHI_V_US, self.props.anchor_size, h_db_ratio)

    def calculate_phi_V_ur(self) -> float:
        """Calculate Design reduced ultimate shear capacity (phi_V_ur)."""
        phi_V_urc = self.calculate_phi_V_urc()  # Concrete edge shear
        phi_V_us = self._get_phi_V_us()  # Steel shear

        # phi_V_ur = minimum of phi_V_urc, phi_V_us
        phi_V_ur = min(phi_V_urc, phi_V_us)
        return phi_V_ur  # Result in kN


# --- ADD DISPATCH FUNCTION ---
def calculate_anchor_capacity(
    thickness_concrete,
    diameter,
    head_diameter,
    effective_embedment_depth,
    edge_distance,
    edge_distance_perp,
    eccentricity,
    spacing,
    number_of_anchors,
    ultimate_tensile_strength,
    concrete_strength,
    is_cracked_concrete,
    length_hook,
    anchor_type,
    shear_prep,
):
    """
    Calculates the design tension and shear capacity based on the anchor type.

    Args:
        properties: An instance of either AnchorProperties (for Chapter 17 calculations)
                    or PostInstalledScrewProperties (for post-installed screw calculations).

    Returns:
        A tuple containing:
        - Design Tension Capacity (N)
        - Design Shear Capacity (N)

    Raises:
        TypeError: If the input properties object is not an instance of
                   AnchorProperties or PostInstalledScrewProperties.
    """
    if anchor_type == "screw":
        screw_props = PostInstalledScrewProperties(
            anchor_size=diameter,
            effective_depth=effective_embedment_depth,
            concrete_strength=concrete_strength,  # Changed to test interpolation
            edge_distance=edge_distance,
            anchor_spacing=spacing,
            is_end_of_row=False,  # Internal anchor
        )
        calculator = PostInstalledScrewCapacity(screw_props)
        tension_capacity_n = calculator.calculate_phi_N_ur() * 1000  # Convert to N
        shear_capacity_n = calculator.calculate_phi_V_ur() * 1000  # Convert to N
        return tension_capacity_n, shear_capacity_n
    else:
        anchor_props = AnchorProperties(
            thickness_concrete=thickness_concrete,
            diameter=diameter,
            head_diameter=head_diameter,
            effective_embedment_depth=effective_embedment_depth,
            edge_distance=edge_distance,
            edge_distance_perp=edge_distance_perp,
            eccentricity=eccentricity,
            spacing=spacing,
            number_of_anchors=number_of_anchors,
            ultimate_tensile_strength=ultimate_tensile_strength,
            concrete_strength=concrete_strength,
            is_cracked_concrete=is_cracked_concrete,
            length_hook=length_hook,
            anchor_type=anchor_type,
            shear_prep=shear_prep,
        )
        calculator = AnchorCapacity_Chapter_17(anchor_props)
        tension_capacity_n = calculator._calc_anchor_tension_strength_17_5_7()
        shear_capacity_n = calculator._calc_anchor_shear_strength_17_5_8()
        return tension_capacity_n, shear_capacity_n
