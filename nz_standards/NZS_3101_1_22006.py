from dataclasses import dataclass
from typing import Optional, List
import numpy as np


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

    # def __post_init__(self):
    #     """Convert and validate input parameters"""

    #     # Helper function to handle tuple conversion
    #     def convert_tuple(value, convert_type):
    #         if isinstance(value, tuple):
    #             return convert_type(value[0])
    #         return convert_type(value)

    #     # Convert required numeric inputs
    #     self.diameter = convert_tuple(self.diameter, float)
    #     self.head_diameter = convert_tuple(self.head_diameter, float)
    #     self.effective_embedment_depth = convert_tuple(
    #         self.effective_embedment_depth, float
    #     )
    #     self.edge_distance = convert_tuple(self.edge_distance, float)
    #     self.eccentricity = convert_tuple(self.eccentricity, float)
    #     self.number_of_anchors = convert_tuple(self.number_of_anchors, int)
    #     self.ultimate_tensile_strength = convert_tuple(
    #         self.ultimate_tensile_strength, float
    #     )
    #     self.concrete_strength = convert_tuple(self.concrete_strength, float)

    #     # Convert optional numeric inputs
    #     if self.edge_distance_perp is not None:
    #         self.edge_distance_perp = convert_tuple(self.edge_distance_perp, float)
    #     if self.spacing is not None:
    #         self.spacing = convert_tuple(self.spacing, float)
    #     if self.length_hook is not None:
    #         self.length_hook = convert_tuple(self.length_hook, float)

    #     # Convert boolean inputs
    #     self.is_cracked_concrete = convert_tuple(self.is_cracked_concrete, bool)
    #     self.shear_prep = convert_tuple(self.shear_prep, bool)

    #     # Validate inputs
    #     if self.diameter <= 0:
    #         raise ValueError("diameter must be positive")
    #     if self.head_diameter <= self.diameter:
    #         raise ValueError("head_diameter must be greater than shaft diameter")
    #     if self.effective_embedment_depth <= 0:
    #         raise ValueError("effective_embedment_depth must be positive")
    #     if self.edge_distance <= 0:
    #         raise ValueError("edge_distance must be positive")
    #     if self.number_of_anchors <= 0:
    #         raise ValueError("number_of_anchors must be positive")
    #     if self.ultimate_tensile_strength <= 0:
    #         raise ValueError("ultimate_tensile_strength must be positive")
    #     if self.concrete_strength <= 0:
    #         raise ValueError("concrete_strength must be positive")

    #     # Validate anchor type
    #     valid_types = ["headed_stud", "headed_bolt", "hooked_bolt"]
    #     if self.anchor_type not in valid_types:
    #         raise ValueError(f"anchor_type must be one of {valid_types}")

    #     # Validate hook length for hooked bolts
    #     if self.anchor_type == "hooked_bolt":
    #         if not self.length_hook:
    #             raise ValueError("length_hook must be provided for hooked bolts")
    #         if not (3 * self.diameter <= self.length_hook <= 4.5 * self.diameter):
    #             raise ValueError("hook length must be between 3do and 4.5do")


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
