"""AS_NZS_1664_11997 Method

Automatically generated by Colaboratory.


This method references the following standard:
AS_NZS_1664_11997 Method, for New Zealand and Australia structures.

Method developed 30 June 2024
(c) Constratum Ltd

Developed - NSh

Reviewed - MB


###Initialise  Dependents and Libraries
"""

import numpy as np
import pandas as pd

# Table 3.3(A) - Minimum Mechenical Properties for Aluminium Alloys
# Define the columns based on the table provided
columns_3_3_a = ["Alloy and temper", "Product", "Thickness range", "Tension_Ftu", "Tension_Fty", "Compression_Fcy",
           "Shear_Fsu", "Shear_Fsy", "Bearing_Fbru", "Bearing_Fbry", "Compressive modulus of elasticity_E"]

# Create an empty DataFrame with the defined columns
table_3_3_A = pd.DataFrame(columns=columns_3_3_a)

# Data from the provided table (manually entered)
data_3_3_a = [
    ["5005-H12", "Sheet & plate", "0.4–50", 124, 96, 90, 76, 55,  234, 152, 70000],
    ["5005-H14", "Sheet & plate", "0.2–25", 145, 117, 103, 83, 69, 276, 172, 70000],
    ["5005-H16", "Sheet", "0.15–1.6", 165, 138, 124, 96, 83, 331, 207, 70000],
    ["5005-H32", "Sheet & plate", "0.4–50", 117, 83, 76, 76, 48, 234, 138, 70000],
    ["5005-H34", "Sheet & plate", "0.2–25", 138, 103, 96, 83, 58, 276, 165, 70000],
    ["5005-H36", "Sheet", "0.15–1.6", 158, 124, 110, 90, 76, 331, 200, 70000],

    ["5050-H32", "Sheet", "0.4–6.3", 152, 110, 96, 96, 62, 303, 186, 70000],
    ["5050-H34", "Sheet", "0.2–6.3", 172, 138, 124, 103, 83, 345, 220, 70000],
    ["5050-H32", "Cold fin. rod & bar* Drawn tube", "All", 152, 110, 103, 90, 62, 303, 186, 70000],
    ["5050-H34", "Cold fin. rod & bar* Drawn tube", "All", 172, 138, 131, 103, 83, 345, 220, 70000],

    ["5052-H32", "Sheet & plate", "All", 214, 158, 145, 131, 90, 448, 269, 70000],
    ["5052-H34", "Cold fin. rod & bar* Drawn tube", "All", 234, 179, 165, 138, 103, 448, 303, 70000],
    ["5052-H36", "Sheet", "0.15–4.1", 255, 200, 179, 152, 117, 483, 317, 71000],
    ["5052-H38", "Sheet", "≤3.25", 268, 220, 207, 152, 124, 510, 338, 70327],
    ["5052-H391", "Sheet", "≤2", 290, 241, 227, 159, 138, 524, 70327],

    ["5083-H111", "Extrusions", "Up to 12", 276, 165, 145, 165, 96, 538, 282, 72000],
    ["5083-H111", "Extrusions", "Over 12", 276, 165, 145, 165, 96, 538, 282, 72000],
    ["5083-H321", "Sheet & plate", "4.8–38", 303, 214, 179, 179, 124, 579, 365, 72000],
    ["5083-H116", "Sheet & plate", "4.8–38", 303, 214, 179, 179, 124, 579, 365, 72000],
    ["5083-H321", "Plate", "38–76", 283, 200, 165, 165, 117, 538, 338, 72000],
    ["5083-H116", "Plate", "38–76", 283, 200, 165, 165, 117, 538, 338, 72000],
    ["5083-H323", "Sheet", "0.4–6.3", 310, 234, 220, 179, 138, 607, 400, 72000],
    ["5083-H343", "Sheet", "0.4–6.3", 345, 269, 255, 200, 158, 655, 455, 72000],

    ["5086-H111", "Extrusions", "Up to 12", 248, 145, 124, 145, 83, 483, 248, 72000],
    ["5086-H111", "Extrusions", "Over 12", 248, 145, 124, 145, 83, 483, 234, 72000],
    ["5086-H112", "Plate", "6–12", 248, 124, 117, 152, 69, 496, 214, 72000],
    ["5086-H112", "Plate", "12-25", 241, 110, 110, 145, 62, 483, 193, 72000],
    ["5086-H112", "Plate", "25–50", 241, 96, 103, 145, 55, 483, 193, 72000],
    ["5086-H112", "Plate", "50–70", 234, 96, 103, 145, 55, 469, 193, 72000],
    ["5086-H116", "Sheet & plate", "All", 276, 193, 179, 165, 110, 538, 331, 72000],
    ["5086-H32", "Sheet & plate", "All", 276, 193, 179, 165, 110, 538, 331, 72000],
    ["5086-H34", "Drawn tube", "All", 303, 234, 220, 179, 138, 579, 400, 72000],

    ["5154-H38", "Sheet", "0.15–3.2", 310, 241, 227, 165, 138, 558, 386, 72000],

    ["5251-H34", "Sheet, plate", "≤25", 231, 179, 159, 131, 103, 434, 303, 70327],

    ["5454-H111", "Extrusions", "Up to 12", 228, 131, 110, 138, 76, 441, 220, 72000],
    ["5454-H111", "Extrusions", "Over 12", 228, 131, 110, 131, 76, 441, 207, 72000],
    ["5454-H112", "Extrusions", "Up to 127", 214, 83, 90, 131, 48, 427, 165, 72000],
    ["5454-H32", "Sheet & plate", "0.5–50", 248, 179, 165, 145, 103, 483, 303, 72000],
    ["5454-H34", "Sheet & plate", "0.5–25", 269, 200, 186, 158, 117, 510, 338, 72000],

    ["5456-H111", "Extrusions", "Up to 12", 290, 179, 152, 172, 103, 565, 303, 72000],
    ["5456-H111", "Extrusions", "Over 12", 290, 179, 152, 165, 103, 565, 290, 72000],
    ["5456-H112", "Extrusions", "Up to 127", 283, 131, 138, 165, 76, 565, 262, 72000],
    ["5456-H116", "Sheet & plate", "4.8–32", 317, 227, 186, 186, 131, 600, 386, 72000],
    ["5456-H321", "Sheet & plate", "4.8–32", 317, 227, 186, 186, 131, 600, 386, 72000],
    ["5456-H116", "Plate", "32-38", 303, 214, 172, 172, 124, 579, 365, 72000],
    ["5456-H321", "Plate", "32-38", 303, 214, 172, 172, 124, 579, 365, 72000],
    ["5456-H116", "Plate", "38-76", 283, 200, 172, 172, 117, 565, 338, 72000],
    ["5456-H321", "Plate", "38-76", 283, 200, 172, 172, 117, 565, 338, 72000],

    ["6005-T5", "Extrusions", "Up to 25", 262, 241, 241, 165, 138, 552, 386, 70000],
    ["6105-T5", "Extrusions", "Up to 25", 262, 241, 241, 165, 138, 552, 386, 70000],
    ["6061-T6", "Sheet & plate", "0.25–102", 290, 241, 241, 186, 138, 607, 400, 70000],
    ["6061-T651", "Sheet & plate", "0.25–102", 290, 241, 241, 186, 138, 607, 400, 70000],
    ["6061-T6", "Extrusions", "Up to 25", 262, 241, 241, 165, 138, 551, 386, 70000],
    ["6061-T6510", "Extrusions", "Up to 25", 262, 241, 241, 165, 138, 551, 386, 70000],
    ["6061-T6511", "Extrusions", "Up to 25", 262, 241, 241, 165, 138, 551, 386, 70000],
    ["6061-T6", "Cold fin. rod & bar", "Up to 200", 290, 241, 241, 172, 138, 607, 386, 70000],
    ["6061-T651", "Cold fin. rod & bar", "Up to 200", 290, 241, 241, 172, 138, 607, 386, 70000],
    ["6061-T6", "Drawn tube", "0.6-12", 290, 241, 241, 186, 138, 607, 386, 70000],
    ["6061-T6", "Pipe", "0.6-12", 290, 241, 241, 186, 138, 607, 386, 70000],
    ["6061-T6", "Pipe", "Up to 25", 290, 241, 241, 186, 138, 607, 386, 70000],
]

# Adding data to the DataFrame
table_3_3_A = pd.concat([table_3_3_A, pd.DataFrame(data_3_3_a, columns=columns_3_3_a)], ignore_index=True)


# Table 3.4(A) - Values of Coefficients Kt and Kc
# Define the columns based on the table provided
columns_3_4_b = ["Alloy and temper", "Non-welded or regions farther than 25 mm from a weld_kt", "Non-welded or regions farther than 25 mm from a weld_kc",
                "Regions within 25 mm of a weld_kt", "Regions within 25 mm of a weld_kc"]

# Create an empty DataFrame with the defined columns
table_3_4_B = pd.DataFrame(columns=columns_3_4_b)

# Data from the provided table
data_3_4_b = [
    ["2014-T6", 1.25, 1.12, None, None],
    ["2014-T651", 1.25, 1.12, None, None],
    ["Alclad 2014-T6", 1.25, 1.12, None, None],
    ["Alclad 2014-T651", 1.25, 1.12, None, None],
    ["6005-T5", 1.0, 1.12, 1.0, 1.0],
    ["6061-T6", 1.0, 1.12, 1.0, 1.0],
    ["6061-T651", 1.0, 1.12, 1.0, 1.0],
    ["6063-T5", 1.0, 1.12, 1.0, 1.0],
    ["6063-T6", 1.0, 1.12, 1.0, 1.0],
    ["6063-T83", 1.0, 1.12, 1.0, 1.0],
    ["6105-T5", 1.0, 1.12, 1.0, 1.0],
    ["6351-T5", 1.0, 1.12, 1.0, 1.0],
    ["All others listed in Table 3.3(A)", 1.0, 1.10, 1.0, 1.0],
]
table_3_4_B = pd.concat([table_3_4_B, pd.DataFrame(data_3_4_b, columns=columns_3_4_b)], ignore_index=True)

# Add the data to the DataFrame using pd.concat

def kt_table_3_4_b(alloy_temper, welded_region):

    if welded_region == "Regions within 25 mm of a weld_kt":
        kt_value = table_3_4_B.loc[table_3_4_B["Alloy and temper"] == alloy_temper,"Regions within 25 mm of a weld_kt"].values[0]
    else:
        kt_value = table_3_4_B.loc[table_3_4_B["Alloy and temper"] == alloy_temper,"Non-welded or regions farther than 25 mm from a weld_kt"].values[0]
        # kt_value = filtered_row_3_4B["Non-welded or regions farther than 25 mm from a weld_kt"].values[0]

    return kt_value

def kc_table_3_4_b(alloy_temper, welded_region):

    if welded_region == "Regions within 25 mm of a weld_kc":
        kc_value = table_3_4_B.loc[table_3_4_B["Alloy and temper"] == alloy_temper,"Regions within 25 mm of a weld_kc"].values[0]
    else:
        kc_value = table_3_4_B.loc[table_3_4_B["Alloy and temper"] == alloy_temper,"Non-welded or regions farther than 25 mm from a weld_kc"].values[0]

    return kc_value


# Tension in Axial, Net Section - Section 3.4.2
def tension_axial_net_section_3_4_2(alloy_temper, product, welded_region, sectio_area):
    φ_y = 0.95
    φ_u = 0.85

    # Get the Tension_Fty* value
    tension_fty_value = table_3_3_A.loc[((table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product)), "Tension_Fty"].values[0]
    tension_ftu_value = table_3_3_A.loc[((table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product)), "Tension_Fty"].values[0]

    fl_yield_stress = φ_y * tension_fty_value
    fl_yield_force = fl_yield_stress * sectio_area
    kt = kt_table_3_4_b(alloy_temper, welded_region)
    fl_ultimate_stress = φ_u * tension_ftu_value/kt
    fl_ultimate_force = fl_ultimate_stress * sectio_area
    tension_capacity = min((fl_yield_force, fl_ultimate_force))
    return tension_capacity


# Compression Capacity - 3.4.8
def define_equivalent_slenderness_ratio_3_4_8_3(alloy_temper, product, Lb, Lt, section_area, rx, ry, G, J, xo, Cw):
    """
    Calculate the compression capacity based on torsional and torsional-flexural buckling.

    Parameters:
    E (float): Modulus of elasticity (MPa)
    G (float): Shear modulus (MPa)
    A (float): Cross-sectional area (mm^2)
    kx (float): Effective length factor for flexural buckling
    ko (float): Effective length factor for torsional buckling
    Lb (float): Unbraced length for bending (mm)
    Lt (float): Unbraced length for twisting (mm)
    rx (float): Radius of gyration about x-axis (mm)
    ry (float): Radius of gyration about y-axis (mm)
    J (float): Torsion constant
    Cw (float): Torsional warping constant

    Returns:
    float: Compression capacity
    """
    kx = 1
    ko = 1

    E = table_3_3_A.loc[
        (table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product), "Compressive modulus of elasticity_E"].values[0]
    # Calculate F_ex (Flexural buckling stress)
    F_ex = (np.pi ** 2 * E) / ((kx * Lb / rx) ** 2)

    # Calculate ro (Polar radius of gyration)
    ro = np.sqrt((1 / rx ** 2) + (1 / ry ** 2))
    beta = 1 - (xo/ro)**2
    # Calculate F_tb (Torsional buckling stress)
    F_tb = (1 / (section_area * ro ** 2)) * (G * J + (np.pi ** 2 * E * Cw) / ((ko * Lt) ** 2))

    # Calculate F_e (Elastic critical stress for torsional-flexural buckling)
    Fe_numerator = (F_ex + F_tb) - np.sqrt((F_ex + F_tb) ** 2 - 4 * beta * F_ex * F_tb)
    F_e = Fe_numerator / 2

    # Calculate equivalent slenderness ratio
    kL_r_e = np.pi * np.sqrt(E / F_e)

    # Calculate the larger slenderness ratio
    kL_r_flexural = kx * Lb / rx
    kL_r = max(kL_r_flexural, kL_r_e)


    return kL_r


def calculate_compression_capacity(alloy_temper, product, welded_region, Lb, Lt, section_area, rx, ry, G, J, xo, Cw):
    """
    Calculate the compression capacity of a column.

    Parameters:
    k (float): Effective length factor
    L (float): Unsupported length
    r (float): Radius of gyration
    Fcy (float): Compressive yield strength
    E (float): Modulus of elasticity
    kc (float): Coefficient from table
    Bc (float, optional): Value from equations, if required
    Cc (float, optional): Value from equations, if required

    Returns:
    float: Compression capacity
    """

    E = table_3_3_A.loc[
        (table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product), "Compressive modulus of elasticity_E"].values[0]
    Fcy = table_3_3_A.loc[
        (table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product), "Compression_Fcy"].values[0]

    kL_r = define_equivalent_slenderness_ratio_3_4_8_3(alloy_temper, product, Lb, Lt, section_area, rx, ry, G, J, xo, Cw)
    kc = kc_table_3_4_b(alloy_temper, welded_region)
    # Calculate slenderness parameter (λ)
    lambda_ = kL_r * (1 / np.pi) * np.sqrt(Fcy / E)

    # Determine φcc based on λ
    if lambda_ <= 1.2:
        phi_cc = 1 - 0.21 * (lambda_ ** 2)
    else:
        phi_cc = 0.14 * lambda_ + 0.58

    phi_cc = min(phi_cc, 0.95)



    # Table 3.3(D) Bc and Cc
    # filtered_row = table_3_3_A[(table_3_3_A["Alloy and temper"] == alloy_temper) & (table_3_3_A["Product"] == product)]

    # compression_fcy = filtered_row["Compression_Fcy"].values[0]

    Bc = Fcy * (1 + (Fcy / 15510)**0.5)
    Dc = (Bc / 10) * (Bc / E) ** 0.5
    Cc = 0.41 * Bc / Dc
    # Calculate S1* and S2* if Bc and Cc are provided
    # Calculate Dc*
    Dc_star = np.pi * Dc * np.sqrt(E/Fcy)
    S1_star = (Bc - Fcy / kc) / Dc_star
    S2_star = (Cc / np.pi) * np.sqrt(Fcy / E)

    # Calculate compression capacity based on λ
    if lambda_ < S1_star:
        compression_capacity = phi_cc * Fcy / kc
    elif lambda_ < S2_star and lambda_ > S1_star:
        compression_capacity = phi_cc * (Bc - Dc_star * lambda_)
    else:
        compression_capacity = (phi_cc / lambda_ ** 2) * Fcy

    return compression_capacity * section_area

