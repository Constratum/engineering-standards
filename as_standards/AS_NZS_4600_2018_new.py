import numpy as np
import pandas as pd

from product_data import constratum_product_data_in

section_database = constratum_product_data_in.get_df_from_file_name(
    "section_database.csv"
)


## Section 5: Connection
### Section 5.4: Screwed Connections
### Screw connections in shear
def calculate_tension_in_connected_part_5_4_2_3(
    d_t, s_f, f_u, a_n=None, single_screw=True
):
    """
    Calculate the nominal tensile capacity (N_t) of the net section of the connected part
    based on the clauses given in the structural standard provided in the image.

    Parameters:
    φ (float): Capacity reduction factor of screwed connection subject to tension. = 0.65 for tension in the connected part - Table 1.6.3
    d_t (float): Nominal screw diameter.
    s_f (float): Spacing of screws perpendicular to the line of the force, or width of sheet, in the case of a single screw.
    f_u (float): Ultimate strength of the material.
    a_n (float, optional): Net area of the connected part. Required if multiple screws in the line parallel to the force.
    single_screw (bool): Indicates whether it's for a single screw/single row of screws or multiple screws in line.

    Returns:
    float: The design tensile force (N_t') on the net section of the connected part.
    """

    # For a single screw or a single row of screws perpendicular to the force
    if single_screw:
        N_t = min((2.5 * d_t / s_f) * a_n * f_u, a_n * f_u)
    else:
        # For multiple screws in the line parallel to the force
        if a_n is None:
            raise ValueError(
                "Net area (a_n) must be provided for multiple screws in line."
            )
        N_t = a_n * f_u

    # The design tensile force (N_t') is then calculated
    phi = 0.65
    N_tension = phi * N_t

    return N_tension


def get_bearing_factor_table_5_4_2_4(d_f, t1):
    """
    Determine the bearing factor (C) based on Table 5.4.2.4

    Parameters:
    d_t (float): Nominal screw diameter.
    t1 (float): Thickness of the sheet in contact with the screw head.

    Returns:
    float: The bearing factor (C).
    """
    # Ratio of fastener diameter to member thickness, d/t
    ratio = d_f / t1

    # Determine C based on the ratio
    if ratio < 6:
        C = 2.7
    elif ratio <= 13:
        C = 3.3 - 0.1 * ratio
    else:
        C = 2.0

    return C


def calculate_nominal_bearing_capacity(t2, t1, d_f, f_u1, f_u2):
    """
    Calculate the nominal bearing capacity (V_b) based on the provided parameters,
    taking into account the appropriate conditions from the standard.

    Parameters:
    t2 (float): Thickness of the sheet not in contact with the screw head.
    t1 (float): Thickness of the sheet in contact with the screw head.
    d_t (float): Nominal screw diameter.
    f_u1 (float): Tensile strength of the sheet in contact with the screw head.
    f_u2 (float): Tensile strength of the sheet not in contact with the screw head.

    Returns:
    float: The nominal bearing capacity (V_b) of the connected part.
    """
    # Get the bearing factor (C) from Table 5.4.2.4
    C = get_bearing_factor_table_5_4_2_4(d_f, t1)

    # Apply the condition for t2/t1 ≤ 1.0
    if t2 / t1 <= 1.0:
        V_b_options = [4.2 * (t1**1.5) * f_u1, C * d_f * f_u1, C * d_f * f_u2]
    # Apply the condition for t2/t1 > 1.0
    else:
        V_b_options = [C * d_f * f_u1, C * d_f * f_u2]

    # V_b shall be taken as the smallest of the calculated options
    V_b = min(V_b_options)

    return V_b


def calculate_effective_pull_over_diameter_5_4_3_2(d_h, t_w, d_w, t1, washer_type):
    """
    Calculate the effective pull-over diameter (d_p) based on clause 5.4.3.2(4) of the standard.

    Parameters:
    d_h (float): Screw head diameter or hex washer head integral washer diameter.
    t_w (float): Steel washer thickness.
    d_w (float): Steel washer diameter.

    Returns:
    float: The effective pull-over diameter (d_p).
    """
    if washer_type == "independent":
        d_p = min(d_h + 2 * t_w + t1, d_w)
    elif washer_type == "integral":
        # Should not be exceeding 20 mm
        d_p = min(d_h, 20)
    return d_p


def calculate_nominal_capacity_pull_out_5_4_3_2(
    d_h, d_f, t_w, d_w, t1, t2, f_u1, f_u2, washer_type
):
    """
    Calculate the nominal pull-out capacity (N_uo) and pull-through capacity (N_w)
    based on clauses 5.4.3.2(2) and 5.4.3.2(3) of the standard.

    Parameters:
    t (float): Thickness of the connected sheet.
    d_p (float): Effective pull-over diameter.
    f_u (float): Ultimate tensile strength of the material.

    Returns:
    float, float: The nominal pull-out capacity (N_uo), pull-through capacity (N_w).
    """

    d_p = calculate_effective_pull_over_diameter_5_4_3_2(d_h, t_w, d_w, t1, washer_type)
    # Nominal pull-out capacity for t > 0.9 mm
    N_uo = 0.85 * t2 * d_f * f_u2
    # Nominal pull-through capacity
    N_ov = 1.5 * t1 * d_p * f_u1

    Nt = min(N_uo, N_ov)
    phi = 0.5

    pull_out_capacity = phi * Nt
    return pull_out_capacity


## Section 7: Direct Strength Method
### Calculation of Compression
#### Appendix D
### Global Buckling


def foc_without_holes_D1_1_1_2(grid_name, le_x, le_y, le_z):
    """
    D1.1.1.1 Section not subject to torsional or flexural-torsional buckling:

    le = effective lenght of member
    r = raduis of gyration of the full, unreduced cross-section

    """
    E = section_database.loc[
        section_database["Section Name"] == grid_name, "E (MPa)"
    ].values[0]
    G = section_database.loc[
        section_database["Section Name"] == grid_name, "G (MPa)"
    ].values[0]
    J = section_database.loc[
        section_database["Section Name"] == grid_name, "Torsion Const J (mm4)"
    ].values[0]
    r_x = section_database.loc[
        section_database["Section Name"] == grid_name, "Radi of Gyration rx (mm)"
    ].values[0]
    r_y = section_database.loc[
        section_database["Section Name"] == grid_name, "Radi of Gyration ry (mm)"
    ].values[0]
    x_o = section_database.loc[
        section_database["Section Name"] == grid_name, "x0_shear (mm)"
    ].values[0]
    y_o = section_database.loc[
        section_database["Section Name"] == grid_name, "y0_shear (mm)"
    ].values[0]
    Iw = section_database.loc[
        section_database["Section Name"] == grid_name, "Warping Const Iw (mm4)"
    ].values[0]

    Ag = section_database.loc[
        section_database["Section Name"] == grid_name, "Area (mm2)"
    ].values[0]
    ro1 = np.sqrt(r_x**2 + r_y**2 + x_o**2 + y_o**2)
    beta = 1 - (x_o / ro1) ** 2
    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the x-axis
    fox = (np.pi**2 * E) / ((le_x / r_x) ** 2)
    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the y-axis
    foy = (np.pi**2 * E) / ((le_y / r_y) ** 2)
    # Elastic buckling stress in an axially loaded compression member for torsional buckling
    # foz = (G * J / Ag * ro1**2) * (1 + (np.pi**2 * E * Iw) / (G * J * le_z**2))
    foz = ((G * J) / (Ag * ro1**2)) * (1 + (np.pi**2 * E * Iw) / (G * J * le_z**2))
    # foxz = (1 / 2 * beta) * (
    #     (fox + foy) - np.sqrt((fox - foz) ** 2 - 4 * beta * foz * fox)
    # )
    foxz = (1 / (2 * beta)) * (
        (fox + foz) - (np.sqrt((fox + foz) ** 2 - (4 * beta * fox * foz)))
    )
    foc = min(foxz, foy)

    return fox, foy, foz, foxz, foc


def weighted_avg_cross_sectional_properties_table_D1_1_2_1(grid_name, Lg):
    x_o_g = section_database.loc[
        section_database["Section Name"] == grid_name, "x0_shear (mm)"
    ].values[0]
    y_o_g = section_database.loc[
        section_database["Section Name"] == grid_name, "y0_shear (mm)"
    ].values[0]

    A_g = section_database.loc[
        section_database["Section Name"] == grid_name, "Area (mm2)"
    ].values[0]

    Ixx_g = section_database.loc[
        section_database["Section Name"] == grid_name, "Ixx (mm4)"
    ].values[0]

    Iyy_g = section_database.loc[
        section_database["Section Name"] == grid_name, "Iyy (mm4)"
    ].values[0]
    A_net = section_database.loc[
        section_database["Section Name"] == grid_name, "Area Net (mm2)"
    ].values[0]

    L_net = section_database.loc[
        section_database["Section Name"] == grid_name, "Length Net x (mm)"
    ].values[0]

    x_o_net = section_database.loc[
        section_database["Section Name"] == grid_name, "x0_shear_net (mm)"
    ].values[0]

    y_o_net = section_database.loc[
        section_database["Section Name"] == grid_name, "y0_shear_net (mm)"
    ].values[0]

    Ixx_net = section_database.loc[
        section_database["Section Name"] == grid_name,
        "Moment of Inertia Ix with Hole (mm4)",
    ].values[0]

    Iyy_net = section_database.loc[
        section_database["Section Name"] == grid_name,
        "Moment of Inertia Iy with Hole (mm4)",
    ].values[0]

    J_net = section_database.loc[
        section_database["Section Name"] == grid_name, "Torsion Const with Hole (mm4)"
    ].values[0]

    """
    Calculate the weighted average of the cross-sectional properties of the member.

    Parameters:
    """

    A_avg = (A_g * Lg + A_net * L_net) / (Lg + L_net)
    Ix_avg = (Ixx_g * Lg + Ixx_net * L_net) / (Lg + L_net)
    Iy_avg = (Iyy_g * Lg + Iyy_net * L_net) / (Lg + L_net)
    J_avg = (J_net * L_net) / (L_net)
    ro1_avg = np.sqrt(x_o_avg**2 + y_o_avg**2 + (Ix_avg + Iy_avg) / A_avg)
    x_o_avg = ((x_o_g * Lg) + (x_o_net * L_net)) / L_net
    y_o_avg = ((y_o_g * Lg) + (y_o_net * L_net)) / L_net

    return A_avg, Ix_avg, Iy_avg, J_avg, ro1_avg, x_o_avg, y_o_avg


def foc_with_holes_D1_1_2_1(
    r_x_avg,
    r_y_avg,
    x_o_avg,
    y_o_avg,
    le_x,
    le_y,
    le_z,
    E,
    Ag,
    Ix_avg,
    Iy_avg,
    J_avg,
    Iw_net,
    G,
):
    ro1_avg = np.sqrt(r_x_avg**2 + r_y_avg**2 + x_o_avg**2 + y_o_avg**2)
    beta = 1 - (x_o_avg / ro1_avg) ** 2
    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the x-axis
    fox = (np.pi**2 * E * Ix_avg) / (Ag * le_x**2)
    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the y-axis
    foy = (np.pi**2 * E * Iy_avg) / (Ag * le_y**2)
    # Elastic buckling stress in an axially loaded compression member for torsional buckling
    # foz = (G * J / Ag * ro1**2) * (1 + (np.pi**2 * E * Iw) / (G * J * le_z**2))
    foz = ((G * J_avg) / (Ag * ro1_avg**2)) * (
        1 + (np.pi**2 * E * Iw_net) / (G * J_avg * le_z**2)
    )
    # foxz = (1 / 2 * beta) * (
    #     (fox + foy) - np.sqrt((fox - foz) ** 2 - 4 * beta * foz * fox)
    # )
    foxz = (1 / (2 * beta)) * (
        (fox + foz) - (np.sqrt((fox + foz) ** 2 - (4 * beta * fox * foz)))
    )
    foc = min(foxz, foy)

    return fox, foy, foz, foxz, foc


def compression_global_buckling_without_holes_7_2_1_2_1(Noc, Ny):
    """
    The nominal member capacity of a member in compression (Nce) for flexural,
    torsional or flexural-torsional buckling shall be calculated as follows:

    """
    λc = (Ny / Noc) ** 0.5

    if λc <= 1.5:
        Nce = (0.658 ** (λc**2)) * Ny
    else:
        Nce = (0.877 / (λc**2)) * Ny

    return Nce


def compression_global_buckling_with_holes_7_2_1_2_2(Noc, Ny):
    """
    The nominal member capacity of a member in compression (Nce) for flexural,
    torsional or flexural-torsional buckling of compression members with holes
    shall be calculated in accordance with Clause 7_2_1_2_1 except that foc shall be determined including the influence of holes.

    """
    λc = (Ny / Noc) ** 0.5

    if λc <= 1.5:
        Nce = (0.658 ** (λc**2)) * Ny
    else:
        Nce = (0.877 / (λc**2)) * Ny

    return Nce


### Local Buckling


def compression_local_buckling_without_holes_7_2_1_3_1(Nce, Nol):
    """
    Calculate the nominal member capacity (Nc) for local buckling.

    Parameters:
    float
        Nominal member capacity in compression for local buckling (N or kips)
    """
    λl = (Nce / Nol) ** 0.5

    if λl <= 0.776:
        Ncl = Nce
    else:
        Ncl = Nce * ((1 - 0.15 * ((Nol / Nce) ** 0.4)) * ((Nol / Nce) ** 0.4))
    return Ncl


def compression_local_buckling_with_holes_7_2_1_3_2(Nce, Nol, Ny_net):
    """
    Calculate the nominal member capacity (Nc) for local buckling.

    float
        Nominal member capacity in compression for local buckling (N or kips)
    """
    λl = (Nce / Nol) ** 0.5

    if λl <= 0.776:
        Ncl = min(Nce, Ny_net)
    else:
        Ncl = min(
            Nce * ((1 - 0.15 * ((Nol / Nce) ** 0.4)) * ((Nol / Nce) ** 0.4)), Ny_net
        )
    return Ncl


### Distortional buckling


def compression_distorsional_buckling_without_holes_7_2_1_4_1(Ny, Nod):
    λd = (Ny / Nod) ** 0.5
    if λd <= 0.561:
        Ncd = Ny
    else:
        Ncd = (1 - 0.25 * (Nod / Ny) ** 0.6) * ((Nod / Ny) ** 0.6) * Ny

    return Ncd


def compression_distorsional_buckling_with_holes_7_2_1_4_2(Ny, Nod, Ny_net):
    λd = (Ny / Nod) ** 0.5
    λd1 = 0.561 * (Ny_net / Ny)
    λd2 = 0.561 * (14 * ((Ny / Ny_net) ** 0.4) - 13)
    Nd2 = (1 - 0.25 * (1 / λd2) ** 1.2) * ((1 / λd2) ** 1.2) * Ny_net

    if λd <= λd1:
        Ncd = Ny_net
    else:
        Ncd = Ny_net - ((Ny_net - Nd2) / (λd2 - λd1)) * (λd - λd1)

    return Ncd


## Compression Capacity


def compression_capacity_7_2_1_1(Ncd, Ncl, Nce):
    """
    Calculate the compression capacity of the member.

    Parameters:
    Ncd : float
        Nominal member capacity in compression for distortional buckling (N or kips)
    Ncl : float
        Nominal member capacity in compression for local buckling (N or kips)
    Nce : float
        Nominal member capacity in compression for flexural, torsional or flexural-torsional buckling (N or kips)

    Returns:
    float
        Compression capacity of the member (N or kips)
    """
    Nc = min(Ncd, Ncl, Nce)
    return Nc


# Design of members subject to bending
## General Buckling
def calc_cb_D_2_1_1_2(fy, le):
    M_max = fy * le**2 / 8
    M3 = 7 * fy * le**2 / 128
    M4 = 3 * fy * le**2 / 32
    M5 = 15 * fy * le**2 / 128
    cb = 12.5 * M_max / (2.5 * M_max + 3 * M3 + 4 * M4 + 3 * M5)

    return cb


def calc_Beta_E2(t, xc, a, b, c, Iy, x_o, section_type):
    beta_w = ((1 * t * xc * a**3) / 12) + t * (xc**3) * a
    beta_f = 0.5 * t * ((b + xc) ** 4 - xc**4) + (
        0.25 * a**2 * t * ((b + xc) ** 2 - xc**2)
    )
    if section_type == "c_shaped":
        beta_l = 0.0
    elif section_type == "outer_lipped":
        beta_l = (2 * c * t * (xc + b) ** 3) + (
            2 / 3 * t * (xc + b) * ((a / 2 + c) ** 3 - (a / 2) ** 3)
        )
    else:
        beta_l = (2 * c * t * (xc + b) ** 3) + (
            2 / 3 * t * (xc + b) * ((a / 2) ** 3 - (a / 2 - c) ** 3)
        )

    beta_y = (beta_w + beta_f + beta_l / Iy) - 2 * x_o

    return beta_y


def calc_Mo_D_2_1_1_2_a(cb, Ag, ro1, foy, foz):
    Mo = cb * Ag * ro1 * (foy * foz) ** 0.5

    return Mo


def calc_Mo_D_2_1_1_2_b(Ag, ro1, beta_y, fox, foy, foz, lips_tension):
    ctf = 1.0  # if the bending moment at any point within an unbraced length is larger than that at both ends of this length, Ctf shall be taken as unity

    if lips_tension == "Yes":
        cs = 1.0
    else:
        cs = -1.0

    Mo = (
        cs
        * Ag
        * fox
        * ((beta_y / 2) + cs * ((beta_y / 2) ** 2 + (ro1**2 * foz / fox)) ** 0.5)
    ) / ctf

    return Mo


def bending_global_buckling_without_holes_7_2_2_2_2(My, Mo):
    if Mo < 0.56 * My:
        Mbe = Mo
    elif Mo >= 0.56 * My and Mo <= 2.78 * My:
        Mbe = (10 / 9) * My * (1 - (10 * My) / (36 * Mo))
    else:
        Mbe = My

    return Mbe


def bending_global_buckling_with_holes_7_2_2_2_3(My, Mo):
    if Mo < 0.56 * My:
        Mbe = Mo
    elif Mo >= 0.56 * My and Mo <= 2.78 * My:
        Mbe = (10 / 9) * My * (1 - (10 * My) / (36 * Mo))
    else:
        Mbe = My

    return Mbe


## Local Buckling
def bending_local_buckling_without_holes_7_2_2_3_2(Mbe, Mol):
    """

    Returns:
    float
        Nominal member capacity in compression for local buckling (N or kips)
    """
    λl = (Mbe / Mol) ** 0.5

    if λl <= 0.776:
        Mbl = Mbe
    else:
        Mbl = Mbe * ((1 - 0.15 * ((Mol / Mbe) ** 0.4)) * ((Mol / Mbe) ** 0.4))
    return Mbl


def bending_local_buckling_with_holes_7_2_2_3_3(Mbe, Mol, My_net):
    λl = (Mbe / Mol) ** 0.5

    if λl <= 0.776:
        Mbl = min(Mbe, My_net)
    else:
        Mbl = min(
            Mbe * ((1 - 0.15 * ((Mol / Mbe) ** 0.4)) * ((Mol / Mbe) ** 0.4)), My_net
        )
    return Mbl


## Distortional Buckling
def bending_distorsional_buckling_without_holes_7_2_2_4_2(My, Mod):
    λd = (My / Mod) ** 0.5
    if λd <= 0.673:
        Mbd = My
    else:
        Mbd = (1 - 0.25 * (Mod / My) ** 0.6) * ((Mod / My) ** 0.6) * My

    return Mbd


def bending_distorsional_buckling_with_holes_7_2_2_4_3(My, Mod, My_net):
    λd = (My / Mod) ** 0.5
    λd1 = 0.673 * (My_net / My) ** 3
    λd2 = 0.673 * (1.7 * ((My / My_net) ** 2.7) - 0.7)
    Md2 = (1 - 0.22 * (1 / λd2)) * ((1 / λd2)) * My

    if λd <= λd1:
        Mbd = My_net
    else:
        Mbd = My_net - ((My_net - Md2) / (λd2 - λd1)) * (λd - λd1)

    return Mbd


def bending_capacity_7_2_2_2(Mbd, Mbl, Mbe):
    """
    Calculate the bending capacity of the member.

    Parameters:
    Mbd : float
        Nominal member capacity in compression for distortional buckling (N or kips)
    Mbl : float
        Nominal member capacity in compression for local buckling (N or kips)
    Mbe : float
        Nominal member capacity in compression for flexural, torsional or flexural-torsional buckling (N or kips)

    Returns:
    float
        Compression capacity of the member (N or kips)
    """
    Mb = min(Mbd, Mbl, Mbe)
    return Mb
