import numpy as np
import pandas as pd

section_database = pd.read_csv("section_database.csv")


def initial_properties(grid_name):
    ### Global buckling

    E = section_database.loc[
        section_database["Section Name"] == grid_name, "E (MPa)"
    ].values[0]
    G = section_database.loc[
        section_database["Section Name"] == grid_name, "G (MPa)"
    ].values[0]
    # effective length
    rx = section_database.loc[
        section_database["Section Name"] == grid_name, "Radi of Gyration rx (mm)"
    ].values[0]
    ry = section_database.loc[
        section_database["Section Name"] == grid_name, "Radi of Gyration ry (mm)"
    ].values[0]
    x_o = section_database.loc[
        section_database["Section Name"] == grid_name, "x0_shear (mm)"
    ].values[0]
    y_o = section_database.loc[
        section_database["Section Name"] == grid_name, "y0_shear (mm)"
    ].values[0]

    J = section_database.loc[
        section_database["Section Name"] == grid_name, "Torsion Const J (mm4)"
    ].values[0]
    Iw = section_database.loc[
        section_database["Section Name"] == grid_name, "Warping Const Iw (mm4)"
    ].values[0]

    Ag = section_database.loc[
        section_database["Section Name"] == grid_name, "Area (mm2)"
    ].values[0]

    return E, G, rx, ry, x_o, y_o, J, Iw, Ag


### Global Buckling


def foc_without_holes(grid_name, le_x, le_y, le_z):
    """
    D1.1.1.1 Section not subject to torsional or flexural-torsional buckling:

    le = effective lenght of member
    r = raduis of gyration of the full, unreduced cross-section

    """
    E, G, r_x, r_y, x_o, y_o, J, Iw, Ag = initial_properties(grid_name)
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


def weighted_avg_cross_sectional_properties(
    A_g, A_net, Lg, L_net, Ig, I_net, J_net, x_o_net, y_o_net, x_o_g, y_o_g
):
    """
    Calculate the weighted average of the cross-sectional properties of the member.

    Parameters:
    """
    A_avg = (A_g * Lg + A_net * L_net) / (Lg + L_net)
    Ix_avg = (Ig * Lg + I_net * L_net) / (Lg + L_net)
    Iy_avg = (Ig * Lg + I_net * L_net) / (Lg + L_net)
    J_avg = (J_net * L_net) / (L_net)
    ro1_avg = np.sqrt(x_o_avg**2 + y_o_avg**2 + (Ix_avg + Iy_avg) / A_avg)
    x_o_avg = (x_o_net * L_net) / (L_net)
    y_o_avg = (y_o_net * L_net) / (L_net)

    return A_avg, Ix_avg, Iy_avg, J_avg, ro1_avg, x_o_avg, y_o_avg


def foc_with_holes(
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


# def landa_C(foc, Ny, Ag):
#     """
#       Calculate the non-dimensional slenderness (λc).

#     Parameters:
#     Ny : float
#         Nominal yield capacity of the member in compression
#     Nc : float
#         Least of the elastic compression member buckling load in flexural, torsional, and flexural-torsional buckling

#     """
#     Noc = Ag * foc
#     λc = np.sqrt(Ny / Noc)

#     return λc


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


def compression_distorsional_buckling_without_holes_without_holes_7_2_1_4_1(Ny, Nod):
    λd = (Ny / Nod) ** 0.5
    if λd <= 0.561:
        Ncd = Ny
    else:
        Ncd = (1 - 0.25 * (Nod / Ny) ** 0.6) * ((Nod / Ny) ** 0.6) * Ny

    return Ncd


def compression_distorsional_buckling_with_holes_without_holes_7_2_1_4_2(
    Ny, Nod, Ny_net
):
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


def compression_capacity(Ncd, Ncl, Nce):
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
