import numpy as np
import pandas as pd

## Section 1.5.2: Design Stresses

data_1_5_2 = {
    "Applicable Standard": [
        "AS/NZS 1163", "AS/NZS 1163", "AS/NZS 1163", 
        
        "AS 1397", "AS 1397", "AS 1397", "AS 1397", "AS 1397", "AS 1397", "AS 1397","AS 1397",
         
        "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", 
        "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", 
        "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594", "AS/NZS 1594",
         
        "AS/NZS 1595", "AS/NZS 1595", "AS/NZS 1595", "AS/NZS 1595", "AS/NZS 1595",
         
        "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", 
        "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", 
        "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", 
        "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678",
        "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678", "AS/NZS 3678",
        "AS/NZS 3678"
        
    ],
    "Grade": [
        "C250 and C250L0", "C350 and C350L0", "C450 and C450L0", "G250", "G300", "G350", "G450", "G500", "G550‡ (t ≥ 0.9 mm)",
        "G550‡ (0.9 mm > t ≥ 0.6 mm)", "G550‡ (t < 0.6 mm)", "HA1", "HA3", "HA4N", "HA200", "HA250, HU250", "HA250/1",
        "HA300, HU300", "HA300/1, HU300/1", "HW350", "HW350", "HA400", "XF300", "XF400", "XF500", "CA220", "CA260", "CW300", "CA350",
        "CA500", "200 (t ≤ 8 mm)", "200 (8 mm < t ≤ 12 mm)", "250, 250L15 (t ≤ 8 mm)", "250, 250L15 (8 mm < t ≤ 12 mm)",
        "250, 250L15 (12 mm < t ≤ 20 mm)", "250, 250L15 (20 mm < t ≤ 25 mm)", "300, 300L15 (t ≤ 8 mm)", "300, 300L15 (8 mm < t ≤ 12 mm)",
        "300, 300L15 (12 mm < t ≤ 20 mm)", "300, 300L15 (20 mm < t ≤ 25 mm)", "350, 350L15 (t ≤ 8 mm)", "350, 350L15 (8 mm < t ≤ 12 mm)",
        "350, 350L15 (12 mm < t ≤ 20 mm)", "350, 350L15 (20 mm < t ≤ 25 mm)", "400, 400L15 (t ≤ 8 mm)", "400, 400L15 (8 mm < t ≤ 12 mm)",
        "400, 400L15 (12 mm < t ≤ 20 mm)", "400, 400L15 (20 mm < t ≤ 25 mm)", "450, 450L15 (t ≤ 8 mm)", "450, 450L15 (8 mm < t ≤ 12 mm)",
        "450, 450L15 (12 mm < t ≤ 20 mm)", "450, 450L15 (20 mm < t ≤ 25 mm)", "WR350, WR350/L0 (t ≤ 8 mm)", "WR350, WR350/L0 (8 mm < t ≤ 12 mm)",
        "WR350, WR350/L0 (12 mm < t ≤ 20 mm)", "WR350, WR350/L0 (20 mm < t ≤ 25 mm)"
    ],
    "Yield stress (fy) MPa": [
        250, 350, 450, 250, 300, 350, 450, 500, 550, 495, 413, 200, 200, 170, 200, 250, 250, 300, 300, 350, 340, 380, 300, 380,
        480, 210, 250, 300, 350, 500, 200, 200, 280, 260, 250, 250, 320, 310, 300, 280, 360, 360, 350, 340, 400, 400, 380, 360, 450,
        450, 450, 420, 340, 340, 340, 340
    ],
    "Tensile strength (fu) MPa": [
        320, 430, 500, 320, 340, 420, 480, 520, 550, 495, 413, 300, 300, 280, 300, 350, 350, 400, 430, 430, 450, 460, 440, 460, 570,
        340, 350, 450, 430, 510, 300, 300, 410, 410, 410, 410, 430, 430, 430, 430, 450, 450, 450, 450, 480, 480, 480, 480, 520,
        520, 520, 500, 450, 450, 450, 450
    ]
}

# Create DataFrame
table_1_5_2 = pd.DataFrame(data_1_5_2)

## Section 3: Axial Tension
def tension_section_capacity_3_2_1(fy, Ag, fu= None, An = None):
    phi_t = 0.8
    Nt_g = phi_t * Ag * fy
    if An != None and fu != None:
        kt = 1.0
        Nt_net = phi_t * (0.85 * kt * An * fu)
        Nt = min(Nt_g, Nt_net)
    else:
        Nt = Nt_g

return Nt
## Section 5: Connection
### Section 5.3: Bolted Connections
def modified_net_shear_area(plate_width, t, hole_diameter, no_bolts):
    """
    Calculate the modified net shear area for bolted connections in cold-formed steel structures.
        
    Returns:
    float: Modified net shear area.
    """
    

    # Calculating net area reduction due to bolt holes
    # Assuming staggered holes impact according to sf (stagger factor)
    net_area_reduction = (plate_width - (no_bolts * hole_diameter)) * t

    return net_area_reduction

### Section 5.3.2: Tearout
def design_shear_force_tearout_5_3_2(fy_bolt, fu_bolt, t, e):
    """
    Calculate the design shear force considering tearout for a connected part.
    
    Parameters:
    t : float
        Thickness of the connected part (in mm or appropriate unit).
    e : float
        Edge distance from the bolt to the nearest edge of an adjacent hole or end of the connected part (in mm or appropriate unit).
    fu : float
        Ultimate tensile strength of the material (in MPa or appropriate unit).
    fy : float
        Yield strength of the material (in MPa or appropriate unit).
        
    Returns:
    Vr_star : float
        The design shear force considering tearout (in kN or appropriate unit).
    """
    # Determine the capacity reduction factor (φ) based on the ratio of fu to fy
    if fu_bolt / fy_bolt >= 1.05:
        phi = 0.70
    else:
        phi = 0.60
    
    # Calculate the nominal shear capacity (Vr)
    Vf = t * e * fu_bolt
    
    # Calculate the design shear force considering tearout (Vr_star)
    Vf_tearout = phi * Vf
    
    return Vf_tearout

# Data for Table 5.3.4.2(A)
data_table_5_3_4_2_a = {
    "Type of bearing": [
        "Single shear and outside sheets of double shear connection with washers under both bolt head and nut",
        "Single shear and outside sheets of double shear connection without washers under both head and nut, or with only one washer",
        "Single shear and outside sheets of double shear connection using oversized or short-slotted holes parallel to the applied load without washers under both head and nut, or with only one washer",
        "Single shear and outside sheets of double shear connection using short-slotted holes perpendicular to the applied load without washers under both head and nut, or with only one washer",
        "Inside sheet of double shear connection with or without washers",
        "Inside sheet of double shear connection using oversized or short slotted holes parallel to the applied load with or without washers",
        "Inside sheet of double shear connection using short slotted holes perpendicular to the applied load with or without washers"
    ],
    "alpha": [1.00, 0.75, 0.70, 0.55, 1.33, 1.10, 0.90]
}
# Create DataFrame for Table 5.3.4.2(A)
table_5_3_4_2_a = pd.DataFrame(data_table_5_3_4_2_a)

#### Table 5_3_4_2_B: Bearing Factor (C)
def table_5_3_4_2_b(t, d):
    """
    t = Thickness of the connected part (in mm or appropriate unit) 
    d = Fastener diameter (in mm or appropriate unit)
    """
    if d/t <10:
        C = 3
    elif d/t >22:
        C = 1.8
    else:
        C = 4 - 0.1*d/t
    return C

def bearing_capacity_5_3_4_2(fu_bolt, bearing_type, d_bolt, t):
    """
    Calculate the nominal bearing capacity for a bolted connection.
    
    Parameters:
    alpha : float
        Modification factor for the type of bearing connection.
    C : float
        Bearing factor.
    d_bolt : float
        Nominal bolt diameter (in mm or appropriate unit).
    t : float
        Base metal thickness (in mm or appropriate unit).
    fu : float
        Tensile strength of the sheet (in MPa or appropriate unit).
        
    Returns:
    Vb : float
        The nominal bearing capacity (in kN or appropriate unit).
    """
    alpha = table_5_3_4_2_a.loc[table_5_3_4_2_a["Type of bearing"] == bearing_type, "alpha"].values[0]
    C = table_5_3_4_2_b(t, d_bolt)
    phi = 0.60  # Capacity reduction factor for bearing capacity without considering bolt hole deformation
    Vb = alpha * C * d_bolt * t * fu_bolt * phi
    return Vb

def bolt_shear_capacity_5_3_5_1(bolt_grade, n_n, Ac, n_x, Ao):
    """
    Calculate the nominal shear capacity of a bolt.

    Parameters:
    f_uf : float
        Minimum tensile strength of a bolt (in MPa).
    n_n : int
        Number of shear planes with threads intercepting the shear plane.
    A_d : float
        Minor diameter area of a bolt (in mm^2).
    n_x : int
        Number of shear planes without threads intercepting the shear plane.
    A_o : float
        Plain shank area of a bolt (in mm^2).

    Returns:
    V_rv : float
        The nominal shear capacity of a bolt (in kN).
    """
    # Capacity reduction factor (φ)
    phi = 0.8
    
    if bolt_grade == "4.6":
        f_uf = 400
    else:
        f_uf = 830

    # Calculate the nominal shear capacity (V_rv)
    V_rv = 0.62 * f_uf * (n_n * Ac + n_x * Ao) * phi

    return V_rv 

def bolt_tension_capacity_5_3_5_2(bolt_grade, As_tension):
    
    phi = 0.9
    
    if bolt_grade == "4.6":
        f_uf = 400
    else:
        f_uf = 830
    Nft = phi * f_uf * As_tension
    return Nft

def bolt_combined_shear_tension_5_3_5_3(Nu, Vu, bolt_grade, As_tension, n_n, Ac, n_x, Ao):
    Nft = bolt_tension_capacity_5_3_5_2(bolt_grade, As_tension)
    Vfv = bolt_shear_capacity(bolt_grade, n_n, Ac, n_x, Ao)
    
    unity = (Nu / Nft)**2 + (Vu / Vfv)**2
    if unity <= 1:
        compliance = True
    else:
        compliance = False
    return unity, compliance

## Section 5: Connections
### Section 5.4: Screwed Connections

def calculate_tension_in_connected_part_5_4_2_3(
    d_f, s_f, f_u, a_n=None, single_screw=True
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
        N_t = min((2.5 * d_f / s_f) * a_n * f_u, a_n * f_u)
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


def calculate_nominal_bearing_capacity_5_4_2_4(t2, t1, d_f, f_u1, f_u2):
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
        v_b = min(
            4.2 * (t2**1.5) * (d_f**0.5) * f_u2,
            C * t1 * d_f * f_u1,
            C * t2 * d_f * f_u2,
        )
    # Apply the condition for t2/t1 > 1.0
    if t2 / t1 >= 2.5:
        v_b = min(C * t1 * d_f * f_u1, C * t2 * d_f * f_u2)
    if 1.0 < t2 / t1 < 2.5:
        v_b_1 = min(
            4.2 * (t2**1.5) * (d_f**0.5) * f_u2,
            C * t1 * d_f * f_u1,
            C * t2 * d_f * f_u2,
        )
        v_b_2 = min(C * t1 * d_f * f_u1, C * t2 * d_f * f_u2)
        v_b = v_b_1 + v_b_2 / 2
    # V_b shall be taken as the smallest of the calculated options

    return v_b


def conc_shear_tearout_5_4_2_5(fu, fy, t, e):
    """
    Calculate the design shear force as limited by end distance shall satifisy
    V_fv<phi*V_fv

    Parameters:
    fu (float): Ultimate tensile strength of the connected part.
    fy (float): Yield strength of the connected part.
    t (float): Thickness of the connected part.
    e (float): Edge distance from the center of the fastener to the edge of the connected part.

    """
    if fu / fy >= 1.05:
        phi = 0.7
    if fu / fy < 1.05:
        phi = 0.6

    v_fv = phi * t * e * fu
    return v_fv


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


# Section 7: Direct Strength Method

### Global Buckling


def foc_without_holes_D1_1_1_2(r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw, G):
    """
    D1.1.1.1 Section not subject to torsional or flexural-torsional buckling:

    le = effective lenght of member
    r = raduis of gyration of the full, unreduced cross-section

    """
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
    print("ro1", ro1)
    return fox, foy, foz, foxz, foc


def weighted_avg_cross_sectional_properties_table_D1_1_2_1(
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


def calc_Mo_D_2_1_1_2_b(Ag, ro1, beta_y, fox, foz, lips_tension):
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
