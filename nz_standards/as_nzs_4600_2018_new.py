import numpy as np

### Global Buckling


def foc_without_holes(r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw):
    """
    D1.1.1.1 Section not subject to torsional or flexural-torsional buckling:

    le = effective lenght of member
    r = raduis of gyration of the full, unreduced cross-section

    """
    G = 80 * 10**3  # MPa
    ro1 = np.sqrt(r_x**2 + r_y**2 + x_o**2 + y_o**2)
    beta = 1 - (x_o / ro1) ** 2

    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the x-axis
    fox = (np.pi**2 * E) / ((le_x / r_x) ** 2)
    # Elastic buckling stress in an axially loaded compression member for flexural buckling about the y-axis
    foy = (np.pi**2 * E) / ((le_y / r_y) ** 2)
    # Elastic buckling stress in an axially loaded compression member for torsional buckling
    foz = (G * J / Ag * ro1**2) * (1 + (np.pi**2 * E * Iw) / (G * J * le_z**2))

    foxz = (1 / 2 * beta) * (
        (fox + foy) - np.sqrt((fox - foz) ** 2 - 4 * beta * foz * fox)
    )

    foc = min(foxz, foy)

    return foc


def landa_C(r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw, fy):
    """
      Calculate the non-dimensional slenderness (λc).

    Parameters:
    Ny : float
        Nominal yield capacity of the member in compression
    Nc : float
        Least of the elastic compression member buckling load in flexural, torsional, and flexural-torsional buckling

    """
    foc = foc_without_holes(r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw)
    Ny = Ag * fy
    Noc = Ag * foc
    λc = np.sqrt(Ny / Noc)


    return λc


def compression_members_without_holes_7_2_1_2_1(
    r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw, fy
):
    """
    Adjust the nominal member capacity (Nc) for flexural, torsional or flexural-torsional buckling
    considering the influence of holes.

    Parameters:
    Nc : float
        Nominal member capacity without considering the holes (N or kips)
    fe : float
        Elastic local buckling stress (MPa or ksi)
    Ae : float
        Effective net area (mm^2 or in^2)

    Returns:
    float
        Adjusted nominal member capacity considering the holes (N or kips)
    """
    λc = landa_C(r_x, r_y, x_o, y_o, le_x, le_y, le_z, E, Ag, J, Iw, fy)

    if λc <= 1.5:
        Nce = 0.658 ** (λc**2) * Ag * fy
    else:
        Nce = 0.877 / (λc**2) * Ag * fy

    return Nce


### Local Buckling


def local_buckling_without_holes_7_2_1_3_1(Nce, Nol):
    """
    Calculate the nominal member capacity (Nc) for local buckling.

    Parameters:
    Ag : float
        Gross area of the member (mm^2 or in^2)
    Fy : float
        Yield stress of the material (MPa or ksi)
    λp : float
        Non-dimensional slenderness ratio for local buckling
    λr : float
        Non-dimensional slenderness ratio for inelastic local buckling
    Na : float
        Elastic local buckling load (N or kips)

    Returns:
    float
        Nominal member capacity in compression for local buckling (N or kips)
    """
    λl = (Nce / Nol) ** 0.5

    if λl <= 0.776:
        Ncl = Nce
    else:
        Ncl = Nce * ((1 - 0.15 * ((Nol / Nce) ** 0.4)) * ((Nol / Nce) ** 0.4))
    return Ncl


### Distortional buckling


def distorsional_buckling_without_holes_without_holes_7_2_1_4_1(Ag, fy, fod):
    Nod = Ag * fod
    λd = (Ag * fy / Nod) ** 0.5
    if λd <= 0.561:
        Ncd = Ag * fy
    else:
        Ncd = (
            (1 - 0.25 * (Nod / (Ag * fy)) ** 0.6) * ((Nod / (Ag * fy)) ** 0.6) * Ag * fy
        )

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
