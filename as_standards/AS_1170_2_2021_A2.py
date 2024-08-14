# AS 1170.2:2021 method

# This method references the following standard:
# NZS 1170.4:2021, Amendment No.1 for New Zealand structures.

# Method developed 23 January 2023
# (c) BVT Consulting Ltd

# Developer - Nima Shokrollahi

import pandas as pd
import numpy as np

## Site wind speed
### Regional wind speeds (VR)


fig3_1_a = pd.DataFrame(
    [
        [
            "Adelaide",
            "A5",
        ],
        [
            "Albany",
            "A1",
        ],
        [
            "Albury/Wodonga",
            "A3",
        ],
        [
            "Alice Springs",
            "A0",
        ],
        [
            "Ballarat",
            "A3",
        ],
        [
            "Bathurst",
            "A0",
        ],
        [
            "Bendigo",
            "A3",
        ],
        [
            "Brisbane",
            "B1",
        ],
        [
            "Broome",
            "C",
        ],
        [
            "Bundaberg",
            "C",
        ],
        [
            "Burnie",
            "A4",
        ],
        [
            "Cairns",
            "C",
        ],
        [
            "Camden",
            "A2",
        ],
        [
            "Canberra",
            "A3",
        ],
        [
            "Carnarvon",
            "D",
        ],
        [
            "Coffs Harbour",
            "A2",
        ],
        [
            "Cooma",
            "A3",
        ],
        [
            "Dampier",
            "D",
        ],
        [
            "Darwin",
            "C",
        ],
        [
            "Derby",
            "C",
        ],
        [
            "Dubbo",
            "A0",
        ],
        [
            "Esperance",
            "A1",
        ],
        [
            "Geelong",
            "A5",
        ],
        [
            "Geraldton",
            "B2",
        ],
        [
            "Gladstone",
            "C",
        ],
        [
            "Gold Coast",
            "B1",
        ],
        [
            "Gosford",
            "A2",
        ],
        [
            "Grafton",
            "B1",
        ],
        [
            "Gippsland",
            "A5",
        ],
        [
            "Goulburn",
            "A3",
        ],
        [
            "Hobart",
            "A4",
        ],
        [
            "Karratha",
            "D",
        ],
        [
            "Katoomba",
            "A3",
        ],
        [
            "Latrobe Valley",
            "A5",
        ],
        [
            "Launceston",
            "A4",
        ],
        [
            "Lismore",
            "A5",
        ],
        [
            "Lorne",
            "A5",
        ],
        [
            "Mackay",
            "C",
        ],
        [
            "Maitland",
            "A2",
        ],
        [
            "Melbourne",
            "A5",
        ],
        [
            "Mittagong",
            "A3",
        ],
        [
            "Morisset",
            "A2",
        ],
        [
            "Newcastle",
            "A2",
        ],
        [
            "Noosa",
            "B1",
        ],
        [
            "Orange",
            "A0",
        ],
        [
            "Perth",
            "A1",
        ],
        [
            "Port Augusta",
            "A5",
        ],
        [
            "Port Lincoln",
            "A5",
        ],
        [
            "Port Hedland",
            "D",
        ],
        [
            "Port Macquarie",
            "A2",
        ],
        [
            "Port Pirie",
            "A5",
        ],
        [
            "Robe",
            "A5",
        ],
        [
            "Rockhampton",
            "C",
        ],
        [
            "Shepparton",
            "A5",
        ],
        [
            "Sydney",
            "A2",
        ],
        [
            "Tamworth",
            "A3",
        ],
        [
            "Taree",
            "A2",
        ],
        [
            "Tennant Creek",
            "A0",
        ],
        [
            "Toowoomba",
            "B1",
        ],
        [
            "Townsville",
            "C",
        ],
        [
            "Tweed Heads",
            "B1",
        ],
        [
            "Uluru",
            "A0",
        ],
        [
            "Wagga Wagga",
            "A3",
        ],
        [
            "Wangaratta",
            "A3",
        ],
        [
            "Whyalla",
            "A5",
        ],
        [
            "Wollongong",
            "A2",
        ],
        [
            "Woomera",
            "A0",
        ],
        [
            "Wyndham",
            "C",
        ],
        [
            "Wyong",
            "A2",
        ],
        [
            "Ballidu",
            "A1",
        ],
        [
            "Corrigin",
            "A0",
        ],
        [
            "Cunderdin",
            "A1",
        ],
        [
            "Dowerin",
            "C",
        ],
        [
            "Goomalling",
            "A1",
        ],
        [
            "Kellerberrin",
            "A0",
        ],
        [
            "Meckering",
            "A1",
        ],
        [
            "Northam",
            "A1",
        ],
        [
            "Wongan Hills",
            "A1",
        ],
        [
            "Wickepin",
            "A0",
        ],
        [
            "York",
            "A1",
        ],
        [
            "Christmas Island",
            "B2",
        ],
        [
            "Cocos Islands",
            "C",
        ],
        [
            "Heard Island",
            "A1",
        ],
        [
            "Lord Howe Island",
            "A2",
        ],
        [
            "Macquarie Island",
            "A1",
        ],
        [
            "Norfolk Island",
            "B1",
        ],
    ],
    columns=["Location", "Wind Region"],
)


def location_wind_region(location):
    return fig3_1_a[fig3_1_a["Location"] == location]["Wind Region"].values[0]


def Vr_table_3_1_A(R, location):

    location_region = location_wind_region(location)
    if (
        location_region == "A0"
        or location_region == "A1"
        or location_region == "A2"
        or location_region == "A3"
        or location_region == "A4"
        or location_region == "A5"
    ):
        if 1 / R < 5:
            Vr = 30
        else:
            Vr = np.round(67 - 41 * ((1 / R) ** -0.1))
    elif location_region == "B1" or location_region == "B2":
        if 1 / R < 5:
            Vr = 26
        else:
            Vr = np.round(106 - 92 * ((1 / R) ** -0.1))
    elif location_region == "C":
        if 1 / R < 5:
            Vr = 23
        else:
            Vr = np.round(122 - 104 * ((1 / R) ** -0.1))
    elif location_region == "D":
        if 1 / R < 5:
            Vr = 23
        else:
            Vr = np.round(156 - 142 * ((1 / R) ** -0.1))

    return Vr


## Md - Direction multiplier - Australian locations
# table_3_2_a = pd.DataFrame({
#     'Cardinal directions': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
#     'Region A0': [0.90, 0.85, 0.85, 0.90, 0.90, 0.95, 1.00, 0.95],
#     'Region A1': [0.90, 0.85, 0.85, 0.80, 0.80, 0.95, 1.00, 0.95],
#     'Region A2': [0.85, 0.75, 0.85, 0.95, 0.95, 0.95, 1.00, 0.95],
#     'Region A3': [0.90, 0.75, 0.75, 0.90, 0.90, 0.95, 1.00, 0.95],
#     'Region A4': [0.85, 0.75, 0.75, 0.80, 0.80, 0.90, 1.00, 1.00],
#     'Region A5': [0.95, 0.80, 0.80, 0.80, 0.80, 0.95, 1.00, 0.95],
#     'Region B1': [0.75, 0.75, 0.85, 0.90, 0.95, 0.95, 0.95, 0.90],
#     'Regions B2, C, D': [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
# }
# )

table_3_2_a = pd.DataFrame(
    [
        [
            "A0",
            1.0,
        ],
        [
            "A1",
            1.0,
        ],
        [
            "A2",
            1.0,
        ],
        [
            "A3",
            1.0,
        ],
        [
            "A4",
            1.0,
        ],
        [
            "A5",
            1.0,
        ],
        [
            "B1",
            0.95,
        ],
        [
            "B2",
            0.9,
        ],
        [
            "C",
            0.9,
        ],
        ["D", 0.9],
    ],
    columns=["Region", "Md"],
)

# def Md_table_3_2_a(location, direction):
#     region = location_wind_region(location)
#     if region in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1']:
#         column = f'Region {region}'
#     elif region in ['B2', 'C', 'D']:
#         column = 'Regions B2, C, D'
#     else:
#         raise ValueError("Invalid region.")

#     return table_3_2_a.loc[direction, column]


def Md_table_3_2_a(location):
    region = location_wind_region(location)
    return table_3_2_a[table_3_2_a["Region"] == region]["Md"].values[0]


### Climate change multiplier (Mc)
Table_3_3 = {
    "Region": ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "C", "D"],
    "Mc": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05],
}


def Mc_table_3_3(location):
    location_region = location_wind_region(location)
    return Table_3_3["Mc"][Table_3_3["Region"].index(location_region)]


### Terrain/height Multiplier (Mzcat)
Table4_1 = pd.DataFrame(
    [
        [3, 0.97, 0.91, 0.87, 0.83, 0.75],
        [5, 1.01, 0.91, 0.87, 0.83, 0.75],
        [10, 1.08, 1.00, 0.92, 0.83, 0.75],
        [15, 1.12, 1.05, 0.97, 0.89, 0.75],
        [20, 1.14, 1.08, 1.01, 0.94, 0.75],
        [30, 1.18, 1.12, 1.06, 1, 0.8],
        [40, 1.21, 1.16, 1.10, 1.04, 0.85],
        [50, 1.23, 1.18, 1.13, 1.07, 0.9],
        [75, 1.27, 1.22, 1.17, 1.12, 0.98],
        [100, 1.31, 1.24, 1.20, 1.16, 1.03],
        [150, 1.36, 1.27, 1.24, 1.21, 1.11],
        [200, 1.39, 1.29, 1.27, 1.24, 1.16],
    ],
    columns=["Height", "TC1", "TC2", "TC2.5", "TC3", "TC4"],
)


def interpolation(height):
    indices = Table4_1["Height"].searchsorted([height], side="right")
    lower_bound_index = indices[0] - 1
    upper_bound_index = indices[0] if indices[0] != len(Table4_1) else indices[0] - 1

    # Handling the boundary condition where height is less than the smallest value in the dataframe
    if lower_bound_index == -1:
        lower_bound_index = 0
        upper_bound_index = 0

    height_low = Table4_1["Height"][lower_bound_index]
    height_high = Table4_1["Height"][upper_bound_index]

    # Check for division by zero
    if height_high == height_low:
        interpolation_hn = 0
    else:
        interpolation_hn = (height - height_low) / (height_high - height_low)

    return interpolation_hn, lower_bound_index, upper_bound_index


def Mz_cat_table_4_1(height, Terrain_category):
    interpolation_height, lower_bound_index, upper_bound_index = interpolation(height)
    mz_cat_low = Table4_1[Terrain_category][lower_bound_index]
    mz_cat_high = Table4_1[Terrain_category][upper_bound_index]
    Mz_cat = mz_cat_low + interpolation_height * (mz_cat_high - mz_cat_low)

    return Mz_cat


## Ms - Shielding multiplier
### For structures more than 25 m should be taken as 1.0
### For structures less than 25 m, Ms is considered as 1.0 conservatively due to the lack of information
def Ms_4_3():
    return 1.0


## Mt - Topographic multiplier
cities_elevelatio = pd.DataFrame(
    [
        ["Adelaide", 50],
        ["Albany", 30],
        ["Albury/Wodonga", 165],
        ["Alice Springs", 576],
        ["Ballarat", 435],
        ["Bathurst", 670],
        ["Bendigo", 225],
        ["Brisbane", 27],
        ["Broome", 19],
        ["Bundaberg", 30],
        ["Burnie", 40],
        ["Cairns", 2.5],
        ["Camden", 70],
        ["Canberra", 577],
        ["Carnarvon", 4],
        ["Coffs Harbour", 5],
        ["Cooma", 800],
        ["Dampier", 5],
        ["Darwin", 30],
        ["Derby", 7],
        ["Dubbo", 275],
        ["Esperance", 30],
        ["Geelong", 15],
        ["Geraldton", 19],
        ["Gladstone", 16],
        ["Gold Coast", 12],
        ["Gosford", 20],
        ["Grafton", 6],
        ["Gippsland", 70],
        ["Goulburn", 690],
        ["Hobart", 17],
        ["Karratha", 9],
        ["Katoomba", 1017],
        ["Latrobe Valley", 60],
        ["Launceston", 15],
        ["Lismore", 12],
        ["Lorne", 5],
        ["Mackay", 11],
        ["Maitland", 10],
        ["Melbourne", 31],
        ["Mittagong", 635],
        ["Morisset", 10],
        ["Newcastle", 9],
        ["Noosa", 9],
        ["Orange", 862],
        ["Perth", 15],
        ["Port Augusta", 7],
        ["Port Lincoln", 9],
        ["Port Hedland", 6],
        ["Port Macquarie", 5],
        ["Port Pirie", 4],
        ["Robe", 4],
        ["Rockhampton", 10],
        ["Shepparton", 117],
        ["Sydney", 58],
        ["Tamworth", 404],
        ["Taree", 5],
        ["Tennant Creek", 377],
        ["Toowoomba", 691],
        ["Townsville", 11],
        ["Tweed Heads", 6],
        ["Uluru", 863],
        ["Wagga Wagga", 147],
        ["Wangaratta", 150],
        ["Whyalla", 20],
        ["Wollongong", 5],
        ["Woomera", 167],
        ["Wyndham", 11],
        ["Wyong", 10],
        ["Ballidu", 314],
        ["Corrigin", 320],
        ["Cunderdin", 240],
        ["Dowerin", 303],
        ["Goomalling", 248],
        ["Kellerberrin", 260],
        ["Meckering", 250],
        ["Northam", 170],
        ["Wongan Hills", 286],
        ["Wickepin", 360],
        ["York", 228],
        ["Christmas Island", 361],
        ["Cocos Islands", 3],
        ["Heard Island", 710],
        ["Lord Howe Island", 2],
        ["Macquarie Island", 20],
        ["Norfolk Island", 98],
    ],
    columns=["Location", "Elevation (m)"],
)


def Mt_4_4(location):
    region = location_wind_region(location)
    Mh = 1.71  # Is considere conservatily
    Mlee = 1.0
    E = cities_elevelatio[cities_elevelatio["Location"] == location][
        "Elevation (m)"
    ].values[0]
    if region == "A4" and E > 500:
        Mt = Mh * Mlee * (1 + 0.00015 * E)
    elif region == "A0":
        Mt = 0.5 + 0.5 * Mh
    else:
        Mt = max(Mh, Mlee)
    return Mt


######
## Section 5 - Aerodynamic shape factor
######
# Table 5.1(A) for buildings without openings greater than 0.5% of the wall area
cpi_building_table_5_1_a = pd.DataFrame(
    {
        "Condition": [
            "One Windward wall permeable",
            "One Windward wall impermeable",
            "Two or three Windward wall permeable",
            "Two or three Windward wall impermeable",
            "All walls permeable",
            "Effectively sealed",
        ],
        "Cpi": [0.8, -0.3, 0.2, -0.3, -0.3, -0.2],
    }
)

# Table 5.1(B) for buildings with openings greater than 0.5% of the wall area
# cpi_building_table_5_1_b = pd.DataFrame({
#     'Ratio': ['0.5 or less', '1', '2', '4', '6 or more'],
#     'Windward': [-0.3, -0.1, 0.7, 0.85, 0.85],
#     'Leeward': [-0.3, -0.3, 0.7, 0.85, 0.85],
#     'Side': [-0.3, -0.3, 0.7, 0.85, 0.85],
#     'Roof': [-0.3, -0.3, 0.7, 0.85, 0.85]
# })
cpi_building_table_5_1_b = pd.DataFrame(
    [
        [0.5, -0.3],
        [1, -0.3],
        [2, 0.7],
        [4, 0.85],
        [6, 0.85],
    ],
    columns=["Ratio", "Cpi"],
)


#     'Ratio': ['0.5 or less', '1', '2', '4', '6 or more'],
#     'Windward': [-0.3, -0.1, 0.7, 0.85, 0.85],
#     'Leeward': [-0.3, -0.3, 0.7, 0.85, 0.85],
#     'Side': [-0.3, -0.3, 0.7, 0.85, 0.85],
#     'Roof': [-0.3, -0.3, 0.7, 0.85, 0.85]
# })
def get_building_cpi_table_5_1(ratio, permeability):
    """
    Get the building Cpi based on the opening ratio, wall type, and permeability.

    :param ratio: Ratio of openings (use '0.5 or less', '1', '2', '4', '6 or more', or 'less than 0.5%')
    :param wall_type: 'Windward', 'Leeward', 'Side', or 'Roof'
    :param permeability: Required for Table 5.1(A) - 'permeable', 'impermeable', 'all permeable', or 'sealed'
    :return: Cpi value or range for the building
    """
    if ratio == "less than 0.5%":

        return cpi_building_table_5_1_a.loc[
            cpi_building_table_5_1_a["Condition"] == permeability, "Cpi"
        ].values[0]
    else:
        return cpi_building_table_5_1_b[cpi_building_table_5_1_b["Ratio"] == ratio][
            "Cpi"
        ].values[0]
        # return cpi_building_table_5_1_b.loc[cpi_building_table_5_1_b['Ratio'] == ratio, wall_type].values[0]


def get_internal_cpi_5_3_3(building_cpi, intenral_condition):
    """
    Get the internal Cpi for ceilings and partitions.

    :param building_cpi: Cpi value for the building
    :param intenral_condition: 'ceiling', 'adjacent', 'sealed', or 'unsealed'
    :return: Total internal Cpi
    """
    additional_cpi = {
        # "ceiling": 0.4,
        "adjacent": 0.2,  # Use abs(building_cpi) + 0.2 or abs(building_cpi) - 0.2, whichever is larger
        "sealed": 0.4,
        "unsealed": 0.3,
    }

    building_cpi = float(building_cpi)

    if intenral_condition == "adjacent":
        return max(abs(building_cpi + 0.2), abs(building_cpi - 0.2))
    else:
        return abs(building_cpi) + additional_cpi[intenral_condition]


def calculate_internal_cpi(ratio, intenral_condition, permeability=None):
    building_cpi = get_building_cpi_table_5_1(ratio, permeability)
    internal_cpi = get_internal_cpi_5_3_3(building_cpi, intenral_condition)
    return internal_cpi


def calculate_kv_5_3_4(A, Vol, is_largest_opening_on_wall=True):
    """
    Calculate the open area/volume factor, Kv.

    :param A: The open area on the wall
    :param Vol: The internal volume
    :param is_largest_opening_on_wall: Boolean indicating if the largest opening is on a wall
    :return: Kv value
    """
    if not is_largest_opening_on_wall:
        return 1.0

    ratio = 100 * (A ** (3 / 2)) / Vol

    if ratio < 0.09:
        return 0.85
    elif ratio > 3:
        return 1.085
    else:
        return 1.01 + 0.15 * np.log10(ratio)


def Cshp_5_2(
    ratio,
    intenral_condition,
    permeability,
    is_largest_opening_on_wall,
    kci=1.0,
    kv=1.0,
):
    cpi = calculate_internal_cpi(ratio, intenral_condition, permeability)
    kv = calculate_kv_5_3_4(0.5, 100, is_largest_opening_on_wall)
    cshp = cpi * kv * kci
    return cshp


def site_wind_speed(p, location, height, Terrain_category):

    Vr = Vr_table_3_1_A(p, location)
    Md = Md_table_3_2_a(location)
    Mt = Mt_4_4(location)
    Ms = Ms_4_3()
    Mz_cat_value = Mz_cat_table_4_1(height, Terrain_category)
    print("Vr: ", Vr, "Md: ", Md, "Mt: ", Mt, "Ms: ", Ms, "Mz_cat: ", Mz_cat_value)
    return Vr * Md * (Mz_cat_value * Ms * Mt)


def calc_wind_pressure(
    v_site,
    ratio,
    intenral_condition,
    permeability,
    kci=1.0,
    kv=1.0,
    Cdyn=1.0,
):

    C_shp = Cshp_5_2(
        ratio,
        intenral_condition,
        permeability,
        kci,
        kv,
    )
    rho_air = 1.2
    print("C_shp: ", C_shp)
    wind_pressure = 0.5 * rho_air * (v_site**2) * C_shp * Cdyn
    return wind_pressure


v = site_wind_speed(1 / 500, "Sydney", 20, "TC2")
print("v: ", v)
wind_pressure = calc_wind_pressure(
    v, 0.5, "sealed", "One Windward wall permeable", 1.0, 1.0, 1.0
)
print(wind_pressure)
