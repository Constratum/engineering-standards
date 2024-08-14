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


fig3_1_b = pd.DataFrame(
    [
        [
            "Akaroa",
            "NZ4",
        ],
        [
            "Alexandra",
            "NZ2",
        ],
        [
            "Arrowtown",
            "NZ2",
        ],
        [
            "Arthurs Pass",
            "NZ2",
        ],
        [
            "Ashburton",
            "NZ2",
        ],
        [
            "Auckland",
            "NZ1",
        ],
        [
            "Balclutha",
            "NZ2",
        ],
        [
            "Blenheim",
            "NZ2",
        ],
        [
            "Bluff",
            "NZ4",
        ],
        [
            "Bulls",
            "NZ2",
        ],
        [
            "Cambridge",
            "NZ1",
        ],
        [
            "Cheviot",
            "NZ2",
        ],
        [
            "Christchurch",
            "NZ2",
        ],
        [
            "Cromwell",
            "NZ2",
        ],
        [
            "Dannevirke",
            "NZ2",
        ],
        [
            "Darfield",
            "NZ2",
        ],
        [
            "Dargaville",
            "NZ1",
        ],
        [
            "Dunedin",
            "NZ2",
        ],
        [
            "Eastbourne - Point Howard",
            "NZ3",
        ],
        [
            "Fairlie",
            "NZ2",
        ],
        [
            "Fielding",
            "NZ2",
        ],
        [
            "Fox Glacier",
            "NZ2",
        ],
        [
            "Foxton/Foxton Beach",
            "NZ2",
        ],
        [
            "Franz Josef",
            "NZ2",
        ],
        [
            "Geraldine",
            "NZ2",
        ],
        [
            "Gisborne",
            "NZ2",
        ],
        [
            "Gore",
            "NZ2",
        ],
        [
            "Greymouth",
            "NZ2",
        ],
        [
            "Hamilton",
            "NZ1",
        ],
        [
            "Hanmer Springs",
            "NZ2",
        ],
        [
            "Harihari",
            "NZ2",
        ],
        [
            "Hastings",
            "NZ2",
        ],
        [
            "Hawera",
            "NZ2",
        ],
        [
            "Hokitika",
            "NZ2",
        ],
        [
            "Huntly",
            "NZ1",
        ],
        [
            "Hutt Valley - south of Taita Gorge",
            "NZ3",
        ],
        [
            "Inglewood",
            "NZ2",
        ],
        [
            "Invercargill",
            "NZ4",
        ],
        [
            "Kaikohe",
            "NZ1",
        ],
        [
            "Kaikoura",
            "NZ2",
        ],
        [
            "Kaitaia",
            "NZ1",
        ],
        [
            "Kawerau",
            "NZ2",
        ],
        [
            "Levin",
            "NZ2",
        ],
        [
            "Manukau City",
            "NZ1",
        ],
        [
            "Mangakino",
            "NZ2",
        ],
        [
            "Marton",
            "NZ2",
        ],
        [
            "Masterton",
            "NZ2",
        ],
        [
            "Matamata",
            "NZ1",
        ],
        [
            "Mataura",
            "NZ2",
        ],
        [
            "Milford Sound",
            "NZ2",
        ],
        [
            "Morrinsville",
            "NZ1",
        ],
        [
            "Mosgiel",
            "NZ2",
        ],
        [
            "Motueka",
            "NZ2",
        ],
        [
            "Mount Maunganui",
            "NZ1",
        ],
        [
            "Mt Cook",
            "NZ2",
        ],
        [
            "Murchison",
            "NZ2",
        ],
        [
            "Murupara",
            "NZ2",
        ],
        [
            "Napier",
            "NZ2",
        ],
        [
            "Nelson",
            "NZ2",
        ],
        [
            "New Plymouth",
            "NZ2",
        ],
        [
            "Ngaruawahia",
            "NZ1",
        ],
        [
            "Oamaru",
            "NZ2",
        ],
        [
            "Oban",
            "NZ3",
        ],
        [
            "Ohakune",
            "NZ2",
        ],
        [
            "Opotiki",
            "NZ2",
        ],
        [
            "Opunake",
            "NZ2",
        ],
        [
            "Otaki",
            "NZ2",
        ],
        [
            "Otira",
            "NZ2",
        ],
        [
            "Otorohanga",
            "NZ2",
        ],
        [
            "Paeroa",
            "NZ1",
        ],
        [
            "Pahiatua",
            "NZ2",
        ],
        [
            "Paihia/Russell",
            "NZ1",
        ],
        [
            "Palmerston ",
            "NZ2",
        ],
        [
            "Palmerston North",
            "NZ2",
        ],
        [
            "Paraparaumu",
            "NZ3",
        ],
        [
            "Patea",
            "NZ2",
        ],
        [
            "Picton",
            "NZ3",
        ],
        [
            "Porirua",
            "NZ3",
        ],
        [
            "Pukekohe",
            "NZ2",
        ],
        [
            "Putaruru",
            "NZ2",
        ],
        [
            "Queenstown",
            "NZ2",
        ],
        [
            "Raetihi",
            "NZ2",
        ],
        [
            "Rangiora",
            "NZ2",
        ],
        [
            "Reefton",
            "NZ2",
        ],
        [
            "Riverton",
            "NZ4",
        ],
        [
            "Rotorua",
            "NZ2",
        ],
        [
            "Ruatoria",
            "NZ2",
        ],
        [
            "Seddon",
            "NZ3",
        ],
        [
            "Springs Junction",
            "NZ2",
        ],
        [
            "St Arnaud",
            "NZ2",
        ],
        [
            "Stratford",
            "NZ2",
        ],
        [
            "Taihape",
            "NZ2",
        ],
        [
            "Takaka ",
            "NZ2",
        ],
        [
            "Taumarunui",
            "NZ2",
        ],
        [
            "Taupo",
            "NZ2",
        ],
        [
            "Tauranga",
            "NZ1",
        ],
        [
            "Te Anau",
            "NZ2",
        ],
        [
            "Te Aroha",
            "NZ1",
        ],
        [
            "Te Awamutu",
            "NZ2",
        ],
        [
            "Te Kuiti",
            "NZ2",
        ],
        [
            "Te Puke",
            "NZ2",
        ],
        [
            "Temuka",
            "NZ2",
        ],
        [
            "Thames",
            "NZ1",
        ],
        [
            "Timaru",
            "NZ2",
        ],
        [
            "Tokoroa",
            "NZ2",
        ],
        [
            "Turangi",
            "NZ2",
        ],
        [
            "Twizel",
            "NZ2",
        ],
        [
            "Upper Hutt",
            "NZ3",
        ],
        [
            "Waihi",
            "NZ1",
        ],
        [
            "Waikanae",
            "NZ2",
        ],
        [
            "Waimate",
            "NZ2",
        ],
        [
            "Wainuiomata",
            "NZ3",
        ],
        [
            "Waiouru",
            "NZ2",
        ],
        [
            "Waipawa",
            "NZ2",
        ],
        [
            "Waipukurau",
            "NZ2",
        ],
        [
            "Wairoa",
            "NZ2",
        ],
        [
            "Waitara",
            "NZ2",
        ],
        [
            "Waiuku",
            "NZ1",
        ],
        [
            "Wanaka",
            "NZ2",
        ],
        [
            "Wanganui",
            "NZ2",
        ],
        [
            "Ward",
            "NZ1",
        ],
        [
            "Warkworth",
            "NZ1",
        ],
        [
            "Wellington",
            "NZ3",
        ],
        [
            "Wellington CBD",
            "NZ3",
        ],
        [
            "Westport",
            "NZ2",
        ],
        [
            "Whakatane",
            "NZ2",
        ],
        [
            "Whangarei",
            "NZ1",
        ],
        [
            "Winton",
            "NZ3",
        ],
        [
            "Woodville",
            "NZ2",
        ],
    ],
    columns=["Location", "Wind Region"],
)


def location_wind_region(location):
    return fig3_1_b[fig3_1_b["Location"] == location]["Wind Region"].values[0]


def Vr_table_3_1_b(R, location):

    location_region = location_wind_region(location)
    if location_region == "NZ1":
        Vr = np.round(61 - 30 * ((1 / R) ** -0.1))
    elif location_region == "NZ2":
        Vr = np.round(71 - 34 * ((1 / R) ** -0.1))
    elif location_region == "NZ3":
        Vr = np.round(63 - 25 * ((1 / R) ** -0.1))
    elif location_region == "NZ4":
        Vr = np.round(61 - 30 * ((1 / R) ** -0.1))
    else:
        Vr = np.nan
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

table_3_2_b = pd.DataFrame(
    [
        [
            "NZ1",
            1.0,
        ],
        [
            "NZ2",
            1.0,
        ],
        [
            "NZ3",
            1.0,
        ],
        [
            "NZ4",
            1.0,
        ],
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


def Md_table_3_2_b(location):
    region = location_wind_region(location)
    return table_3_2_b[table_3_2_b["Region"] == region]["Md"].values[0]


### Climate change multiplier (Mc)
table_3_3 = pd.DataFrame(
    [
        [
            "NZ1",
            1.0,
        ],
        [
            "NZ2",
            1.0,
        ],
        [
            "NZ3",
            1.0,
        ],
        [
            "NZ4",
            1.0,
        ],
    ],
    columns=["Region", "Mc"],
)


def Mc_table_3_3(location):
    region = location_wind_region(location)
    return table_3_3[table_3_3["Region"] == region]["Mc"].values[0]


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
        ["Akaroa", 0],  # Akaroa Harbour
        ["Alexandra", 138],
        ["Arrowtown", 480],
        ["Arthurs Pass", 739],
        ["Ashburton", 93],
        ["Auckland", 0],  # Sea level
        ["Balclutha", 9],
        ["Blenheim", 31],
        ["Bluff", 0],  # Sea level
        ["Bulls", 29],
        ["Cambridge", 40],
        ["Cheviot", 100],
        ["Christchurch", 20],
        ["Cromwell", 220],
        ["Dannevirke", 210],
        ["Darfield", 177],
        ["Dargaville", 8],
        ["Dunedin", 0],  # Sea level
        ["Eastbourne - Point Howard", 2],
        ["Fairlie", 296],
        ["Fielding", 74],
        ["Fox Glacier", 300],
        ["Foxton/Foxton Beach", 5],
        ["Franz Josef", 150],
        ["Geraldine", 106],
        ["Gisborne", 0],  # Sea level
        ["Gore", 123],
        ["Greymouth", 0],  # Sea level
        ["Hamilton", 39],
        ["Hanmer Springs", 385],
        ["Harihari", 61],
        ["Hastings", 14],
        ["Hawera", 75],
        ["Hokitika", 5],
        ["Huntly", 8],
        ["Hutt Valley - south of Taita Gorge", 0],  # Sea level
        ["Inglewood", 227],
        ["Invercargill", 0],  # Sea level
        ["Kaikohe", 200],
        ["Kaikoura", 5],
        ["Kaitaia", 80],
        ["Kawerau", 60],
        ["Levin", 30],
        ["Manukau City", 0],  # Sea level
        ["Mangakino", 440],
        ["Marton", 145],
        ["Masterton", 110],
        ["Matamata", 50],
        ["Mataura", 40],
        ["Milford Sound", 0],  # Sea level
        ["Morrinsville", 38],
        ["Mosgiel", 16],
        ["Motueka", 10],
        ["Mount Maunganui", 0],  # Sea level
        ["Mt Cook", 760],
        ["Murchison", 180],
        ["Murupara", 60],
        ["Napier", 2],
        ["Nelson", 0],  # Sea level
        ["New Plymouth", 0],  # Sea level
        ["Ngaruawahia", 15],
        ["Oamaru", 0],  # Sea level
        ["Oban", 0],  # Sea level
        ["Ohakune", 610],
        ["Opotiki", 0],  # Sea level
        ["Opunake", 0],  # Sea level
        ["Otaki", 10],
        ["Otira", 500],
        ["Otorohanga", 50],
        ["Paeroa", 5],
        ["Pahiatua", 200],
        ["Paihia/Russell", 0],  # Sea level
        ["Palmerston ", 270],
        ["Palmerston North", 20],
        ["Paraparaumu", 0],  # Sea level
        ["Patea", 0],  # Sea level
        ["Picton", 0],  # Sea level
        ["Porirua", 0],  # Sea level
        ["Pukekohe", 97],
        ["Putaruru", 190],
        ["Queenstown", 310],
        ["Raetihi", 564],
        ["Rangiora", 33],
        ["Reefton", 216],
        ["Riverton", 10],
        ["Rotorua", 290],
        ["Ruatoria", 44],
        ["Seddon", 20],
        ["Springs Junction", 306],
        ["St Arnaud", 640],
        ["Stratford", 315],
        ["Taihape", 500],
        ["Takaka ", 40],
        ["Taumarunui", 190],
        ["Taupo", 360],
        ["Tauranga", 0],  # Sea level
        ["Te Anau", 210],
        ["Te Aroha", 60],
        ["Te Awamutu", 53],
        ["Te Kuiti", 66],
        ["Te Puke", 30],
        ["Temuka", 30],
        ["Thames", 5],
        ["Timaru", 10],
        ["Tokoroa", 320],
        ["Turangi", 365],
        ["Twizel", 465],
        ["Upper Hutt", 60],
        ["Waihi", 55],
        ["Waikanae", 10],
        ["Waimate", 40],
        ["Wainuiomata", 25],
        ["Waiouru", 815],
        ["Waipawa", 150],
        ["Waipukurau", 137],
        ["Wairoa", 10],
        ["Waitara", 10],
        ["Waiuku", 0],  # Sea level
        ["Wanaka", 277],
        ["Wanganui", 0],  # Sea level
        ["Ward", 37],
        ["Warkworth", 60],
        ["Wellington", 0],  # Sea level
        ["Wellington CBD", 0],  # Sea level
        ["Westport", 0],  # Sea level
        ["Whakatane", 0],  # Sea level
        ["Whangarei", 0],  # Sea level
        ["Winton", 54],
        ["Woodville", 95],
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
    Mt = Mh * Mlee * (1 + 0.00015 * E)

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
    num, denom = p.split("/")
    R = int(num) / int(denom)
    Vr = Vr_table_3_1_b(R, location)
    Md = Md_table_3_2_b(location)
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


v = site_wind_speed("1 / 500", "Auckland", 20, "TC2")
print("v: ", v)
wind_pressure = calc_wind_pressure(
    v, 0.5, "sealed", "One Windward wall permeable", 1.0, 1.0, 1.0
)
print(wind_pressure)