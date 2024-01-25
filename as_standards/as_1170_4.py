#This method references the following standard:
#AS 1170.4:2007, R2018 - Earthquake actions in Australia

#Created by *Nima SHokrollahi* on *29 June 2023* 

#Copyright (c) Constratum Ltd 2022




import pandas as pd
import numpy as np


# Earthquake Design Category
table2_1 = pd.DataFrame(
    [
        [2, "<=0.05", "<=0.05", "<=0.08", "<=0.11", "<=0.14", "<=12", "I"],
        [2, "<=0.05", "<=0.05", "<=0.08", "<=0.11", "<=0.14", ">12, <50", "II"],
        [2, "<=0.05", "<=0.05", "<=0.08", "<=0.11", "<=0.14", ">50", "III"],
        [
            2,
            ">0.05, <=0.08",
            ">0.05, <=0.08",
            ">0.08, <=0.12",
            ">0.11, <=0.17",
            ">0.14, <=0.21",
            "<50",
            "II",
        ],
        [
            2,
            ">0.05, <=0.08",
            ">0.05, <=0.08",
            ">0.08, <=0.12",
            ">0.11, <=0.17",
            ">0.14, <=0.21",
            ">=50",
            "III",
        ],
        [2, ">0.08", ">0.08", ">0.12", ">0.17", ">0.21", "<25", "II"],
        [2, ">0.08", ">0.08", ">0.12", ">0.17", ">0.21", ">=25", "III"],
        [3, "<=0.08", "<=0.08", "<=0.12", "<=0.17", "<=0.21", "<50", "II"],
        [3, "<=0.08", "<=0.08", "<=0.12", "<=0.17", "<=0.21", ">=50", "III"],
        [3, ">0.08", ">0.08", ">0.12", ">0.17", ">0.21", "<25", "II"],
        [3, ">0.08", ">0.08", ">0.12", ">0.17", ">0.21", ">=25", "III"],
        [4, "", "", "", "", "", "<12", "II"],
        [4, "", "", "", "", "", ">=12", "III"],
    ],
    columns=[
        "Importance level",
        "kpz_E",
        "kpz_D",
        "kpz_C",
        "kpz_B",
        "kpz_A",
        "h_n",
        "EDC",
    ],
)


def check_condition(value, condition):
    # Split the condition into individual comparisons
    comparisons = condition.split(", ")

    # Handle empty condition
    if condition == "":
        return True

    for comparison in comparisons:
        if "<=" in comparison:
            op, num = comparison.split("<=")
            if not (value <= float(num)):
                return False
        elif ">=" in comparison:
            op, num = comparison.split(">=")
            if not (value >= float(num)):
                return False
        elif "<" in comparison:
            op, num = comparison.split("<")
            if not (value < float(num)):
                return False
        elif ">" in comparison:
            op, num = comparison.split(">")
            if not (value > float(num)):
                return False

    return True


def get_EDC(IL, kpz, subsoil, h_n):
    kpz_subsoil = "kpz_" + subsoil

    # Iterate over each row in the DataFrame
    for index, row in table2_1.iterrows():
        if row["Importance level"] == IL:
            if check_condition(kpz, row[kpz_subsoil]) and check_condition(
                h_n, row["h_n"]
            ):
                return row["EDC"]

    return "EDC not found"
# Site Hazard
## Annual probability of exceedance (P) and probability factor (kp)
### Table 3.1 - Probability Factor ($k_p$)

table3_1 = pd.DataFrame(
    {
        "Annual probability of exceedance (P)": [
            "1/2500",
            "1/2000",
            "1/1500",
            "1/1000",
            "1/800",
            "1/500",
            "1/250",
            "1/200",
            "1/100",
            "1/50",
            "1/25",
            "1/20",
        ],
        "Probability factor (kp)": [
            1.8,
            1.7,
            1.5,
            1.3,
            1.25,
            1.0,
            0.75,
            0.7,
            0.5,
            0.35,
            0.25,
            0.2,
        ],
    }
)

# table3_1


def return_period_factor(P):
    kp = table3_1.loc[
        table3_1["Annual probability of exceedance (P)"] == P, "Probability factor (kp)"
    ].squeeze()

    return kp


# @title Table 3.2 - Hazard factor ( 𝑍 ) For Specific Australian Locations { vertical-output: true }

table3_2 = pd.DataFrame(
    [
        ["Adelaide", 0.10],
        ["Albany", 0.08],
        ["Albury/Wodonga", 0.09],
        ["Alice Springs", 0.08],
        ["Ballarat", 0.08],
        ["Bathurst", 0.08],
        ["Bendigo", 0.09],
        ["Brisbane", 0.05],
        ["Broome", 0.12],
        ["Bundaberg", 0.11],
        ["Burnie", 0.07],
        ["Cairns", 0.06],
        ["Camden", 0.09],
        ["Canbern", 0.08],
        ["Carnarvon", 0.09],
        ["Coffs Harbour", 0.05],
        ["Cooma", 0.08],
        ["Dampier", 0.12],
        ["Darwin", 0.09],
        ["Derby", 0.09],
        ["Dubbo", 0.08],
        ["Esperance", 0.09],
        ["Geelong", 0.10],
        ["Geraldton", 0.09],
        ["Gladstone", 0.09],
        ["Gold Coast", 0.05],
        ["Gosford", 0.09],
        ["Grafton", 0.05],
        ["Gippsland", 0.10],
        ["Goulburn", 0.09],
        ["Hobart", 0.03],
        ["Karratha", 0.12],
        ["Katoomba", 0.09],
        ["Latrobe Valley", 0.10],
        ["Launceston", 0.04],
        ["Lismore", 0.05],
        ["Lorne", 0.10],
        ["Mackay", 0.07],
        ["Maitland", 0.10],
        ["Melbourne", 0.08],
        ["Mittagong", 0.09],
        ["Morisset", 0.10],
        ["Newcastle", 0.11],
        ["Noosa", 0.08],
        ["Orange", 0.08],
        ["Perth", 0.09],
        ["Port Augusta", 0.11],
        ["Port Lincoln", 0.10],
        ["Port Hedland", 0.12],
        ["Port Macquarie", 0.06],
        ["Port Pirie", 0.10],
        ["Robe", 0.10],
        ["Rockhampton", 0.08],
        ["Shepparton", 0.09],
        ["Sydney", 0.08],
        ["Tamworth", 0.07],
        ["Taree", 0.08],
        ["Tennant Creek", 0.13],
        ["Toowoomba", 0.06],
        ["Townsville", 0.07],
        ["Tweed Heads", 0.05],
        ["Uluru", 0.08],
        ["Wagga Wagga", 0.09],
        ["Wangaratta", 0.09],
        ["Whyalla", 0.09],
        ["Wollongong", 0.09],
        ["Woomera", 0.08],
        ["Wyndham", 0.09],
        ["Wyong", 0.10],
        ["Ballidu", 0.15],
        ["Corrigin", 0.14],
        ["Cunderdin", 0.22],
        ["Dowerin", 0.20],
        ["Goomalling", 0.16],
        ["Kellerberrin", 0.14],
        ["Meckering", 0.20],
        ["Northam", 0.14],
        ["Wongan Hills", 0.15],
        ["Wickepin", 0.15],
        ["York", 0.14],
        ["Christmas Island", 0.15],
        ["Cocos Islands", 0.08],
        ["Heard Island", 0.10],
        ["Lord Howe Island", 0.06],
        ["Macquarie Island", 0.60],
        ["Norfolk Island", 0.08],
    ],
    columns=["Location", "Z"],
)

# table3_3
# Minimum KpZ values for Australia
table3_3 = pd.DataFrame(
    {
        "Annual probability of exceedance (P)": [
            "1/500",
            "1/1000",
            "1/1500",
            "1/2000",
            "1/2500",
        ],
        "Minimum value of kpZ": [
            0.08,
            0.10,
            0.12,
            0.14,
            0.15,
        ],
    }
)

def min_kp_z(P):
    min_kpZ = table3_3.loc[
        table3_3["Annual probability of exceedance (P)"] == P, "Minimum value of kpZ"
    ].squeeze()

    return min_kpZ

# location = "Albany" #@param ['Adelaide','Albany','Albury/Wodonga','Alice Springs','Ballarat','Bathurst','Bendigo','Brisbane','Broome','Burnie','Camden','Canbern','Carnarvon','Coffs Harbour','Cooma','Dampier','Darwin','Derby','Dubbo','Esperance','Geelong','Geraldton','Gladstone','Gold Coast','Gosford','Grafton','Gippsland','Goulburn','Karratha','Katoomba','Latrobe Valley','Launceston','Lismore','Mackay','Melbourne','Mittagong', 'Morisset', 'Newcastle', 'Noosa', 'Orange', 'Perth', 'Port Augusta', 'Port Lincoln', 'Port Hedland', 'Port Macquarie', 'Port Pirie', 'Robe', 'Rockhampton', 'Shepparton', 'Sydney', 'Sydney', 'Sydney', 'Tennant Creek', 'Toowoomba', 'Toowoomba', 'Tweed Heads', 'Uluru', 'Wagga Wagga', 'Wangaratta', 'Whyalla', 'Wollongong', 'Woomera', 'Wyndham', 'Wyong', 'Ballidu', 'Corrigin', 'Cunderdin', 'Dowerin', 'Goomalling', 'Kellerberrin', 'Meckering', 'Northam', 'Wongan Hills', 'Wickepin', 'York', 'Christmas Island', 'Christmas Island', 'Heard Island', 'Lord Howe Island', 'Lord Howe Island', 'Lord Howe Island']

# Section 6 - Equivalent static analysis
# @title Table 6.4 - Spectral shape factor, $C_h(T)$ - General { vertical-output: true }

table6_4 = pd.DataFrame(
    {
        "A Strong rock": [
            2.35,
            2.35,
            2.35,
            2.35,
            1.76,
            1.41,
            1.17,
            1.01,
            0.88,
            0.78,
            0.70,
            0.59,
            0.47,
            0.37,
            0.26,
            0.17,
            0.12,
            0.086,
            0.066,
            0.052,
            0.042,
        ],
        "B Rock": [
            2.94,
            2.94,
            2.94,
            2.94,
            2.20,
            1.76,
            1.47,
            1.26,
            1.10,
            0.98,
            0.88,
            0.73,
            0.59,
            0.46,
            0.33,
            0.21,
            0.15,
            0.11,
            0.083,
            0.065,
            0.053,
        ],
        "C Shallow soil": [
            3.68,
            3.68,
            3.68,
            3.68,
            3.12,
            2.50,
            2.08,
            1.79,
            1.56,
            1.39,
            1.25,
            1.04,
            0.83,
            0.65,
            0.47,
            0.30,
            0.21,
            0.15,
            0.12,
            0.093,
            0.075,
        ],
        "D Deep or very soft soil": [
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.30,
            2.83,
            2.48,
            2.20,
            1.98,
            1.65,
            1.32,
            1.03,
            0.74,
            0.48,
            0.33,
            0.24,
            0.19,
            0.15,
            0.12,
        ],
        "E Very soft soil": [
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.42,
            3.08,
            2.57,
            2.05,
            1.60,
            1.16,
            0.74,
            0.51,
            0.38,
            0.29,
            0.23,
            0.18,
        ],
    },
    index=[
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.5,
        1.7,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ],
)

#table6_4.plot(table=True, figsize=(15, 10))

# @title Table 6.4(1) - Spectral shape factor, $C_h(T)$ - Modal analysis, numerical integration time history analysis, vertical loading and parts. { vertical-output: true }
table6_4_1 = pd.DataFrame(
    {
        "A Strong rock": [
            0.8,
            2.35,
            2.35,
            2.35,
            1.76,
            1.41,
            1.17,
            1.01,
            0.88,
            0.78,
            0.70,
            0.59,
            0.47,
            0.37,
            0.26,
            0.17,
            0.12,
            0.086,
            0.066,
            0.052,
            0.042,
        ],
        "B Rock": [
            1.0,
            2.94,
            2.94,
            2.94,
            2.20,
            1.76,
            1.47,
            1.26,
            1.10,
            0.98,
            0.88,
            0.73,
            0.59,
            0.46,
            0.33,
            0.21,
            0.15,
            0.11,
            0.083,
            0.065,
            0.053,
        ],
        "C Shallow soil": [
            1.3,
            3.68,
            3.68,
            3.68,
            3.12,
            2.50,
            2.08,
            1.79,
            1.56,
            1.39,
            1.25,
            1.04,
            0.83,
            0.65,
            0.47,
            0.30,
            0.21,
            0.15,
            0.12,
            0.093,
            0.075,
        ],
        "D Deep or very soft soil": [
            1.1,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.30,
            2.83,
            2.48,
            2.20,
            1.98,
            1.65,
            1.32,
            1.03,
            0.74,
            0.48,
            0.33,
            0.24,
            0.19,
            0.15,
            0.12,
        ],
        "E Very soft soil": [
            1.1,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.68,
            3.42,
            3.08,
            2.57,
            2.05,
            1.60,
            1.16,
            0.74,
            0.51,
            0.38,
            0.29,
            0.23,
            0.18,
        ],
    },
    index=[
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.5,
        1.7,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ],
)

#table6_4_1.plot(table=True, figsize=(15, 10))


# spectral_method = "modal, numerical, parts (table 6.4(1))" #@param ["General (table 6.4)", "modal, numerical, parts (table 6.4(1))"]


# Spectral Shape factor
def spectral_shape_factor(Subsoil_Type, T, spectral_method):
    if spectral_method == "General (table 6.4)":
        table = table6_4
    else:
        table = table6_4_1

    # linear interpolation
    a = table.index.values
    b = table[Subsoil_Type].to_numpy()

    ChT = np.interp(T, a, b)

    return ChT


# Hazard Factor
def hazard_factor(location):
    Z = table3_2.loc[table3_2["Location"] == location, "Z"]
    Z = Z.squeeze()

    return Z


# Section 8 - Design of parts and components
## Section 8.1.4 - Parts and components
### Simple method to define the design action


def height_amplification_factor(h_x, h_n):
    if h_n >= 12:
        kc = 2 / h_n
    else:
        kc = 0.17

    ax = 1 + kc * h_x

    return ax


def part_horizontal_design_action_simple_method(
    IL, Subsoil_Type, kp, Z, ChT, h_x, h_n, Ic, ac, Rc, Wc, P
):
    min_kpz = min_kp_z(P)
    kpZ = max(kp * Z, min_kpz)
    ax = height_amplification_factor(h_x, h_n)
    subsoil_type = Subsoil_Type.split(" ")[0]
    if type(IL) == str:
        importance_level = int(IL.split("L")[1])
    else:
        importance_level = IL
    EDC = get_EDC(importance_level, kpZ, subsoil_type, h_n)
    if EDC == "EDC not found":
        fc = 0.01
    elif EDC == "I":
        # clause 5.3
        fc = 0.1 * Wc
    else:
        # Section 8
        fc = max(kpZ * ChT * ax * (Ic * ac / Rc) * Wc, 0.05 * Wc)

    return fc, kpZ, ax
