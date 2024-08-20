# -*- coding: utf-8 -*-
"""ASNZS_1170_0_2002.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vXdyWK3uu6MncNY87YmABolL_-iVVzg2

#AS/NZS 1170.0:2002.py method

This method references the following standard:
AS/NZS 1170.0:2002 (incorporating Amendment Nos 1,2,3,4 and 5), for New Zealand structures.

Method developed 28 August 2021
(c) BVT Consulting Ltd

Developed - MMB

Reviewed - SB

05.07.2022 - v 1.1 - Lookup function added to find load factors for load combinations given in Section 4.

###Initialise  Dependents and Libraries
"""

import pandas as pd

"""#3.4 Annual Probability of Exceedance, $P$

Given the design working life, $N$, and the importance level, $IL$, this function returns the annual probability of exceedance,$P$ , for wind, snow and earthquake Ultimate limit states, and service limit states for SLS1 and SLS2, as given in Table 3.3.

Note this function also includes SLS2 annual probabilities as given in NZS 1170.5, table 8.1, note 6.
"""
# @title Table 3.3 - Annual probability of exceedence { vertical-output: true }
# recreate table 3.3
table3_3 = pd.DataFrame(
    {
        "Wind ULS": [
            "1/100",
            "1/25",
            "1/100",
            "1/250",
            "1/1000",
            "1/25",
            "1/250",
            "1/500",
            "1/1000",
            "1/50",
            "1/250",
            "1/500",
            "1/1000",
            "1/100",
            "1/500",
            "1/1000",
            "1/2500",
            "1/250",
            "1/1000",
            "1/2500",
            "*",
        ],
        "Snow ULS": [
            "1/50",
            "1/25",
            "1/50",
            "1/100",
            "1/250",
            "1/25",
            "1/50",
            "1/100",
            "1/250",
            "1/25",
            "1/50",
            "1/100",
            "1/250",
            "1/50",
            "1/150",
            "1/250",
            "1/500",
            "1/150",
            "1/250",
            "1/500",
            "*",
        ],
        "Earthquake ULS": [
            "1/100",
            "1/25",
            "1/100",
            "1/250",
            "1/1000",
            "1/25",
            "1/250",
            "1/500",
            "1/1000",
            "1/50",
            "1/250",
            "1/500",
            "1/1000",
            "1/100",
            "1/500",
            "1/1000",
            "1/2500",
            "1/250",
            "1/1000",
            "1/2500",
            "*",
        ],
        "SLS1": [
            "1/25",
            "-",
            "1/25",
            "1/25",
            "1/25",
            "-",
            "1/25",
            "1/25",
            "1/25",
            "-",
            "1/25",
            "1/25",
            "1/25",
            "-",
            "1/25",
            "1/25",
            "1/25",
            "-",
            "1/25",
            "1/25",
            "1/25",
        ],
        "SLS2": [
            " ",
            " ",
            " ",
            " ",
            " ",
            "-",
            "-",
            "-",
            "1/250",
            "-",
            "-",
            "-",
            "1/250",
            "-",
            "1/100",
            "1/250",
            "1/500",
            "-",
            "-",
            "-",
            "*",
        ],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ("Construction equipment", 2),
            ("Less than 6 months", 1),
            ("Less than 6 months", 2),
            ("Less than 6 months", 3),
            ("Less than 6 months", 4),
            ("5 years", 1),
            ("5 years", 2),
            ("5 years", 3),
            ("5 years", 4),
            ("25 years", 1),
            ("25 years", 2),
            ("25 years", 3),
            ("25 years", 4),
            ("50 years", 1),
            ("50 years", 2),
            ("50 years", 3),
            ("50 years", 4),
            ("100 years or more", 1),
            ("100 years or more", 2),
            ("100 years or more", 3),
            ("100 years or more", 4),
        ],
        names=["Design working life", "Importance level"],
    ),
)

table3_3
# @title table_F2_cyclonic - Annual probability of exceedence - Australia { vertical-output: true }
# recreate table F2

table_F2 = pd.DataFrame(
    {
        "Wind ULS": [
            "1/100",
            "1/25",
            "1/50",
            "1/100",
            "1/100",
            "1/200",
            "1/500",
            "1/1000",
            "1/100",
            "1/500",
            "1/1000",
            "1/2500",
            "1/500",
            "1/1000",
            "1/2500",
            "*",
        ],
        "Snow ULS": [
            "1/50",
            "1/25",
            "1/50",
            "1/100",
            "1/25",
            "1/50",
            "1/100",
            "1/250",
            "1/100",
            "1/150",
            "1/200",
            "1/500",
            "1/200",
            "1/250",
            "1/500",
            "*",
        ],
        "Earthquake ULS": [
            "-",
            "-",
            "-",
            "-",
            "-",
            "1/250",
            "1/500",
            "1/1000",
            "1/250",
            "1/500",
            "1/1000",
            "1/2500",
            "1/250",
            "1/1000",
            "1/2500",
            "*",
        ],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ("Construction equipment", 2),
            ("Less than 5 years", 1),
            ("Less than 5 years", 2),
            ("Less than 5 years", 3),
            ("25 years", 1),
            ("25 years", 2),
            ("25 years", 3),
            ("25 years", 4),
            ("50 years", 1),
            ("50 years", 2),
            ("50 years", 3),
            ("50 years", 4),
            ("100 years or more", 1),
            ("100 years or more", 2),
            ("100 years or more", 3),
            ("100 years or more", 4),
        ],
        names=["Design working life", "Importance level"],
    ),
)
# @title annual_probability_of_exceedence(N,IL,LS) { run: "auto", vertical-output: true }
# @markdown Design Working Life:
N = "50 years"  # @param ["Construction equipment", "Less than 6 months", "5 years", "25 years", "50 years", "100 years or more"]
# @markdown Importance Level:
IL = 4  # @param ["1", "2", "3", "4"] {type:"raw"}
# @markdown Limit State:
LS = "Earthquake ULS"  # @param ["Wind ULS", "Snow ULS", "Earthquake ULS", "SLS1", "SLS2"]


def annual_probability_of_exceedence_nzl(N, IL, LS):
    if type(IL) == str:
        index = ["IL1", "IL2", "IL3", "IL4"].index(IL)
        IL = [1, 2, 3, 4][index]

    P = table3_3.loc[(N, IL), LS]

    return P


def annual_probability_of_exceedence_aus(N, IL, LS):
    if type(IL) == str:
        index = ["IL1", "IL2", "IL3", "IL4"].index(IL)
        IL = [1, 2, 3, 4][index]

    P = table_F2.loc[(N, IL), LS]

    return P


# print("P =",annual_probability_of_exceedence(N,IL,LS))

"""#Table 4.1 Combinations of Actions - Imposed load factors

Given the Character of imposed action, this function returns the short term, long term, combination and earthquake factors. 
"""

# @title Table 4.1 - Short-term, long-term and combination factors { vertical-output: true }
# recreate table 4.1
table4_1 = pd.DataFrame(
    {
        "Short-term factor": [
            0.7,
            0.7,
            0.7,
            0.7,
            1.0,
            1.0,
            0.7,
            0.7,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "Long-term factor": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.6,
            0.6,
            0.4,
            0.0,
            0.6,
            0.4,
            0.6,
            0.0,
            0.0,
            1.0,
        ],
        "Combination factor": [
            0.4,
            0.4,
            0.4,
            0.4,
            0.6,
            0.6,
            0.4,
            0.0,
            0.6,
            0.4,
            0.4,
            0.0,
            0.0,
            1.2,
        ],
        "Earthquake combination factor": [
            0.3,
            0.3,
            0.3,
            0.3,
            0.6,
            0.6,
            0.3,
            0.0,
            0.3,
            0.3,
            0.3,
            0.0,
            0.0,
            1.0,
        ],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ("Distributed imposed actions", "Residential and domestic floors"),
            ("Distributed imposed actions", "Office floors"),
            ("Distributed imposed actions", "Parking floors"),
            ("Distributed imposed actions", "Retail floors"),
            ("Distributed imposed actions", "Storage floors"),
            ("Distributed imposed actions", "Other floors"),
            ("Distributed imposed actions", "Roofs used for floor type activities"),
            ("Distributed imposed actions", "All other roofs"),
            ("Concentrated imposed actions", "Floors"),
            ("Concentrated imposed actions", "Floors of domestic housing"),
            ("Concentrated imposed actions", "Roofs used for floor type activities"),
            ("Concentrated imposed actions", "All other roofs"),
            ("Concentrated imposed actions", "Balustrades"),
            (
                "Concentrated imposed actions",
                "Long-term installed machinery, tare weight",
            ),
        ]
    ),
)

table4_1

# @title imposed_load_factors(action_type,action_character) { run: "auto", vertical-output: true }

action_type = "Distributed imposed actions"  # @param ["Distributed imposed actions", "Concentrated imposed actions"]
distributed_action_character = "Storage floors"  # @param ["Residential and domestic floors", "Office floors", "Parking floors", "Retail floors", "Storage floors", "Other floors", "Roofs used for floor type activities", "All other roofs"]
concentrated_action_character = "Floors"  # @param ["Floors", "Floors of domestic housing", "Roofs used for floor type activities", "All other roofs", "Balustrades", "Long-term installed machinery, tare weight"]

if action_type == "Distributed imposed actions":
    action_character = distributed_action_character
elif action_type == "Concentrated imposed actions":
    action_character = concentrated_action_character


def imposed_load_factors(action_type, action_character):

    df1 = table4_1.loc[(action_type, action_character)]

    return df1


imposed_load_factors(action_type, action_character)

"""#4.2 Combinations of actions for ultimate and serviceability limit states

Given an action type and action character for imposed loads, this function returns a summary of all action combinations as a dataframe.
"""

# @title Section 4 - Combination of actions table for ULS and SLS { vertical-output: true }


def action_combinations(action_type, action_character):

    df1 = imposed_load_factors(action_type, action_character)
    PsiS = df1.iloc[0]
    PsiL = df1.iloc[1]
    PsiC = df1.iloc[2]
    PsiE = df1.iloc[3]

    df2 = pd.DataFrame(
        {
            "G, permanent action": [
                0.9,
                1.35,
                1.2,
                1.2,
                1,
                1.2,
                1.35,
                1.2,
                1.2,
                1.2,
                0.9,
                1,
                1.2,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            "Q, imposed or live action": [
                0,
                0,
                1.5,
                PsiC,
                PsiE,
                PsiC,
                0,
                1.5,
                1.5 * PsiL,
                PsiC,
                0,
                PsiE,
                PsiC,
                0,
                PsiS,
                PsiL,
                0,
                0,
                0,
            ],
            "W, wind action": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            "E, earthquake action": [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            "S, other actions": [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("ULS stability", "stabilising permanent action only"),
                ("ULS stability", "destabilising permanent action only"),
                ("ULS stability", "permanent and imposed action"),
                ("ULS stability", "permanent, wind and imposed action"),
                ("ULS stability", "permanent, earthquake and imposed action"),
                ("ULS stability", "permanent, other actions and imposed action"),
                ("ULS strength", "permanent action only"),
                ("ULS strength", "permanent and imposed action"),
                ("ULS strength", "permanent and long term imposed action"),
                ("ULS strength", "permanent, wind and imposed action"),
                ("ULS strength", "permanent and wind action reversal"),
                ("ULS strength", "permanent, earthquake and imposed action"),
                ("ULS strength", "permanent, other actions and imposed action"),
                ("SLS", "permanent action only"),
                ("SLS", "short term imposed action only"),
                ("SLS", "long term imposed action only"),
                ("SLS", "wind action only"),
                ("SLS", "earthquake action only"),
                ("SLS", "other actions only"),
            ]
        ),
    )

    return df2


Section_4_2and3_df = action_combinations(action_type, action_character)
Section_4_2and3_df

# @title imposed_load_factors(action_type,action_character) { run: "auto", vertical-output: true }

action = "Q, imposed or live action"  # @param ["G, permanent action", "Q, imposed or live action","W, wind action","E, earthquake action","S, other actions"]
load_case_general = "ULS strength"  # @param ["ULS stability", "ULS strength", "SLS"]
load_case_specific = "permanent and imposed action"  # @param ["stabilising permanent action only", "destabilising permanent action only", "permanent and imposed action", "permanent, wind and imposed action", "permanent, earthquake and imposed action", "permanent, other actions and imposed action","permanent action only","permanent and long term imposed action","permanent and wind action reversal","short term imposed action only","long term imposed action only","wind action only","earthquake action only","other actions only"]


def Section_4_2and3_load_combination_factors(
    action, load_case_general, load_case_specific
):

    Section_4_2and3_df = action_combinations(action_type, action_character)
    series1 = Section_4_2and3_df.loc[(load_case_general, load_case_specific)]
    load_factor = series1[action]

    return load_factor


print(
    f"Load factor for selected action is: {Section_4_2and3_load_combination_factors(action, load_case_general, load_case_specific)}"
)

"""#7.2.1 ULS stability confirmation method

Given stabilising design actions, design capacity and destabilising actions, this function returns a unity number and whether compliance = true or false.

$$E_{d,stb} + R_d \ge E_{d,dst}$$
"""


def uls_stability(Edstb, Rd, Eddst):

    if Edstb + Rd >= Eddst:
        compliance = True
    else:
        compliance = False

    unity = Eddst / (Edstb + Rd)

    return compliance, unity


"""#7.2.2 ULS strength confirmation method

Given design action effect and design capacity, this function returns a unity number and whether compliance = true or false.

$$R_d \ge E_{d}$$
"""


def uls_strength(Rd, Ed):

    if Rd >= Ed:
        compliance = True
    else:
        compliance = False

    unity = Ed / Rd

    return compliance, unity


"""#7.3 SLS confirmation method

Given a servicability parameter from design actions and a limiting servicability parameter, this function returns a unity number and whether compliance = true or false.

$$\delta \le \delta_l$$
"""


def sls(delta, delta_l):

    if delta <= delta_l:
        compliance = True
    else:
        compliance = False

    unity = delta / delta_l

    return compliance, unity
