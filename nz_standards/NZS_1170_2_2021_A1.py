#!/usr/bin/env python
# coding: utf-8

# # AS 1170.2:2021 method
# 
# This method references the following standard:
# NZS 1170.4:2021, for New Zealand structures.
# 
# Method developed 23 January 2023
# (c) BVT Consulting Ltd
# 
# Developer - Nima Shokrollahi

# In[1]:


import pandas as pd
import numpy as np


# ## Site wind speed

# ![image.png](attachment:image.png)

# ### Regional wind speeds ($V_R$)

# In[2]:



fig3_1_b = pd.DataFrame([
['Akaroa','NZ4',],
['Alexandra','NZ2',],
['Arrowtown','NZ2',],
['Arthurs Pass','NZ2',],
['Ashburton','NZ2',],
['Auckland','NZ1',],
['Balclutha','NZ2',],
['Blenheim','NZ2',],
['Bluff','NZ4',],
['Bulls','NZ2',],
['Cambridge','NZ1',],
['Cheviot','NZ2',],
['Christchurch','NZ2',],
['Cromwell','NZ2',],
['Dannevirke','NZ2',],
['Darfield','NZ2',],
['Dargaville','NZ1',],
['Dunedin','NZ2',],
['Eastbourne - Point Howard','NZ3',],
['Fairlie','NZ2',],
['Fielding','NZ2',],
['Fox Glacier','NZ2',],
['Foxton/Foxton Beach','NZ2',],
['Franz Josef','NZ2',],
['Geraldine','NZ2',],
['Gisborne','NZ2',],
['Gore','NZ2',],
['Greymouth','NZ2',],
['Hamilton','NZ1',],
['Hanmer Springs','NZ2',],
['Harihari','NZ2',],
['Hastings','NZ2',],
['Hawera','NZ2',],
['Hokitika','NZ2',],
['Huntly','NZ1',],
['Hutt Valley - south of Taita Gorge','NZ3',],
['Inglewood','NZ2',],
['Invercargill','NZ4',],
['Kaikohe','NZ1',],
['Kaikoura','NZ2',],
['Kaitaia','NZ1',],
['Kawerau','NZ2',],
['Levin','NZ2',],
['Manukau City','NZ1',],
['Mangakino','NZ2',],
['Marton','NZ2',],
['Masterton','NZ2',],
['Matamata','NZ1',],
['Mataura','NZ2',],
['Milford Sound','NZ2',],
['Morrinsville','NZ1',],
['Mosgiel','NZ2',],
['Motueka','NZ2',],
['Mount Maunganui','NZ1',],
['Mt Cook','NZ2',],
['Murchison','NZ2',],
['Murupara','NZ2',],
['Napier','NZ2',],
['Nelson','NZ2',],
['New Plymouth','NZ2',],
['Ngaruawahia','NZ1',],
['Oamaru','NZ2',],
['Oban','NZ3',],
['Ohakune','NZ2',],
['Opotiki','NZ2',],
['Opunake','NZ2',],
['Otaki','NZ2',],
['Otira','NZ2',],
['Otorohanga','NZ2',],
['Paeroa','NZ1',],
['Pahiatua','NZ2',],
['Paihia/Russell','NZ1',],
['Palmerston ','NZ2',],
['Palmerston North','NZ2',],
['Paraparaumu','NZ3',],
['Patea','NZ2',],
['Picton','NZ3',],
['Porirua','NZ3',],
['Pukekohe','NZ2',],
['Putaruru','NZ2',],
['Queenstown','NZ2',],
['Raetihi','NZ2',],
['Rangiora','NZ2',],
['Reefton','NZ2',],
['Riverton','NZ4',],
['Rotorua','NZ2',],
['Ruatoria','NZ2',],
['Seddon','NZ3',],
['Springs Junction','NZ2',],
['St Arnaud','NZ2',],
['Stratford','NZ2',],
['Taihape','NZ2',],
['Takaka ','NZ2',],
['Taumarunui','NZ2',],
['Taupo','NZ2',],
['Tauranga','NZ1',],
['Te Anau','NZ2',],
['Te Aroha','NZ1',],
['Te Awamutu','NZ2',],
['Te Kuiti','NZ2',],
['Te Puke','NZ2',],
['Temuka','NZ2',],
['Thames','NZ1',],
['Timaru','NZ2',],
['Tokoroa','NZ2',],
['Turangi','NZ2',],
['Twizel','NZ2',],
['Upper Hutt','NZ3',],
['Waihi','NZ1',],
['Waikanae','NZ2',],
['Waimate','NZ2',],
['Wainuiomata','NZ3',],
['Waiouru','NZ2',],
['Waipawa','NZ2',],
['Waipukurau','NZ2',],
['Wairoa','NZ2',],
['Waitara','NZ2',],
['Waiuku','NZ1',],
['Wanaka','NZ2',],
['Wanganui','NZ2',],
['Ward','NZ1',],
['Warkworth','NZ1',],
['Wellington','NZ3',],
['Wellington CBD','NZ3',],
['Westport','NZ2',],
['Whakatane','NZ2',],
['Whangarei','NZ1',],
['Winton','NZ3',],
['Woodville','NZ2',]],
columns = ["Location","Wind Region"]                       
)

import numpy as np
def location_wind_region(location):
    return fig3_1_b[fig3_1_b["Location"] == location]["Wind Region"].values[0]


def wind_region_speeds(V, location):
    
    location_region = location_wind_region(location)
    if location_region == "NZ1":
        Vr = np.round(61 - 30 * ((1/V) ** -0.1))
    elif location_region == "NZ2":
        Vr = np.round(71 - 34 * ((1/V) ** -0.1))
    elif location_region == "NZ3":
        Vr = np.round(63 - 25 * ((1/V) ** -0.1))
    elif location_region == "NZ4":
        Vr = np.round(61 - 30 * ((1/V) ** -0.1))
    else:
        Vr = np.nan
    return Vr



Table4_1 = pd.DataFrame([
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
    [200, 1.39, 1.29, 1.27, 1.24, 1.16]],
    columns = ["Height", "TC1", "TC2", "TC2.5", "TC3", "TC4"]
)

def interpolation(height):
    indices = Table4_1['Height'].searchsorted([height], side='right')
    lower_bound_index = max(0, indices[0] - 1)
    upper_bound_index = indices[0] if indices[0] != len(Table4_1) else indices[0] - 1
    height_low = Table4_1['Height'][lower_bound_index]
    height_high = Table4_1['Height'][upper_bound_index]
    interpolation_hn = (height - height_low) / (height_high - height_low)
    return interpolation_hn, lower_bound_index, upper_bound_index


def Mz_cat(height, Terrain_category):
    interpolation_height, lower_bound_index, upper_bound_index = interpolation(height)
    mz_cat_low = Table4_1[Terrain_category][lower_bound_index]
    mz_cat_high = Table4_1[Terrain_category][upper_bound_index]
    Mz_cat = mz_cat_low + interpolation_height * (mz_cat_high - mz_cat_low)
    
    return Mz_cat




def site_wind_speed(p, location, height, Terrain_category, Md, Ms, Mt):
    
    Vr = wind_region_speeds(p, location)
    Mz_cat_value = Mz_cat(height, Terrain_category)
    
    return Vr * Md * (Mz_cat_value * Ms * Mt)

def calc_wind_pressure(v_site):
    
    partition_overall_pressure_factor = 0.4
    #Density of air (kg/m3)
    rho_air = 1.2
    #Partition and building assumed not to be dynamically sensitive
    wind_dynamic_response_factor = 1.0
    
    wind_pressure = 0.5 * rho_air * (v_site**2) * partition_overall_pressure_factor * wind_dynamic_response_factor
    return wind_pressure
