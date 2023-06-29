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


#NZ and AS wind regions
table3_1_b = pd.DataFrame([
["Adelaide", 'A1',],
["Albany", 'A1',],
["Albury/Wodonga", 'A1',],
["Alice Springs", 'A4',],
["Ballarat", 'A1',],
["Bathurst", 'A1',],
["Bendigo", 'A1',],
["Brisbane", 'B',],
["Broome", 'C',],
["Bundaberg", 'B',],
["Burnie", 'A3',],
["Cairns", 'C',],
["Camden", 'A3',],
["Canberra", 'A3',],
["Carnarvon", 'D',],
["Coffs Harbour", 'B',],
["Cooma", 'A3',],
["Dampier", 'D',],
["Darwin", 'C',],
["Derby", 'C',],
["Dubbo", 'A4',],
["Esperance", 'A1',],
["Geelong", 'A5',],
["Geraldton", 'B',],
["Gladstone", 'C',],
["Gold Coast", 'B',],
["Gosford", 'A3',],
["Grafton", 'A4',],
["Gippsland", 'A3',],
["Goulburn", 'A3',],
["Hobart", 'A3',],
["Karratha", 'C',],
["Katoomba", 'A3',],
["Latrobe Valley", 'A1',],
["Launceston", 'A3',],
["Lismore", 'A4',],
["Lorne", 'A1',],
["Mackay", 'C',],
["Maitland", 'A2',],
["Melbourne", 'A5',],
["Mittagong", 'A2',],
["Morisset", 'A2',],
["Newcastle", 'A2',],
["Noosa", 'B',],
["Orange", 'A1',],
["Perth", 'A1',],
["Port Augusta", 'A1',],
["Port Lincoln", 'A1',],
["Port Hedland", 'D',],
["Port Macquarie", 'A2',],
["Port Pirie", 'A1',],
["Robe", 'A1',],
["Rockhampton", 'C',],
["Shepparton", 'A1',],
["Sydney", 'A2',],
["Tamworth", 'A4',],
["Taree", 'A2',],
["Tennant Creek", 'A4',],
["Toowoomba", 'A4',],
["Townsville", 'C',],
["Tweed Heads", 'B',],
["Uluru", 'A4',],
["Wagga Wagga", 'A3',],
["Wangaratta", 'A1',],
["Whyalla", 'A1',],
["Wollongong", 'A2',],
["Woomera", 'A1',],
["Wyndham", 'A1',],
["Wyong", 'A2',],
["Ballidu", 'A1',],
["Corrigin", 'A1',],
["Cunderdin", 'A1',],
["Dowerin", 'A1',],
["Goomalling", 'A1',],
["Kellerberrin", 'A1',],
["Meckering", 'A1',],
["Northam", 'A1',],
["Wongan Hills", 'A1',],
["Wickepin", 'A1',],
["York", 'A1',],
["Christmas Island", 'B',],
["Cocos Islands", 'C',],
["Heard Island", 'A1',],
["Lord Howe Island", 'A1',],
["Macquarie Island", 'A1',],
["Norfolk Island", 'B',],],
columns = ["Location","Wind Region"]                       
)


#table3_1_b


# ## Wind region speeds

# In[3]:

# for R< 50 years
table3_1 = pd.DataFrame([
['1/25', 37, 37, 37, 37, 37, 37, 37, 43, 39, 47, 53],   
['1/50', 39, 39, 39, 39, 39, 39, 39, 45, 44, 52, 60],
['1/100', 41, 41, 41, 41, 41, 41, 41, 47, 48, 56, 66],
['1/200', 43, 43, 43, 43, 43, 43, 43, 49, 52, 61, 72],
['1/250', 43, 43, 43, 43, 43, 43, 43, 49, 53, 62, 74],
['1/500', 45, 45, 45, 45, 45, 45, 45, 51, 57, 66, 80],
['1/1000', 46, 46, 46, 46, 46, 46, 46, 53, 60, 70, 85],
['1/2000', 48, 48, 48, 48, 48, 48, 48, 54, 63, 73, 90],
['1/2500', 48, 48, 48, 48, 48, 48, 48, 55, 64, 74, 91]],
columns = ["V value", "A1", "A2", "A3", "A4", "A5","A6", "A7", "W", "B", "C", "D"]
)

# for R>= 50 years
table3_1_50 = pd.DataFrame([
['1/25', 37, 37, 37, 37, 37, 37, 37, 43, 39, 49.35, 58.3],   
['1/50', 39, 39, 39, 39, 39, 39, 39, 45, 44, 54.6, 66],
['1/100', 41, 41, 41, 41, 41, 41, 41, 47, 48, 58.8, 72.6],
['1/200', 43, 43, 43, 43, 43, 43, 43, 49, 52, 64.05, 79.2],
['1/250', 43, 43, 43, 43, 43, 43, 43, 49, 53, 65.1, 81.4],
['1/500', 45, 45, 45, 45, 45, 45, 45, 51, 57, 69.3, 88],
['1/1000', 46, 46, 46, 46, 46, 46, 46, 53, 60, 73.5, 93.5],
['1/2000', 48, 48, 48, 48, 48, 48, 48, 54, 63, 76.65, 99],
['1/2500', 48, 48, 48, 48, 48, 48, 48, 55, 64, 77.7, 100.1]],
columns = ["V value", "A1", "A2", "A3", "A4", "A5","A6", "A7", "W", "B", "C", "D"]
)

# In[4]:


def location_wind_region(location):
    return table3_1_b[table3_1_b["Location"] == location]["Wind Region"].values[0]

def wind_region_speed(p, location, design_working_life):
    location_region = location_wind_region(location)
    design_working_life = design_working_life.split()
    year = int(design_working_life[0])
    if year < 50:
        wind_region_speed = table3_1[table3_1["V value"] == p][location_region].values[0]
    else:
        wind_region_speed = table3_1_50[table3_1_50["V value"] == p][location_region].values[0]
    return wind_region_speed


# In[5]:


#Vr = wind_region_speed(1/50, "Auckland")
#Vr


# ### Terrain/height Multiplier ($M_zcat$)

# ![image.png](attachment:image.png)

# In[6]:


Table4_1 = pd.DataFrame([
    [3, 0.99, 0.95, 0.91, 0.87, 0.83, 0.75],
    [5, 1.05, 0.98, 0.91, 0.87, 0.83, 0.75],
    [10, 1.12, 1.06, 1, 0.915, 0.83, 0.75],
    [15, 1.16, 1.05, 1.05, 0.97, 0.89, 0.75],
    [20, 1.19, 1.135, 1.08, 1.01, 0.94, 0.75],
    [30, 1.22, 1.17, 1.12, 1.06, 1, 0.8],
    [40, 1.24, 1.2, 1.16, 1.1, 1.04, 0.85],
    [50, 1.25, 1.215, 1.18, 1.125, 1.07, 0.9],
    [75, 1.27, 1.245, 1.22, 1.17, 1.12, 0.98],
    [100, 1.29, 1.265, 1.24, 1.2, 1.16, 1.03],
    [150, 1.31, 1.29, 1.27, 1.24, 1.21, 1.11],
    [200, 1.32, 1.305, 1.29, 1.265, 1.24, 1.16]],
    columns = ["Height", "TC1", "TC1.5", "TC2", "TC2.5", "TC3", "TC4"]
)

#Table4_1
    


# In[7]:


def interpolation(height):
    indices = Table4_1['Height'].searchsorted([height], side='right')
    lower_bound_index = indices[0] - 1
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
    


# ### Calculate site wind

# In[9]:


def site_wind_speed(p, location, design_working_life, height, Terrain_category):
    
    Md = 1.0 #wind_direction_multiplier
    Ms = 1.0 #shielding_multiplier
    Mh = 1.0 #hill_multiplier
    elevation_multiplier = 1.0
    Mlee = 1.0 #lee_multiplier
    Mt = 1.0 #topographic_multiplier
    
    Vr = wind_region_speed(p, location, design_working_life)
    Mz_cat_value = Mz_cat(height, Terrain_category)
    
    return Vr * Md * (Mz_cat_value * Ms * Mt)


# In[10]:


#v_site = site_wind_speed(1/1000, "Auckland", 20, "TC2")
#print("site_wind:", v_site)


# ### Calculate wind pressure

# In[ ]:


def calc_wind_pressure(v_site):
    
    partition_overall_pressure_factor = 0.4
    #Density of air (kg/m3)
    rho_air = 1.2
    #Partition and building assumed not to be dynamically sensitive
    wind_dynamic_response_factor = 1.0
    
    wind_pressure = 0.5 * rho_air * (v_site**2) * partition_overall_pressure_factor * wind_dynamic_response_factor
    return wind_pressure

