#!/usr/bin/env python
# coding: utf-8

# # Grad Project Analysis Notebook
# 
# ## Data Sampling and Collection 
# 
# The data used for this project is from **Dataset A: Space Exploration** of **Topic 3: Emerging Research and Technologies**, which includes data from reports of outer space exploration focusing on faraway exoplanets. The two data sets I will be using for this project are **kepler_exoplanet_search.csv** ([Source](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi)) and **kepler_planetary_system_composite.csv** ([Source](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars)), which are both data sets from the NASA Exoplanet Archive of the NASA Exoplanet Science Institute. (Note: The kepler search dataset that was provided in DataHub is missing many parameters, so I will also be using a suppllemenaty external dataset to fill in the missing information: **koi_composite.csv** - [Source](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)).
# 
# The dataset, **kepler_exoplanet_search.csv**, collected from the Kelper Space Observatory focuses on finding candidate exoplanets that may be habitable. These candidate exoplanets are called Kepler objects of interests. The external dataset I used **koi_composite.csv** includes values for some of the characteristics/parameters of the Kepler objects of interest that are missing in the first dataset. On the other hand, the dataset, **kepler_planetary_system_composite.csv**, contains all the confirmed planets outside of the solar system, including those from the Kepler missions. A bias that may have arisen from the collection of the Kepler exoplanet data was that the Kepler Space Telescope is trained to look in areas where objects of interest or masses may have already been detected or are expected to be detected, so , and also there may be observational bias towards the detection of larger planets.
# 
# The goal of the project is to determine which features can be used to create a machine learning model to predict if a CANDIDATE kepler object of interest should be a CONFIRMED exoplanet or if it was a FALSE POSITIVE.

# #### Import Libraries


import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ## Data Cleaning
# 
# **Read in Kepler Exoplanet Search Dataset and Kepler Planetary System Composite Dataset**



kep_search = pd.read_csv("data/kepler_exoplanet_search.csv")
koi_comp = pd.read_csv("data/koi_composite.csv")
kep_comp = pd.read_csv("data/kepler_planetary_system_composite.csv")


# #### Kepler Exoplanet Search Dataset
# 
# - **Granularity:** Each entry represents a kepler object of interest (possible exoplanet) that has been discovered on the Kepler missions
# - The dataset includes many invalid entries. Many of the features only include NaN entries.



kep_search.head()


# We will remove all the features that only have NaN entries and create a new dataframe **kep_search_clean**



#function that removes all the columns with only NaN values
def rm_nan_features(df):
    clean_df = pd.DataFrame()
    for column in df.columns:
        if not all(pd.isnull(df[column])):
            clean_df[column] = df[column]
    return clean_df

#remove all columns with only NaN values from the Kepler Exoplanet Search dataset
kep_search_clean = rm_nan_features(kep_search)
kep_search_clean.head()


# #### Kepler Object of Interest Composite Dataset
# 
# - **Granularity:** Each entry represents a kepler object of interest (possible exoplanet) that has been discovered on the Kepler missions
# - This dataset contains data for the information on various characteristics of the Kepler objects of interest that were missing from the provided Kepler Search dataset, such as planet radius, transit duration, transit depth, planet equilibrium temperature, stellar effective temperature, and insolation flux.
# - This dataset does not seem to contain invalid entries.



koi_comp[['kepid','koi_prad','koi_insol', 'koi_teq','koi_depth','koi_duration','koi_steff', 'koi_impact']].head()


# The data from the **Kepler Object of Interest Composite Dataset** will be merged with **Kepler Exoplanet Search Dataset** by the specific Kepler of Interests to form a new dataframe called **koi_merged**.



koi_merged = kep_search_clean.merge(koi_comp[['kepid','koi_prad','koi_insol', 'koi_teq', 'koi_depth','koi_duration','koi_steff']],
                                    on = ['kepid']).drop_duplicates()
koi_merged.head()


# **Kepler Planetary System Composite Dataset**
# 
# - **Granularity:** Each entry in the dataset contains information on a confirmed exoplanet from outside of our solar system
# - This dataset also contains several features/columns with invalid values (all NaNs). 



kep_comp.head()


# For this dataset, we will also remove all the features that only have NaN entries and create a new dataframe **kep_comp_clean**.



kep_comp_clean = rm_nan_features(kep_comp)
kep_comp_clean.head()
