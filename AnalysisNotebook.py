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

# In[1]:


import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
