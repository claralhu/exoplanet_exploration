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

# ## Exploratory Data Analysis
# 
# Because we are interested in using both the dataset on Kepler objects of interest and the Kepler Planetary System Composite Dataset, we need to find the features that both have in common:
# 
# - Finding common columns(features) between both of the datasets.
# - Explore the datasets to see which features are useable
# 
# Through looking at our dataset, we find that these are the features both datasets have in common:
# 
# - Planet Radius 
# - Planetary Insolation
# - Equilibrium Temperature 
# - Transit Depth
# - Transit Duration
# - Stellar Effective Temperature
# - Declination Angle
# - Ascension Angle
# 
# 
# After going through the datasets, we find that the features that we would be able to use from the **Kepler Exoplanet Search Dataset** are **koi_prad**, **koi_period**, **koi_insol**, **koi_teq**, **koi_depth**, **koi_duration**, **koi_steff**, **dec,** and **ra** in addition to identification like name(**kepler_name**), and the objects disposition (**koi_disposition**) which tells whether it is a Candidate, Confirmed, or False Positive. 
# 
# The corresponding features from the **Kepler Planetary System Composite Dataset** are **pl_name, pl_rade, pl_orbper, pl_insol, pl_eqt, pl_trandep, pl_trandur, st_teff, dec,** and **ra**. Because all of the planets in this dataset are already confirmed to be planets, we need to add a column for **disposition** before merging the two datasets together
# 



koi_merged.columns


kep_comp_clean.columns


# ### Combine the two datasets
# 
# 1. Add disposition column to the **Kepler Planetary Systems Composite** Dataset
# 2. Keep only the features present in both datasets
# 3. Rename the features in the **koi_merged** dataframe so that they are consistent with the feature names in the **kep_comp_clean** dataframe
# 4. Combine the dataframes into one
# 5. Remove duplicate entries



#add a column for the disposition to the Kepler Planetary Systems Composit Dataset
kep_comp_clean["disposition"] = "CONFIRMED"

#Features to keep that are in both data frames
koi_ft = ['koi_disposition', 'kepler_name','koi_prad', 'koi_period', 'koi_insol', 'koi_teq', 'koi_depth', 'koi_duration', 'koi_steff', 'ra', 'dec']
comp_ft = ['disposition','pl_name', 'pl_rade', 'pl_orbper', 'pl_insol', 'pl_eqt', 'pl_trandep', 'pl_trandur', 'st_teff', 'ra', 'dec']

koi_merged_short = koi_merged[koi_ft]
kep_comp_short = kep_comp_clean[comp_ft]

#Rename the columns in the koi_merged dataframe to match the columns in the kep_comp_clean dataframe
koi_merged_short = koi_merged_short.rename(columns={"koi_disposition": "disposition", 
                                                    "kepler_name": "pl_name", 
                                                    "koi_prad": "pl_rade",
                                                    "koi_period": "pl_orbper",
                                                    "koi_insol": "pl_insol",
                                                    "koi_teq": "pl_eqt",
                                                    "koi_depth": "pl_trandep",
                                                    "koi_duration": "pl_trandur",
                                                    "koi_steff": "st_teff"})

#Concatonate the two dataframes together
kep_exo = pd.concat([koi_merged_short, kep_comp_short]).drop_duplicates().reset_index()
kep_exo = kep_exo.drop("pl_name", axis = 1)

#Transformations on different features
kep_exo["log_orbper"] = np.log(kep_exo["pl_orbper"])
kep_exo["log_radius"] = np.log(kep_exo["pl_rade"])
kep_exo["log_trandep"] = np.log(kep_exo["pl_trandep"])
kep_exo["log_trandur"] = np.log(kep_exo["pl_trandur"])
kep_exo["log_eqt"] = np.log(kep_exo["pl_eqt"])
kep_exo["log_steff"] = np.log(kep_exo["st_teff"])

#Remove rows with NaN or infinite, which are invalid values
kep_exo = kep_exo.replace([np.inf, -np.inf], np.nan)
kep_exo = kep_exo.dropna()
kep_exo.head()


# ### Separate Test Data from Training Data
# 
# The test dataset I will be using for this project is made up of all the CANDIDATE kepler exoplanets that have not yet been confirmed as a planet or as a false positive. The training data is made up of the training and validation data tha have been CONFIRMED or noted as FALSE POSITIVE.
# 
# In the training data, a column **planet** will be added. Entries with a **disposition** of CONFIRMED should have the value 1 in the **planet** column and entries with a FALSE POSITIVE **disposition** should have the value 0 under the **planet** column.



test = kep_exo[kep_exo['disposition'] == "CANDIDATE"]
train_val = kep_exo[kep_exo['disposition'] != "CANDIDATE"] 
train_val['planet'] = (train_val['disposition'] == 'CONFIRMED').astype(int)
train_val.head()
test.head()


# ### Visualizations
# 
# The first histogram shows the proportions of CONFIRMED, CANDIDATE, and FALSE POSITIVE exoplanets in the original **Kepler Exoplanet Search** dataset.  
# 
# The second histogram shows the proportions of CONFIRMED and FALSE POSITIVE exoplanets in the training dataset.
# 



sns.histplot(data = kep_search, x = 'koi_disposition', hue = 'koi_disposition', stat = 'probability', legend = False)
plt.title('Proportion of Exoplanets in Kepler Exoplanet Search')
plt.xlabel('Disposition')
plt.savefig('figures/barchart_exosearch.png')
plt.show();

sns.histplot(data = train_val, x = 'disposition', hue = 'disposition', stat = 'probability', legend = False)
plt.title('Proportion of Exoplanets in Training Dataset')
plt.xlabel('Disposition')
plt.savefig('figures/barchart_training.png')
plt.show();


# The following histogram shows the count of the CONFIRMED Kepler Exoplanets in the Kepler Planetary System Composite Dataset. These are the entries that overlap in the two data sets.


kep_comp['Kepler'] = kep_comp['pl_name'].str.contains('Kepler').astype(str)
sns.histplot(data = kep_comp, x = 'Kepler', hue = 'Kepler', legend = False)
plt.title('Count of Kepler Exoplanets in the Planetary System Composite Dataset')
plt.xlabel('Kepler Exoplanet')
plt.savefig('figures/barchart_propkepler.png')
plt.show();


# The correlation heatmap explores if there are any preliminary relationships between the features.
# 
# - The features ra and dec seem to be highly correlated, so only one of them needs to be kept as a feature
# - In addition, pl_eqt and st_teff; pl_eqt and pl_insol; and pl_trandur and pl_orbper seem to have some level of correlation



sns.heatmap(train_val[['planet', 'pl_rade', 'pl_orbper', 'pl_insol', 
                      'pl_eqt', 'pl_trandep', 'pl_trandur', 'st_teff', 'ra', 'dec']].corr(), 
            vmin=-1, vmax=1, cmap='PuOr')
plt.title("Correlation Heat Map")
plt.savefig('figures/corr_heatmap.png')
plt.show();


# Many of the features need log transformations. After transforming various features with a log transformations, the following jointplots were made to explore the differences in distribution of the features between CONFIRMED and FALSE POSITIVE objects.
# 
# While some areas overlap on the plots, there are clear differences in the distributions of log transformed transit depth, log transformed orbital period, log transformed radius, and the equilibrium temperature between CONFIRMED and FALSE POSITIVE kepler objects, so these features may be useful in classifying the Kepler objects of interest.


sns.jointplot(data = train_val, x = "log_orbper", y = "log_trandep", hue = "disposition", kind = 'kde')
plt.suptitle("Log Transit Depth and Log Orbital Period for Kepler Objects")
plt.savefig('figures/kdeplot_logtrandep_logorbper.png')
plt.show();


sns.jointplot(data = train_val, x = 'log_radius', y ="pl_eqt",  hue = "disposition", kind = "kde")
plt.suptitle("Equilibrium Temperature and Log Radius for Kepler Objects")
plt.ylim(top = 4500)
plt.savefig('figures/kdeplot_logradius_eqt.png')
plt.show();


# For the features Transit Duration (and log transit duration), Declination angle, and Ascension angle, the distribution for CONFIRMED and FALSE POSITIVE exoplanets almost completely overlapped, so those features would not be of much use to the model.



fig, axs = plt.subplots(figsize=(15,6), ncols=3, nrows=1)
plot1 = sns.kdeplot(data = train_val, x = "log_trandur", hue = "disposition", ax = axs[0])
plot1.set(title = 'Log Transit Duration for Kepler Objects')

plot2 = sns.kdeplot(data = train_val, x = "ra", hue = "disposition", ax = axs[1])
plot2.set(title= 'Ascension Angle of Kepler Objects')

plot3 = sns.kdeplot(data = train_val, x = "dec", hue = "disposition", ax = axs[2])
plot3.set(title = 'Declination Angle of Kepler Objects')

plt.savefig('figures/kdeplot_unused_features.png')
plt.show()


# After exploring the data through EDA and visualizations, the features we will be using for the model are log radius, log orbital period, equilibrium temperature, and log transit depth.



features = ['log_radius', 'log_orbper', 'log_trandep', 'pl_eqt']
