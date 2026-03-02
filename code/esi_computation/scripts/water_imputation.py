"""
Preprocessing and emission estimation for soil organic carbon.

This module performs:
- standard cleaning steps
- creation of tillage level column
- assignment of IPCC coefficients for estimating SOC
- estimation of SOC and corresponding (negative) emissions

Authors:
    Dario Marone
    Giorgio Parlato

Date:
    2026-01-12
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import pandas as pd
import numpy as np



# GIS
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
import rasterio
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from fiona.crs import from_epsg

# ML
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor


# -----------------------------------------------------------------------------
# Paths and other constants
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# CIMMYT irrigation dataset (version 6.0)
PATH_CIMMYT_MEXICO_WATER = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\5.-Irrigacion_2012-2022_01.xlsx"
)
SHEET_CIMMYT_MEXICO_WATER = "Data"

# Path to water tariffs
# Source: compiled from https://www.gob.mx/conagua/documentos/ley-federal-de-derechos-386262
PATH_WATER_TARIFFS = (
    PATH_HOME / PATH_DATA / r"data\raw\water\water_tariffs_conagua.xlsx"
)

# Path to CIMMYT logbook dataset (version 6.0)
PATH_CIMMYT_MEXICO_LOGBOOK = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\1.-Farmer_Plot_Logbook_2012-2022_03.xlsx"
)

SHEET_CIMMYT_MEXICO_LOGBOOK = "Data"

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_03_agronomic_features.feather"
)

# Path to reference table
PATH_REF_TABLE = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_01_reference.feather"
)

# Path to CIMMYT cost-revenue dataset (Mexico)
PATH_CIMMYT_MEXICO_ECONOMICS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\6.-Costs and revenues_2012-2022_02.xlsx"
)
SHEET_CIMMYT_MEXICO_ECONOMICS = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = r'/Users/giorgioparlato/Library/CloudStorage/Box-Box/SU/1.Projects/p - ESI_CYMMIT/CYMMIT_python_code_review/data/processed/tables'

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"
ENV_SQL_DB = r'/Users/giorgioparlato/Library/CloudStorage/Box-Box/SU/1.Projects/p - ESI_CYMMIT/CYMMIT_python_code_review/from_dario/cimmytExt.db'

# Path to climatic zones raster
# Reference: European Soil Data Centre (ESDAC), esdac.jrc.ec.europa.eu, European Commission, Joint Research Centre
# Source: https://esdac.jrc.ec.europa.eu/content/support-renewable-energy-directive#tabs-0-description=1
PATH_CLIMATE_ZONES = PATH_HOME / PATH_DATA / r"data\raw\geospatial\raster\climate_zones"

# Path to soil classes
# Source: WebCoverageService("https://maps.isric.org/mapserv?map=/map/wrb.map", version='1.0.0')
# PATH_SOIL_CLASS = PATH_HOME / PATH_DATA / r"data\raw\geospatial\raster\soil_classes.tif"


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# ID columns in costs and revenues dataset
ID_KEYS_ECONOMICS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "innovation_practices_id",
    "conventional_practices_id",
]

# Year column
YEAR_COL = ["year"]

# Other key variables
WATER_MANAGEMENT_VARS = [
    "water_source",
    "hydric_regime",
    "irrigation_type",
    "irrigation_system",
    "winter_sumer_season",
    "practices_type",
    'average_irrigation_time_hr_ha',
    'irrigation_time_hr_ha'
]

# Key columns from logbook
logbook_cols = ["year", "logbook_id", "state", "practice_id", 'agroecological_hub', "conservation_agriculture"
]

# Water consumption variables
WATER_CONSUMPTION_VARS = ["total_water_consumption_per_irrigation_lt", "water_consumption_lt"]

# ID columns in the irrigation dataset
ID_KEYS_WATER = ["farmer_id", "plot_id", "logbook_id", "practice_id", "id_irrigation"]

# Efficiency parameters
# Reference: Rohwer, J., Gerten, D., & Lucht, W. (2007). Development of functional irrigation types for improved global crop modelling.
# Source: adapted from Table 6 in https://www.pik-potsdam.de/en/output/publications/pikreports/.files/pr104.pdf
efficiency_parameters_map = {
    "SURFACE IRRIGATION": 0.420,
    "SPRINKLER IRRIGATION": 0.713,
    "DRIP IRRIGATION": 0.855,
}

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load water tariffs from CONAGUA
water_tariffs_df = pd.read_excel(
    io=PATH_WATER_TARIFFS,
)

# Load dataset on irrigation
water_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_WATER, sheet_name=SHEET_CIMMYT_MEXICO_WATER
)

agronomic_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES)

reference_df = pd.read_feather(PATH_REF_TABLE)

logbook_df = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_LOGBOOK, sheet_name=SHEET_CIMMYT_MEXICO_LOGBOOK, header=1
)

soil_classes = rasterio.open(PATH_SOIL_CLASS, driver="GTiff")

economics_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_ECONOMICS, sheet_name=SHEET_CIMMYT_MEXICO_ECONOMICS, header=1
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
logbook_df = logbook_df.clean_names(
    axis='columns', # Clean the labels on the columns
    strip_underscores=True, # Remove outer underscores from all column names 
    case_type='lower', # Make all characters lower case
    remove_special=True, # Preserve only letters, numbers, and underscores
    strip_accents=True, # Remove accents 
    preserve_original_labels=True, # Retrievable using df.original_labels
    enforce_string=True, # Convert all column names to string type
)

### rename column innovation_conventional_practices_id to practice_id in logbook_df
logbook_df.rename(columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True)

# Clean column names
water_df_clean_columns = water_df_raw.copy()
water_df_clean_columns = water_df_raw.clean_names(
    axis="columns",
    strip_underscores=True,
    case_type="lower",
    remove_special=True,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
)

# Rename selected columns
water_df_clean_columns.rename(
    columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True
)

# Convert id columns to string type
water_df_dtypes = water_df_clean_columns.copy()
water_df_dtypes[ID_KEYS_WATER] = water_df_dtypes[ID_KEYS_WATER].astype("string")

# Select variables of interest
water_df_vars = water_df_dtypes.copy()
water_df_vars = water_df_vars[
    ID_KEYS_WATER + YEAR_COL + WATER_MANAGEMENT_VARS + WATER_CONSUMPTION_VARS
]

# Assign project irrigation efficiency parameters based on irrigation type
water_df_efficiency = water_df_vars.copy()


water_df_efficiency["irr_efficiency_param"] = water_df_efficiency[
    "irrigation_type"
].map(efficiency_parameters_map)


#logbook_df['practice_id'] = logbook_df['practice_id'].astype('string')
logbook_df['logbook_id'] = logbook_df['logbook_id'].astype('string')
logbook_df['practice_id'] = logbook_df['practice_id'].astype('string')
logbook_df_merge = logbook_df.loc[:, logbook_cols]

# Economics DF
economics_df_clean_columns = economics_df_raw.copy()
economics_df_clean_columns = economics_df_clean_columns.clean_names(
    axis="columns",
    strip_underscores=True,
    case_type="lower",
    remove_special=True,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
)

economics_df_dtypes = economics_df_clean_columns.copy()
economics_df_dtypes[ID_KEYS_ECONOMICS] = economics_df_dtypes[ID_KEYS_ECONOMICS].astype(
    "string"
)

# Split the original dataframe between the innovation and conventional parts
# Create the innovation dataframe and rename practices id column
economics_df_innov = economics_df_dtypes.iloc[:, 0:24].copy()
economics_df_innov.rename(
    columns={"innovation_practices_id": "practice_id"}, inplace=True
)

# Create the conventional dataframe and rename columns accordingly
innov_cols_to_exclude = economics_df_clean_columns.columns[9:24]
economics_df_conv = economics_df_clean_columns.drop(
    columns=innov_cols_to_exclude
).copy()
economics_df_conv.columns = economics_df_conv.columns.str.replace(
    r"_1$", "", regex=True
)
economics_df_conv.rename(
    columns={"conventional_practices_id": "practice_id"}, inplace=True
)

# Concatenate the conventional and innovation dataframes
economics_df_concat = pd.concat([economics_df_conv, economics_df_innov]).sort_index()
economics_df_concat.dropna(subset=["practice_id"], inplace=True)

# Rename columns
economics_df_concat.rename(
    columns={"irrigation": "irrigation_cost_mxn_ha"}, inplace=True
)

# -----------------------------------------------------------------------------
# Merging dataframes
# -----------------------------------------------------------------------------
### merge reference_df, on=["practice_id", "logbook_id", "year"], but only adding these columns: plot_type, latitude_n, longitude_w
reference_df = reference_df.loc[:, ["practice_id", "logbook_id", "year", "plot_type", "latitude_n", "longitude_w"]]


irrigation_df_impute = water_df_efficiency.merge(
    reference_df, on=["practice_id", "logbook_id", "year"], how="left") 


irrigation_df_impute_1 = irrigation_df_impute.merge(
    logbook_df_merge, on=["logbook_id", "practice_id", "year"], how="left", indicator=True)  ##to check merge


## create copy of agronomic_df only with "practice_id", "year", maize_grain_yield_t_ha

agronomic_df_merge_irrigation = agronomic_df.loc[:, ["practice_id", "year", "area_ha", "maize_grain_yield_t_ha"]]

irrigation_df_impute_2 = irrigation_df_impute_1.merge(
    agronomic_df_merge_irrigation, on=["practice_id", "year"], how="left") # , indicator=True to check merge

# add soil type and climate code to dataset
lon, lat = irrigation_df_impute_2['longitude_w'].values, irrigation_df_impute_2['latitude_n'].values
coords = list(zip(lon, lat))

# Use rasterio.sample for efficiency
soil_type = [val[0] for val in soil_classes.sample(coords)]
# Add to dataframe

irrigation_df_impute_2['soil_type'] = soil_type
irrigation_df_impute_2['soil_type'] = irrigation_df_impute_2['soil_type'].astype('category')

# Sample raster values at those point locations
with rasterio.open(PATH_CLIMATE_ZONES) as climate_zone_raster:
    values = list(climate_zone_raster.sample(coords))
    # Flatten the result if it's single-band
    values = [val[0] for val in values]

# Assign to GeoDataFrame
irrigation_df_impute_2['climate_code'] = values
irrigation_df_impute_2['climate_code'] = irrigation_df_impute_2['climate_code'].astype('category')

# Merge environmental variables from ENV_SQL_DB
conn = sqlite3.connect(ENV_SQL_DB)
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)

env_data_query = "SELECT * FROM env_data;"
env_data_df_full = pd.read_sql(env_data_query, conn)

env_data_df = env_data_df_full.loc[:, ['logbook_id', "year", "pcp", "tmax", "tmin"]]

env_data_df['logbook_id'].dtype
env_data_df['logbook_id'] = env_data_df['logbook_id'].astype('string')


irrigation_df_impute_3 = irrigation_df_impute_2.merge(
    env_data_df, on=["logbook_id", "year"], how="left") # , indicator=True to check merge

# Add our own SQL
conn = sqlite3.connect(PATH_SQL_DB)
soc_query = "SELECT * FROM table_09_emissions_soc;"
soc_df_full = pd.read_sql_query(soc_query, conn)

soc_df = soc_df_full.loc[:, ['logbook_id',"practice_id", "year"]]


soc_df['logbook_id'].dtype
soc_df['logbook_id'] = soc_df['logbook_id'].astype('string')

irrigation_df_impute_4 = irrigation_df_impute_3.merge(
    soc_df, on=["logbook_id", "practice_id", "year"], how="left", 
   # indicator=True # # to check merge
    ) 

irr_cost_df = economics_df_concat[
    [
        "logbook_id",
        "practice_id",
        "year",
        "irrigation_cost_mxn_ha",
    ]
].copy()


# Merge with irrigation cost
irrigation_df_impute_5 = irrigation_df_impute_4.merge(
    irr_cost_df,
    on=["logbook_id", "practice_id", "year"],
    how="left",
#    indicator=True
)
# Merge with tariff
irrigation_df_impute_6 = irrigation_df_impute_5.merge(water_tariffs_df, on="year", how="left")

# If total_water_consumption_per_irrigation_lt is zero, and water_consumption_lt is not NA, then set adj_total_water_consumption_per_irrigation_lt to water_consumption_lt*area_ha*(average_irrigation_time_hr_ha OR irrigation_time_hr_ha (whichever not NA)) - if total_water_consumption_per_irrigation_lt is zero, and water_consumption_lt is NA, set total_water_consumption_per_irrigation_lt to NA
irrigation_df_impute_6["adj_total_water_consumption_per_irrigation_lt_ha"] = np.where(
    (irrigation_df_impute_6["total_water_consumption_per_irrigation_lt"] == 0)
    & (irrigation_df_impute_6["water_consumption_lt"].notna()),
    irrigation_df_impute_6["water_consumption_lt"]
    * irrigation_df_impute_6[["average_irrigation_time_hr_ha", "irrigation_time_hr_ha"]]
    .bfill(axis=1)
    .iloc[:, 0],
    np.where(
        (irrigation_df_impute_6["total_water_consumption_per_irrigation_lt"] == 0)
        & (irrigation_df_impute_6["water_consumption_lt"].isna()),
        np.nan,
        irrigation_df_impute_6["total_water_consumption_per_irrigation_lt"],
    ),
)

# Introduce new variable number of irrigation events per year which counts the number of rows per logbookd id and year in irrigation_df_impute_5
irrigation_df_impute_6["irr_events_per_year"] = (
    irrigation_df_impute_6
        .groupby(["practice_id", "year"])["practice_id"]
        .transform("count")
)

# Set final dataset for ML
irrigation_df_gb = irrigation_df_impute_6.copy()

# filter out observations with NA in area_ha and latitude_n
irrigation_df_gb = irrigation_df_gb[
    irrigation_df_gb["area_ha"].notna() & irrigation_df_gb["latitude_n"].notna()
]

# -----------------------------------------------------------------------------
# Gradient Boosting
# -----------------------------------------------------------------------------
# Set target and features variables
# adj_total_water_consumption_per_irrigation_lt_ha or total_water_consumption_per_irrigation_lt
target = 'adj_total_water_consumption_per_irrigation_lt_ha'

numeric_features = [
    'area_ha',
    'year',
    "latitude_n",
    "longitude_w",
    'irr_efficiency_param',
    'irrigation_cost_mxn_ha',
    "pcp",
    "tmax",
    "tmin",
    "irr_events_per_year",
]

categorical_features = [
    'hydric_regime',
    "plot_type",
   "soil_type",
    "water_source",
    "climate_code", 
   "state",
    "winter_sumer_season",
    "agroecological_hub",
]

# Set parameters for imputation and testing
X = irrigation_df_gb[numeric_features + categorical_features]
y = irrigation_df_gb[target]

X_categorical = X.select_dtypes(include=['object'])

X_numerical = X.select_dtypes(exclude=['object'])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x_categorical_ohe = pd.DataFrame(
    ohe.fit_transform(X_categorical),
    index=X.index,
    columns=ohe.get_feature_names_out(X_categorical.columns)
)

x = pd.concat([X_numerical, x_categorical_ohe], axis=1)

mask_known = ~pd.isna(y)
mask_missing = pd.isna(y)

X_known = x[mask_known]
y_known = y[mask_known]

X_missing = x[mask_missing]

X_train, X_test, y_train, y_test = train_test_split(
    X_known,
    y_known,
    test_size=0.2,
    random_state=123
)

# Fit best model on all known data
final_gbr = HistGradientBoostingRegressor(
    l2_regularization=0.1,
    max_depth=12,
    learning_rate=0.07,
    max_iter=600,
    random_state=42,
    min_samples_leaf = 10
)

final_gbr.fit(X_known, y_known)

y_filled = y.copy()
y_filled.loc[mask_missing] = final_gbr.predict(X_missing)

irrigation_df_gb["water_hybrid_ml_lt"] = y_filled

# adjust negative and zero values to minimum observed
min_observed = irrigation_df_gb.loc[~mask_missing, target].min()
irrigation_df_gb["water_hybrid_ml_lt"] = irrigation_df_gb["water_hybrid_ml_lt"].clip(lower=min_observed)


# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file

irrigation_df_gb.to_feather(PATH_EXPORT_TABLES / r"irrigation_df_ml_imputed.feather")

# END
exit("Execution complete.")
