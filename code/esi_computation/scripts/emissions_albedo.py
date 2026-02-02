"""
Estimation of emissions equivalent to albedo change.

Authors:
    Dario Marone
    Giorgio Parlato

Date:
    2025-12-16
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# Processed tables
PATH_PROCESSED_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# Map of vegetation types and albedo estimates (higher values mean brighter surfaces)
# Sources available in the project documentation
VEGETATION_ALBEDO_MAP = {
    "tropical forest": 0.15,
    "temperate forest": 0.16,
    "boreal forest": 0.25,
    "cool climate grassland": 0.43,
    "warm climate grassland": 0.22,
    "barren land": 0.55,
}

# Albedo estimate for cropland in North America
ALBEDO_CROP_NORTH_AMERICA = 0.25

# GWP from Wang et al. (2023), 0.0259 kgCO2 / m2 / 0.01 brightening / year
# 1 ha is 10000 m2
# This is equivalent to 259 kgCO2eq / ha / 0.01 brightening / year OR 0.259 tCO2eq / ha / 0.01 brightening / year
GWP_ALBEDO = 0.259

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Table storing original vegetation types at each location
veg_type_df = pd.read_feather(
    PATH_PROCESSED_TABLES / "reference_table_enriched.feather",
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "year", "veg_type"],
)

# Dataframe storing plot areas
areas_df = pd.read_feather(
    PATH_PROCESSED_TABLES / "agronomic_features.feather",
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "year", "area_ha"],
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Merge dataframes above
albedo_df = veg_type_df.merge(
    areas_df,
    on=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"],
    how="inner",
)

# Map the albedo estimates to the corresponding vegetation types
albedo_df["albedo_original"] = albedo_df["veg_type"].map(VEGETATION_ALBEDO_MAP)

# Compute the albedo change relative to the current value (cropland)
albedo_df["albedo_change"] = albedo_df["albedo_original"] - ALBEDO_CROP_NORTH_AMERICA

# Compute the equivalent emissions from albedo changes
albedo_df["albedo_emissions_tco2eq_ha"] = albedo_df["albedo_change"] * GWP_ALBEDO / 0.01
albedo_df["albedo_emissions_tco2eq"] = (
    albedo_df["albedo_emissions_tco2eq_ha"] * albedo_df["area_ha"]
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
albedo_df.to_feather(PATH_PROCESSED_TABLES / r"table_10_emissions_albedo.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

albedo_df.to_sql(
    name="table_10_emissions_albedo", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
