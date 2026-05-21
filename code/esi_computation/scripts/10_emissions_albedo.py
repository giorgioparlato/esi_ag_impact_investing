"""
Estimation of emissions equivalent to albedo change.
Main operations:
    - map albedo estimates to vegetation types
    - calculate albedo change
    - estimate equivalent emissions
    - export results as feather file and to SQL database

Authors:Dario Marone, Giorgio Parlato
Date: 2026-02-19
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
from paths import PATH_PROCESSED_TABLES, PATH_SQL_DB

# Map of vegetation types and albedo estimates (higher values mean brighter surfaces)
# Source: model output (LPJmL)
VEGETATION_ALBEDO_MAP = {
    "tropical forest": 0.15,
    "temperate forest": 0.16,
    "boreal forest": 0.25,
    "cool climate grassland": 0.43,
    "warm climate grassland": 0.22,
    "barren land": 0.55,
}

# Albedo estimate for cropland in North America
# Source: model output (LPJmL)
ALBEDO_CROP_NORTH_AMERICA = 0.25

# GWP of albedo, 0.0259 kgCO2 / m2 / 0.01 brightening / year
# Source: https://doi.org/10.1016/j.ecolind.2023.110690
# Reference: Wang, K., Zhao, D., Zhu, Y., Gao, X., Deng, S., Chen, Z., ... & Cui, Y. (2023). Albedo-dominated biogeophysical warming effects induced by vegetation restoration on the Loess Plateau, China. Ecological Indicators, 154, 110690.
# 1 ha is 10000 m2
# This is equivalent to 259 kgCO2eq / ha / 0.01 brightening / year OR 0.259 tCO2eq / ha / 0.01 brightening / year
GWP_ALBEDO = 0.259

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Table storing original vegetation types at each location
veg_type_df = pd.read_feather(
    PATH_PROCESSED_TABLES / "table_02_reference_enriched.feather",
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "year", "veg_type"],
)

# Dataframe storing plot areas
areas_df = pd.read_feather(
    PATH_PROCESSED_TABLES / "table_03_agronomic_features.feather",
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "area_ha"],
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Merge dataframes above
albedo_df = veg_type_df.merge(
    areas_df,
    on=["farmer_id", "plot_id", "logbook_id", "practice_id"],
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
# Write feather file
albedo_df.to_feather(PATH_PROCESSED_TABLES / "table_10_emissions_albedo.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    albedo_df.to_sql(
        name="table_10_emissions_albedo",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
