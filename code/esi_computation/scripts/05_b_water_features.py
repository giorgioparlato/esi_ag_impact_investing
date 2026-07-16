"""
Estimation of water consumption.
Main operations:
    - cleaning column names, adjusting data types, selecting variables
    - concatenation / aggregation of water features
    - extraction of irrigation cost
    - merging
    - calculation of water metrics
    - export results as feather file and to SQL database

Authors: Dario Marone, Giorgio Parlato
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import numpy as np
import pandas as pd

from paths import (
    PATH_WATER_RANDOM_FOREST,
    PATH_PROCESSED_TABLES,
    PATH_SQL_DB,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# ID columns in the irrigation dataset
ID_KEYS_WATER = ["farmer_id", "plot_id", "logbook_id", "practice_id", "id_irrigation"]

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
]

WATER_CONSUMPTION_VARS = [
    "total_water_consumption_per_year_ha",
    "water_hybrid_ml_lt",
]

# Emission factors for irrigation equipment and application (precalculated)
# Reference: Juárez-Hernández, S., & Pardo, C. S. (2019). Irrigation energy use and related greenhouse gas emissions of maize production in Mexico. International journal of water resources development.
# Source: based on values in Table 5 and 7 in https://doi.org/10.1080/07900627.2018.1482739;
# average in tCO2eq per m3 of the values for summer and winter season,
# for pressurized systems only, excluding groundwater pumping
EF_IRR_EQUIP = 1.61319e-05
EF_IRR_APP = 0.000148952

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# # Load dataset on irrigation (legacy code, used previously to ML imputation)
# water_df_raw = pd.read_excel(
#     io=PATH_CIMMYT_MEXICO_WATER,
#     sheet_name=SHEET_CIMMYT_MEXICO_WATER
# )


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# # Clean column names (legacy code, used previously to ML imputation)
# water_df_clean_columns = water_df_raw.copy()
# water_df_clean_columns = water_df_clean_columns.clean_names(
#     axis="columns",
#     strip_underscores=True,
#     case_type="lower",
#     remove_special=True,
#     strip_accents=True,
#     preserve_original_labels=True,
#     enforce_string=True,
# )

# Load Random Forest dataset
water_df_metrics = pd.read_feather(
    PATH_WATER_RANDOM_FOREST,
)

# Compute reported water consumption in cubic meters per hectare
# This column only stores reported (original )values
water_df_metrics["water_cons_rep_m3_ha"] = np.where(
    water_df_metrics["total_water_consumption_per_year_ha"].notna(),
    water_df_metrics["total_water_consumption_per_year_ha"] * 10**-3,
    np.nan,
)

# Compute reported water consumption in cubic meters per hectare, adjusted for irrigation efficiency
# Safe to remove if not used in the analysis
water_df_metrics["water_cons_rep_adj_m3_ha"] = np.where(
    water_df_metrics["total_water_consumption_per_year_ha"].notna()
    & water_df_metrics["irr_efficiency_param"].notna(),
    water_df_metrics["total_water_consumption_per_year_ha"]
    * 10**-3
    * water_df_metrics["irr_efficiency_param"],
    np.nan,
)

# Create a new column using both the ML estimates and reported values
water_df_metrics["water_cons_hyb_ml_m3_ha"] = (
    water_df_metrics["water_hybrid_ml_lt"] * 10**-3
)

# Create a new column using the ML estimates, adjusted for irrigation efficiency
# Safe to remove if not used in the analysis
water_df_metrics["water_cons_hyb_ml_adj_m3_ha"] = (
    water_df_metrics["water_hybrid_ml_lt"]
    * 10**-3
    * water_df_metrics["irr_efficiency_param"]
)

water_df_totals = water_df_metrics.copy()

# Compute total water consumption (reported, hybrid)
water_df_totals["water_cons_rep_m3"] = (
    water_df_totals.water_cons_rep_m3_ha * water_df_totals.area_ha
)

# Safe to remove if not used in the analysis
water_df_totals["water_cons_rep_adj_m3"] = (
    water_df_totals.water_cons_rep_adj_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_hyb_ml_m3"] = (
    water_df_totals.water_cons_hyb_ml_m3_ha * water_df_totals.area_ha
)

# Safe to remove if not used in the analysis
water_df_totals["water_cons_hyb_ml_adj_m3"] = (
    water_df_totals.water_cons_hyb_ml_adj_m3_ha * water_df_totals.area_ha
)

# Convert to cubic decameters for calculation of water ESI
water_df_totals["water_cons_hyb_ml_dam3"] = (
    water_df_totals.water_cons_hyb_ml_m3 * 10**-3
)

# Safe to remove if not used in the analysis
water_df_totals["water_cons_hyb_ml_adj_dam3"] = (
    water_df_totals.water_cons_hyb_ml_adj_m3 * 10**-3
)

# Create df for estimating emissions from water use
water_df_emissions = water_df_totals.copy()

# Emissions based on non-adjusted hybrid water consumption estimates (ML method), per hectare basis
water_df_emissions["irr_app_ml_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_m3_ha"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_ml_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (ML method), per hectare basis
# Safe to remove if not used in the analysis
water_df_emissions["irr_app_ml_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_adj_m3_ha"] * EF_IRR_APP,
    np.nan,
)

# Safe to remove if not used in the analysis
water_df_emissions["irr_equip_ml_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_adj_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on non-adjusted hybrid water consumption estimates (ML method), total
water_df_emissions["irr_app_ml_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_m3"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_ml_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (ML method), total
# Safe to remove if not used in the analysis
water_df_emissions["irr_app_ml_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_adj_m3"] * EF_IRR_APP,
    np.nan,
)

# Safe to remove if not used in the analysis
water_df_emissions["irr_equip_ml_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_ml_adj_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
water_df_final = water_df_emissions.copy()

water_df_final.to_feather(PATH_PROCESSED_TABLES / "table_05_water_features.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    water_df_final.to_sql(
        name="table_05_water_features",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
