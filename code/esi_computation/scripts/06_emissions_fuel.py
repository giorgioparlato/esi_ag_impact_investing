"""
Estimation of emissions from fuel consumption.
Main operations:
    - imputation of missing fuel consumption values
    - computation of emissions from fuel consumption
    - aggregation of emissions and related variables at practice ID level
    - export results as feather file and to SQL database

Authors: Dario Marone, Giorgio Parlato
Date: 2026-02-19
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# File paths and sheet names
# Get home directory (useful if you are working on different machines)
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# Path to CIMMYT dataset storing fuel consumption data (version 6.0)
PATH_CIMMYT_MEXICO_ACTIVITIES = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\3._Labor_harvest_activities_2012-2022_02.xlsx"
)

SHEET_CIMMYT_MEXICO_ACTIVITIES = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# Path to reference table
PATH_REFERENCE_TABLE = (
    PATH_HOME / PATH_DATA / PATH_EXPORT_TABLES / r"table_01_reference.feather"
)

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_03_agronomic_features.feather"
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ID_KEYS_ACTIVITIES = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "activity_id",
]

FUEL_COLUMNS = [
    "power_source",
    "group_activity",
    "fuel_consumption_l_ha_motor1",
    "fuel_consumption_l_ha_motor2",
]

# Emission factors in kilograms per liter
# Comprising emissions from upstream processing of fuel and on-site emissions from fuel consumption
# Source: table A9 in https://linkinghub.elsevier.com/retrieve/pii/S0360544218325350
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
EF_CO2_KG_L = 3.0
EF_N2O_KG_L = 1.0 * 10**-3
EF_CH4_KG_L = 2.6 * 10**-3

# IPCC AR6 global warming potential (GWP) values relative to CO2
# Source: https://ghgprotocol.org/sites/default/files/2024-08/Global-Warming-Potential-Values%20%28August%202024%29.pdf
GWP_AR6_CH4 = 29.8
GWP_AR6_N2O = 273

# Combined CO2 equivalent emission factor in kilograms per liter
EF_DIESEL_KG_CO2EQ_L = (
    EF_CO2_KG_L + (EF_CH4_KG_L * GWP_AR6_CH4) + (EF_N2O_KG_L * GWP_AR6_N2O)
)

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the activities dataset
activities_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_ACTIVITIES,
    sheet_name=SHEET_CIMMYT_MEXICO_ACTIVITIES,
    header=1,
)

# Load clean id-year pairs
ids_year_df = pd.read_feather(
    PATH_REFERENCE_TABLE, ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

# Load plot areas
areas_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES, columns=["practice_id", "year", "area_ha"]
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
activities_df_clean_columns = activities_df_raw.clean_names(
    axis="columns",
    column_names=None,
    strip_underscores=True,
    case_type="lower",
    remove_special=False,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
    truncate_limit=None,
)

activities_df_clean_columns.rename(
    columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True
)

# Create data frame storing fuel consumption data
fuel_df = activities_df_clean_columns[ID_KEYS_ACTIVITIES + FUEL_COLUMNS]

# Convert id columns to string type
fuel_df_dtypes = fuel_df.copy()
fuel_df_dtypes[ID_KEYS_ACTIVITIES] = fuel_df_dtypes[ID_KEYS_ACTIVITIES].astype("string")

# Merge with clean id-year pairs
fuel_df_clean_years = fuel_df_dtypes.merge(
    ids_year_df, on=["farmer_id", "plot_id", "logbook_id", "practice_id"], how="left"
)

# Create a single column to store fuel consumption observations
fuel_df_one_col = fuel_df_clean_years.copy()
fuel_df_one_col["fuel_consumption_l_ha"] = np.select(
    [
        (fuel_df_one_col["fuel_consumption_l_ha_motor1"].notna()),
        (fuel_df_one_col["fuel_consumption_l_ha_motor2"].notna()),
    ],
    [
        fuel_df_one_col["fuel_consumption_l_ha_motor1"],
        fuel_df_one_col["fuel_consumption_l_ha_motor2"],
    ],
    default=np.nan,
)

# Impute missing values with mean fuel consumption by activity type,
# only for records under motorized power sources
fuel_df_imputation = fuel_df_one_col.copy()

mask_fuel_motorized = fuel_df_imputation["power_source"].str.contains(
    "MOTORIZED", na=False
)

fuel_df_imputation.loc[mask_fuel_motorized, "fuel_consumption_l_ha_imputed"] = (
    fuel_df_imputation[mask_fuel_motorized]
    .groupby("group_activity")["fuel_consumption_l_ha"]
    .transform(lambda x: x.fillna(x.mean()))
)

# Create flag indicating the status of fuel consumption values after imputation
cond_not_applicable = ~fuel_df_imputation["power_source"].str.contains(
    "MOTORIZED", na=True
)
cond_reported = fuel_df_imputation["fuel_consumption_l_ha"].notna()
cond_imputed = (
    fuel_df_imputation["fuel_consumption_l_ha"].isna()
    & fuel_df_imputation["fuel_consumption_l_ha_imputed"].notna()
)
cond_missing = (
    fuel_df_imputation["power_source"].str.contains("MOTORIZED")
    & fuel_df_imputation["fuel_consumption_l_ha_imputed"].isna()
)

choice_not_applicable = "NOT APPLICABLE"
choice_reported = "REPORTED"
choice_imputed = "IMPUTED"
choice_missing = "MISSING"

conditions = [cond_not_applicable, cond_reported, cond_imputed, cond_missing]
choices = [choice_not_applicable, choice_reported, choice_imputed, choice_missing]

fuel_df_imputation["status"] = np.select(
    condlist=conditions, choicelist=choices, default="UNKNOWN"
)

# Compute emissions from fuel (diesel) consumption
# Approach based on Juarez-Hernandez et al.
# Compute emissions from fuel consumption in CO_2 equivalent tons per hectare
fuel_df_emissions = fuel_df_imputation.copy()
fuel_df_emissions["fuel_cons_t_co2eq_ha"] = (
    fuel_df_emissions.fuel_consumption_l_ha_imputed * EF_DIESEL_KG_CO2EQ_L
) / 1000

fuel_df_power = fuel_df_emissions.copy()
mask_power_nan = fuel_df_power["power_source"].isna()
fuel_df_power["missing_power_source"] = mask_power_nan

# Group and aggregate variables selectively
group_cols = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]


def join_unique(x):
    return " | ".join(sorted(x.dropna().unique()))


def sum_min_count(x):
    return x.sum(min_count=1)


def power_source_status(x):
    if not x.any():
        return "COMPLETE"
    elif x.all():
        return "UNKNOWN"
    else:
        return "INCOMPLETE"


fuel_df_agg = (
    fuel_df_power.groupby(group_cols, sort=False)
    .agg(
        activities=("group_activity", join_unique),
        activity_count=("group_activity", "nunique"),
        power_sources=("power_source", join_unique),
        power_source_count=("power_source", "nunique"),
        fuel_cons_l_ha=("fuel_consumption_l_ha_imputed", sum_min_count),
        fuel_cons_status=("status", join_unique),
        fuel_cons_t_co2eq_ha=("fuel_cons_t_co2eq_ha", sum_min_count),
        power_source_availability=("missing_power_source", power_source_status),
    )
    .reset_index()
)

# Merge with areas from the agronomic dataframe
areas_df = areas_df.drop_duplicates(keep=False).copy()
fuel_df_totals = fuel_df_agg.copy()
fuel_df_totals = fuel_df_totals.merge(areas_df, on=["practice_id", "year"], how="left")

# Compute total fuel consumption
fuel_df_totals["fuel_cons_t_co2eq"] = (
    fuel_df_totals.fuel_cons_t_co2eq_ha * fuel_df_totals.area_ha
)

# Final DataFrame
fuel_df_final = fuel_df_totals.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "power_sources",
    "power_source_count",
    "power_source_availability",
    "activities",
    "activity_count",
    "fuel_cons_l_ha",
    "fuel_cons_status",
    "fuel_cons_t_co2eq_ha",
    "fuel_cons_t_co2eq",
]

fuel_df_final = fuel_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
fuel_df_final.to_feather(PATH_EXPORT_TABLES / r"table_06_emissions_fuel.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

fuel_df_final.to_sql(
    name="table_06_emissions_fuel", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
