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

# -----------------------------------------------------------------------------
# Paths and other constants
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# Path to enriched reference table
PATH_REF_TABLE_ENRICHED = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\reference_table_enriched.feather"
)

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\agronomic_features.feather"
)

# Path to agricultural inputs table
PATH_AG_INPUTS = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\emissions_ag_inputs.feather"
)

# Path to crop residues table
PATH_CROP_RESIDUES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\emissions_residues.feather"
)

# CIMMYT dataset on field activities
PATH_CIMMYT_MEXICO_ACTIVITIES = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\BEM_e-Agrology_Mexico\3._Labor_harvest_activities_2012-2022_01.xlsx"
)

SHEET_CIMMYT_MEXICO_ACTIVITIES = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

KEY_IDS = ["farmer_id", "plot_id", "logbook_id", "practice_id", "activity_id"]

KEY_VARS = ["year", "tillage_bem", "tillage_land"]

# Map IPCC coefficients based on all the variables
# Land use coefficients
FLU_MAPPING = {
    # Temperate/Boreal
    "Warm Temperate Moist": 0.69,
    "Warm Temperate Dry": 0.80,
    "Cool Temperate Moist": 0.69,
    "Cool Temperate Dry": 0.80,
    # Tropical
    "Tropical Wet": 0.48,
    "Tropical Moist": 0.48,
    "Tropical Dry": 0.58,
    "Tropical Montane": 0.64,
}

# Tillage coefficients
FTILL_MAPPING = {
    # Traditional tillage = Full tillage
    "TRADITIONAL TILLAGE": {"default": 1.00},
    # Minimum tillage = Reduced
    "MINIMUM-REDUCED TILLAGE": {
        "Cool Temperate Dry": 1.02,
        "Cool Temperate Moist": 1.08,
        "Tropical Dry": 1.09,
        "Tropical Moist": 1.15,
        "Tropical Wet": 1.15,
        "Tropical Montane": 1.09,
    },
    # Zero tillage = No-till
    "ZERO TILLAGE": {
        "Cool Temperate Dry": 1.10,
        "Cool Temperate Moist": 1.15,
        "Tropical Dry": 1.17,
        "Tropical Moist": 1.22,
        "Tropical Wet": 1.22,
        "Tropical Montane": 1.16,
    },
    "MINIMUM-REDUCED TILLAGE | TRADITIONAL TILLAGE": {
        "Cool Temperate Dry": (1.02 + 1) / 2,
        "Cool Temperate Moist": (1.08 + 1) / 2,
        "Tropical Dry": (1.09 + 1) / 2,
        "Tropical Moist": (1.15 + 1) / 2,
        "Tropical Wet": (1.15 + 1) / 2,
        "Tropical Montane": (1.09 + 1) / 2,
    },
    "MINIMUM-REDUCED TILLAGE | ZERO TILLAGE": {
        "Cool Temperate Dry": (1.02 + 1.10) / 2,
        "Cool Temperate Moist": (1.08 + 1.15) / 2,
        "Tropical Dry": (1.09 + 1.17) / 2,
        "Tropical Moist": (1.15 + 1.22) / 2,
        "Tropical Wet": (1.15 + 1.22) / 2,
        "Tropical Montane": (1.09 + 1.22) / 2,
    },
    "MINIMUM-REDUCED TILLAGE | TRADITIONAL TILLAGE | ZERO TILLAGE": {
        "Cool Temperate Dry": (1.02 + 1 + 1.10) / 3,
        "Cool Temperate Moist": (1.08 + 1 + 1.15) / 3,
        "Tropical Dry": (1.09 + 1 + 1.17) / 3,
        "Tropical Moist": (1.15 + 1 + 1.22) / 3,
        "Tropical Wet": (1.15 + 1 + 1.22) / 3,
        "Tropical Montane": (1.09 + 1 + 1.22) / 3,
    },
}

# Input coefficients
FI_MAPPING = {
    # LOW
    ("Low", "Cool Temperate Dry"): 0.95,
    ("Low", "Cool Temperate Moist"): 0.92,
    ("Low", "Tropical Dry"): 0.95,
    ("Low", "Tropical Moist"): 0.92,
    ("Low", "Tropical Wet"): 0.92,
    ("Low", "Tropical Montane"): 0.94,
    # MEDIUM — all get 1.00
    ("Medium", "any"): 1.00,
    # HIGH without manure
    ("High_no_manure", "Cool Temperate Dry"): 1.04,
    ("High_no_manure", "Cool Temperate Moist"): 1.11,
    ("High_no_manure", "Tropical Dry"): 1.04,
    ("High_no_manure", "Tropical Moist"): 1.11,
    ("High_no_manure", "Tropical Wet"): 1.11,
    ("High_no_manure", "Tropical Montane"): 1.08,
    # HIGH with manure
    ("High_with_manure", "Cool Temperate Dry"): 1.37,
    ("High_with_manure", "Cool Temperate Moist"): 1.44,
    ("High_with_manure", "Tropical Dry"): 1.37,
    ("High_with_manure", "Tropical Moist"): 1.44,
    ("High_with_manure", "Tropical Wet"): 1.44,
    ("High_with_manure", "Tropical Montane"): 1.41,
}

MANURE_TYPES = {
    "Cattle Manure",
    "General Compost",
    "Sheep Manure",
    "Poultry Manure",
    "Swine Manure",
    "Cattle Manure | General Compost",
    "Horses/Mules/Asses Manure",
    "General Compost | Poultry Manure",
    "Cattle Manure | Poultry Manure",
    "Cattle Manure | Sheep Manure",
    "General Compost | Sheep Manure",
    "Poultry Manure | Sheep Manure",
    "Poultry Manure | Swine Manure",
    "Horses/Mules/Asses Manure | Swine Manure",
}

# BAU coefficient
FTILL_BAU = 1

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Activities dataset
activities_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_ACTIVITIES,
    sheet_name=SHEET_CIMMYT_MEXICO_ACTIVITIES,
    header=1,
)

# Climate zones and SOC reference values
soc_df_ref = pd.read_feather(
    PATH_REF_TABLE_ENRICHED,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "year",
        "climate_class_corrected",
        "soc_ref",
    ],
)

# Areas
soc_agronomic_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES,
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "year", "area_ha"],
)

# Organic fertilizers categories and nitrogen amounts
soc_ag_inputs_df = pd.read_feather(
    PATH_AG_INPUTS,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "year",
        "categories_organic",
        "nitrogen_org_t_ha",
    ],
)

# Crop residues
soc_crop_res_df = pd.read_feather(
    PATH_CROP_RESIDUES,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "year",
        "crop_residues_returned_total_pct",
    ],
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
activities_df_clean_columns = activities_df_raw.copy()
activities_df_clean_columns = activities_df_clean_columns.clean_names(
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

# Rename columns
activities_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
    },
    inplace=True,
)

# Select variables of interest
activities_df_variables = activities_df_clean_columns[KEY_IDS + KEY_VARS].copy()

# Convert id columns to string type
activities_df_dtypes = activities_df_variables.copy()
activities_df_dtypes[KEY_IDS] = activities_df_dtypes[KEY_IDS].astype("string")

# Clean year column
activities_df_year = activities_df_dtypes.copy()
activities_df_year["year"] = activities_df_year["year"].apply(
    lambda x: str(x)[0:4] if pd.notna(x) and isinstance(x, str) else x
)
activities_df_year["year"] = pd.to_numeric(activities_df_year["year"], errors="coerce")
activities_df_year["year"] = activities_df_year["year"].astype("Int64")


# Merge tillage_bem and tillage_land columns
soc_tillage_df = activities_df_year.copy()
soc_tillage_df["tillage"] = np.where(
    soc_tillage_df["tillage_bem"].notna(),
    soc_tillage_df["tillage_bem"],
    soc_tillage_df["tillage_land"],
)

# Drop the original tillage columns
soc_tillage_df.drop(columns=["tillage_bem", "tillage_land"], inplace=True)


# Aggregate at the practices id level
def join_unique(x):
    return " | ".join(sorted(x.dropna().unique()))


soc_tillage_df_agg = soc_tillage_df.copy()
soc_tillage_df_agg = soc_tillage_df_agg.groupby(
    ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"], as_index=False
).agg(tillage=("tillage", join_unique), tillage_count=("tillage", "nunique"))

# Merge
labels = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
soc_df_merged = soc_df_ref.copy()
soc_df_merged = soc_df_merged.merge(soc_agronomic_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_crop_res_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_ag_inputs_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_tillage_df_agg, on=labels, how="left")


# Assigning input coefficients
# Classify FI level
def classify_fi_level(row):
    n_input = row.get("nitrogen_org_t_ha", 0)
    category = row.get("categories_organic", "")
    residues = row.get("crop_residues_returned_total_pct", 0)

    # Treat missing values as 0 or empty string
    n_input = 0 if pd.isna(n_input) else n_input
    residues = 0 if pd.isna(residues) else residues
    category = "" if pd.isna(category) else category

    if category in MANURE_TYPES and n_input > 0 and residues == 100:
        return "High_with_manure"
    elif n_input > 0 and residues == 100:
        return "High_no_manure"
    elif residues == 100 or n_input > 0 and residues >= 50:
        return "Medium"
    else:
        return "Low"


soc_df_mapping = soc_df_merged.copy()

soc_df_mapping["fi_level"] = soc_df_mapping.apply(classify_fi_level, axis=1)


# Assign FI value
def map_fi(row):
    key = (row["fi_level"], row["climate_class_corrected"])
    if row["fi_level"] == "Medium":
        return 1.00
    return FI_MAPPING.get(key, None)


soc_df_mapping["fi"] = soc_df_mapping.apply(map_fi, axis=1)


# Assign FI values for business as usual - Low inputs
def map_fi_bau(row):
    climate = row.get("climate_class_corrected")
    return FI_MAPPING.get(("Low", climate), None)


soc_df_mapping["fi_bau"] = soc_df_mapping.apply(map_fi_bau, axis=1)

# Assign FLU directly from climate_class
soc_df_mapping["flu"] = soc_df_mapping["climate_class_corrected"].map(FLU_MAPPING)

# Fill missing values for tillage with 'TRADITIONAL TILLAGE'
soc_df_mapping["tillage"] = soc_df_mapping["tillage"].fillna("TRADITIONAL TILLAGE")


def map_ftill(row):
    till_type = row.get("tillage")
    climate = row.get("climate_class_corrected", "")

    if pd.isna(till_type) or till_type == "TRADITIONAL TILLAGE":
        return 1.00

    return FTILL_MAPPING.get(till_type, {}).get(climate, 1.00)


# Apply the mapping
soc_df_mapping["ftill"] = soc_df_mapping.apply(map_ftill, axis=1)

# SOC estimation
soc_df_estimation = soc_df_mapping.copy()

# SOC under BAU scenario
soc_df_estimation["soc_0"] = (
    soc_df_estimation["soc_ref"]
    * soc_df_estimation["flu"]
    * FTILL_BAU
    * soc_df_estimation["fi_bau"]
    * soc_df_estimation["area_ha"]
)

# SOC under current scenario
soc_df_estimation["soc_1"] = (
    soc_df_estimation["soc_ref"]
    * soc_df_estimation["flu"]
    * soc_df_estimation["ftill"]
    * soc_df_estimation["fi"]
    * soc_df_estimation["area_ha"]
)

# SOC change (annual, over a 20 years period)
soc_df_estimation["soc_delta"] = (
    soc_df_estimation["soc_1"] - soc_df_estimation["soc_0"]
) / 20

# Conversion to (negative) carbon dioxide equivalents, to reflect carbon sequestration
# Source: https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator#:~:text=To%20convert%20a%20quantity%20of,carbon%20dioxide%2C%20multiply%20by%203.67.
soc_df_estimation["soc_delta_t_co2eq"] = soc_df_estimation["soc_delta"] * 3.67 * -1

# Calculate on a per hectare basis
soc_df_estimation["soc_delta_t_co2eq_ha"] = (
    soc_df_estimation["soc_delta"] * 3.67 * -1 / soc_df_estimation["area_ha"]
)

# Final DataFrame
soc_df_final = soc_df_estimation.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "area_ha",
    "climate_class_corrected",
    "flu",
    "tillage",
    "tillage_count",
    "ftill",
    "categories_organic",
    "nitrogen_org_t_ha",
    "crop_residues_returned_total_pct",
    "fi_bau",
    "fi",
    "soc_ref",
    "soc_0",
    "soc_1",
    "soc_delta",
    "soc_delta_t_co2eq_ha",
    "soc_delta_t_co2eq",
]

soc_df_final = soc_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
soc_df_final.to_feather(PATH_EXPORT_TABLES / r"emissions_soc.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

soc_df_final.to_sql(name="emissions_soc", con=con, if_exists="replace", index=False)

# END
exit("Execution complete.")
