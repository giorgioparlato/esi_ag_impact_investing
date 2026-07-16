"""
Preprocessing of the logbook dataset by CIMMYT.
Main operations:
    - cleaning column names
    - adjusting data types
    - dropping rows missing key information
    - dropping plots with multiple coordinates
    - selecting variables
    - exporting the cleaned reference table as a feather file and to an SQL database

Authors: Dario Marone, Giorgio Parlato
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
from paths import (
    PATH_CIMMYT_LOGBOOK,
    SHEET_CIMMYT_LOGBOOK,
    PATH_PROCESSED_TABLES,
    PATH_SQL_DB,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Define the id columns
ID_COLUMNS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
]

# Define the key columns that must not be missing
KEY_COLUMNS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "latitude_n",
    "longitude_w",
    "plot_type",
    "practices_type",
]

# Define other variables of interest
OTHER_VARS = [
    "state",
    "id_state_inegi",
    "municipality",
    "id_municipality_inegi",
    "land_ownership",
    "property_type",
    "winter_summer_season",
    "hydric_regime",
    "agroecological_hub",
    "technification_level",
    "predominant_topografic_relief",
    "soil_texture",
    "soil_erosion_level",
    "average_annual_precipitation_mm",
    "average_annual_temperature_c",
    "main_incline",
    "total_area_of_the_plots_sown_by_producer_ha",
    "total_plot_area_ha_data_obtained_by_survey_with_the_farmer",
    "area_ha_approach_data_calculated_by_the_delimitation_of_the_plot_in_satellite_image",
    "conservation_agriculture",
    "fertilization",
    "planting_arrangements",
    "water_conservation",
    "integrated_pest_management",
    "improved_varieties",
    "soil_conservation",
    "commercialization",
    "postharvest",
    "other_innovation",
]

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the dataset
logbook_df_raw = pd.read_excel(
    io=PATH_CIMMYT_LOGBOOK, sheet_name=SHEET_CIMMYT_LOGBOOK, header=1
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
logbook_df_clean_columns = logbook_df_raw.copy()
logbook_df_clean_columns = logbook_df_raw.clean_names(
    axis="columns",
    strip_underscores=True,
    case_type="lower",
    remove_special=True,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
)

logbook_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
        "planting_arregements": "planting_arrangements",
    },
    inplace=True,
)

# Strip trailing spaces
logbook_df_clean_columns["practices_type"] = logbook_df_clean_columns[
    "practices_type"
].str.strip()

# Convert mixed columns to string type to prevent downstream issues
# The string conversion preserves missing values in the id columns
logbook_df_adjust_dtypes = logbook_df_clean_columns.copy()
logbook_df_adjust_dtypes[ID_COLUMNS] = logbook_df_adjust_dtypes[ID_COLUMNS].astype(
    "string"
)

# Ensure the year column is of integer type
logbook_df_adjust_dtypes["year"] = pd.to_numeric(
    logbook_df_adjust_dtypes["year"], errors="coerce"
).astype("Int64")

# Drop rows with missing values in the key columns
logbook_df_dropna = logbook_df_adjust_dtypes.copy()
logbook_df_dropna.dropna(subset=KEY_COLUMNS, inplace=True)

# Drop plot ids with multiple coordinates
ids_coords_df = logbook_df_dropna[KEY_COLUMNS].copy()

# Create combined coordinates column
ids_coords_df["coordinates"] = list(
    zip(ids_coords_df["latitude_n"], ids_coords_df["longitude_w"]),
)

# Set values as strings
ids_coords_df["coordinates"] = ids_coords_df["coordinates"].astype(str)

# Count unique coordinates by plot id
nunique_coords_by_id = ids_coords_df.groupby("plot_id")["coordinates"].nunique()

# Get the problematic ids
problematic_ids = nunique_coords_by_id[nunique_coords_by_id > 1].index.values

# Create a mask for dropping
bad_records_mask = ids_coords_df["plot_id"].isin(problematic_ids)

# Drop the problematic ids
logbook_df_bad_records = logbook_df_dropna[~bad_records_mask].copy()

# Select variables of interest
logbook_df_clean = logbook_df_bad_records[KEY_COLUMNS + OTHER_VARS].copy()

# Reindex the DataFrame
logbook_df_final = logbook_df_clean.reindex(
    axis=1, labels=KEY_COLUMNS + OTHER_VARS, copy=True
)

# Sort values
logbook_df_final = logbook_df_final.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
logbook_df_final.to_feather(PATH_PROCESSED_TABLES / "table_01_reference.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    logbook_df_final.to_sql(
        name="table_01_reference",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
