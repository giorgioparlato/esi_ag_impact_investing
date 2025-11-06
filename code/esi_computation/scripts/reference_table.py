"""
Module Name: reference_table
Authors: Dario Marone, Giorgio Parlato
Date: 2025-11-03
Description:
    This script loads, cleans, and transforms the logbook dataset by CIMMYT.
    The main operations are:
        - Cleaning column names, adjusting data types
        - Dropping rows with missing coordinates
        - Dropping plots with multiple coordinates
        - Selecting variables
        - Pivoting to long format
    The output is a reference table, including entity, space, and time indexes.
    These variables form the foundation for any Earth system impact analysis.
    Other essential attributes can be included as needed depending on the user needs.
    In this case, the plot type and practices type are also included as additional variables.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
# File paths and sheet names
# Get home directory (useful if you are working on different machines)
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# Path to CIMMYT logbook dataset (Mexico)
PATH_CIMMYT_MEXICO_LOGBOOK = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\BEM_e-Agrology_Mexico\1.-Farmer_Plot_Logbook_2012-2022_02.xlsx"
)

SHEET_CIMMYT_MEXICO_LOGBOOK = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# Define the id columns
ID_COLUMNS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "innovation_practices_id",
    "conventional_practices_id",
]

# Define the key columns that must not be missing
KEY_COLUMNS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "innovation_practices_id",
    "conventional_practices_id",
    "year",
    "latitude_n",
    "longitude_w",
]

# Define other variables of interest
OTHER_VARS = [
    "plot_type",
    "property_type",
    "land_ownership",
    "state",
    "municipality",
    "agroecological_hub",
    "technification_level",
    "winter_summer_season",
    "conservation_agriculture",
    "fertilization",
    "planting_arrangements",
    "water_conservation",
    "integrated_pest_management",
    "improved_varieties",
    "soil_conservation",
    "commercialization",
    "postharvest",
]

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the dataset
logbook_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_LOGBOOK, sheet_name=SHEET_CIMMYT_MEXICO_LOGBOOK, header=0
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
    columns={"planting_arregements": "planting_arrangements"}, inplace=True
)

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
# Excluding conventional_practices_id column from key columns during this operation
# Otherwise all extension plots (without conventional ids) would be dropped
logbook_df_dropna = logbook_df_adjust_dtypes.copy()
columns_for_dropna = [col for col in KEY_COLUMNS if col != "conventional_practices_id"]
logbook_df_dropna.dropna(subset=columns_for_dropna, inplace=True)

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

# Unpivot the DataFrame to long format
logbook_df_melted = pd.melt(
    logbook_df_clean,
    id_vars=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "year",
        "latitude_n",
        "longitude_w",
    ]
    + OTHER_VARS,
    value_vars=["innovation_practices_id", "conventional_practices_id"],
    value_name="practice_id",
)

# Drop rows where no practice id is available
logbook_df_melted.dropna(subset=["practice_id"], inplace=True)

# Map previous column names to value consistent with the rest of the dataset
practices_map = {
    "innovation_practices_id": "INNOVATION",
    "conventional_practices_id": "CONVENTIONAL",
}

# Replace the values using the map
logbook_df_melted["variable"] = logbook_df_melted["variable"].replace(practices_map)

# Rename the column
logbook_df_melted.rename(columns={"variable": "practices_type"}, inplace=True)

# Reindex the DataFrame
logbook_df_final = logbook_df_melted.reindex(
    axis=1,
    labels=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "longitude_w",
        "latitude_n",
        "year",
        "practices_type",
    ]
    + OTHER_VARS,
    copy=True,
)

# Sort values
logbook_df_final = logbook_df_final.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write the final DataFrame as a feather file
logbook_df_final.to_feather(PATH_EXPORT_TABLES / r"reference_table.feather")

# Write the DataFrame to SQL
con = sqlite3.connect(PATH_SQL_DB)
logbook_df_final.to_sql(
    name="reference_table", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
