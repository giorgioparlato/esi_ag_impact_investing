#!/usr/bin/env python3    # Use Python 3 interpreter from PATH (Unix/Linux/macOS)
# -*- coding: utf-8 -*-    # Ensure file uses UTF-8 encoding (for special characters)

"""
Module Name: reference_table
Authors: Dario Marone, Giorgio Parlato
Date: 2025-10-03
Description:
    This script loads, cleans, and transforms the logbook dataset by CIMMYT.
    The output is a reference table, including entity, space, and time indexes.
    These variables form the foundation for any Earth system impact analysis.
    Other essential attributes can be included as needed depending on the user needs.
    In this case, the plot type and practices type are also included as additional variables.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import logging # Event logging system
from pathlib import Path # Object-oriented filesystem paths
import numpy as np # Fundamental package for scientific computing
import pandas as pd # Data analysis toolkit
from pandas.api.types import infer_dtype # Function to catch mixed type columns
import janitor # Python implementation of the R package janitor for data cleaning
import warnings # Warning control
warnings.filterwarnings("ignore", category=DeprecationWarning)

#-----------------------------------------------------------------------------
# Logging configuration
#-----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

#-----------------------------------------------------------------------------
# Paths
#-----------------------------------------------------------------------------
# Home directory
HOME = Path.home()

# Open CIMMYT dataset (default)
PATH_CIMMYT_MEXICO_LOGBOOK = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\CYMMIT\BEM_e-Agrology_Mexico\1.-Farmer_Plot_Logbook_2012-2022_02.xlsx"
)

SHEET_CIMMYT_MEXICO_LOGBOOK = "Data"

# Inflection Finance dataset
PATH_INFLECTION_FINANCE_LOGBOOK = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\DAT-EAgrology\DAT-EAgrology\1.-Farmer_Plot_Logbook_2012-2022.xlsx"
)

SHEET_INFLECTION_FINANCE_LOGBOOK = "DATA"

# Outputh path
OUTPUT_PATH = HOME / r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data\Intermediate Tables"

#-----------------------------------------------------------------------------
# Data cleaning workflow
#-----------------------------------------------------------------------------
# Load the dataset
# Note: the default path is for the open CIMMYT dataset on Mexico
logging.info(
    "Loading dataset from %s (sheet: %s)",
    PATH_CIMMYT_MEXICO_LOGBOOK,
    SHEET_CIMMYT_MEXICO_LOGBOOK
)

logbook_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_LOGBOOK,
    sheet_name=SHEET_CIMMYT_MEXICO_LOGBOOK,
    header=0
)

logging.info(
    "Loaded DataFrame with shape %s",
    logbook_df_raw.shape
)

# Clean the column names
logging.info(
    "Cleaning column names: %s",
    logbook_df_raw.columns.values
)

logbook_df_clean_cols = logbook_df_raw.clean_names(
    axis='columns', # Clean the labels on the columns
    strip_underscores=True, # Remove outer underscores from all column names 
    case_type='lower', # Make all characters lower case
    remove_special=True, # Preserve only letters, numbers, and underscores
    strip_accents=True, # Remove accents 
    preserve_original_labels=True, # Retrievable using df.original_labels
    enforce_string=True, # Convert all column names to string type
)

logging.info(
    "Completed cleaning column names: %s", 
    logbook_df_clean_cols.columns.values
)

# Any data that cannot be linked to an entity and located across time / space is unusable
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

# Creat ad-hoc DataFrame
logbook_df_missing = logbook_df_clean_cols.copy()

# Check for missing values in those columns
missing_counts = logbook_df_missing[KEY_COLUMNS].isna().sum()
missing_counts = missing_counts[missing_counts > 0]

if not missing_counts.empty:
    logging.warning(
        "Missing values found in the following key columns:\n%s",
        missing_counts
    )
else:
    logging.info(
        "No missing values found in key columns."
    )

# Before proceding with dropping rows where relevant we check dtypes
# Detect columns with mixed types
logging.info(
    "Detecting columns with mixed data types"
)

logbook_df_adjust_dtypes = logbook_df_missing.copy() # Copy previous DataFrame
col_types = logbook_df_adjust_dtypes.apply(infer_dtype) # Infer dtypes
mixed_cols = col_types[col_types.str.contains("mixed")] # Isolate columns with mixed types
mixed_col_names = mixed_cols.index.tolist() # List column names 

if not mixed_cols.empty:
    logging.warning("Columns with mixed types detected:\n%s",
                    mixed_cols
    )
else:
    logging.info(
        "No mixed-type columns detected."
    )

# Convert mixed columns to string type to prevent downstream issues
# Also convert the farmer_id column for consistency
# We are preserving NaNs in the string conversion
logging.info(
    "Converting mixed-integer columns and farmer_id to string type."
)

logbook_df_adjust_dtypes[mixed_col_names + ["farmer_id"]] = (
    logbook_df_adjust_dtypes[mixed_col_names + ["farmer_id"]]
    .astype('string')
)

logging.info(
    "Converted mixed-integer columns and farmer_id to string type:\n%s",
    logbook_df_adjust_dtypes[
        mixed_col_names + ['farmer_id']
        ].apply(infer_dtype)
)

# Ensure the year column is of integer type
logging.info(
    "Ensuring year column is of integer type."
)

logbook_df_adjust_dtypes["year"] = pd.to_numeric(
    logbook_df_adjust_dtypes["year"],
    errors="coerce",
).astype("Int64")

logging.info(
    "Year column is of type: %s",
    logbook_df_adjust_dtypes["year"].dtype
)

# Drop rows with missing values in the KEY_COLUMNS (except for conventional_practices_id)
# Define key identifier to track lost entities
KEY_ID = "plot_id"

# Excluding conventional_practices_id column from key columns during this operation
# Otherwise all extension plots would be dropped
columns_for_dropna = [col for col in KEY_COLUMNS if col != "conventional_practices_id"]

logging.info(
    "Dropping rows with missing values in the KEY_COLUMNS, except for conventional_practices_id."
)

logbook_df_dropna = logbook_df_adjust_dtypes.copy() # Copy previous dataframe
before_dropna = logbook_df_dropna.shape[0] # Get shape before dropping
ids_before_dropna = logbook_df_dropna[KEY_ID].nunique() # Count unique ids before dropping
logbook_df_dropna.dropna(subset=columns_for_dropna, inplace=True) # Drop rows with missing values in the subset
after_dropna = logbook_df_dropna.shape[0] # Get new shape
ids_after_dropna = logbook_df_dropna[KEY_ID].nunique() # Get new count of unique ids

logging.info(
    "Dropped %d rows due to missing key fields."
    , 
    before_dropna - after_dropna
)

logging.info(
    "Lost %d unique %s due to missing key fields.",
    ids_before_dropna - ids_after_dropna, KEY_ID
)

# Check if there are any duplicated rows for the KEY_COLUMNS
logging.info(
    "Checking if any duplicate rows exist for the KEY_COLUMNS."
)

logbook_df_dupes = logbook_df_dropna.copy() # Copy last DataFrame
dupes_mask = logbook_df_dupes.duplicated(subset=KEY_COLUMNS, keep=False) # Boolean mask of duplicates
dupes = logbook_df_dupes[dupes_mask] # Filter rows that are duplicates

if not dupes.empty:
    logging.warning(
        "Duplicated rows found for the following subset:\n%s",
        KEY_COLUMNS
    )
else:
    logging.info("No duplicate rows found.")

# Check for the presence of inconsistence mappings between plot ids and coordinates
logging.info(
    "Checking for inconsistent mappings between plot ids and coordinates."
)

# Creat ad-hoc df
id_coord_check_df = logbook_df_dupes[KEY_COLUMNS].copy()

# Create column storing coordinate pairs
id_coord_check_df["coordinates"] = list(
    zip(id_coord_check_df["latitude_n"], id_coord_check_df["longitude_w"]), 
)

# Cast values as strings
id_coord_check_df["coordinates"] = (
    id_coord_check_df["coordinates"].astype(str)
 ) 

# Identifying problematic ids with multiple coordinates
# Group ids by number of unique coordinate pairs
nunique_coords_by_plot = (
    id_coord_check_df.groupby(KEY_ID)["coordinates"].nunique()
)

# Store index values to an array
problematic_ids = (
    nunique_coords_by_plot[nunique_coords_by_plot > 1].index.values
)

# Count problematic ids
problematic_ids_count = len(set(problematic_ids))

if len(problematic_ids) > 0:
    logging.warning("%s plot ids linked to multiple coordinates: %s",
                    problematic_ids_count, problematic_ids
    )
else:
    logging.info(
        "No plot ids linked to multiple coordinates found."
    )

# Collect, save, and drop bad records
logging.info(
    "Collecting, saving, and dropping bad records."
)

# Create a mask
bad_records_mask = (
    id_coord_check_df[KEY_ID].isin(problematic_ids)
) 

# Apply the mask and create a DataFrame with the bad records
bad_records = logbook_df_dupes[bad_records_mask] 

logging.warning(
    "Found and saved %d problematic rows.",
    bad_records.shape[0]
)

# Drop bad records
# Create a new DataFrame without the bad records
logbook_df_bad_records = logbook_df_dupes[~bad_records_mask].copy() 
logging.info(
    "Dropped %d rows. Remaining rows: %d. Lost %d plot ids.",
    bad_records.shape[0],
    logbook_df_bad_records.shape[0],
    bad_records[KEY_ID].nunique())

# Select variables of interest 
logging.info(
    "Selecting variables of interest."
)

OTHER_VARS = [
    "plot_type"
]

logbook_df_clean = logbook_df_bad_records[KEY_COLUMNS + OTHER_VARS].copy()

logging.info(
    "Selected the following variables of interest: %s",
    KEY_COLUMNS + OTHER_VARS
)

# Unpivot the DataFrame to long format
logging.info(
    "Unpivoting the DataFrame from wide to long format, and cleaning it."
)

logbook_df_melted = pd.melt(
    logbook_df_clean,
    id_vars=[
        'farmer_id',
        'plot_id',
        'logbook_id',
        'year',
        'latitude_n',
        'longitude_w',
        'plot_type'
    ],
    value_vars=["innovation_practices_id", "conventional_practices_id"],
    value_name="practice_id"
)

# Drop rows where no practice id is available
logbook_df_melted.dropna(subset=["practice_id"], inplace=True)

# Map previous column names to value consistent with the rest of the dataset
practices_map = {
    "innovation_practices_id": "INNOVATION",
    "conventional_practices_id": "CONVENTIONAL"
}

# Replace the values using the map
logbook_df_melted["variable"] = logbook_df_melted["variable"].replace(practices_map)

# Rename the column
logbook_df_melted.rename(columns={"variable":"practices_type"}, inplace=True)

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
        "plot_type",
        "practices_type"
        ],
        copy=True
)

# Sort values
logbook_df_final = logbook_df_final.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

logging.info(
    "Final DataFrame is of shape: %s.",
    logbook_df_final.shape
)

# Write the final DataFrame as a feather file
logging.info(
    "Writing the final DataFrame to a feather file."
)

logbook_df_final.to_feather(OUTPUT_PATH / "out_reference_table.feather")

logging.info(
    "Reference table ready to use."
)