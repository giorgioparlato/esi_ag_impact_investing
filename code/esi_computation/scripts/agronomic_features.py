"""
Module Name: agronomic_features
Authors: Dario Marone, Giorgio Parlato
Date: 2025-10-27
Description:
    This script applies a series of preprocessing steps to the "2.-Sowing_harvest_yields_2012-2022_01.xlsx" dataset by CIMMYT.
    The main operations are unit conversions for yields and selective aggregation of area-crop-product-yield variables.
    The output is a table including agronomic features aggregated at the level of innovation or conventional practice IDs.
    The features include plot areas (with scale classification), type and number of crops / products, and yield variables.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import janitor
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Project directory
PATH_PROJECT_DIRECTORY = r"OneDrive - Kungl. Vetenskapsakademien\esi_cimmyt"

# CIMMYT dataset storing agronomic features
PATH_CIMMYT_MEXICO_AGRONOMIC_FEATURES = (
    PATH_HOME
    / PATH_PROJECT_DIRECTORY
    / r"data\raw\cimmyt\BEM_e-Agrology_Mexico\2.-Sowing_harvest_yields_2012-2022_01.xlsx"
)

# Sheet name
SHEET_CIMMYT_MEXICO_AGRONOMIC_FEATURES = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_PROJECT_DIRECTORY / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_PROJECT_DIRECTORY / r"data\processed\sql_db\cimmyt.db"

# ID columns
ID_COLUMNS = ["farmer_id", "plot_id", "logbook_id", "practice_id"]

# Year column
YEAR_COLUMN = ["year"]

# Area-related columns
AREA_COLUMNS = ["area_ha"]

# Other columns
OTHER_COLUMNS = [
    "crop",
    "product_obtained",
    "actual_yield_unit_ha",
    "unit_of_measure_ha",
]

# Conversion factor from bales to metric tons (https://www.convertunits.com/from/bales/to/metric+ton)
CF_BALE_TON = 0.2177243376

# Conversion factor from kilograms to metric tons
CF_KG_TON = 0.001

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the CIMMYT dataset storing agronomic features
agronomic_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_AGRONOMIC_FEATURES,
    sheet_name=SHEET_CIMMYT_MEXICO_AGRONOMIC_FEATURES,
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
agronomic_df_clean_cols = agronomic_df_raw.clean_names(
    axis="columns",  # Clean the labels on the columns
    strip_underscores=True,  # Remove outer underscores from all column names
    case_type="lower",  # Make all characters lower case
    remove_special=True,  # Preserve only letters, numbers, and underscores
    strip_accents=True,  # Remove accents
    preserve_original_labels=True,  # Retrievable using df.original_labels
    enforce_string=True,  # Convert all column names to string type
)

# Rename selected columns
agronomic_df_clean_cols.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
        "area_ha_sowing_whit_innovation_or_conventional_practices": "area_ha",
    },
    inplace=True,
)

# Convert id columns to string type
agronomic_df_adjust_dtypes = agronomic_df_clean_cols.copy()
agronomic_df_adjust_dtypes[ID_COLUMNS] = agronomic_df_adjust_dtypes[ID_COLUMNS].astype(
    "string"
)

# Select variables of interest
agronomic_df_vars = agronomic_df_adjust_dtypes.copy()
agronomic_df_vars = agronomic_df_vars[
    ID_COLUMNS + YEAR_COLUMN + AREA_COLUMNS + OTHER_COLUMNS
]

# Conversions from bales and kgs to metric tons of yield per hectare
agronomic_df_units = agronomic_df_vars.copy()
agronomic_df_units["yield_t_ha"] = np.select(
    [
        (agronomic_df_units["unit_of_measure_ha"] == "TON"),
        (agronomic_df_units["unit_of_measure_ha"] == "BALE"),
        (agronomic_df_units["unit_of_measure_ha"] == "KG"),
    ],
    [
        agronomic_df_units["actual_yield_unit_ha"],
        agronomic_df_units["actual_yield_unit_ha"] * CF_BALE_TON,
        agronomic_df_units["actual_yield_unit_ha"] * CF_KG_TON,
    ],
    default=np.nan,
)

# Drop unnecessary columns
agronomic_df_units.drop(
    columns=["actual_yield_unit_ha", "unit_of_measure_ha"], inplace=True
)

# Group and aggregate variables selectively
agronomic_df_agg = agronomic_df_units.copy()

# Group and aggregate variables selectively
agronomic_df_agg = agronomic_df_units.copy()

agronomic_df_agg = (
    agronomic_df_agg.groupby(
        ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
    )
    .agg(
        area_ha=("area_ha", "mean"),
        crops=("crop", lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))),
        crop_count=("crop", "nunique"),
        products=(
            "product_obtained",
            lambda x: ", ".join(sorted(x.dropna().astype(str).unique())),
        ),
        product_count=("product_obtained", "nunique"),
        maize_grain_yield_t_ha=(
            "yield_t_ha",
            lambda y: y[
                (agronomic_df_agg.loc[y.index, "crop"] == "MAIZE")
                & (agronomic_df_agg.loc[y.index, "product_obtained"] == "GRAIN")
            ].sum(min_count=1),
        ),
        maize_forage_yield_t_ha=(
            "yield_t_ha",
            lambda y: y[
                (agronomic_df_agg.loc[y.index, "crop"] == "MAIZE")
                & (agronomic_df_agg.loc[y.index, "product_obtained"] == "FORAGE")
            ].sum(min_count=1),
        ),
    )
    .reset_index()
)

# Calculate total yields
agronomic_df_agg["maize_grain_yield_t"] = (
    agronomic_df_agg["maize_grain_yield_t_ha"] * agronomic_df_agg["area_ha"]
)
agronomic_df_agg["maize_forage_yield_t"] = (
    agronomic_df_agg["maize_forage_yield_t_ha"] * agronomic_df_agg["area_ha"]
)

# Insert column with areas in square kilometers
agronomic_df_sqkm = agronomic_df_agg.copy()
position = agronomic_df_sqkm.columns.get_loc("area_ha")
new_col = agronomic_df_sqkm["area_ha"] / 100
agronomic_df_sqkm.insert(position + 1, "area_km2", new_col)

# Insert column with scale classification
# Based on: https://doi.org/10.1371/journal.pone.0292528.t001
agronomic_df_scale = agronomic_df_sqkm.copy()
scale_col = np.select(
    [
        (agronomic_df_scale["area_ha"] < 5),
        (agronomic_df_scale["area_ha"] >= 5) & (agronomic_df_scale["area_ha"] <= 50),
        (agronomic_df_scale["area_ha"] > 50),
    ],
    ["SMALL SCALE", "MIDDLE SCALE", "LARGE SCALE"],
    default=pd.NA,
)
position = agronomic_df_scale.columns.get_loc("area_km2")
agronomic_df_scale.insert(position + 1, "scale", scale_col)

# Final DataFrame
agronomic_df_final = agronomic_df_scale.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()


# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
agronomic_df_final.to_feather(PATH_EXPORT_TABLES / r"agronomic_features.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

agronomic_df_final.to_sql(
    name="agronomic_features", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
