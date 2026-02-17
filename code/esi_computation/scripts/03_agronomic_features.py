"""
Preprocessing of the sowing-harvest-yields dataset by CIMMYT.
Main operations:
    - clean column names and adjust data types
    - unit conversions for yields
    - transformation of original yield column into separate columns for maize grain and forage yields
    - aggregation at the practice ID level
    - insertion of area in square kilometers and scale classification
    - export results as feather file and to SQL database

Authors: Dario Marone, Giorgio Parlato
Date: 2026-02-13
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
# Paths
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# CIMMYT dataset storing agronomic features
PATH_CIMMYT_MEXICO_AGRONOMIC_FEATURES = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\BEM_e-Agrology_Mexico\2.-Sowing_harvest_yields_2012-2022_01.xlsx"
)

# Sheet name
SHEET_CIMMYT_MEXICO_AGRONOMIC_FEATURES = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
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

# Conversion factor from bales (US) to metric tons
# Source: https://www.convertunits.com/from/bales/to/metric+ton
CF_BALE_TON = 0.2177243376

# Conversion factor from kilograms to metric tons
# Source: https://www.google.com/search?q=kilograms+to+metric+tons&oq=kilograms+to+metric+tons&gs_lcrp=EgZjaHJvbWUyCQgAEEUYORiABDIGCAEQBhhA0gEIMzg2OGowajSoAgGwAgHxBW6Qdp7Y_Xbz8QVukHae2P128w&sourceid=chrome&ie=UTF-8
CF_KG_TON = 0.001

SUPPORTED_UNITS = ["TON", "BALE", "KG"]

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

supported_units = agronomic_df_units["unit_of_measure_ha"].isin(SUPPORTED_UNITS)

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

# Define masks
maize_grain_mask = (agronomic_df_units["crop"] == "MAIZE") & (
    agronomic_df_units["product_obtained"] == "GRAIN"
)
maize_forage_mask = (agronomic_df_units["crop"] == "MAIZE") & (
    agronomic_df_units["product_obtained"] == "FORAGE"
)

# Compute yield fields for masks
agronomic_df_units["maize_grain_yield_t_ha"] = agronomic_df_units["yield_t_ha"].where(
    maize_grain_mask
)
agronomic_df_units["maize_forage_yield_t_ha"] = agronomic_df_units["yield_t_ha"].where(
    maize_forage_mask
)

# Supported-unit flags for masks
agronomic_df_units["grain_unit_supported"] = supported_units & maize_grain_mask
agronomic_df_units["forage_unit_supported"] = supported_units & maize_forage_mask

# Group and aggregate variables selectively
group_cols = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]


def join_unique(x):
    return ", ".join(sorted(x.dropna().unique()))


def sum_min_count(x):
    return x.sum(min_count=1)


def unit_support_status(x):
    if not x.any():
        return "NOT APPLICABLE"
    elif x.all():
        return "SUPPORTED"
    else:
        return "UNSUPPORTED"


agronomic_df_agg = (
    agronomic_df_units.groupby(group_cols, sort=False)
    .agg(
        area_ha=("area_ha", "mean"),
        crops=("crop", join_unique),
        crop_count=("crop", "nunique"),
        products=("product_obtained", join_unique),
        product_count=("product_obtained", "nunique"),
        maize_grain_yield_t_ha=("maize_grain_yield_t_ha", sum_min_count),
        maize_forage_yield_t_ha=("maize_forage_yield_t_ha", sum_min_count),
        unit_support_status_grain=("grain_unit_supported", unit_support_status),
        unit_support_status_forage=("forage_unit_supported", unit_support_status),
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
# Source: https://doi.org/10.1371/journal.pone.0292528.t001
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
agronomic_df_final.to_feather(
    PATH_EXPORT_TABLES / r"table_03_agronomic_features.feather"
)

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

agronomic_df_final.to_sql(
    name="table_03_agronomic_features", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
