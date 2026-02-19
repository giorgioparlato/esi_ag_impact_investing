"""
Clean and reshape cost and revenue data, compute economic metrics.

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
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# CIMMYT costs and revenues dataset (version  6.0)
PATH_CIMMYT_MEXICO_ECONOMICS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\6.-Costs and revenues_2012-2022_02.xlsx"
)

SHEET_CIMMYT_MEXICO_ECONOMICS = "Data"

# Agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_03_agronomic_features.feather"
)

# Export path for processed tables
PATH_PROCESSED_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load dataset on costs and revenues
economics_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_ECONOMICS, sheet_name=SHEET_CIMMYT_MEXICO_ECONOMICS, header=1
)

# Load plot areas
areas_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES, columns=["practice_id", "year", "area_ha"]
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean columns
economics_df_clean_columns = economics_df_raw.copy()
economics_df_clean_columns = economics_df_clean_columns.clean_names(
    axis="columns",
    strip_underscores=True,
    case_type="lower",
    remove_special=True,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
)

# Convert id columns to string type
economics_df_dtypes = economics_df_clean_columns.copy()
economics_df_dtypes[ID_KEYS_ECONOMICS] = economics_df_dtypes[ID_KEYS_ECONOMICS].astype(
    "string"
)

# Copy previous table
economics_df_reshape = economics_df_dtypes.copy()

# Columns that should not be unpivoted
id_vars = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "plot_type",
    "state",
    "municipality",
    "year",
    "winter_summer_season",
    "hydric_regime",
]

# Rename column names
new_cols = {}
for col in economics_df_reshape.columns:
    if col in id_vars:
        continue
    elif col == "innovation_practices_id":
        new_cols[col] = "practice_id_INNOVATION"
    elif col == "conventional_practices_id":
        new_cols[col] = "practice_id_CONVENTIONAL"
    elif col.endswith("_1"):
        new_cols[col] = col.replace("_1", "_CONVENTIONAL")
    else:
        new_cols[col] = f"{col}_INNOVATION"

economics_df_reshape = economics_df_reshape.rename(columns=new_cols)

# List of wide format variable names to unpivot (common prefix)
stubs = [
    c.replace("_INNOVATION", "")
    for c in economics_df_reshape.columns
    if c.endswith("_INNOVATION")
]

# Reshape from wide to long
economics_df_reshape = pd.wide_to_long(
    economics_df_reshape,
    stubnames=stubs,
    i=id_vars,
    j="practices_type",
    sep="_",
    suffix=r"(INNOVATION|CONVENTIONAL)",
).reset_index()

economics_df_reshape = economics_df_reshape.dropna(subset=["practice_id"])

economics_df_reshape.rename(
    columns={
        "total_costs_ha": "total_cost_mxn_ha",
        "income_ha": "total_revenue_mxn_ha",
    },
    inplace=True,
)

# Compute financial metrics
economics_df_metrics = economics_df_reshape.merge(
    areas_df, on=["practice_id", "year"], how="left"
)

# Total cost
economics_df_metrics["total_cost_mxn"] = (
    economics_df_metrics["total_cost_mxn_ha"] * economics_df_metrics["area_ha"]
)

# Total revenue
economics_df_metrics["total_revenue_mxn"] = (
    economics_df_metrics["total_revenue_mxn_ha"] * economics_df_metrics["area_ha"]
)

# Net income per hectare
economics_df_metrics["net_income_mxn_ha"] = (
    economics_df_metrics["total_revenue_mxn_ha"]
    - economics_df_metrics["total_cost_mxn_ha"]
)

# Total net income
economics_df_metrics["net_income_mxn"] = (
    economics_df_metrics["net_income_mxn_ha"] * economics_df_metrics["area_ha"]
)

# Operating expense ratio
economics_df_metrics["oer"] = (
    economics_df_metrics["total_cost_mxn"] / economics_df_metrics["total_revenue_mxn"]
)

# Retrun on investment
economics_df_metrics["roi"] = (
    economics_df_metrics["net_income_mxn"] / economics_df_metrics["total_cost_mxn"]
)

# Replace inifinities with NaN
economics_df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)


# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
economics_df_metrics.to_feather(
    PATH_PROCESSED_TABLES / r"table_04_economic_features.feather"
)

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

economics_df_metrics.to_sql(
    name="table_04_economic_features", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
