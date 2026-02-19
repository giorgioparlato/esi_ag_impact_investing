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
Date: 2026-02-19
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
# File paths and sheet names
# Get home directory (useful if you are working on different machines)
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# # CIMMYT irrigation dataset (version 6.0)
# PATH_CIMMYT_MEXICO_WATER = (
#     PATH_HOME
#     / PATH_DATA
#     / r"data\raw\cimmyt\dataverse_files\5.-Irrigacion_2012-2022_01.xlsx"
# )

# SHEET_CIMMYT_MEXICO_WATER = "Data"

# Dataset with gradient boosting imputed irrigation data
PATH_WATER_GRADIENT_BOOSTING = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\irrigation_df_GB_imputed.feather"
)

# Path to water tariffs
# Source: compiled from https://www.gob.mx/conagua/documentos/ley-federal-de-derechos-386262
PATH_WATER_TARIFFS = (
    PATH_HOME / PATH_DATA / r"data\raw\water\water_tariffs_conagua.xlsx"
)

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_03_agronomic_features.feather"
)

# Path to CIMMYT cost-revenue dataset (Mexico)
PATH_CIMMYT_MEXICO_ECONOMICS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\6.-Costs and revenues_2012-2022_02.xlsx"
)

SHEET_CIMMYT_MEXICO_ECONOMICS = "Data"

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

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
    "total_water_consumption_per_irrigation_lt",
    "water_hybrid_gb_all_poisson_lt",
]

# Emission factors for irrigation equipment and application (precalculated)
# Reference: Juárez-Hernández, S., & Pardo, C. S. (2019). Irrigation energy use and related greenhouse gas emissions of maize production in Mexico. International journal of water resources development.
# Source: based on values in Table 7 in https://doi.org/10.1080/07900627.2018.1482739;
# average in tCO2eq per m3 of the values for summer and winter season,
# for pressurized systems only, excluding groundwater pumping
EF_IRR_EQUIP = 1.61319e-05
EF_IRR_APP = 0.000148952


# Helper functions for aggregation
def imputation_status(x: pd.Series) -> str:
    # Count values
    n_reported = (x == 0).sum()
    n_imputed = (x == 1).sum()
    n_missing = x.isna().sum()
    n_total = len(x)

    # Case 1: everything missing
    if n_missing == n_total:
        return "MISSING"

    # Case 2: all reported
    if n_reported == n_total:
        return "REPORTED"

    # Case 3: all imputed
    if n_imputed == n_total:
        return "IMPUTED"

    # Case 4: mix of reported and imputed (no missing)
    if n_reported > 0 and n_imputed > 0 and n_missing == 0:
        return "REPORTED AND IMPUTED"

    # Case 5: mix including missing
    return "PARTIALLY MISSING"


# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# # Load dataset on irrigation
# water_df_raw = pd.read_excel(
#     io=PATH_CIMMYT_MEXICO_WATER,
#     sheet_name=SHEET_CIMMYT_MEXICO_WATER
# )

# Load gradient boosting dataset
water_df_gb = pd.read_feather(
    PATH_WATER_GRADIENT_BOOSTING,
)

# Load dataset on costs and revenues
economics_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_ECONOMICS, sheet_name=SHEET_CIMMYT_MEXICO_ECONOMICS, header=1
)

# Load water tariffs from CONAGUA
water_tariffs_df = pd.read_excel(
    io=PATH_WATER_TARIFFS,
)

# Load plot areas
areas_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES, columns=["practice_id", "year", "area_ha"]
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# # Clean column names
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

# # Rename selected columns
# water_df_clean_columns.rename(
#     columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True
# )

# Convert id columns to string type
water_df_dtypes = water_df_gb.copy()
water_df_dtypes[ID_KEYS_WATER] = water_df_dtypes[ID_KEYS_WATER].astype("string")

economics_df_dtypes = economics_df_clean_columns.copy()
economics_df_dtypes[ID_KEYS_ECONOMICS] = economics_df_dtypes[ID_KEYS_ECONOMICS].astype(
    "string"
)

# Select variables of interest
water_df_vars = water_df_dtypes.copy()
water_df_vars = water_df_vars[
    ID_KEYS_WATER + YEAR_COL + WATER_MANAGEMENT_VARS + WATER_CONSUMPTION_VARS
]

# Assign project irrigation efficiency parameters based on irrigation type
# Reference: Rohwer, J., Gerten, D., & Lucht, W. (2007). Development of functional irrigation types for improved global crop modelling.
# Source: adapted from Table 6 in https://www.pik-potsdam.de/en/output/publications/pikreports/.files/pr104.pdf
water_df_efficiency = water_df_vars.copy()

efficiency_parameters_map = {
    "SURFACE IRRIGATION": 0.420,
    "SPRINKLER IRRIGATION": 0.713,
    "DRIP IRRIGATION": 0.855,
}

water_df_efficiency["irr_efficiency_param"] = water_df_efficiency[
    "irrigation_type"
].map(efficiency_parameters_map)

# Add a column to flag whether the value is reported or estimated
water_df_gbflag = water_df_efficiency.copy()

conditions_gb_method = [
    water_df_gbflag["water_hybrid_gb_all_poisson_lt"]
    == water_df_gbflag["total_water_consumption_per_irrigation_lt"],
    water_df_gbflag["water_hybrid_gb_all_poisson_lt"]
    != water_df_gbflag["total_water_consumption_per_irrigation_lt"],
]

choices_gb_method = [0, 1]

water_df_gbflag["flag_est_method_gb"] = np.select(
    conditions_gb_method, choices_gb_method, default=np.nan
)

# Concatenate water sources and irrigation types, aggregate water consumption data
water_df_agg = water_df_gbflag.copy()
water_df_agg = (
    water_df_agg.groupby(["farmer_id", "plot_id", "logbook_id", "practice_id", "year"])
    .agg(
        hydric_regime=(
            "hydric_regime",
            lambda x: (
                " | ".join(sorted(x.dropna().astype(str).unique()))
                if x.notna().all()
                else np.nan
            ),
        ),
        water_sources=(
            "water_source",
            lambda x: (
                " | ".join(sorted(x.dropna().astype(str).unique()))
                if x.notna().all()
                else np.nan
            ),
        ),
        water_sources_count=("water_source", "nunique"),
        irrigation_types=(
            "irrigation_type",
            lambda x: (
                " | ".join(sorted(x.dropna().astype(str).unique()))
                if x.notna().all()
                else np.nan
            ),
        ),
        irrigation_types_count=("irrigation_type", "nunique"),
        irrigation_events_count=("id_irrigation", "nunique"),
        irr_efficiency_param=("irr_efficiency_param", "mean"),
        total_water_consumption_per_irrigation_lt=(
            "total_water_consumption_per_irrigation_lt",
            lambda x: x.sum(min_count=1) if x.notna().all() else np.nan,
        ),
        water_hybrid_gb_all_poisson_lt=(
            "water_hybrid_gb_all_poisson_lt",
            lambda x: x.sum(min_count=1) if x.notna().all() else np.nan,
        ),
        flag_est_method_gb=("flag_est_method_gb", imputation_status),
    )
    .reset_index()
)

# Split the original dataframe between the innovation and conventional parts
# Create the innovation dataframe and rename practices id column
economics_df_innov = economics_df_dtypes.iloc[:, 0:24].copy()
economics_df_innov.rename(
    columns={"innovation_practices_id": "practice_id"}, inplace=True
)

# Create the conventional dataframe and rename columns accordingly
innov_cols_to_exclude = economics_df_clean_columns.columns[9:24]
economics_df_conv = economics_df_clean_columns.drop(
    columns=innov_cols_to_exclude
).copy()
economics_df_conv.columns = economics_df_conv.columns.str.replace(
    r"_1$", "", regex=True
)
economics_df_conv.rename(
    columns={"conventional_practices_id": "practice_id"}, inplace=True
)

# Concatenate the conventional and innovation dataframes
economics_df_concat = pd.concat([economics_df_conv, economics_df_innov]).sort_index()

economics_df_concat.dropna(subset=["practice_id"], inplace=True)

# Rename columns
economics_df_concat.rename(
    columns={"irrigation": "irrigation_cost_mxn_ha"}, inplace=True
)

irr_cost_df = economics_df_concat[
    [
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "year",
        "irrigation_cost_mxn_ha",
    ]
].copy()

# Merge with irrigation cost
water_df_merged = water_df_agg.merge(
    irr_cost_df,
    on=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"],
    how="left",
)

# Merge with water tariffs
water_df_merged = water_df_merged.merge(water_tariffs_df, on="year", how="left")

# Create ad-hoc DataFrame
water_df_metrics = water_df_merged.copy()

# Compute reported water consumption in cubic meters per hectare
water_df_metrics["water_cons_rep_m3_ha"] = np.where(
    water_df_metrics["total_water_consumption_per_irrigation_lt"].notna(),
    water_df_metrics["total_water_consumption_per_irrigation_lt"] * 10**-3,
    np.nan,
)

# Compute reported water consumption in cubic meters per hectare, adjusted for irrigation efficiency
water_df_metrics["water_cons_rep_adj_m3_ha"] = np.where(
    water_df_metrics["total_water_consumption_per_irrigation_lt"].notna()
    & water_df_metrics["irr_efficiency_param"].notna(),
    water_df_metrics["total_water_consumption_per_irrigation_lt"]
    * 10**-3
    * water_df_metrics["irr_efficiency_param"],
    np.nan,
)

# Compute water consumption estimates in cubic meters per hectare
water_df_metrics["water_cons_est_m3_ha"] = np.where(
    water_df_metrics["irrigation_cost_mxn_ha"].notna()
    & water_df_metrics["mxn_m3"].notna(),
    water_df_metrics["irrigation_cost_mxn_ha"] / water_df_metrics["mxn_m3"],
    np.nan,
)

# Compute water consumption estimates in cubic meters per hectare, adjusted for irrigation efficiency
water_df_metrics["water_cons_est_adj_m3_ha"] = np.where(
    water_df_metrics["irrigation_cost_mxn_ha"].notna()
    & water_df_metrics["mxn_m3"].notna()
    & water_df_metrics["irr_efficiency_param"].notna(),
    water_df_metrics["irrigation_cost_mxn_ha"]
    / water_df_metrics["mxn_m3"]
    * water_df_metrics["irr_efficiency_param"],
    water_df_metrics["irrigation_cost_mxn_ha"] / water_df_metrics["mxn_m3"],
)

# Create a new column to store both reported and estimated values
water_df_metrics["water_cons_hyb_m3_ha"] = water_df_metrics[
    "water_cons_rep_m3_ha"
].where(
    water_df_metrics["water_cons_rep_m3_ha"] > 0.0,
    water_df_metrics["water_cons_est_m3_ha"],
)

# Create a new column to store both reported and estimated values, adjusted for irrigation efficiency
water_df_metrics["water_cons_hyb_adj_m3_ha"] = water_df_metrics[
    "water_cons_rep_adj_m3_ha"
].where(
    water_df_metrics["water_cons_rep_adj_m3_ha"] > 0.0,
    water_df_metrics["water_cons_est_adj_m3_ha"],
)

# Add a column to flag whether the value is reported or estimated
conditions_cost_method = [
    water_df_metrics["water_cons_hyb_m3_ha"]
    == water_df_metrics["water_cons_rep_m3_ha"],
    water_df_metrics["water_cons_hyb_m3_ha"]
    == water_df_metrics["water_cons_est_m3_ha"],
]

choices_cost_method = ["REPORTED", "ESTIMATED"]

water_df_metrics["flag_est_method_cost"] = np.select(
    conditions_cost_method, choices_cost_method, default="MISSING"
)

# Create a new column using the gradient boosting estimates
water_df_metrics["water_cons_hyb_gb_poisson_m3_ha"] = (
    water_df_metrics["water_hybrid_gb_all_poisson_lt"] * 10**-3
)

# Create a new column using the gradient boosting estimates, adjusted for irrigation efficiency
water_df_metrics["water_cons_hyb_gb_poisson_adj_m3_ha"] = (
    water_df_metrics["water_hybrid_gb_all_poisson_lt"]
    * 10**-3
    * water_df_metrics["irr_efficiency_param"]
)

# Merge with areas from the agronomic dataframe
areas_df = areas_df.drop_duplicates(keep=False).copy()
water_df_totals = water_df_metrics.copy()
water_df_totals = water_df_totals.merge(
    areas_df, on=["practice_id", "year"], how="left"
)

# Compute total water consumption (reported, estimated, hybrid)
water_df_totals["water_cons_rep_m3"] = (
    water_df_totals.water_cons_rep_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_rep_adj_m3"] = (
    water_df_totals.water_cons_rep_adj_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_est_m3"] = (
    water_df_totals.water_cons_est_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_est_adj_m3"] = (
    water_df_totals.water_cons_est_adj_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_hyb_m3"] = (
    water_df_totals.water_cons_hyb_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_hyb_adj_m3"] = (
    water_df_totals.water_cons_hyb_adj_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_hyb_gb_poisson_m3"] = (
    water_df_totals.water_cons_hyb_gb_poisson_m3_ha * water_df_totals.area_ha
)

water_df_totals["water_cons_hyb_gb_poisson_adj_m3"] = (
    water_df_totals.water_cons_hyb_gb_poisson_adj_m3_ha * water_df_totals.area_ha
)

# Convert to cubic decameters for calculation of water ESI
water_df_totals["water_cons_hyb_dam3"] = water_df_totals.water_cons_hyb_m3 * 10**-3
water_df_totals["water_cons_hyb_adj_dam3"] = (
    water_df_totals.water_cons_hyb_adj_m3 * 10**-3
)
water_df_totals["water_cons_hyb_gb_poisson_dam3"] = (
    water_df_totals.water_cons_hyb_gb_poisson_m3 * 10**-3
)
water_df_totals["water_cons_hyb_gb_poisson_adj_dam3"] = (
    water_df_totals.water_cons_hyb_gb_poisson_adj_m3 * 10**-3
)

# Emissions based on non-adjusted hybrid water consumption estimates (cost method), per hectare basis
water_df_emissions = water_df_totals.copy()

water_df_emissions["irr_app_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_m3_ha"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on non-adjusted hybrid water consumption estimates (GB method), per hectare basis
water_df_emissions["irr_app_gb_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_m3_ha"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_gb_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (cost method), per hectare basis
water_df_emissions["irr_app_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_adj_m3_ha"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_adj_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (GB method), per hectare basis
water_df_emissions["irr_app_gb_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_adj_m3_ha"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_gb_adj_tco2eq_ha"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_adj_m3_ha"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on non-adjusted hybrid water consumption estimates (cost method), total
water_df_emissions["irr_app_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_m3"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on non-adjusted hybrid water consumption estimates (GB method), total
water_df_emissions["irr_app_gb_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_m3"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_gb_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (cost method), total
water_df_emissions["irr_app_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_adj_m3"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_adj_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# Emissions based on adjusted hybrid water consumption estimates (GB method), total
water_df_emissions["irr_app_gb_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_adj_m3"] * EF_IRR_APP,
    np.nan,
)

water_df_emissions["irr_equip_gb_adj_tco2eq"] = np.where(
    water_df_emissions["irrigation_types"].str.contains("DRIP|SPRINKLER"),
    water_df_emissions["water_cons_hyb_gb_poisson_adj_m3"] * EF_IRR_EQUIP,
    np.nan,
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
water_df_final = water_df_emissions.copy()
water_df_final.to_feather(PATH_EXPORT_TABLES / r"table_05_water_features.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)
water_df_final.to_sql(
    name="table_05_water_features", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
