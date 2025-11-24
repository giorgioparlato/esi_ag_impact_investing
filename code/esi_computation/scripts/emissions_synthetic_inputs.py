"""
Preprocessing and emission estimation for synthetic agricultural inputs.

This module processes the CIMMYT dataset
'4.-Agricultural supplies_2012-2022_01.xlsx' and performs:

- standard cleaning steps
- imputation of missing values for product types, amounts, and n content
- computation of emissions associated with synthetic input production,
  transport and distribution, and application
- creation of data-quality flags and missingness diagnostics
- aggregation of emissions and related variables at the level of
  innovation or conventional practice IDs

Authors:
    Dario Marone
    Giorgio Parlato

Date:
    2025-11-24
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
# Paths and constants
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# CIMMYT dataset on agricultural supplies
PATH_CIMMYT_MEXICO_SYNTHETIC_INPUTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\BEM_e-Agrology_Mexico\4.-Agricultural supplies_2012-2022_01.xlsx"
)

# Sheet name
SHEET_CIMMYT_MEXICO_SYNTHETIC_INPUTS = "Agricultural supplies"

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\agronomic_features.feather"
)

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

KEY_IDS = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "id_aplicacionesdeinsumos",
]

KEY_VARS = [
    "year",
    "activity_performed",
    "kind_of_product",
    "product_category",
    "product_name_applied",
    "units_ha",
    "amount_of_product_applied",
    "nitrogen_n",
    "phosphorus_p",
    "potassium_k",
]

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load dataset
ag_supplies_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_SYNTHETIC_INPUTS,
    sheet_name=SHEET_CIMMYT_MEXICO_SYNTHETIC_INPUTS,
)

# Load plot areas
areas_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES, columns=["practice_id", "year", "area_ha"]
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
ag_supplies_df_clean_columns = ag_supplies_df_raw.copy()
ag_supplies_df_clean_columns = ag_supplies_df_clean_columns.clean_names(
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
ag_supplies_df_clean_columns.rename(
    columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True
)

# Create DataFrame storing data on synthetic inputs
ag_supplies_df_vars = ag_supplies_df_clean_columns[KEY_IDS + KEY_VARS].copy()

# Convert id columns to string type
ag_supplies_df_dtypes = ag_supplies_df_vars.copy()
ag_supplies_df_dtypes[KEY_IDS] = ag_supplies_df_dtypes[KEY_IDS].astype("string")

# Clean year column
# Create new DataFrame
ag_supplies_df_year = ag_supplies_df_dtypes.copy()

# Slice year strings to extract the first four digits, then cast the column to int
ag_supplies_df_year["year"] = ag_supplies_df_year.year.apply(
    lambda x: x[0:4] if isinstance(x, str) else x
).astype(int)

# Imputation of missing values - product types
ag_supplies_df_prod = ag_supplies_df_year.copy()

# Get the most frequent kind_of_product per product_name_applied
most_common_kinds = (
    ag_supplies_df_prod.dropna(subset=["kind_of_product"])
    .groupby("product_name_applied")["kind_of_product"]
    .agg(lambda x: x.mode().iat[0])
)

# Get the most frequent kind_of_product per product_category
most_common_categories = (
    ag_supplies_df_prod.dropna(subset=["kind_of_product"])
    .groupby("product_category")["kind_of_product"]
    .agg(lambda x: x.mode().iat[0])
)

# Create column for imputation
ag_supplies_df_prod["kind_of_product_imp"] = ag_supplies_df_prod["kind_of_product"]

# Fill by product_name_applied
ag_supplies_df_prod.loc[
    ag_supplies_df_prod["kind_of_product"].isna()
    & ag_supplies_df_prod["product_name_applied"].isin(most_common_kinds.index),
    "kind_of_product_imp",
] = ag_supplies_df_prod["product_name_applied"].map(most_common_kinds)

# Fill remaining by category
ag_supplies_df_prod.loc[
    ag_supplies_df_prod["kind_of_product_imp"].isna()
    & ag_supplies_df_prod["product_category"].isin(most_common_categories.index),
    "kind_of_product_imp",
] = ag_supplies_df_prod["product_category"].map(most_common_categories)

# Imputation of missing values - product amounts
# Impute missing amounts for all product categories and units
ag_supplies_df_quant = ag_supplies_df_prod.copy()

ag_supplies_df_quant["amount_of_product_applied_imputed"] = (
    ag_supplies_df_quant.groupby(["product_category", "units_ha"])[
        "amount_of_product_applied"
    ].transform(lambda s: s.fillna(s.mean()))
)
# Imputation of missing values - nitrogen quantities
# Original column
n = ag_supplies_df_quant["nitrogen_n"]

# Groups
g = ag_supplies_df_quant.groupby("product_name_applied")["nitrogen_n"]

# Group-level flags
has_zero = g.transform(lambda s: (s == 0.0).any()).astype(bool)
has_pos = g.transform(lambda s: (s > 0.0).any()).astype(bool)
all_nan = g.transform(lambda s: s.isna().all()).astype(bool)

# Values
mean_pos_per_name = g.transform(lambda s: s[s > 0.0].mean())
mean_per_category = ag_supplies_df_quant.groupby("product_category")[
    "nitrogen_n"
].transform("mean")

# Assign based on conditions
ag_supplies_df_quant["nitrogen_n_imputed"] = n.copy()
nan_mask = n.isna()

mask_only_zero_or_nan = nan_mask & has_zero & ~has_pos & ~all_nan
ag_supplies_df_quant.loc[mask_only_zero_or_nan, "nitrogen_n_imputed"] = 0.0

mask_has_positive = nan_mask & has_pos
ag_supplies_df_quant.loc[mask_has_positive, "nitrogen_n_imputed"] = mean_pos_per_name[
    mask_has_positive
]

mask_all_nan_group = nan_mask & all_nan
ag_supplies_df_quant.loc[mask_all_nan_group, "nitrogen_n_imputed"] = mean_per_category[
    mask_all_nan_group
]

# Estimation
# Input production
ag_supplies_df_estimation = ag_supplies_df_quant.copy()

# Non-organic / non-"other" synthetic inputs
mask_synth = ~ag_supplies_df_estimation["kind_of_product"].isin(
    ["BIOLOGICO/ETOLOGICO/ORGANICO", "OTRO"]
)

# Initialize column
ag_supplies_df_estimation["ef_synth_input_prod_kg_co2eq_unit"] = np.nan

# Lookup for pest control products
pest_keys = [
    ("HERBICIDE", "KG_HA"),
    ("HERBICIDE", "L_HA"),
    ("FUNGICIDE", "KG_HA"),
    ("FUNGICIDE", "L_HA"),
    ("INSECTICIDE", "KG_HA"),
    ("INSECTICIDE", "L_HA"),
]

pest_values = [
    14.09768,  # HERBICIDE & KG_HA
    6.223658,  # HERBICIDE & L_HA
    8.202714,  # FUNGICIDE & KG_HA
    7.46783,  # FUNGICIDE & L_HA
    2.697334,  # INSECTICIDE & KG_HA
    9.13278,  # INSECTICIDE & L_HA
]

pest_factors = pd.Series(
    pest_values,
    index=pd.MultiIndex.from_tuples(pest_keys, names=["product_category", "units_ha"]),
)

mask_pest = mask_synth & ag_supplies_df_estimation["product_category"].isin(
    ["HERBICIDE", "FUNGICIDE", "INSECTICIDE"]
)

# Map (category, units) → factor
idx = list(
    zip(
        ag_supplies_df_estimation.loc[mask_pest, "product_category"],
        ag_supplies_df_estimation.loc[mask_pest, "units_ha"],
    )
)

ag_supplies_df_estimation.loc[mask_pest, "ef_synth_input_prod_kg_co2eq_unit"] = [
    pest_factors.get(key, np.nan) for key in idx
]
# Specific fertilizer factors (KG_HA)
fert_specific = {
    "UREA": 1.124768,
    "SULFATO DE AMONIO": 0.64399,
    "FOSFATO DIAMONICO (DAP)": 1.26697,
    "CLORURO DE POTASIO": 0.6247946,
    "FOSFATO MONOAMONICO": 1.05447,
    "SUPERFOSFATO SIMPLE": 0.2173244,
    "SUPERFOSFATO TRIPLE": 0.630778,
}

mask_fert_kg = (
    mask_synth
    & ag_supplies_df_estimation["product_category"].eq("FERTILIZER")
    & ag_supplies_df_estimation["units_ha"].eq("KG_HA")
)

# Assign specific factors by product_name_applied
ag_supplies_df_estimation.loc[mask_fert_kg, "ef_synth_input_prod_kg_co2eq_unit"] = (
    ag_supplies_df_estimation.loc[mask_fert_kg, "product_name_applied"].map(
        fert_specific
    )
)

# For remaining FERTILIZER / KG_HA (i.e. not in dict), assign default 1.1519
mask_fert_default = (
    mask_fert_kg & ag_supplies_df_estimation["ef_synth_input_prod_kg_co2eq_unit"].isna()
)

ag_supplies_df_estimation.loc[
    mask_fert_default, "ef_synth_input_prod_kg_co2eq_unit"
] = 1.1519
# Compute emissions from synthetic input production
ag_supplies_df_estimation["synth_input_prod_t_co2eq_ha"] = (
    ag_supplies_df_estimation["amount_of_product_applied_imputed"]
    * ag_supplies_df_estimation["ef_synth_input_prod_kg_co2eq_unit"]
    / 1000
)
# Import transportation plus domestic transportation and distribution of synthetic inputs
# Initialize column
ag_supplies_df_estimation["ef_synth_input_tran_dist_kg_co2eq_kg"] = np.nan

# Fertilizers (specific products, KG_HA)

fert_tran_specific = {
    "UREA": 6.91e-02,
    "CLORURO DE POTASIO": 5.45e-02,
    "FOSFATO DIAMONICO (DAP)": 4.26e-02,
    "FOSFATO MONOAMONICO": 5.10e-02,
    "SULFATO DE AMONIO": 3.65e-02,
    "SUPERFOSFATO SIMPLE": 3.51e-02,
    "SUPERFOSFATO TRIPLE": 3.51e-02,
}

mask_fert_kg = (
    mask_synth
    & ag_supplies_df_estimation["product_category"].eq("FERTILIZER")
    & ag_supplies_df_estimation["units_ha"].eq("KG_HA")
)

# Assign specific fertilizer factors
ag_supplies_df_estimation.loc[mask_fert_kg, "ef_synth_input_tran_dist_kg_co2eq_kg"] = (
    ag_supplies_df_estimation.loc[mask_fert_kg, "product_name_applied"].map(
        fert_tran_specific
    )
)

# Default for other FERTILIZER / KG_HA (NPK compounds)
mask_fert_default = (
    mask_fert_kg
    & ag_supplies_df_estimation["ef_synth_input_tran_dist_kg_co2eq_kg"].isna()
)

ag_supplies_df_estimation.loc[
    mask_fert_default, "ef_synth_input_tran_dist_kg_co2eq_kg"
] = 7.43e-02  # NPK

# Pesticides (INSECTICIDE / HERBICIDE / FUNGICIDE, KG_HA)

pest_tran_factors = {
    "INSECTICIDE": 1.71e-01,
    "HERBICIDE": 6.74e-02,
    "FUNGICIDE": 4.34e-01,
}

mask_pest_kg = (
    mask_synth
    & ag_supplies_df_estimation["product_category"].isin(pest_tran_factors.keys())
    & ag_supplies_df_estimation["units_ha"].eq("KG_HA")
)

ag_supplies_df_estimation.loc[mask_pest_kg, "ef_synth_input_tran_dist_kg_co2eq_kg"] = (
    ag_supplies_df_estimation.loc[mask_pest_kg, "product_category"].map(
        pest_tran_factors
    )
)
# Compute emissions from the import and domestic transportation plus the distribution of synthetic inputs
ag_supplies_df_estimation["synth_input_tran_dist_t_co2eq_ha"] = (
    ag_supplies_df_estimation["amount_of_product_applied_imputed"]
    * ag_supplies_df_estimation["ef_synth_input_tran_dist_kg_co2eq_kg"]
    / 1000
)
# Input application
# Initialize column with NaN (default for everything)
ag_supplies_df_estimation["ef_synth_input_app_kg_co2eq_unit"] = np.nan

# UREA, FERTILIZER, KG_HA
mask_urea = (
    mask_synth
    & ag_supplies_df_estimation["product_category"].eq("FERTILIZER")
    & ag_supplies_df_estimation["product_name_applied"].eq("UREA")
    & ag_supplies_df_estimation["units_ha"].eq("KG_HA")
)

ag_supplies_df_estimation.loc[mask_urea, "ef_synth_input_app_kg_co2eq_unit"] = (
    0.7333333333
)

# Other synthetic FERTILIZER / INSECTICIDE / FUNGICIDE / HERBICIDE with N > 0, KG_HA, not UREA
mask_n_inputs = (
    mask_synth
    & ag_supplies_df_estimation["product_category"].isin(
        ["FERTILIZER", "INSECTICIDE", "FUNGICIDE", "HERBICIDE"]
    )
    & ag_supplies_df_estimation["product_name_applied"].ne("UREA")
    & ag_supplies_df_estimation["units_ha"].eq("KG_HA")
    & (ag_supplies_df_estimation["nitrogen_n"] > 0)
)

ag_supplies_df_estimation.loc[mask_n_inputs, "ef_synth_input_app_kg_co2eq_unit"] = (
    4.682857143 * ag_supplies_df_estimation.loc[mask_n_inputs, "nitrogen_n"] / 100
)
# Compute emissions from synthetic input application
ag_supplies_df_estimation["synth_input_app_t_co2eq_ha"] = (
    ag_supplies_df_estimation["amount_of_product_applied_imputed"]
    * ag_supplies_df_estimation["ef_synth_input_app_kg_co2eq_unit"]
    / 1000
)

# Flags
ag_supplies_df_flags = ag_supplies_df_estimation.copy()


def make_imputation_flag(original: pd.Series, imputed: pd.Series) -> pd.Series:
    flag = pd.Series(np.nan, index=original.index, dtype="float")

    # original present → 0
    flag.loc[original.notna()] = 0

    # original missing, imputed present → 1
    flag.loc[original.isna() & imputed.notna()] = 1

    # original missing, imputed missing → stay NaN
    return flag


ag_supplies_df_flags["flag_imputed_product"] = make_imputation_flag(
    ag_supplies_df_flags["kind_of_product"], ag_supplies_df_flags["kind_of_product_imp"]
)

ag_supplies_df_flags["flag_imputed_amount"] = make_imputation_flag(
    ag_supplies_df_flags["amount_of_product_applied"],
    ag_supplies_df_flags["amount_of_product_applied_imputed"],
)

ag_supplies_df_flags["flag_imputed_nitrogen"] = make_imputation_flag(
    ag_supplies_df_flags["nitrogen_n"], ag_supplies_df_flags["nitrogen_n_imputed"]
)
missing_ef_prod = ag_supplies_df_flags["ef_synth_input_prod_kg_co2eq_unit"].isna()
missing_ef_tran = ag_supplies_df_flags["ef_synth_input_tran_dist_kg_co2eq_kg"].isna()
missing_ef_app = ag_supplies_df_flags["ef_synth_input_app_kg_co2eq_unit"].isna()

ag_supplies_df_flags["flag_missing_ef_prod"] = missing_ef_prod
ag_supplies_df_flags["flag_missing_ef_tran"] = missing_ef_tran
ag_supplies_df_flags["flag_missing_ef_app"] = missing_ef_app
# Emission itself is missing
missing_est_prod = ag_supplies_df_flags["synth_input_prod_t_co2eq_ha"].isna()

# Amount is still missing even after imputation: flag_imputed_amount is NaN
missing_amount = ag_supplies_df_flags["flag_imputed_amount"].isna()

# EF missing is already encoded in your flag
missing_ef_prod = ag_supplies_df_flags["flag_missing_ef_prod"]  # boolean

ag_supplies_df_flags["reason_missing_prod"] = np.select(
    [
        ~missing_est_prod,  # estimate is present
        missing_amount & ~missing_ef_prod,  # only amount missing
        missing_ef_prod & ~missing_amount,  # only EF missing
        missing_amount & missing_ef_prod,  # both missing
    ],
    [
        pd.NA,
        "MISSING AMOUNT",
        "MISSING EF",
        "MISSING BOTH",
    ],
    default=pd.NA,
)

missing_est_tran = ag_supplies_df_flags["synth_input_tran_dist_t_co2eq_ha"].isna()
missing_ef_tran = ag_supplies_df_flags["flag_missing_ef_tran"]

ag_supplies_df_flags["reason_missing_tran"] = np.select(
    [
        ~missing_est_tran,
        missing_amount & ~missing_ef_tran,
        missing_ef_tran & ~missing_amount,
        missing_amount & missing_ef_tran,
    ],
    [
        pd.NA,
        "MISSING AMOUNT",
        "MISSING EF",
        "MISSING BOTH",
    ],
    default=pd.NA,
)

missing_est_app = ag_supplies_df_flags["synth_input_app_t_co2eq_ha"].isna()
missing_ef_app = ag_supplies_df_flags["flag_missing_ef_app"]

ag_supplies_df_flags["reason_missing_app"] = np.select(
    [
        ~missing_est_app,
        missing_amount & ~missing_ef_app,
        missing_ef_app & ~missing_amount,
        missing_amount & missing_ef_app,
    ],
    [
        pd.NA,
        "MISSING AMOUNT",
        "MISSING EF",
        "MISSING BOTH",
    ],
    default=pd.NA,
)

ag_supplies_df_flags.reason_missing_tran.value_counts(dropna=False)
# Aggregation
ag_supplies_df_agg = ag_supplies_df_flags.copy()

# Group and aggregate variables selectively
group_cols = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]


def join_unique(x):
    return " | ".join(sorted(x.dropna().unique()))


def join_reasons(s):
    if s.notna().any():
        return " | ".join(sorted(set(s.dropna())))
    else:
        return "NOT MISSING"


def sum_min_count(x):
    return x.sum(min_count=1)


def sum_if_notna_all(x):
    return x.sum(min_count=1) if x.notna().all() else np.nan


def imputation_status(x: pd.Series) -> str:
    # Count values
    n_reported = (x == 0).sum()
    n_imputed = (x == 1).sum()
    n_missing = x.isna().sum()
    n_total = len(x)

    # Case 1: everything missing
    if n_missing == n_total:
        return "MISSING ONLY"

    # Case 2: all reported
    if n_reported == n_total:
        return "REPORTED ONLY"

    # Case 3: all imputed
    if n_imputed == n_total:
        return "IMPUTED ONLY"

    # Case 4: mix of reported and imputed (no missing)
    if n_reported > 0 and n_imputed > 0 and n_missing == 0:
        return "REPORTED AND IMPUTED"

    # Case 5: mix including missing
    return "PARTIALLY MISSING"


def ef_availability_status(x: pd.Series) -> str:
    # x is a boolean Series: True = missing, False = available
    all_missing = x.all()
    all_available = ~x.any()  # same as (x == False).all()

    if all_missing:
        return "ALL MISSING"
    elif all_available:
        return "ALL AVAILABLE"
    else:
        return "PARTIALLY MISSING"


ag_supplies_df_agg = (
    ag_supplies_df_agg.groupby(group_cols, sort=False)
    .agg(
        activities=("activity_performed", join_unique),
        activity_count=("activity_performed", "nunique"),
        products=("kind_of_product_imp", join_unique),
        product_count=("kind_of_product_imp", "nunique"),
        synth_input_prod_t_co2eq_ha=("synth_input_prod_t_co2eq_ha", sum_if_notna_all),
        synth_input_tran_dist_t_co2eq_ha=(
            "synth_input_tran_dist_t_co2eq_ha",
            sum_if_notna_all,
        ),
        synth_input_app_t_co2eq_ha=("synth_input_app_t_co2eq_ha", sum_if_notna_all),
        flag_imp_prod=("flag_imputed_product", imputation_status),
        flag_imp_amount=("flag_imputed_amount", imputation_status),
        flag_imp_nitrogen=("flag_imputed_nitrogen", imputation_status),
        flag_missing_ef_prod=("flag_missing_ef_prod", ef_availability_status),
        flag_missing_ef_tran=("flag_missing_ef_tran", ef_availability_status),
        flag_missing_ef_app=("flag_missing_ef_app", ef_availability_status),
        reason_missing_est_prod=("reason_missing_prod", join_reasons),
        reason_missing_est_tran=("reason_missing_tran", join_reasons),
        reason_missing_est_app=("reason_missing_app", join_reasons),
    )
    .reset_index()
)
# Totals
# Merge with areas from the agronomic dataframe
ag_supplies_df_totals = ag_supplies_df_agg.copy()

ag_supplies_df_totals = ag_supplies_df_totals.merge(
    areas_df, on=["practice_id", "year"], how="left"
)
# Compute total emission estimates
ag_supplies_df_totals["synth_input_prod_t_co2eq"] = (
    ag_supplies_df_totals["synth_input_prod_t_co2eq_ha"]
    * ag_supplies_df_totals["area_ha"]
)

ag_supplies_df_totals["synth_input_tran_dist_t_co2eq"] = (
    ag_supplies_df_totals["synth_input_tran_dist_t_co2eq_ha"]
    * ag_supplies_df_totals["area_ha"]
)

ag_supplies_df_totals["synth_input_app_t_co2eq"] = (
    ag_supplies_df_totals["synth_input_app_t_co2eq_ha"]
    * ag_supplies_df_totals["area_ha"]
)

# Final table
# Final DataFrame
ag_supplies_df_final = ag_supplies_df_totals.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "activities",
    "activity_count",
    "products",
    "product_count",
    "synth_input_prod_t_co2eq_ha",
    "synth_input_prod_t_co2eq",
    "synth_input_tran_dist_t_co2eq_ha",
    "synth_input_tran_dist_t_co2eq",
    "synth_input_app_t_co2eq_ha",
    "synth_input_app_t_co2eqflag_imp_prod",
    "flag_imp_amount",
    "flag_imp_nitrogen",
    "flag_missing_ef_prod",
    "flag_missing_ef_tran",
    "flag_missing_ef_app",
    "reason_missing_est_prod",
    "reason_missing_est_tran",
    "reason_missing_est_app",
]

ag_supplies_df_final = ag_supplies_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
ag_supplies_df_final.to_feather(
    PATH_EXPORT_TABLES / r"emissions_synthetic_inputs.feather"
)

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

ag_supplies_df_final.to_sql(
    name="emissions_synthetic_inputs", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
