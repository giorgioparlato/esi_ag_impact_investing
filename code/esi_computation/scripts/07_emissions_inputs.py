"""
Estimation of emissions from synthetic and organic agricultural inputs.
Main operations:
    - standard cleaning steps
    - imputation of missing values for product types, amounts, n content and moisture levels
    - computation of emissions associated with synthetic input production,
    transport and distribution, and application
    - computation of emissions associated with organic input application
    - aggregation of emissions and related variables at the practice ID level
    - creation of data-quality flags and missingness diagnostics
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
# Home directory
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# CIMMYT dataset on agricultural supplies (version 6.0)
PATH_CIMMYT_MEXICO_INPUTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\cimmyt\dataverse_files\4.-Agricultural supplies_2012-2022_01.xlsx"
)

# Sheet names
SHEET_CIMMYT_MEXICO_SYNTHETIC_INPUTS = "Agricultural supplies"
SHEET_CIMMYT_MEXICO_ORGANIC_INPUTS = "Organic Fertilization"

# Path to agronomic features table
PATH_AGRONOMIC_FEATURES = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_03_agronomic_features.feather"
)

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KEY_IDS_SYNTHETIC = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "application_id",
]

KEY_IDS_ORGANIC = ["logbook_id", "practice_id", "application_id"]

KEY_VARS_SYNTHETIC = [
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

KEY_VARS_ORGANIC = [
    "kind_of_product",
    "product_name_applied",
    "units_ha",
    "amount_of_product_applied",
    "product_moisture_in_the_application",
]

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load synthetic inputs dataset
syn_inputs_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_INPUTS, sheet_name=SHEET_CIMMYT_MEXICO_SYNTHETIC_INPUTS
)

# Load organic inputs dataset
org_inputs_df_raw = pd.read_excel(
    io=PATH_CIMMYT_MEXICO_INPUTS, sheet_name=SHEET_CIMMYT_MEXICO_ORGANIC_INPUTS
)

# Load plot areas
areas_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES, columns=["practice_id", "year", "area_ha"]
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
syn_inputs_df_clean_columns = syn_inputs_df_raw.copy()
syn_inputs_df_clean_columns = syn_inputs_df_clean_columns.clean_names(
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

org_inputs_df_clean_columns = org_inputs_df_raw.copy()
org_inputs_df_clean_columns = org_inputs_df_clean_columns.clean_names(
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
syn_inputs_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
        "id_aplicacionesdeinsumos": "application_id",
    },
    inplace=True,
)

org_inputs_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
        "id_de_la_aplicacion_clave_primaria": "application_id",
        "product_applied_type": "product_name_applied",
        "amount_of_product_applied_t_ha": "amount_of_product_applied",
    },
    inplace=True,
)

# Add columns
org_inputs_df_clean_columns["kind_of_product"] = "BIOLOGICO/ETOLOGICO/ORGANICO"
org_inputs_df_clean_columns["units_ha"] = "T_HA"

# Create DataFrame storing data on synthetic inputs
syn_inputs_df_variables = syn_inputs_df_clean_columns[
    KEY_IDS_SYNTHETIC + KEY_VARS_SYNTHETIC
].copy()

# Create DataFrame storing data on organic inputs
org_inputs_df_variables = org_inputs_df_clean_columns[
    KEY_IDS_ORGANIC + KEY_VARS_ORGANIC
].copy()

# Convert id columns to string type
syn_inputs_df_dtypes = syn_inputs_df_variables.copy()
syn_inputs_df_dtypes[KEY_IDS_SYNTHETIC] = syn_inputs_df_dtypes[
    KEY_IDS_SYNTHETIC
].astype("string")

# Convert id columns to string type
org_inputs_df_dtypes = org_inputs_df_variables.copy()
org_inputs_df_dtypes[KEY_IDS_ORGANIC] = org_inputs_df_dtypes[KEY_IDS_ORGANIC].astype(
    "string"
)

# Merge
syn_inputs_df_merge = syn_inputs_df_dtypes.copy()
org_inputs_df_merge = org_inputs_df_dtypes.copy()

merge_keys = ["logbook_id", "practice_id", "application_id"]

inputs_df_merge = pd.merge(
    syn_inputs_df_merge, org_inputs_df_merge, how="outer", on=merge_keys
)

cols_to_fill = ["farmer_id", "plot_id", "year"]

for col in cols_to_fill:
    inputs_df_merge[col] = inputs_df_merge.groupby("logbook_id")[col].ffill()
    inputs_df_merge[col] = inputs_df_merge.groupby("logbook_id")[col].bfill()

harmonize_cols = [
    "amount_of_product_applied",
    "kind_of_product",
    "product_name_applied",
    "units_ha",
]

for col in harmonize_cols:
    inputs_df_merge[col] = inputs_df_merge[f"{col}_y"].combine_first(
        inputs_df_merge[f"{col}_x"]
    )
    inputs_df_merge.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "application_id",
    "year",
    "activity_performed",
    "kind_of_product",
    "product_category",
    "product_name_applied",
    "amount_of_product_applied",
    "units_ha",
    "product_moisture_in_the_application",
    "nitrogen_n",
    "phosphorus_p",
    "potassium_k",
]

inputs_df_merge = inputs_df_merge.reindex(columns=new_index)

# Clean year column
# Create new DataFrame
inputs_df_year = inputs_df_merge.copy()

# Slice year strings to extract the first four digits, then cast the column to int (allowing for missing values)
inputs_df_year["year"] = inputs_df_year["year"].apply(
    lambda x: str(x)[0:4] if pd.notna(x) and isinstance(x, str) else x
)

inputs_df_year["year"] = pd.to_numeric(inputs_df_year["year"], errors="coerce")

inputs_df_year["year"] = inputs_df_year["year"].astype("Int64")

# Imputation
inputs_df_imp_product = inputs_df_year.copy()

# Find the most frequent product type by product name
most_common_kinds = (
    inputs_df_imp_product.dropna(subset=["kind_of_product"])
    .groupby("product_name_applied")["kind_of_product"]
    .agg(lambda x: x.mode().iat[0])
)

# Find the most frequent product type by product category
most_common_categories = (
    inputs_df_imp_product.dropna(subset=["kind_of_product"])
    .groupby("product_category")["kind_of_product"]
    .agg(lambda x: x.mode().iat[0])
)

# Create column for imputation
inputs_df_imp_product["kind_of_product_imp"] = inputs_df_imp_product["kind_of_product"]

# Fill by product name
inputs_df_imp_product.loc[
    inputs_df_imp_product["kind_of_product"].isna()
    & inputs_df_imp_product["product_name_applied"].isin(most_common_kinds.index),
    "kind_of_product_imp",
] = inputs_df_imp_product["product_name_applied"].map(most_common_kinds)

# Fill remaining by category
inputs_df_imp_product.loc[
    inputs_df_imp_product["kind_of_product_imp"].isna()
    & inputs_df_imp_product["product_category"].isin(most_common_categories.index),
    "kind_of_product_imp",
] = inputs_df_imp_product["product_category"].map(most_common_categories)

# Impute missing amounts by product name or category and units
inputs_df_imp_amount = inputs_df_imp_product.copy()
imputation_column = "amount_of_product_applied"
target_column = "amount_of_product_applied_imputed"

inputs_df_imp_amount[target_column] = inputs_df_imp_amount.groupby(
    ["product_name_applied", "units_ha"]
)[imputation_column].transform(lambda s: s.fillna(s.mean()))

mask_still_missing = inputs_df_imp_amount[target_column].isna()
category_means = inputs_df_imp_amount.groupby("product_category")[
    imputation_column
].mean()
inputs_df_imp_amount.loc[mask_still_missing, target_column] = inputs_df_imp_amount.loc[
    mask_still_missing, "product_category"
].map(category_means)

# Convert pesticide units from l_ha to kg_ha so we can more easily deal with them
# Density values taken from product-specific pages, sources included in appendix table

# Density mapping (kg/L) by product name
DENSITY_MAP = {
    # INSECTICIDES
    "PALGUS":           1.0176,
    "DENIN 19 CE":      0.9,
    "LORSBAN 480":      1.149,
    "ARRIVO 200 CE":    0.9413,
    "CIPERMETRINA":     0.9413,
    "KARATE ZEON 5CS":  1.057,
    "ENGEO":            1.119,
    "CORAGEN 20 SC":    1.09,
    "AMPLIGO":          1.08,

    # FUNGICIDES
    "FOLICUR 250 EW":               0.97,
    "HEADLINE":                     1.05,
    "CONSIST MAX":                  1.16,
    "PRIORI XTRA":                  1.1,
    "PRIORI EXTRA":                 1.1,
    "QUILT":                        1.07,
    "OPUS":                         1.05,
    "TEBUCUR":                      0.97,

    # HERBICIDES
    "PARAQUAT 200":     1.08,
    "GRAMOXONE":        1.073,
    "2,4-D AMINA 720":  1.2,
    "HIERBAMINA":       1.17,
    "GLIFOSATO 360 SA": 1.165,
    "FAENA":            1.25,
    "FINALE":           1.11,
    "GRAMOCIL":         1.088,
    "LAUDIS":           1.22,
    "ESTERON 47M":      1.025,
    "PRIMAGRAM GOLD":   1.1,
}

# Generic (category-level) fallback densities (average for the category of all values gathered from product-specific sources)
GENERIC_DENSITY = {
    "INSECTICIDE": 1.033,
    "FUNGICIDE":   1.053,
    "HERBICIDE":   1.151,
}

RELEVANT_CATEGORIES = set(GENERIC_DENSITY.keys())


def get_density(product_name, category: str) -> float:
    """Return density for a given product name, falling back to category generic."""
    if pd.isna(product_name):
        return GENERIC_DENSITY.get(category)
    return DENSITY_MAP.get(str(product_name).strip(), GENERIC_DENSITY.get(category))


def convert_L_to_KG(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where product_category is INSECTICIDE, FUNGICIDE, or HERBICIDE
    and units_ha == 'L_HA', multiply amount_of_product_applied and
    amount_of_product_applied_imputed by the product-specific density (kg/L),
    falling back to the category generic average for unknown product names.
    """
    df = df.copy()

    mask = (
        df["product_category"].isin(RELEVANT_CATEGORIES) &
        (df["units_ha"] == "L_HA")
    )

    density = df.loc[mask].apply(
        lambda row: get_density(row["product_name_applied"], row["product_category"]),
        axis=1
    )

    df.loc[mask, "amount_of_product_applied"]         *= density.values
    df.loc[mask, "amount_of_product_applied_imputed"] *= density.values
    df.loc[mask, "units_ha"]                           = "KG_HA"

    return df


# Apply
inputs_df_imp_new_units = convert_L_to_KG(inputs_df_imp_amount)

# Imputing nitrogen content
inputs_df_imp_nitrogen = inputs_df_imp_new_units.copy()

# Original column
n = inputs_df_imp_nitrogen["nitrogen_n"]

# Groups
g = inputs_df_imp_nitrogen.groupby("product_name_applied")["nitrogen_n"]

# Group-level flags
has_zero = g.transform(lambda s: (s == 0.0).any()).astype(bool)
has_pos = g.transform(lambda s: (s > 0.0).any()).astype(bool)
all_nan = g.transform(lambda s: s.isna().all()).astype(bool)

# Values
mean_pos_per_name = g.transform(lambda s: s[s > 0.0].mean())
mean_per_category = inputs_df_imp_nitrogen.groupby("product_category")[
    "nitrogen_n"
].transform("mean")

# Assign based on conditions
inputs_df_imp_nitrogen["nitrogen_n_imputed"] = n.copy()
nan_mask = n.isna()

mask_only_zero_or_nan = nan_mask & has_zero & ~has_pos & ~all_nan
inputs_df_imp_nitrogen.loc[mask_only_zero_or_nan, "nitrogen_n_imputed"] = 0.0

mask_has_positive = nan_mask & has_pos
inputs_df_imp_nitrogen.loc[mask_has_positive, "nitrogen_n_imputed"] = mean_pos_per_name[
    mask_has_positive
]

mask_all_nan_group = nan_mask & all_nan
inputs_df_imp_nitrogen.loc[mask_all_nan_group, "nitrogen_n_imputed"] = (
    mean_per_category[mask_all_nan_group]
)

# Impute missing values for moisture content to enable convertion to dry amounts downstream
# We still have 250 missing values after
inputs_df_imp_moisture = inputs_df_imp_nitrogen.copy()

# Filter for organic products only
cond_organic = (
    inputs_df_imp_moisture["kind_of_product"] == "BIOLOGICO/ETOLOGICO/ORGANICO"
)

# Count frequency of moisture values by product name
moisture_counts_by_product = (
    inputs_df_imp_moisture[cond_organic]
    .groupby("product_name_applied")["product_moisture_in_the_application"]
    .value_counts(dropna=True)
)

# Unstack result
moisture_counts_by_product_df = moisture_counts_by_product.unstack(fill_value=0)

# Find most frequent value with a 50% threshold
inputs_df_imp_moisture["product_moisture_imputed"] = inputs_df_imp_moisture[
    "product_moisture_in_the_application"
]

for product in moisture_counts_by_product_df.index:
    counts = moisture_counts_by_product_df.loc[product]

    if counts.sum() == 0:
        continue

    most_frequent_moisture = counts.idxmax()
    percentage = counts.max() / counts.sum()

    if percentage > 0.50:
        inputs_df_imp_moisture.loc[
            (cond_organic)
            & (inputs_df_imp_moisture["product_name_applied"] == product)
            & (inputs_df_imp_moisture["product_moisture_imputed"].isna()),
            "product_moisture_imputed",
        ] = most_frequent_moisture

inputs_df_imp_estimation_syn = inputs_df_imp_moisture.copy()

# Non-organic / non-"other" synthetic inputs
mask_synth = ~inputs_df_imp_estimation_syn["kind_of_product"].isin(
    ["BIOLOGICO/ETOLOGICO/ORGANICO", "OTRO"]
)

# Initialize column
inputs_df_imp_estimation_syn["ef_synth_input_prod_kg_co2eq_unit"] = np.nan

# Lookup for pest control products
pest_keys = [
    ("HERBICIDE", "KG_HA"),
    ("HERBICIDE", "L_HA"),
    ("FUNGICIDE", "KG_HA"),
    ("FUNGICIDE", "L_HA"),
    ("INSECTICIDE", "KG_HA"),
    ("INSECTICIDE", "L_HA"),
]

# Source: compiled from table A9 in supplementary material of reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
pest_values = [
    14.130,  # HERBICIDE & KG_HA
    6.238,  # HERBICIDE & L_HA
    8.222,  # FUNGICIDE & KG_HA
    7.485,  # FUNGICIDE & L_HA
    2.704,  # INSECTICIDE & KG_HA
    9.154,  # INSECTICIDE & L_HA
]

pest_factors = pd.Series(
    pest_values,
    index=pd.MultiIndex.from_tuples(pest_keys, names=["product_category", "units_ha"]),
)

mask_pest = mask_synth & inputs_df_imp_estimation_syn["product_category"].isin(
    ["HERBICIDE", "FUNGICIDE", "INSECTICIDE"]
)

# Map (category, units) → factor
idx = list(
    zip(
        inputs_df_imp_estimation_syn.loc[mask_pest, "product_category"],
        inputs_df_imp_estimation_syn.loc[mask_pest, "units_ha"],
    )
)

inputs_df_imp_estimation_syn.loc[mask_pest, "ef_synth_input_prod_kg_co2eq_unit"] = [
    pest_factors.get(key, np.nan) for key in idx
]

# Specific fertilizer factors (KG_HA)
# Source: compiled from table A9 in supplementary material of reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
fert_specific = {
    "UREA": 1.134,
    "SULFATO DE AMONIO": 0.647,
    "FOSFATO DIAMONICO (DAP)": 1.272,
    "CLORURO DE POTASIO": 0.626,
    "FOSFATO MONOAMONICO": 1.058,
    "SUPERFOSFATO SIMPLE": 0.218,
    "SUPERFOSFATO TRIPLE": 0.633,
}

mask_fert_kg = (
    mask_synth
    & inputs_df_imp_estimation_syn["product_category"].isin(["FERTILIZER", "FOLIAR FERTILIZER", "BIOFERTILIZER"]) ## later on we include foliar and biofertilizers in the other synthetic inputs (if product category is chemichal)
    & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")
)

# Assign specific factors by product_name_applied
inputs_df_imp_estimation_syn.loc[mask_fert_kg, "ef_synth_input_prod_kg_co2eq_unit"] = (
    inputs_df_imp_estimation_syn.loc[mask_fert_kg, "product_name_applied"].map(
        fert_specific
    )
)

# For remaining FERTILIZER / KG_HA (i.e. not in dict), assign default 1.1519
mask_fert_default = (
    mask_fert_kg
    & inputs_df_imp_estimation_syn["ef_synth_input_prod_kg_co2eq_unit"].isna()
)

# Source: compiled from table A9 in supplementary material of reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
inputs_df_imp_estimation_syn.loc[
    mask_fert_default, "ef_synth_input_prod_kg_co2eq_unit"
] = 1.108


# Bayfolan specific nitrogen
mask_bayfolan_l = inputs_df_imp_estimation_syn["product_name_applied"].eq(
"BAYFOLAN FORTE"
) & inputs_df_imp_estimation_syn["units_ha"].eq("L_HA")

inputs_df_imp_estimation_syn.loc[mask_bayfolan_l, "units_ha"] = "KG_HA"

# Assign conversion factor for Bayfolan Forte based on density to convert from L_HA to KG_HA
# Source: https://www.micultivo.bayer.com.mx/content/dam/regional-folders/latam/norla/mexico/spanish/image/product-msds/BAYFOLAN-FORTE-Hoja-de-seguridad-2024.pdf
inputs_df_imp_estimation_syn.loc[
    mask_bayfolan_l, ["amount_of_product_applied", "amount_of_product_applied_imputed"]
] = (
    inputs_df_imp_estimation_syn.loc[
        mask_bayfolan_l, ["amount_of_product_applied", "amount_of_product_applied_imputed"]
    ] * 1.22
)

# Assign nitrogen content percentage for all Bayfolan Forte in KG_HA
# Source: https://www.micultivo.bayer.com.mx/content/dam/regional-folders/latam/norla/mexico/spanish/image/producto-ficha-tecnica/Bayfolan-Forte-Ficha-tecnica.pdf

mask_bayfolan = inputs_df_imp_estimation_syn["product_name_applied"].eq(
    "BAYFOLAN FORTE"
    ) & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")

inputs_df_imp_estimation_syn.loc[mask_bayfolan, "nitrogen_n"] = 11.47

# Compute emissions from synthetic input production
inputs_df_imp_estimation_syn["synth_input_prod_t_co2eq_ha"] = (
    inputs_df_imp_estimation_syn["amount_of_product_applied_imputed"]
    * inputs_df_imp_estimation_syn["ef_synth_input_prod_kg_co2eq_unit"]
    / 1000
)

# Initialize column
inputs_df_imp_estimation_syn["ef_synth_input_tran_dist_kg_co2eq_kg"] = np.nan

# Fertilizers (specific products, KG_HA)
# Source: compiled from table A10 in supplementary material of reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
fert_tran_dist_specific = {
    "UREA": 0.069,
    "CLORURO DE POTASIO": 0.055,
    "FOSFATO DIAMONICO (DAP)": 0.043,
    "FOSFATO MONOAMONICO": 0.051,
    "SULFATO DE AMONIO": 0.036,
    "SUPERFOSFATO SIMPLE": 0.035,
    "SUPERFOSFATO TRIPLE": 0.035,
}

mask_fert_kg = (
    mask_synth
    & inputs_df_imp_estimation_syn["product_category"].isin(
        ["FERTILIZER", "FOLIAR FERTILIZER", "BIOFERTILIZER"] ## later on we include biofertilizers in the other synthetic inputs (if product category is chemichal)
    )
    & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")
)

# Assign specific fertilizer factors
inputs_df_imp_estimation_syn.loc[
    mask_fert_kg, "ef_synth_input_tran_dist_kg_co2eq_kg"
] = inputs_df_imp_estimation_syn.loc[mask_fert_kg, "product_name_applied"].map(
    fert_tran_dist_specific
)

# Default for other FERTILIZER / KG_HA (NPK compounds)
mask_fert_default = (
    mask_fert_kg
    & inputs_df_imp_estimation_syn["ef_synth_input_tran_dist_kg_co2eq_kg"].isna()
)

# Source: compiled from table A10 in supplementary material of reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
inputs_df_imp_estimation_syn.loc[
    mask_fert_default, "ef_synth_input_tran_dist_kg_co2eq_kg"
] = 0.074  # NPK

# Pesticides (INSECTICIDE / HERBICIDE / FUNGICIDE, KG_HA)
# Reference:
pest_tran_factors = {
    "INSECTICIDE": 0.171,
    "HERBICIDE": 0.067,
    "FUNGICIDE": 0.434,
}

mask_pest_kg = (
    mask_synth
    & inputs_df_imp_estimation_syn["product_category"].isin(pest_tran_factors.keys())
    & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")
)

inputs_df_imp_estimation_syn.loc[
    mask_pest_kg, "ef_synth_input_tran_dist_kg_co2eq_kg"
] = inputs_df_imp_estimation_syn.loc[mask_pest_kg, "product_category"].map(
    pest_tran_factors
)

# Compute emissions from the import and domestic transportation plus the distribution of synthetic inputs
inputs_df_imp_estimation_syn["synth_input_tran_dist_t_co2eq_ha"] = (
    inputs_df_imp_estimation_syn["amount_of_product_applied_imputed"]
    * inputs_df_imp_estimation_syn["ef_synth_input_tran_dist_kg_co2eq_kg"]
    / 1000
)

# Initialize column with NaN (default for everything)
inputs_df_imp_estimation_syn["ef_synth_input_app_kg_co2eq_unit"] = np.nan

# UREA, FERTILIZER, KG_HA
mask_urea = (
    mask_synth
    # & inputs_df_imp_estimation_syn["product_category"].eq("FERTILIZER")
    & inputs_df_imp_estimation_syn["product_name_applied"].eq("UREA")
    & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")
)

# Source: compiled from equation 20 in reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
inputs_df_imp_estimation_syn.loc[mask_urea, "ef_synth_input_app_kg_co2eq_unit"] = 0.73

# Other synthetic FERTILIZER / INSECTICIDE / FUNGICIDE / HERBICIDE with N > 0, KG_HA, not UREA
mask_n_inputs = (
    mask_synth
    & inputs_df_imp_estimation_syn["product_category"].isin(
        ["FERTILIZER", "INSECTICIDE", "FUNGICIDE", "HERBICIDE", "FOLIAR FERTILIZER", "BIOFERTILIZER"]
    )
    & inputs_df_imp_estimation_syn["product_name_applied"].ne("UREA")
    & inputs_df_imp_estimation_syn["units_ha"].eq("KG_HA")
    & (inputs_df_imp_estimation_syn["nitrogen_n"] >= 0)
)

# Source: compiled from equation 20 in reference
# Reference: Juárez-Hernández, S., Usón, S., & Pardo, C. S. (2019). Assessing maize production systems in Mexico from an energy, exergy, and greenhouse-gas emissions perspective. Energy, 170, 199-211.
inputs_df_imp_estimation_syn.loc[mask_n_inputs, "ef_synth_input_app_kg_co2eq_unit"] = (
    4.29 * inputs_df_imp_estimation_syn.loc[mask_n_inputs, "nitrogen_n"] / 100
)

# Compute emissions from synthetic input application
inputs_df_imp_estimation_syn["synth_input_app_t_co2eq_ha"] = (
    inputs_df_imp_estimation_syn["amount_of_product_applied_imputed"]
    * inputs_df_imp_estimation_syn["ef_synth_input_app_kg_co2eq_unit"]
    / 1000
)

# Conversion to dry equivalents
inputs_df_imp_estimation_org = inputs_df_imp_estimation_syn.copy()

# Filter the data to get rows where product_moisture_in_the_application is "HUMEDO" and count occurrences
humid_counts = inputs_df_imp_estimation_org.loc[
    inputs_df_imp_estimation_org["product_moisture_imputed"] == "HUMEDO",
    "product_name_applied",
].value_counts(dropna=False)

# Manual selection of the most frequent humid products
# "OTRO" is excluded since we don't know what it refers to
# "BIOL", "TE DE LOMBRICOMPOSTA", and "LIXIVIADO DE LOMBRIZ" are excluded because they can hardly be assigned a nitrogen content level
product_selection = [
    "ESTIERCOL DE BOVINO",
    "LOMBRICOMPOSTA",
    "COMPOSTA",
    "BOCASHI (ABONO ORGANICO FERMENTADO)",
    "ESTIERCOL DE OVINO",
    "GALLINAZA",
]

# Set moisture parameters
# Source 1: https://landresources.montana.edu/soilfertility/documents/PDF/pres/ManureCompostComFertilizer_GalCoCropSch2006.pdf
# Source 2: https://compost.css.cornell.edu/monitor/monitormoisture.html
moisture_parameters = {
    "ESTIERCOL DE BOVINO": 0.8,
    "LOMBRICOMPOSTA": 0.50,
    "COMPOSTA": 0.50,
    "BOCASHI (ABONO ORGANICO FERMENTADO)": 0.50,
    "ESTIERCOL DE OVINO": 0.7,
    "GALLINAZA": 0.75,
}

# Conversion factors to metric tons
# If a unit is not mapped (e.g., L_HA, UNITS_HA, etc.), it will be left as is or set to NaN.
conversion_factors = {
    "T_HA": 1.0,
    "KG_HA": 0.001,
    "L_HA": 0.001,  # Assuming density ~1
}

# Convert to dry equivalents for the selected products
inputs_df_imp_estimation_org["moisture_parameter"] = inputs_df_imp_estimation_org[
    "product_name_applied"
].map(moisture_parameters)

# Harmonize amount to metric tons per hectare
inputs_df_imp_estimation_org["unit_conversion_factor"] = inputs_df_imp_estimation_org[
    "units_ha"
].map(conversion_factors)

# Apply conversion: amount * factor. If factor is NaN (for non-mapped units), the result will be NaN.
inputs_df_imp_estimation_org["amount_harmonized_t_ha"] = (
    inputs_df_imp_estimation_org["amount_of_product_applied_imputed"]
    * inputs_df_imp_estimation_org["unit_conversion_factor"]
)

cond_organic = (
    inputs_df_imp_estimation_org["kind_of_product"] == "BIOLOGICO/ETOLOGICO/ORGANICO"
)

cond_humid = (
    cond_organic
    & (inputs_df_imp_estimation_org["product_moisture_imputed"] == "HUMEDO")
    & (inputs_df_imp_estimation_org["product_name_applied"].isin(product_selection))
    & (inputs_df_imp_estimation_org["moisture_parameter"].notna())
)

cond_dry = cond_organic & (
    inputs_df_imp_estimation_org["product_moisture_imputed"] == "SECO"
)

result_humid = inputs_df_imp_estimation_org["amount_harmonized_t_ha"] * (
    1 - inputs_df_imp_estimation_org["moisture_parameter"]
)
result_dry = inputs_df_imp_estimation_org["amount_harmonized_t_ha"]

inputs_df_imp_estimation_org["dry_organic_amount_t_ha"] = np.select(
    condlist=[cond_humid, cond_dry],
    choicelist=[result_humid, result_dry],
    default=np.nan,
)

# Map manure types to macro-categories (AI-generated)
manure_mapping = {
    "ESTIERCOL DE BOVINO": "Cattle Manure",
    "ESTIERCOL DE GANADO MULAR": "Horses/Mules/Asses Manure",
    "ESTIERCOL DE MULAR": "Horses/Mules/Asses Manure",
    "ESTIERCOL DE EQUINO": "Horses/Mules/Asses Manure",
    "ESTIERCOL DE CABALLO": "Horses/Mules/Asses Manure",
    "ESTIERCOL DE PORCINO": "Swine Manure",
    "ESTIERCOL DE OVINO": "Sheep Manure",
    "ESTIERCOL DE CAPRINO": "Sheep Manure",
    # "ESTIERCOL DE MURCIÉLAGO": "General Compost",
    # "ESTIERCOL DE HORMIGA ARRIERA": "General Compost",
    # "EXCRETA DE HORMIGA ARRIERA": "General Compost",
    "HUMUS DE LOMBRIZ LÍQUIDO": "General Compost",
    "LIXIVIADO DE LOMBRIZ": "General Compost",
    "LIXIVIADO DE HUMUS DE LOMBRIZ": "General Compost",
    "HUMUS SÓLIDO DE LOMBRIZ": "General Compost",
    "LIXIVIADO DE LOMBRIZ ROJA CALIFORNIANA": "General Compost",
    "COMPOSTA": "General Compost",
    "LOMBRICOMPOSTA": "General Compost",
    "COMPOSTA CASERA": "General Compost",
    "COMPOSTA BOVINA": "General Compost",
    "COMPOSTA OVINA": "General Compost",
    "COMPOSTA DE GALLINAZA": "General Compost",
    "COMPOSTA GALLINAZA": "General Compost",
    "GALLINAZA": "Poultry Manure",
    "COMPOSTA DE BORREGO": "General Compost",
    "COMPOSTA MINERALIZADA": "General Compost",
    "COMPOSTA A BASE DE ESTIERCOL DE CERDO": "General Compost",
    "COMPOSTA DE VARIOS ESTIERCOLES": "General Compost",
    "COMPOSTA CON ABONO DE BOVINO": "General Compost",
    "BOCASHI (ABONO ORGÁNICO FERMENTADO)": "General Compost",
    "SUPERMAGRO": "General Compost",
    # "BIOFOL": "General Compost",
    # "BIOL": "General Compost",
    "ABONO FOLIAR FERMENTADO": "General Compost",
}

# Assign default values for nitrogen content (on a dry basis) based on category
# Source for livestock manure: table 5.5c in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch05_Cropland.pdf
# Source for general compost: https://umaine.edu/soiltestinglab/wp-content/uploads/sites/227/2016/07/Compost-Report-Interpretation-Guide.pdf
nitrogen_content = {
    "General Compost": 1.5,
    "Cattle Manure": 2.9,
    "Sheep Manure": 3.3,
    "Poultry Manure": 5.1,
    "Swine Manure": 4.1,
    "Horses/Mules/Asses Manure": 1.3,
}

cond_organic = (
    inputs_df_imp_estimation_org["kind_of_product"] == "BIOLOGICO/ETOLOGICO/ORGANICO"
)

# Initialize new columns to NaN to ensure non-biologico rows are clearly marked
inputs_df_imp_estimation_org["category"] = pd.NA
inputs_df_imp_estimation_org["nitrogen_content_pct"] = np.nan
inputs_df_imp_estimation_org["nitrogen_org_t_ha"] = np.nan

# Map products to macro-categories for organic fertilizers
inputs_df_imp_estimation_org.loc[cond_organic, "category"] = (
    inputs_df_imp_estimation_org.loc[cond_organic, "product_name_applied"].map(
        manure_mapping
    )
)

# Assign nitrogen content based on macro-category
inputs_df_imp_estimation_org.loc[cond_organic, "nitrogen_content_pct"] = (
    inputs_df_imp_estimation_org.loc[cond_organic, "category"].map(nitrogen_content)
)

# Estimate nitrogen content
inputs_df_imp_estimation_org.loc[cond_organic, "nitrogen_org_t_ha"] = (
    inputs_df_imp_estimation_org.loc[cond_organic, "dry_organic_amount_t_ha"]
    * inputs_df_imp_estimation_org.loc[cond_organic, "nitrogen_content_pct"].div(100)
)

# Parameters
# Emission factor for N2O emissions from N inputs
# Source: table 11.1 in https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/4_Volume4/V4_11_Ch11_N2O&CO2.pdf
EF_N2ON_KG_N_KG = 0.01

# Conversion factor from N2O-N emissions to N2O for reporting purposes
# Source: page 10 in https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/4_Volume4/V4_11_Ch11_N2O&CO2.pdf#page=10.19
CF_N2ON_N2O = 44 / 28

# Global warming potential of nitrous oxide
# Source: https://ghgprotocol.org/sites/default/files/2024-08/Global-Warming-Potential-Values%20%28August%202024%29.pdf
GWP_AR6_N2O = 273

inputs_df_imp_estimation_org["org_fert_app_t_co2eq_ha"] = (
    inputs_df_imp_estimation_org["nitrogen_org_t_ha"]
    * 1000
    * EF_N2ON_KG_N_KG
    * CF_N2ON_N2O
    * GWP_AR6_N2O
    / 1000
)


# Helper functions for flags
def imputation_flag(original: pd.Series, imputed: pd.Series) -> pd.Series:
    flag = pd.Series(np.nan, index=original.index, dtype="float")
    flag.loc[original.notna()] = 0
    flag.loc[original.isna() & imputed.notna()] = 1
    return flag


inputs_df_imp_flags = inputs_df_imp_estimation_org.copy()

# Flags for imputations (1, 0, NaN)
inputs_df_imp_flags["flag_imputed_product_type"] = imputation_flag(
    inputs_df_imp_flags["kind_of_product"], inputs_df_imp_flags["kind_of_product_imp"]
)

inputs_df_imp_flags["flag_imputed_amount"] = imputation_flag(
    inputs_df_imp_flags["amount_of_product_applied"],
    inputs_df_imp_flags["amount_of_product_applied_imputed"],
)

inputs_df_imp_flags["flag_imputed_nitrogen"] = imputation_flag(
    inputs_df_imp_flags["nitrogen_n"], inputs_df_imp_flags["nitrogen_n_imputed"]
)

inputs_df_imp_flags["flag_imputed_moisture_level"] = imputation_flag(
    inputs_df_imp_flags["product_moisture_in_the_application"],
    inputs_df_imp_flags["product_moisture_imputed"],
)

# Flags for missing emission factors (True, False)
missing_ef_prod = inputs_df_imp_flags["ef_synth_input_prod_kg_co2eq_unit"].isna()
missing_ef_tran = inputs_df_imp_flags["ef_synth_input_tran_dist_kg_co2eq_kg"].isna()

# Application EF applies primarily to fertilizers, not pesticides
missing_ef_app = (
    inputs_df_imp_flags["ef_synth_input_app_kg_co2eq_unit"].isna()
    & ~inputs_df_imp_flags["product_category"].isin(
        ["HERBICIDE", "FUNGICIDE", "INSECTICIDE"]
    )
)

inputs_df_imp_flags["flag_missing_ef_prod"] = missing_ef_prod
inputs_df_imp_flags["flag_missing_ef_tran"] = missing_ef_tran
inputs_df_imp_flags["flag_missing_ef_app"] = missing_ef_app

# Flags for missing moisture and nitrogen parameters (True, False)
missing_param_moisture = inputs_df_imp_flags["moisture_parameter"].isna()
inputs_df_imp_flags["flag_missing_moisture_parameter"] = missing_param_moisture

missing_param_nitrogen = inputs_df_imp_flags["nitrogen_content_pct"].isna()
inputs_df_imp_flags["flag_missing_nitrogen_parameter"] = missing_param_nitrogen

# Flag for failed unit conversion (True, False)
failed_unit_conversion = inputs_df_imp_flags["amount_harmonized_t_ha"].isna()
inputs_df_imp_flags["flag_failed_unit_conversion"] = failed_unit_conversion

# Flag for reason missing synthetic emissions
cond_not_chemical = inputs_df_imp_flags["kind_of_product_imp"] != "QUIMICO"
missing_est_prod = inputs_df_imp_flags["synth_input_prod_t_co2eq_ha"].isna()
missing_amount = inputs_df_imp_flags["flag_imputed_amount"].isna()
missing_ef_prod = inputs_df_imp_flags["flag_missing_ef_prod"]
fc = inputs_df_imp_flags["flag_failed_unit_conversion"]

nc = cond_not_chemical
me = missing_est_prod
ma = missing_amount
ef = missing_ef_prod

inputs_df_imp_flags["flag_missing_emissions_syn_prod"] = np.select(
    [
        nc,
        ~me,
        me & fc,
        me & ma & ef,
        me & ma & ~ef,
        me & ef & ~ma,
        me & ~ma & ~ef,
    ],
    [
        "NOT APPLICABLE",
        "NOT MISSING",
        "UNIT CONVERSION",
        "MISSING AMOUNT + EF",
        "MISSING AMOUNT",
        "MISSING EMISSION FACTOR",
        "NO INPUTS MISSING (ESTIMATE ERROR)",
    ],
    default="OTHER/UNFLAGGED MISSING",
)

missing_est_tran = inputs_df_imp_flags["synth_input_tran_dist_t_co2eq_ha"].isna()
missing_amount = inputs_df_imp_flags["flag_imputed_amount"].isna()
missing_ef_tran = inputs_df_imp_flags["flag_missing_ef_tran"]

me = missing_est_tran
ma = missing_amount
ef = missing_ef_tran

inputs_df_imp_flags["flag_missing_emissions_syn_tran"] = np.select(
    [
        nc,
        ~me,
        me & fc,
        me & ma & ef,
        me & ma & ~ef,
        me & ef & ~ma,
        me & ~ma & ~ef,
    ],
    [
        "NOT APPLICABLE",
        "NOT MISSING",
        "UNIT CONVERSION",
        "MISSING AMOUNT + EF",
        "MISSING AMOUNT",
        "MISSING EMISSION FACTOR",
        "NO INPUTS MISSING (ESTIMATE ERROR)",
    ],
    default="OTHER/UNFLAGGED MISSING",
)

missing_est_app = inputs_df_imp_flags["synth_input_app_t_co2eq_ha"].isna()
missing_amount = inputs_df_imp_flags["flag_imputed_amount"].isna()
missing_ef_app = inputs_df_imp_flags["flag_missing_ef_app"]

me = missing_est_app
ma = missing_amount
ef = missing_ef_app

inputs_df_imp_flags["flag_missing_emissions_syn_app"] = np.select(
    [
        nc,
        ~me,
        me & fc,
        me & ma & ef,
        me & ma & ~ef,
        me & ef & ~ma,
        me & ~ma & ~ef,
    ],
    [
        "NOT APPLICABLE",
        "NOT MISSING",
        "UNIT CONVERSION",
        "MISSING AMOUNT + EF",
        "MISSING AMOUNT",
        "MISSING EMISSION FACTOR",
        "NO INPUTS MISSING (ESTIMATE ERROR)",
    ],
    default="OTHER/UNFLAGGED MISSING",
)

# Flag for reason of missing organic emissions
cond_not_organic = (
    inputs_df_imp_flags["kind_of_product_imp"] != "BIOLOGICO/ETOLOGICO/ORGANICO"
)

# Define reasons
missing_estimate = inputs_df_imp_flags["org_fert_app_t_co2eq_ha"].isna()
missing_quantity = inputs_df_imp_flags["flag_imputed_amount"].isna()
failed_conversion = inputs_df_imp_flags["flag_failed_unit_conversion"]
missing_moisture_param = inputs_df_imp_flags["flag_missing_moisture_parameter"]
missing_nitrogen_pct = inputs_df_imp_flags["flag_missing_nitrogen_parameter"]

# Convenience variables for clarity
no = cond_not_organic
mq = missing_quantity
fc = failed_conversion
mm = missing_moisture_param
mn = missing_nitrogen_pct
me = missing_estimate

# Create flag
inputs_df_imp_flags["flag_missing_emissions_org_app"] = np.select(
    [
        no,
        ~me,
        me & mq & fc & mm & mn,
        me & mq & fc & mm,
        me & mq & fc & mn,
        me & mq & mm & mn,
        me & fc & mm & mn,
        me & mq & fc,
        me & mq & mm,
        me & mq & mn,
        me & fc & mm,
        me & fc & mn,
        me & mm & mn,
        me & mq,
        me & fc,
        me & mm,
        me & mn,
        me & ~mq & ~fc & ~mm & ~mn,
    ],
    [
        "NOT APPLICABLE",
        "AVAILABLE",
        "MISSING Q + UC + MP + NP",
        "MISSING Q + UC + MP",
        "MISSING Q + UC + NP",
        "MISSING Q + MP + NP",
        "MISSING UC + MP + NP",
        "MISSING Q + UC",
        "MISSING Q + MP",
        "MISSING Q + NP",
        "MISSING UC + MP",
        "MISSING UC + NP",
        "MISSING MP + NP",
        "MISSING QUANTITY",
        "UNIT CONVERSION",
        "MISSING MOISTURE PARAMETER",
        "MISSING NITROGEN PARAMETER",
        "NO INPUTS MISSING (ESTIMATE ERROR)",
    ],
    default="OTHER/UNFLAGGED MISSING",  # Use a sensible default for unhandled combinations
)

# Flag for the use of pesticides
inputs_df_imp_flags["flag_pesticides_use"] = (
    inputs_df_imp_flags["product_category"]
    .isin(["INSECTICIDE", "HERBICIDE", "FUNGICIDE"])
    .astype("Int64")
)


# Helper functions for aggregation
# Numerical
def sum_applicable_if_all_available(series, kind_of_product_series, applicable_kind):
    applicable_rows = kind_of_product_series == applicable_kind
    applicable_emissions = series[applicable_rows]
    applicable_kinds = kind_of_product_series[applicable_rows]

    if applicable_kinds.isna().any():
        return np.nan

    required_count = len(applicable_emissions)
    non_nan_count = applicable_emissions.notna().sum()

    if non_nan_count == required_count and required_count > 0:
        return applicable_emissions.sum()
    else:
        return np.nan


# Categorical
def join_unique(x):
    return " | ".join(sorted(x.dropna().unique()))


def join_reasons(s):
    if s.notna().any():
        return " | ".join(sorted(set(s.dropna())))
    else:
        return "AVAILABLE"


# Helper functions for aggregation
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


# Group and aggregate variables selectively
inputs_df_aggregation = inputs_df_imp_flags.copy()

group_cols = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]

APPLICABLE_KIND_SYNTHETIC = "QUIMICO"
APPLICABLE_KIND_ORGANIC = "BIOLOGICO/ETOLOGICO/ORGANICO"

inputs_df_aggregation = (
    inputs_df_aggregation.groupby(group_cols, sort=False)
    .agg(
        # Context
        activities=("activity_performed", join_unique),
        activity_count=("activity_performed", "nunique"),
        product_types=("kind_of_product_imp", join_unique),
        product_types_count=("kind_of_product_imp", "nunique"),
        product_categories=("product_category", join_unique),
        product_categories_count=("product_category", "nunique"),
        categories_organic=("category", join_unique),
        # Emission estimates
        synth_input_prod_t_co2eq_ha=(
            "synth_input_prod_t_co2eq_ha",
            lambda x: sum_applicable_if_all_available(
                x,
                inputs_df_aggregation.loc[x.index, "kind_of_product_imp"],
                APPLICABLE_KIND_SYNTHETIC,
            ),
        ),
        synth_input_tran_dist_t_co2eq_ha=(
            "synth_input_tran_dist_t_co2eq_ha",
            lambda x: sum_applicable_if_all_available(
                x,
                inputs_df_aggregation.loc[x.index, "kind_of_product_imp"],
                APPLICABLE_KIND_SYNTHETIC,
            ),
        ),
        synth_input_app_t_co2eq_ha=(
            "synth_input_app_t_co2eq_ha",
            lambda x: sum_applicable_if_all_available(
                x,
                inputs_df_aggregation.loc[x.index, "kind_of_product_imp"],
                APPLICABLE_KIND_SYNTHETIC,
            ),
        ),
        nitrogen_org_t_ha=(
            "nitrogen_org_t_ha",
            lambda x: sum_applicable_if_all_available(
                x,
                inputs_df_aggregation.loc[x.index, "kind_of_product_imp"],
                APPLICABLE_KIND_ORGANIC,
            ),
        ),
        org_fert_app_t_co2eq_ha=(
            "org_fert_app_t_co2eq_ha",
            lambda x: sum_applicable_if_all_available(
                x,
                inputs_df_aggregation.loc[x.index, "kind_of_product_imp"],
                APPLICABLE_KIND_ORGANIC,
            ),
        ),
        # Flags
        flag_missing_emissions_syn_app=("flag_missing_emissions_syn_app", join_reasons),
        flag_missing_emissions_syn_prod=(
            "flag_missing_emissions_syn_prod",
            join_reasons,
        ),
        flag_missing_emissions_syn_tran=(
            "flag_missing_emissions_syn_tran",
            join_reasons,
        ),
        flag_missing_emissions_org_app=("flag_missing_emissions_org_app", join_reasons),
        flag_imputed_product_type=("flag_imputed_product_type", imputation_status),
        flag_imputed_amount=("flag_imputed_amount", imputation_status),
        flag_imputed_nitrogen=("flag_imputed_nitrogen", imputation_status),
        flag_imputed_moisture_level=("flag_imputed_moisture_level", imputation_status),
        flag_pesticides_use=("flag_pesticides_use", "max"),
    )
    .reset_index()
)

# Merge with areas from the agronomic dataframe
inputs_df_totals = inputs_df_aggregation.copy()

inputs_df_totals = inputs_df_totals.merge(
    areas_df, on=["practice_id", "year"], how="left"
)

# Compute total emission estimates
inputs_df_totals["synth_input_prod_t_co2eq"] = (
    inputs_df_totals["synth_input_prod_t_co2eq_ha"] * inputs_df_totals["area_ha"]
)

inputs_df_totals["synth_input_tran_dist_t_co2eq"] = (
    inputs_df_totals["synth_input_tran_dist_t_co2eq_ha"] * inputs_df_totals["area_ha"]
)

inputs_df_totals["synth_input_app_t_co2eq"] = (
    inputs_df_totals["synth_input_app_t_co2eq_ha"] * inputs_df_totals["area_ha"]
)

inputs_df_totals["org_fert_app_t_co2eq"] = (
    inputs_df_totals["org_fert_app_t_co2eq_ha"] * inputs_df_totals["area_ha"]
)

# Final DataFrame
inputs_df_final = inputs_df_totals.sort_values(
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
    "product_types",
    "product_types_count",
    "product_categories",
    "product_categories_count",
    "categories_organic",
    "synth_input_prod_t_co2eq_ha",
    "synth_input_tran_dist_t_co2eq_ha",
    "synth_input_app_t_co2eq_ha",
    "nitrogen_org_t_ha",
    "org_fert_app_t_co2eq_ha",
    "flag_missing_emissions_syn_app",
    "flag_missing_emissions_syn_prod",
    "flag_missing_emissions_syn_tran",
    "flag_missing_emissions_org_app",
    "flag_imputed_product_type",
    "flag_imputed_amount",
    "flag_imputed_nitrogen",
    "flag_imputed_moisture_level",
    "flag_pesticides_use",
    "area_ha",
    "synth_input_prod_t_co2eq",
    "synth_input_tran_dist_t_co2eq",
    "synth_input_app_t_co2eq",
    "org_fert_app_t_co2eq",
]

inputs_df_final = inputs_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write as feather file
inputs_df_final.to_feather(PATH_EXPORT_TABLES / r"table_07_emissions_ag_inputs.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

inputs_df_final.to_sql(
    name="table_07_emissions_ag_inputs", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
