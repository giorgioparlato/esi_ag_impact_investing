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
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import janitor 
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# Paths and other constants
# -----------------------------------------------------------------------------

from paths import (
    PATH_CIMMYT_LOGBOOK,
    SHEET_CIMMYT_LOGBOOK,
    PATH_CIMMYT_WATER,
    SHEET_CIMMYT_WATER,
    PATH_CIMMYT_ECONOMICS,
    SHEET_CIMMYT_ECONOMICS,
    PATH_REFERENCE_TABLE,
    PATH_AGRONOMIC_FEATURES,
    PATH_WATER_TARIFFS,
    PATH_CLIMATE_ZONES,
    PATH_SOIL_CLASS,
    PATH_PROCESSED_TABLES,
    PATH_SQL_DB,
    PATH_SQL_DB_EXT,
)

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

# Other key variables
WATER_MANAGEMENT_VARS = [
    "water_source",
    "hydric_regime",
    "irrigation_type",
    "irrigation_system",
    "winter_sumer_season",
    "practices_type",
    "average_irrigation_time_hr_ha",
    "irrigation_time_hr_ha",
]

# Key columns from logbook
LOGBOOK_COLS = [
    "year",
    "logbook_id",
    "state",
    "practice_id",
    "agroecological_hub",
    "conservation_agriculture",
]

# Water consumption variables
WATER_CONSUMPTION_VARS = [
    "total_water_consumption_per_irrigation_lt",
    "water_consumption_lt",
]

# ID columns in the irrigation dataset
ID_KEYS_WATER = ["farmer_id", "plot_id", "logbook_id", "practice_id", "id_irrigation"]

# Efficiency parameters
# Reference: Rohwer, J., Gerten, D., & Lucht, W. (2007). Development of functional irrigation types for improved global crop modelling.
# Source: adapted from Table 6 in https://www.pik-potsdam.de/en/output/publications/pikreports/.files/pr104.pdf
EFFICIENCY_MAP = {
    "SURFACE IRRIGATION": 0.420,
    "SPRINKLER IRRIGATION": 0.713,
    "DRIP IRRIGATION": 0.855,
}

# ---------------------------------------------------------------------------
# Helper: concatenate unique sorted values or return NaN
# ---------------------------------------------------------------------------
def _join_unique(s: pd.Series) -> str | float:
    """Join unique non-NA values with ' | ', or return NaN if any value is NA."""
    if s.isna().any():
        return np.nan
    return " | ".join(sorted(s.astype(str).unique()))


CLEAN_NAMES_KW = dict(
    axis="columns",
    strip_underscores=True,
    case_type="lower",
    remove_special=True,
    strip_accents=True,
    preserve_original_labels=True,
    enforce_string=True,
)


# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load water tariffs from CONAGUA
water_tariffs_df = pd.read_excel(io=PATH_WATER_TARIFFS)

# Load datasets to merge with irrigation
agronomic_df = pd.read_feather(PATH_AGRONOMIC_FEATURES)
reference_df = pd.read_feather(PATH_REFERENCE_TABLE)

logbook_df = (pd.read_excel(
    io=PATH_CIMMYT_LOGBOOK, sheet_name=SHEET_CIMMYT_LOGBOOK, header=1
).clean_names(**CLEAN_NAMES_KW))
logbook_df.rename(columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True)

economics_df_raw = (pd.read_excel(
    io=PATH_CIMMYT_ECONOMICS, sheet_name=SHEET_CIMMYT_ECONOMICS, header=1
).clean_names(**CLEAN_NAMES_KW))

# ---- Water dataset ----
water_df = (
    pd.read_excel(PATH_CIMMYT_WATER, sheet_name=SHEET_CIMMYT_WATER)
    .clean_names(**CLEAN_NAMES_KW)
    .rename(columns={"innovation_conventional_practices_id": "practice_id"})
)
water_df[ID_KEYS_WATER] = water_df[ID_KEYS_WATER].astype("string")
water_df = water_df[ID_KEYS_WATER + ["year"] + WATER_MANAGEMENT_VARS + WATER_CONSUMPTION_VARS]

# Map irrigation efficiency
water_df["irr_efficiency_param"] = water_df["irrigation_type"].map(EFFICIENCY_MAP)

# Adjusted water consumption: use water_consumption_lt × time when total is zero
time_fallback = (
    water_df[["average_irrigation_time_hr_ha", "irrigation_time_hr_ha"]]
    .bfill(axis=1)
    .iloc[:, 0]
)
total_is_zero = water_df["total_water_consumption_per_irrigation_lt"] == 0
lt_available = water_df["water_consumption_lt"].notna()

water_df["adj_total_water_cons"] = np.where(
    total_is_zero & lt_available,
    water_df["water_consumption_lt"] * time_fallback,
    np.where(total_is_zero & ~lt_available, np.nan,
                water_df["total_water_consumption_per_irrigation_lt"]),
)

water_df["irr_events_per_year"] = (
    water_df.groupby(["practice_id", "year"])["practice_id"].transform("count")
)

# ---- Aggregate to practice-year level ----
water_agg = (
    water_df
    .groupby(["farmer_id", "plot_id", "logbook_id", "practice_id", "year"])
    .agg(
        hydric_regime=("hydric_regime", _join_unique),
        water_sources=("water_source", _join_unique),
        water_sources_count=("water_source", "nunique"),
        irrigation_types=("irrigation_type", _join_unique),
        irrigation_types_count=("irrigation_type", "nunique"),
        irrigation_events_count=("id_irrigation", "nunique"),
        irr_efficiency_param=("irr_efficiency_param", "mean"),
        total_water_consumption_per_year_ha=(
            "adj_total_water_cons",
            lambda x: x.sum(min_count=1) if x.notna().all() else np.nan,
        ),
        water_consumption_flag=(
            "adj_total_water_cons",
            lambda x: (
                "reported" if x.notna().all() else
                "missing" if x.isna().all() else
                "partially_missing"
            ),
        ),
    )
    .reset_index()
)

# ---- Merge reference table (plot_type, coords) ----
reference_df = reference_df[["practice_id", "logbook_id", "year",
                                "plot_type", "latitude_n", "longitude_w"]]
df = water_agg.merge(reference_df, on=["practice_id", "logbook_id", "year"], how="left")

# ---- Merge logbook variables ----
logbook_df["logbook_id"] = logbook_df["logbook_id"].astype("string")
logbook_df["practice_id"] = logbook_df["practice_id"].astype("string")
df = df.merge(
    logbook_df[LOGBOOK_COLS],
    on=["logbook_id", "practice_id", "year"],
    how="left",
)

# ---- Merge agronomic (area, yield) ----
df = df.merge(
    agronomic_df[["practice_id", "year", "area_ha"]],
    on=["practice_id", "year"],
    how="left",
)

# ---- Sample soil type and climate zone from rasters ----
coords = list(zip(df["longitude_w"].values, df["latitude_n"].values))

with rasterio.open(PATH_SOIL_CLASS) as src:
    df["soil_type"] = pd.Categorical([v[0] for v in src.sample(coords)])

with rasterio.open(PATH_CLIMATE_ZONES / "CLIMATE_ZONE.rst") as src:
    df["climate_code"] = pd.Categorical([v[0] for v in src.sample(coords)])

# ---- Merge SQL environmental data (pcp, tmax, tmin) ----
with sqlite3.connect(PATH_SQL_DB_EXT) as conn:
    env_df = pd.read_sql("SELECT logbook_id, year, pcp, tmax, tmin FROM env_data", conn)
env_df["logbook_id"] = env_df["logbook_id"].astype("string")
df = df.merge(env_df, on=["logbook_id", "year"], how="left")

# ---- Merge economics (irrigation cost) ----
econ_raw = (
    pd.read_excel(PATH_CIMMYT_ECONOMICS, sheet_name="Data", header=1)
    .clean_names(**CLEAN_NAMES_KW)
)
ID_KEYS_ECONOMICS = ["farmer_id", "plot_id", "logbook_id",
                "innovation_practices_id", "conventional_practices_id"]
econ_raw[ID_KEYS_ECONOMICS] = econ_raw[ID_KEYS_ECONOMICS].astype("string")

# Split innovation / conventional, then concatenate
econ_innov = econ_raw.iloc[:, :24].copy()
econ_innov.rename(columns={"innovation_practices_id": "practice_id"}, inplace=True)

innov_cols_to_exclude = econ_raw.columns[9:24]
econ_conv = econ_raw.drop(columns=innov_cols_to_exclude).copy()
econ_conv.columns = econ_conv.columns.str.replace(r"_1$", "", regex=True)
econ_conv.rename(columns={"conventional_practices_id": "practice_id"}, inplace=True)

econ = pd.concat([econ_conv, econ_innov]).dropna(subset=["practice_id"])
econ.rename(columns={"irrigation": "irrigation_cost_mxn_ha"}, inplace=True)

df = df.merge(
    econ[["logbook_id", "practice_id", "year", "irrigation_cost_mxn_ha"]],
    on=["logbook_id", "practice_id", "year"],
    how="left",
)

# ---- Merge water tariffs ----
df = df.merge(water_tariffs_df, on="year", how="left")

# ── Configuration ────────────────────────────────────────────────────
TARGET = "total_water_consumption_per_year_ha"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "latitude_n",
    "longitude_w",
    "irrigation_cost_mxn_ha",
    "pcp",
    "tmax",
    "tmin",
    "irrigation_events_count",
]

CATEGORICAL_FEATURES = [
    "hydric_regime",
    "plot_type",
    "soil_type",
    "water_sources",
    "climate_code",
    "agroecological_hub",
]

# Best hyperparameters from GridSearchCV (refined from RandomizedSearchCV)
SURFACE_RF_PARAMS = {
    "max_depth": None,
    "max_features": 0.7,
    "min_samples_leaf": 2,
    "n_estimators": 720,
}

DRIP_RF_PARAMS = {
    "max_depth": None,
    "max_features": 0.7,
    "min_samples_leaf": 3,
    "n_estimators": 800,
}


# ── Functions ────────────────────────────────────────────────────────
def build_and_impute(df, irr_type, model_params, cat_features):
    """Train RF on known data for one irrigation type, return imputed values."""
    type_mask = df["irrigation_types"] == irr_type
    X_raw = df.loc[type_mask, NUMERIC_FEATURES + cat_features]
    y_raw = df.loc[type_mask, TARGET]

    known = ~pd.isna(y_raw)
    missing = pd.isna(y_raw)

    X_known = X_raw[known]
    y_known = y_raw[known]
    X_missing = X_raw[missing]

    # OHE fitted on known data only
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_known[cat_features])

    def encode(data):
        num = data[NUMERIC_FEATURES]
        cat_ohe = pd.DataFrame(
            encoder.transform(data[cat_features]),
            index=data.index,
            columns=encoder.get_feature_names_out(cat_features),
        )
        return pd.concat([num, cat_ohe], axis=1)

    # Train on all known data
    model = RandomForestRegressor(
        **model_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(encode(X_known), y_known)

    # Predict (RF guarantees non-negative predictions with non-negative training data)
    imputed = model.predict(encode(X_missing))

    print(f"{irr_type}: trained on {known.sum():,}, imputing {missing.sum():,}")
    print(f"  Imputed mean: {imputed.mean():,.0f}, median: {np.median(imputed):,.0f}")

    return df.loc[type_mask & missing].index, imputed


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load and filter
    df = df[~df["irrigation_types"].str.contains(r"\|", na=False)].copy()
    print(f"Loaded {len(df):,} observations")

    y = df[TARGET]
    mask_known = ~pd.isna(y)

    # Initialize hybrid variable with observed values
    df["water_hybrid_ml_lt"] = np.nan
    df.loc[mask_known, "water_hybrid_ml_lt"] = y[mask_known]

    # Impute surface and drip separately
    for irr_type, params in [
        ("SURFACE IRRIGATION", SURFACE_RF_PARAMS),
        ("DRIP IRRIGATION", DRIP_RF_PARAMS),
    ]:
        idx, imputed = build_and_impute(df, irr_type, params, CATEGORICAL_FEATURES)
        df.loc[idx, "water_hybrid_ml_lt"] = imputed

    df["water_use_source"] = np.select(
        [y.notna(),
        y.isna() & df["water_hybrid_ml_lt"].notna(),
        ],
        ["observed",
         "imputed",
        ],
        default="missing",
    )

    # Summary
    print(f"\nFinal variable:")
    print(f"  Observed:  {mask_known.sum():,}")
    print(f"  Imputed:   {df['water_hybrid_ml_lt'].notna().sum() - mask_known.sum():,}")
    print(f"  Remaining NaN: {df['water_hybrid_ml_lt'].isna().sum():,} (sprinkler)")
    print("\nwater_use_source counts:")
    print(df["water_use_source"].value_counts(dropna=False))

    for irr_type in ["SURFACE IRRIGATION", "DRIP IRRIGATION"]:
        subset = df.loc[df["irrigation_types"] == irr_type, "water_hybrid_ml_lt"]
        print(f"\n  {irr_type}: n={len(subset):,}, "
              f"mean={subset.mean():,.0f}, median={subset.median():,.0f}")


# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
df.to_feather(PATH_PROCESSED_TABLES / "irrigation_df_ml_imputed.feather")

# # Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    df.to_sql(
        name="irrigation_df_ml_imputed",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
