"""
Estimation of water consumption.
Main operations:
    - cleaning column names, adjusting data types, selecting variables
    - concatenation / aggregation of water features
    - merging
    - calculation of water metrics
    - calculation of irrigation emissions
    - export results as feather file and to SQL database

Only reported water consumption is used: practices without a reported value keep a
missing water-use estimate.

Authors: Dario Marone, Giorgio Parlato
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import janitor


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
from paths import (
    PATH_CIMMYT_LOGBOOK,
    SHEET_CIMMYT_LOGBOOK,
    PATH_CIMMYT_WATER,
    SHEET_CIMMYT_WATER,
    PATH_REFERENCE_TABLE,
    PATH_AGRONOMIC_FEATURES,
    PATH_WATER_FEATURES,
    PATH_SQL_DB,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Key columns from logbook
LOGBOOK_COLS = [
    "year",
    "logbook_id",
    "state",
    "practice_id",
    "agroecological_hub",
    "conservation_agriculture",
]

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

# Water consumption variables
WATER_CONSUMPTION_VARS = [
    "total_water_consumption_per_irrigation_lt",
    "water_consumption_lt",
]

# ID columns in the irrigation dataset
ID_KEYS_WATER = ["farmer_id", "plot_id", "logbook_id", "practice_id", "id_irrigation"]

# Practice-year grouping keys
GROUP_KEYS = ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]

# Pressurized systems are the only ones the irrigation emission factors apply to
PRESSURIZED_IRRIGATION_TYPES = ["DRIP IRRIGATION", "SPRINKLER IRRIGATION"]

# Emission factors for irrigation equipment and application (precalculated)
# Reference: Juárez-Hernández, S., & Pardo, C. S. (2019). Irrigation energy use and related greenhouse gas emissions of maize production in Mexico. International journal of water resources development.
# Source: based on values in Table 5 and 7 in https://doi.org/10.1080/07900627.2018.1482739;
# average in tCO2eq per m3 of the values for summer and winter season,
# for pressurized systems only, excluding groundwater pumping
EF_IRR_EQUIP = 1.61319e-05
EF_IRR_APP = 0.000148952


# ---------------------------------------------------------------------------
# Helper: concatenate unique sorted values or return NaN
# ---------------------------------------------------------------------------
def _join_unique(s: pd.Series) -> str | float:
    """Join unique non-NA values with ' | ', or return NaN if any value is NA."""
    if s.isna().any():
        return np.nan
    return " | ".join(sorted(s.astype(str).unique()))


# ---------------------------------------------------------------------------
# Helper: sum a group only when it is complete
# ---------------------------------------------------------------------------
def _sum_if_complete(s: pd.Series) -> float:
    """Sum the group, or return NaN if any value is missing."""
    return s.sum(min_count=1) if s.notna().all() else np.nan


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
# Load datasets to merge with irrigation after aggregation
agronomic_df = pd.read_feather(PATH_AGRONOMIC_FEATURES)
reference_df = pd.read_feather(PATH_REFERENCE_TABLE)

logbook_df = pd.read_excel(
    io=PATH_CIMMYT_LOGBOOK, sheet_name=SHEET_CIMMYT_LOGBOOK, header=1
).clean_names(**CLEAN_NAMES_KW)
logbook_df.rename(
    columns={"innovation_conventional_practices_id": "practice_id"}, inplace=True
)

# ---- Water dataset ----
water_df = (
    pd.read_excel(PATH_CIMMYT_WATER, sheet_name=SHEET_CIMMYT_WATER)
    .clean_names(**CLEAN_NAMES_KW)
    .rename(columns={"innovation_conventional_practices_id": "practice_id"})
)
water_df[ID_KEYS_WATER] = water_df[ID_KEYS_WATER].astype("string")
water_df = water_df[
    ID_KEYS_WATER + ["year"] + WATER_MANAGEMENT_VARS + WATER_CONSUMPTION_VARS
]

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Total water consumption: use water_consumption_lt × time when total is zero
time_fallback = (
    water_df[["average_irrigation_time_hr_ha", "irrigation_time_hr_ha"]]
    .bfill(axis=1)
    .iloc[:, 0]
)
total_is_zero = water_df["total_water_consumption_per_irrigation_lt"] == 0
lt_available = water_df["water_consumption_lt"].notna()

water_df["total_water_cons"] = np.where(
    total_is_zero & lt_available,
    water_df["water_consumption_lt"] * time_fallback,
    np.where(
        total_is_zero & ~lt_available,
        np.nan,
        water_df["total_water_consumption_per_irrigation_lt"],
    ),
)

# ---- Irrigation emissions (event level) ----
# The emission factors apply to pressurized systems only, so emissions are computed
# per irrigation event and summed over the pressurized events of each practice-year.
# Practice-years without a pressurized event are absent here and keep a missing value.
pressurized_events = water_df[
    water_df["irrigation_type"].isin(PRESSURIZED_IRRIGATION_TYPES)
].copy()

pressurized_events["irr_app_tco2eq_ha"] = (
    pressurized_events["total_water_cons"] * 10**-3 * EF_IRR_APP
)
pressurized_events["irr_equip_tco2eq_ha"] = (
    pressurized_events["total_water_cons"] * 10**-3 * EF_IRR_EQUIP
)

emissions_agg = (
    pressurized_events.groupby(GROUP_KEYS)
    .agg(
        irr_app_tco2eq_ha=("irr_app_tco2eq_ha", _sum_if_complete),
        irr_equip_tco2eq_ha=("irr_equip_tco2eq_ha", _sum_if_complete),
    )
    .reset_index()
)

# ---- Aggregate to practice-year level ----
water_agg = (
    water_df.groupby(GROUP_KEYS)
    .agg(
        hydric_regime=("hydric_regime", _join_unique),
        water_sources=("water_source", _join_unique),
        water_sources_count=("water_source", "nunique"),
        irrigation_types=("irrigation_type", _join_unique),
        irrigation_types_count=("irrigation_type", "nunique"),
        irrigation_events_count=("id_irrigation", "nunique"),
        total_water_consumption_per_year_ha=("total_water_cons", _sum_if_complete),
        water_consumption_flag=(
            "total_water_cons",
            lambda x: (
                "reported"
                if x.notna().all()
                else "missing"
                if x.isna().all()
                else "partially_missing"
            ),
        ),
    )
    .reset_index()
)

# ---- Attach the event-level irrigation emissions ----
water_agg = water_agg.merge(emissions_agg, on=GROUP_KEYS, how="left")

# ---- Merge reference table (plot_type, coords) ----
reference_df = reference_df[
    ["practice_id", "logbook_id", "year", "plot_type", "latitude_n", "longitude_w"]
]
df = water_agg.merge(reference_df, on=["practice_id", "logbook_id", "year"], how="left")

# ---- Merge logbook variables ----
logbook_df["logbook_id"] = logbook_df["logbook_id"].astype("string")
logbook_df["practice_id"] = logbook_df["practice_id"].astype("string")
df = df.merge(
    logbook_df[LOGBOOK_COLS],
    on=["logbook_id", "practice_id", "year"],
    how="left",
)

# ---- Merge agronomic (area) ----
df = df.merge(
    agronomic_df[["practice_id", "area_ha"]],
    on="practice_id",
    how="left",
)

# -----------------------------------------------------------------------------
# Water metrics
# -----------------------------------------------------------------------------
water_df_metrics = df.copy()

# Compute reported water consumption in cubic meters per hectare
water_df_metrics["water_cons_rep_m3_ha"] = np.where(
    water_df_metrics["total_water_consumption_per_year_ha"].notna(),
    water_df_metrics["total_water_consumption_per_year_ha"] * 10**-3,
    np.nan,
)

water_df_totals = water_df_metrics.copy()

# Compute total reported water consumption
water_df_totals["water_cons_rep_m3"] = (
    water_df_totals.water_cons_rep_m3_ha * water_df_totals.area_ha
)

# Convert to cubic decameters for calculation of water ESI
water_df_totals["water_cons_rep_dam3"] = water_df_totals.water_cons_rep_m3 * 10**-3

# -----------------------------------------------------------------------------
# Irrigation emissions
# -----------------------------------------------------------------------------
# Create df for estimating emissions from water use
water_df_emissions = water_df_totals.copy()

# The per hectare emissions were computed per irrigation event above; scale to totals
water_df_emissions["irr_app_tco2eq"] = (
    water_df_emissions.irr_app_tco2eq_ha * water_df_emissions.area_ha
)

water_df_emissions["irr_equip_tco2eq"] = (
    water_df_emissions.irr_equip_tco2eq_ha * water_df_emissions.area_ha
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
water_df_final = water_df_emissions.copy()

water_df_final.to_feather(PATH_WATER_FEATURES)

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    water_df_final.to_sql(
        name="table_05_water_features",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
