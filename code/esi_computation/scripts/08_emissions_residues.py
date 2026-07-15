"""
Estimation of emissions from crop residues management.
Main operations:
    - clean and select columns
    - computation of emissions associated with
    - aggregation of emissions and related variables at the practice ID level
    - creation of data-quality flags
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
from paths import (
    PATH_CIMMYT_ACTIVITIES,
    SHEET_CIMMYT_ACTIVITIES,
    PATH_AGRONOMIC_FEATURES,
    PATH_PROCESSED_TABLES,
    PATH_SQL_DB,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KEY_IDS = ["farmer_id", "plot_id", "logbook_id", "practice_id", "activity_id"]

KEY_VARS = [
    "year",
    "group_activity",
    "field_grazing_perc_harvest",
    "burned_perc_harvest",
    "incorporated_perc_harvest",
    "chopped_and_left_for_coverage_perc_harvest",
    "detained_for_coverage_perc_harvest",
    "removed_from_plot_baled_in_nunches_perc_harvest",
]

# Applicable activities
APPLICABLE_ACTIVITIES = ["MANUAL HARVEST", "MOTORIZED HARVEST"]

# Columns with percentages of resiudes returned to the field
RESIDUES_RETURNED = [
    "incorporated_perc_harvest",  # Percentage of residue incorporated into the soil
    "chopped_and_left_for_coverage_perc_harvest",  # Percentage of residue crushed to serve as coverage
    "detained_for_coverage_perc_harvest",  # Percentage of residue left standing for coverage
]

# Residue left on the SURFACE as cover (FAO "permanent soil cover"); excludes incorporated (buried)
RESIDUES_SURFACE_COVER = [
    "chopped_and_left_for_coverage_perc_harvest",
    "detained_for_coverage_perc_harvest",
]

# Columns with percentages of resiudes removed from the field
RESIDUES_REMOVED = [
    "field_grazing_perc_harvest",  # Percentage of residue left for pasture in the field
    "burned_perc_harvest",  # Percentage of residue subjected to burning
    "removed_from_plot_baled_in_nunches_perc_harvest",  # Percentage of residue removed from the plot (packed, bundled, etc.)
]

# Parameters
# Dry matter fraction of harvested product
# Source: Table 11.1A in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
DRY_MATTER_FRACTION_MAIZE = (
    0.87  # 1 kg fresh weight is equivalent to 0.87 kg dry matter for maize
)
DRY_MATTER_FRACTION_NON_N_FIXING_FORAGE = 0.90

# Slope
# Source: Table 11.2 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
SLOPE_MAIZE = 1.03

# Intercept
# Source: Table 11.2 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
INTERCEPT_MAIZE = 0.61

# Ratio of below-ground residues to above-ground biomass
# Source: Table 11.2 in https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/4_Volume4/V4_11_Ch11_N2O&CO2.pdf
BELOW_ABOVE_RATIO_MAIZE = 0.22

# N content of above-ground residues in kg N / kg d.m.
# Source: table 11.1A in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
N_CONTENT_ABOVE_GROUND = 0.006

# N content of below-ground residues in kg N / kg d.m.
# Source: table 11.1A in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
N_CONTENT_BELOW_GROUND = 0.007

# Combustion factor for post harvest burning of maize field residues
# Source: table 2.6 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch02_Generic%20Methods.pdf
COMBUSTION_FACTOR_MAIZE = 0.8

# Emission factor for N2O emissions from N inputs, kg N2O–N / kg N input
# Source: table 11.1 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
EF_N2ON_N = 0.01

# Conversion from N2O-N emissions to N2O for reporting purposes
# Source: page 11 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
N2ON_N2O = 44 / 28

# Global warming potential of nitrous oxide
# Source: https://ghgprotocol.org/sites/default/files/2024-08/Global-Warming-Potential-Values%20%28August%202024%29.pdf
GWP_AR6_N2O = 273

# Emission factors for the burning of cropland, in g / kg
# Source: Equation 20 in https://verra.org/wp-content/uploads/2018/03/VM0017-SALM-Methodolgy-v1.0.pdf
EF_CH4 = 2.7
EF_N2O = 0.07

# Global warming potential values relative to CO2
# Source: https://ghgprotocol.org/sites/default/files/2024-08/Global-Warming-Potential-Values%20%28August%202024%29.pdf
GWP_AR6_CH4 = 27

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the activities dataset
residues_df_raw = pd.read_excel(
    io=PATH_CIMMYT_ACTIVITIES,
    sheet_name=SHEET_CIMMYT_ACTIVITIES,
    header=1,
)

# Load agronomic data and areas
agronomic_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES,
    columns=[
        "practice_id",
        "area_ha",
        "maize_grain_yield_t_ha",
        "maize_forage_yield_t_ha",
    ],
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
residues_df_clean_columns = residues_df_raw.copy()
residues_df_clean_columns = residues_df_clean_columns.clean_names(
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
residues_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
    },
    inplace=True,
)

# Select variables of interest
residues_df_variables = residues_df_clean_columns[KEY_IDS + KEY_VARS].copy()

# Convert id columns to string type
residues_df_dtypes = residues_df_variables.copy()
residues_df_dtypes[KEY_IDS] = residues_df_dtypes[KEY_IDS].astype("string")

# Clean year column
residues_df_year = residues_df_dtypes.copy()
residues_df_year["year"] = residues_df_year["year"].apply(
    lambda x: str(x)[0:4] if pd.notna(x) and isinstance(x, str) else x
)
residues_df_year["year"] = pd.to_numeric(residues_df_year["year"], errors="coerce")
residues_df_year["year"] = residues_df_year["year"].astype("Int64")

# We create a dataframe to deal with percentages
residues_df_pct = residues_df_year.copy()
applicable_rows = residues_df_pct["group_activity"].isin(APPLICABLE_ACTIVITIES)
residues_df_pct = residues_df_pct[applicable_rows]
residues_df_pct.head()

# Calculate the sum of returned and removed residue percentages
sum_residues = residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].sum(
    axis=1, skipna=True
)

# Calculate group-level medians using complete rows, to use as fallback for rows with incomplete data
mask_all_reported = (
    residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].notna().all(axis=1)
)

group_median_returned = (
    residues_df_pct.loc[mask_all_reported, RESIDUES_RETURNED]
    .sum(axis=1)
    .groupby(residues_df_pct.loc[mask_all_reported, "group_activity"])
    .median()
)

group_median_surface_cover = (
    residues_df_pct.loc[mask_all_reported, RESIDUES_SURFACE_COVER]
    .sum(axis=1)
    .groupby(residues_df_pct.loc[mask_all_reported, "group_activity"])
    .median()
)

group_median_removed = (
    residues_df_pct.loc[mask_all_reported, RESIDUES_REMOVED]
    .sum(axis=1)
    .groupby(residues_df_pct.loc[mask_all_reported, "group_activity"])
    .median()
)

# Calculate total percentages of residues removed and returned
# Applied to rows where any percentage is available and the sum is greater or equal to 80%
residues_df_pct["residues_removed_perc_harvest"] = np.where(
    (residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].notna().any(axis=1))
    & (sum_residues >= 80),
    residues_df_pct[RESIDUES_REMOVED].sum(axis=1, skipna=True),
    residues_df_pct["group_activity"].map(group_median_removed),
)

residues_df_pct["residues_returned_perc_harvest"] = np.where(
    (residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].notna().any(axis=1))
    & (sum_residues >= 80),
    residues_df_pct[RESIDUES_RETURNED].sum(axis=1, skipna=True),
    residues_df_pct["group_activity"].map(group_median_returned),
)

residues_df_pct["surface_cover_perc_harvest"] = np.where(
    (residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].notna().any(axis=1))
    & (sum_residues >= 80),
    residues_df_pct[RESIDUES_SURFACE_COVER].sum(axis=1, skipna=True),
    residues_df_pct["group_activity"].map(group_median_surface_cover),
)

# Dealing with cases where the sum of percentages exceeds 100%
# The adjustment scales down the percentages proportionally to ensure the total does not exceed 100%
exceeding_rows = residues_df_pct[
    residues_df_pct["residues_returned_perc_harvest"]
    + residues_df_pct["residues_removed_perc_harvest"]
    > 100
]

for idx, row in exceeding_rows.iterrows():
    total = (
        row["residues_returned_perc_harvest"] + row["residues_removed_perc_harvest"]
    )  # Total percentage before adjustment
    ratio_returned = (
        row["residues_returned_perc_harvest"] / total
    )  # Proportion of returned residue in the total
    ratio_removed = (
        row["residues_removed_perc_harvest"] / total
    )  # Proportion of removed residue in the total
    residues_df_pct.at[idx, "residues_returned_perc_harvest"] = (
        ratio_returned * 100
    )  # Adjusted percentage of returned residue
    residues_df_pct.at[idx, "residues_removed_perc_harvest"] = (
        ratio_removed * 100
    )  # Adjusted percentage of removed residue
    ratio_cover = (
        row["surface_cover_perc_harvest"] / total
    )  # Proportion of surface cover in the total
    residues_df_pct.at[idx, "surface_cover_perc_harvest"] = (ratio_cover * 100)

# Define cases for flag
# case_normal: actual sums used (any percentage available AND sum >= 80%)
case_normal = (
    residues_df_pct[RESIDUES_RETURNED + RESIDUES_REMOVED].notna().any(axis=1)
) & (sum_residues >= 80)

# case_assumption: group medians used (fallback for incomplete data)
case_assumption = ~case_normal

# Flag indicating whether the total percent was computed under normal case (0) or assumption case (1)
residues_df_pct["flag_total_pct"] = np.select(
    [case_normal, case_assumption], [0, 1], np.nan
)


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
        return "ASSUMED ONLY"

    # Case 4: mix of reported and imputed (no missing)
    if n_reported > 0 and n_imputed > 0 and n_missing == 0:
        return "REPORTED AND ASSUMED"

    # Case 5: mix including missing
    return "PARTIALLY MISSING"


# Aggregate at the practices id level
residues_df_agg = residues_df_pct.copy()
residues_df_agg = residues_df_agg.groupby(
    ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"], as_index=False
).agg(
    field_grazing_perc_harvest=("field_grazing_perc_harvest", "mean"),
    burned_perc_harvest=("burned_perc_harvest", "mean"),
    removed_from_plot_baled_in_nunches_perc_harvest=(
        "removed_from_plot_baled_in_nunches_perc_harvest",
        "mean",
    ),
    residues_returned_perc_harvest=("residues_returned_perc_harvest", "mean"),
    residues_removed_perc_harvest=("residues_removed_perc_harvest", "mean"),
    surface_cover_perc_harvest=("surface_cover_perc_harvest", "mean"),
    flag_total_pct=("flag_total_pct", imputation_status),
)

# Merge with agronomic data
residues_df_merge = residues_df_agg.merge(
    agronomic_df, on="practice_id", how="left"
)

residues_df_emissions = residues_df_merge.copy()

# Compute the harvested dry matter for the grain and forage yield, in kg d.m. / ha
residues_df_emissions["maize_grain_yield_kg_dm_ha"] = (
    residues_df_emissions["maize_grain_yield_t_ha"] * 1000 * DRY_MATTER_FRACTION_MAIZE
)

residues_df_emissions["maize_forage_yield_kg_dm_ha"] = (
    residues_df_emissions["maize_forage_yield_t_ha"]
    * 1000
    * DRY_MATTER_FRACTION_NON_N_FIXING_FORAGE
)

# Compute the above-ground residue dry matter for the grain yield in Mg/ha (i.e. metric tons per ha)
# Note: assumed to be zero for the forage yield
# Source: table 11.2 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch11_Soils_N2O_CO2.pdf
residues_df_emissions["maize_grain_ag_residue_dm_t_ha"] = np.where(
    residues_df_emissions["maize_grain_yield_kg_dm_ha"].ge(1e-6),
    (
        (residues_df_emissions["maize_grain_yield_kg_dm_ha"] / 1000) * SLOPE_MAIZE
        + INTERCEPT_MAIZE
    ),
    np.nan,
)

residues_df_emissions["maize_forage_ag_residue_dm_t_ha"] = np.where(
    residues_df_emissions["maize_forage_yield_kg_dm_ha"].ge(1e-6), 0.0, np.nan
)

# Compute the above-ground biomass
residues_df_emissions["maize_grain_ag_biomass_dm_t_ha"] = (
    residues_df_emissions["maize_grain_yield_kg_dm_ha"] / 1000
    + residues_df_emissions["maize_grain_ag_residue_dm_t_ha"]
)

residues_df_emissions["maize_forage_ag_biomass_dm_t_ha"] = (
    residues_df_emissions["maize_forage_yield_kg_dm_ha"] / 1000
    + residues_df_emissions["maize_forage_ag_residue_dm_t_ha"]
)

# Compute the below-ground residue dry matter for the grain and forage yield
# Based on equation and parameters in table 11.2 in https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/4_Volume4/V4_11_Ch11_N2O&CO2.pdf
residues_df_emissions["maize_grain_bg_residue_dm_t_ha"] = (
    residues_df_emissions["maize_grain_ag_biomass_dm_t_ha"] * BELOW_ABOVE_RATIO_MAIZE
)

residues_df_emissions["maize_forage_bg_residue_dm_t_ha"] = (
    residues_df_emissions["maize_forage_ag_biomass_dm_t_ha"] * BELOW_ABOVE_RATIO_MAIZE
)

# Compute the total crop residue dry matter for grain, forage and both, in t per ha
residues_df_emissions["maize_grain_residue_dm_t_ha"] = residues_df_emissions[
    ["maize_grain_ag_residue_dm_t_ha", "maize_grain_bg_residue_dm_t_ha"]
].sum(axis=1, skipna=False, min_count=1)

residues_df_emissions["maize_forage_residue_dm_t_ha"] = residues_df_emissions[
    ["maize_forage_ag_residue_dm_t_ha", "maize_forage_bg_residue_dm_t_ha"]
].sum(axis=1, skipna=False, min_count=1)

residues_df_emissions["maize_residues_dm_t_ha"] = residues_df_emissions[
    ["maize_grain_residue_dm_t_ha", "maize_forage_residue_dm_t_ha"]
].sum(axis=1, skipna=True, min_count=1)

# Total crop residues returned (grain and forage), in t per ha
residues_df_emissions["maize_residues_returned_dm_t_ha"] = residues_df_emissions[
    "maize_residues_dm_t_ha"
] * (residues_df_emissions["residues_returned_perc_harvest"].fillna(0) / 100)

residues_df_emissions["maize_residues_surface_cover_dm_t_ha"] = (
    residues_df_emissions["maize_residues_dm_t_ha"]
    * residues_df_emissions["surface_cover_perc_harvest"].fillna(0) / 100
)

# Compute the crop residue N content
# Calculate total_loss: percentage of residues lost due to burning and removal
# Burned residues lose (100% - combustion_factor)%, plus all removed residues are lost
# The complement represents residues that return to the soil
residues_df_emissions["total_loss"] = (
    1
    - residues_df_emissions["residues_returned_perc_harvest"].fillna(0) / 100
    + residues_df_emissions["burned_perc_harvest"].fillna(0)
    / 100
    * (1 - COMBUSTION_FACTOR_MAIZE)
).clip(0, 1)

grain_ag_n_kg_ha = (
    residues_df_emissions["maize_grain_ag_residue_dm_t_ha"]
    * 1000
    * N_CONTENT_ABOVE_GROUND
    * (1 - residues_df_emissions["total_loss"])
)

grain_bg_n_kg_ha = (
    residues_df_emissions["maize_grain_bg_residue_dm_t_ha"]
    * 1000
    * N_CONTENT_BELOW_GROUND
)

forage_bg_n_kg_ha = (
    residues_df_emissions["maize_forage_bg_residue_dm_t_ha"]
    * 1000
    * N_CONTENT_BELOW_GROUND
)

# Total N from grain crop residues
residues_df_emissions["maize_grain_residues_n_kg_ha"] = np.where(
    residues_df_emissions["maize_grain_yield_kg_dm_ha"] > 1e-6,
    (grain_ag_n_kg_ha + grain_bg_n_kg_ha),
    np.nan,
)

# Total N from maize crop residues
residues_df_emissions["maize_forage_residues_n_kg_ha"] = np.where(
    residues_df_emissions["maize_forage_yield_kg_dm_ha"] > 1e-6,
    (forage_bg_n_kg_ha),
    np.nan,
)

# Compute direct nitrous oxide emissions from crop residues in t CO2eq
# (crop residues term in the inputs term of equation 11.1)
residues_df_emissions["maize_grain_residues_returned_tco2eq_ha"] = (
    residues_df_emissions["maize_grain_residues_n_kg_ha"]
    * (EF_N2ON_N * N2ON_N2O * GWP_AR6_N2O)
    / 1000
)

residues_df_emissions["maize_forage_residues_returned_tco2eq_ha"] = (
    residues_df_emissions["maize_forage_residues_n_kg_ha"]
    * (EF_N2ON_N * N2ON_N2O * GWP_AR6_N2O)
    / 1000
)

residues_df_emissions["maize_residues_returned_tco2eq_ha"] = residues_df_emissions[
    [
        "maize_grain_residues_returned_tco2eq_ha",
        "maize_forage_residues_returned_tco2eq_ha",
    ]
].sum(axis=1, skipna=True, min_count=1)

# Compute the non-CO2 emissions from crop residues burning
residues_df_emissions["maize_grain_residues_burnt_tco2eq_ha"] = (
    residues_df_emissions["maize_grain_residue_dm_t_ha"]
    * 1000
    * residues_df_emissions["burned_perc_harvest"].div(100)
    * COMBUSTION_FACTOR_MAIZE
    * (EF_CH4 * GWP_AR6_CH4 + EF_N2O * GWP_AR6_N2O)
    * 1e-6
)

# Compute total emission estimates
residues_df_totals = residues_df_emissions.copy()

residues_df_totals["maize_grain_residues_returned_tco2eq"] = (
    residues_df_totals["maize_grain_residues_returned_tco2eq_ha"]
    * residues_df_totals["area_ha"]
)

residues_df_totals["maize_forage_residues_returned_tco2eq"] = (
    residues_df_totals["maize_forage_residues_returned_tco2eq_ha"]
    * residues_df_totals["area_ha"]
)

residues_df_totals["maize_residues_returned_tco2eq"] = (
    residues_df_totals["maize_residues_returned_tco2eq_ha"]
    * residues_df_totals["area_ha"]
)

residues_df_totals["maize_grain_residues_burnt_tco2eq"] = (
    residues_df_totals["maize_grain_residues_burnt_tco2eq_ha"]
    * residues_df_totals["area_ha"]
)

# Final DataFrame
residues_df_final = residues_df_totals.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "field_grazing_perc_harvest",
    "burned_perc_harvest",
    "removed_from_plot_baled_in_nunches_perc_harvest",
    "residues_removed_perc_harvest",
    "residues_returned_perc_harvest",
    "flag_total_pct",
    "area_ha",
    "maize_grain_yield_t_ha",
    "maize_grain_yield_kg_dm_ha",
    "maize_grain_ag_residue_dm_t_ha",
    "maize_grain_ag_biomass_dm_t_ha",
    "maize_grain_bg_residue_dm_t_ha",
    "maize_grain_residue_dm_t_ha",
    "maize_grain_residues_n_kg_ha",
    "maize_grain_residues_returned_tco2eq_ha",
    "maize_grain_residues_burnt_tco2eq_ha",
    "maize_grain_residues_returned_tco2eq",
    "maize_grain_residues_burnt_tco2eq",
    "maize_forage_yield_t_ha",
    "maize_forage_yield_kg_dm_ha",
    "maize_forage_ag_biomass_dm_t_ha",
    "maize_forage_ag_residue_dm_t_ha",
    "maize_forage_bg_residue_dm_t_ha",
    "maize_forage_residue_dm_t_ha",
    "maize_forage_residues_n_kg_ha",
    "maize_forage_residues_returned_tco2eq_ha",
    "maize_forage_residues_returned_tco2eq",
    "maize_residues_dm_t_ha",
    "maize_residues_returned_dm_t_ha",
    "surface_cover_perc_harvest",
    "maize_residues_surface_cover_dm_t_ha",
    "maize_residues_returned_tco2eq_ha",
    "maize_residues_returned_tco2eq",
]

residues_df_final = residues_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
residues_df_final.to_feather(
    PATH_PROCESSED_TABLES / "table_08_emissions_residues.feather"
)

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    residues_df_final.to_sql(
        name="table_08_emissions_residues",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")