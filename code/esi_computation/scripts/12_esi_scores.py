"""
Computation of ESI scores.
Main operations:
    - query necessary variables from SQL database
    - compute absolute and intensity ESI scores
    - export results as feather file and to SQL database

Note: this script uses **scaled** ESI metrics for improved interpretability
Authors: Dario Marone, Giorgio Parlato
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
from paths import PATH_PROCESSED_TABLES, PATH_SQL_DB

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Multi-table query
QUERY = """
SELECT 
    table_01_reference.farmer_id, 
    table_01_reference.plot_id, 
    table_01_reference.logbook_id, 
    table_01_reference.practice_id, 
    table_01_reference.year,
    table_01_reference.practices_type,
    table_01_reference.hydric_regime,

    table_02_reference_enriched.luc_metric_scaled,
    table_02_reference_enriched.ro_metric_scaled,
    table_02_reference_enriched.cc_metric_rate_scaled,

    table_03_agronomic_features.area_km2,
    table_03_agronomic_features.crops,
    table_05_b_water_features.water_cons_hyb_ml_dam3,
    table_05_b_water_features.water_sources,
    table_11_emissions_net_ghg.net_ghg_emissions_tco2eq,
    table_11_emissions_net_ghg.net_ghg_emissions_wfuel_tco2eq,
    table_11_emissions_net_ghg.albedo_emissions_tco2eq,
    table_11_emissions_net_ghg.flag_net_ghg,
    table_11_emissions_net_ghg.flag_net_ghg_wfuel,
    table_04_economic_features.total_revenue_mxn

FROM table_01_reference

LEFT JOIN table_02_reference_enriched USING (practice_id)
LEFT JOIN table_03_agronomic_features USING (practice_id)
LEFT JOIN table_05_b_water_features USING (practice_id)
LEFT JOIN table_11_emissions_net_ghg USING (practice_id)
LEFT JOIN table_04_economic_features USING (practice_id)
"""

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Connection
conn = sqlite3.connect(PATH_SQL_DB)

df_in = pd.read_sql_query(QUERY, conn)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Land ESI scores
df_scores = df_in.copy()


def normalize_condition(condition: pd.Series) -> pd.Series:
    """Return a plain boolean mask, treating missing conditions as false."""
    return condition.astype("boolean").fillna(False).astype(bool)


def append_reason(
    status: pd.Series, condition: pd.Series, label: str
) -> pd.Series:
    """Append a label to rows where a status condition applies."""
    condition = normalize_condition(condition)
    current = status.loc[condition]
    status.loc[condition] = np.where(
        current.eq(""),
        label,
        current + " | " + label,
    )
    return status


def build_score_flag(
    score_col: str,
    unavailable_rules: list[tuple[str, pd.Series]],
    available_notes: list[tuple[str, pd.Series]] | None = None,
) -> pd.Series:
    """Describe why a score is unavailable and note assumptions in available scores."""
    score_available = df_scores[score_col].notna()
    status = pd.Series("", index=df_scores.index, dtype="string")

    for label, condition in unavailable_rules:
        status = append_reason(status, ~score_available & condition, label)

    status = status.mask(status.eq("") & ~score_available, "UNAVAILABLE - UNCLASSIFIED")
    status = status.mask(score_available, "AVAILABLE")

    for label, condition in available_notes or []:
        status = append_reason(status, score_available & condition, label)

    return status


df_scores["esi_land"] = df_scores["area_km2"] * df_scores["luc_metric_scaled"]

# Water ESI scores
valid_water_source = ~df_scores.water_sources.str.contains(
    "SUBTERRANEAN", case=False, na=False
)

water_conditions = [
    df_scores.hydric_regime.eq("RAINFED"),
    df_scores.hydric_regime.eq("IRRIGATION")
    & df_scores.water_sources.eq("WASTEWATER (E.G. INDUSTRIAL AND/OR DOMESTIC USE)"),
    valid_water_source
    & df_scores.water_cons_hyb_ml_dam3.notna()
    & df_scores.ro_metric_scaled.notna(),
]

water_choices = [0, 0, df_scores.water_cons_hyb_ml_dam3 * df_scores.ro_metric_scaled]

df_scores["esi_water"] = np.select(water_conditions, water_choices, default=np.nan)

# Climate ESI scores
df_scores["esi_climate"] = (
    df_scores["net_ghg_emissions_tco2eq"] * df_scores["cc_metric_rate_scaled"]
)

# Climate ESI scores
df_scores["esi_climate_wfuel"] = (
    df_scores["net_ghg_emissions_wfuel_tco2eq"] * df_scores["cc_metric_rate_scaled"]
)

# Albedo ESI scores
df_scores["esi_climate_albedo"] = (
    df_scores["albedo_emissions_tco2eq"] * df_scores["cc_metric_rate_scaled"]
)

# Total ESI scores
df_scores["esi_total"] = df_scores[
    ["esi_land", "esi_water", "esi_climate", "esi_climate_albedo"]
].sum(axis=1, skipna=False, min_count=1)

df_scores["esi_total_wfuel"] = df_scores[
    ["esi_land", "esi_water", "esi_climate_wfuel", "esi_climate_albedo"]
].sum(axis=1, skipna=False, min_count=1)

# Excluding farms that cultivate crops other than MAIZE
esi_cols = [
    "esi_total_wfuel",
    "esi_total",
    "esi_climate_albedo",
    "esi_climate_wfuel",
    "esi_climate",
    "esi_water",
    "esi_land",
]
df_scores.loc[df_scores["crops"] != "MAIZE", esi_cols] = np.nan

# ESI score availability and quality flags
maize_rows = df_scores.crops.eq("MAIZE")
missing_crop = df_scores.crops.isna()
non_maize_rows = df_scores.crops.notna() & ~maize_rows

crop_rules = [
    ("MISSING CROP CLASSIFICATION", missing_crop),
    ("NOT APPLICABLE - CROP IS NOT MAIZE", non_maize_rows),
]

df_scores["flag_esi_land"] = build_score_flag(
    "esi_land",
    crop_rules
    + [
        ("MISSING AREA", maize_rows & df_scores.area_km2.isna()),
        (
            "MISSING LAND ESI COEFFICIENT",
            maize_rows & df_scores.luc_metric_scaled.isna(),
        ),
    ],
)

rainfed_rows = df_scores.hydric_regime.eq("RAINFED")
wastewater_rows = df_scores.hydric_regime.eq("IRRIGATION") & df_scores.water_sources.eq(
    "WASTEWATER (E.G. INDUSTRIAL AND/OR DOMESTIC USE)"
)
calculated_water_rows = maize_rows & ~rainfed_rows & ~wastewater_rows
subterranean_rows = df_scores.water_sources.str.contains(
    "SUBTERRANEAN", case=False, na=False
)
missing_hydric_regime = calculated_water_rows & df_scores.hydric_regime.isna()
unsupported_hydric_regime = calculated_water_rows & ~df_scores.hydric_regime.isin(
    ["RAINFED", "IRRIGATION"]
)

df_scores["flag_esi_water"] = build_score_flag(
    "esi_water",
    crop_rules
    + [
        (
            "MISSING HYDRIC REGIME",
            missing_hydric_regime,
        ),
        (
            "UNSUPPORTED HYDRIC REGIME",
            unsupported_hydric_regime & ~df_scores.hydric_regime.isna(),
        ),
        (
            "MISSING WATER SOURCE",
            calculated_water_rows & df_scores.water_sources.isna(),
        ),
        (
            "SUBTERRANEAN WATER SOURCE NOT SCORED",
            calculated_water_rows & subterranean_rows,
        ),
        (
            "MISSING WATER USE",
            calculated_water_rows & df_scores.water_cons_hyb_ml_dam3.isna(),
        ),
        (
            "MISSING WATER ESI COEFFICIENT",
            calculated_water_rows & df_scores.ro_metric_scaled.isna(),
        ),
    ],
    available_notes=[
        ("RAINFED ZERO", maize_rows & rainfed_rows),
        ("WASTEWATER ZERO", maize_rows & wastewater_rows),
        ("HYDRIC REGIME MISSING", missing_hydric_regime),
        (
            "HYDRIC REGIME UNSUPPORTED",
            unsupported_hydric_regime & ~df_scores.hydric_regime.isna(),
        ),
        (
            "WATER SOURCE MISSING",
            calculated_water_rows & df_scores.water_sources.isna(),
        ),
    ],
)

df_scores["flag_esi_climate"] = build_score_flag(
    "esi_climate",
    crop_rules
    + [
        (
            "INCOMPLETE NET GHG EMISSIONS",
            maize_rows & ~df_scores.flag_net_ghg.eq("AVAILABLE"),
        ),
        (
            "MISSING CLIMATE COEFFICIENT",
            maize_rows & df_scores.cc_metric_rate_scaled.isna(),
        ),
    ],
)

df_scores["flag_esi_climate_wfuel"] = build_score_flag(
    "esi_climate_wfuel",
    crop_rules
    + [
        (
            "INCOMPLETE NET GHG EMISSIONS WITH FUEL",
            maize_rows & ~df_scores.flag_net_ghg_wfuel.eq("AVAILABLE"),
        ),
        (
            "MISSING CLIMATE COEFFICIENT",
            maize_rows & df_scores.cc_metric_rate_scaled.isna(),
        ),
    ],
)

df_scores["flag_esi_climate_albedo"] = build_score_flag(
    "esi_climate_albedo",
    crop_rules
    + [
        (
            "MISSING ALBEDO EMISSIONS",
            maize_rows & df_scores.albedo_emissions_tco2eq.isna(),
        ),
        (
            "MISSING CLIMATE COEFFICIENT",
            maize_rows & df_scores.cc_metric_rate_scaled.isna(),
        ),
    ],
)

df_scores["flag_esi_total"] = build_score_flag(
    "esi_total",
    crop_rules
    + [
        ("INCOMPLETE LAND SCORE", maize_rows & df_scores.esi_land.isna()),
        ("INCOMPLETE WATER SCORE", maize_rows & df_scores.esi_water.isna()),
        ("INCOMPLETE CLIMATE SCORE", maize_rows & df_scores.esi_climate.isna()),
        (
            "INCOMPLETE ALBEDO SCORE",
            maize_rows & df_scores.esi_climate_albedo.isna(),
        ),
    ],
)

df_scores["flag_esi_total_wfuel"] = build_score_flag(
    "esi_total_wfuel",
    crop_rules
    + [
        ("INCOMPLETE LAND SCORE", maize_rows & df_scores.esi_land.isna()),
        ("INCOMPLETE WATER SCORE", maize_rows & df_scores.esi_water.isna()),
        (
            "INCOMPLETE CLIMATE SCORE WITH FUEL",
            maize_rows & df_scores.esi_climate_wfuel.isna(),
        ),
        (
            "INCOMPLETE ALBEDO SCORE",
            maize_rows & df_scores.esi_climate_albedo.isna(),
        ),
    ],
)


# ESI intensity variables
def compute_esi_intensity_vars(df):
    """
    Automates ESI intensity calculations.
    """
    numerators = {
        "total": "esi_total",
        "total_wfuel": "esi_total_wfuel",
        "land": "esi_land",
        "water": "esi_water",
        "climate": "esi_climate",
        "climate_wfuel": "esi_climate_wfuel",
        "climate_albedo": "esi_climate_albedo",
    }

    denominators = {"area": "area_km2", "revenue": "total_revenue_mxn"}

    for prefix, metric_col in numerators.items():
        for suffix, denom_col in denominators.items():
            new_col_name = f"{prefix}_esi_intensity_{suffix}"

            df[new_col_name] = np.where(
                df[denom_col] > 0, df[metric_col].div(df[denom_col]), np.nan
            )

    return df


df_scores = compute_esi_intensity_vars(df_scores)

df_scores["flag_esi_intensity_area"] = np.select(
    [df_scores.area_km2.isna(), df_scores.area_km2.le(0)],
    ["MISSING AREA", "NON-POSITIVE AREA"],
    default="AVAILABLE",
)

df_scores["flag_esi_intensity_revenue"] = np.select(
    [df_scores.total_revenue_mxn.isna(), df_scores.total_revenue_mxn.le(0)],
    ["MISSING REVENUE", "NON-POSITIVE REVENUE"],
    default="AVAILABLE",
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
df_scores.to_feather(PATH_PROCESSED_TABLES / "table_12_esi_scores.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    df_scores.to_sql(
        name="table_12_esi_scores",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
