"""
Computation of ESI scores.
Main operations:
    - query necessary variables from SQL database
    - compute absolute and intensity ESI scores
    - export results as feather file and to SQL database

Note: this script uses **scaled** ESI metrics for improved interpretability
Authors: Dario Marone, Giorgio Parlato
Date: 2026-02-19
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
    table_05_water_features.water_cons_hyb_ml_dam3,
    table_05_water_features.water_sources,
    table_11_emissions_net_ghg.net_ghg_emissions_tco2eq,
    table_11_emissions_net_ghg.net_ghg_emissions_wfuel_tco2eq,
    table_11_emissions_net_ghg.albedo_emissions_tco2eq,
    table_04_economic_features.total_revenue_mxn

FROM table_01_reference

LEFT JOIN table_02_reference_enriched USING (practice_id)
LEFT JOIN table_03_agronomic_features USING (practice_id)
LEFT JOIN table_05_water_features USING (practice_id)
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
df_scores["esi_land"] = df_scores["area_km2"] * df_scores["luc_metric_scaled"]

# Water ESI scores
valid_water_source = ~df_scores.water_sources.str.contains(
    "SUBTERRANEAN", case=False, na=False
)

water_conditions = [
    valid_water_source
    & df_scores.water_cons_hyb_ml_dam3.notna()
    & df_scores.ro_metric_scaled.notna(),
    df_scores.hydric_regime.eq("RAINFED"),
    df_scores.hydric_regime.eq("IRRIGATION")
    & df_scores.water_sources.eq("WASTEWATER (E.G. INDUSTRIAL AND/OR DOMESTIC USE)"),
]

water_choices = [df_scores.water_cons_hyb_ml_dam3 * df_scores.ro_metric_scaled, 0, 0]

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
