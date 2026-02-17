"""
Computation of ESI scores.
Main operations:
    - query necessary variables from SQL database
    - compute absolute and intensity ESI scores
    - export results as feather file and to SQL database

Note: this script uses **scaled** ESI metrics for improved interpretability
Authors: Dario Marone, Giorgio Parlato
Date: 2026-02-13
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
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# Processed tables
PATH_PROCESSED_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Multi-table query
QUERY = """
SELECT 
    table_01_reference_table.farmer_id, 
    table_01_reference_table.plot_id, 
    table_01_reference_table.logbook_id, 
    table_01_reference_table.practice_id, 
    table_01_reference_table.year,
    table_01_reference_table.practices_type,

    table_02_reference_table_enriched.luc_metric_scaled,
    table_02_reference_table_enriched.ro_metric_scaled,
    table_02_reference_table_enriched.cc_metric_rate_scaled,

    table_03_agronomic_features.area_km2,
    table_05_water_features.hydric_regime,
    table_05_water_features.water_cons_hyb_dam3,
    table_11_emissions_net_ghg.net_ghg_emissions_tco2eq,
    table_04_economic_features.total_revenue_mxn

FROM table_01_reference_table

LEFT JOIN table_02_reference_table_enriched USING (practice_id)
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
water_conditions = [
    df_scores.water_cons_hyb_dam3.notna() & df_scores.ro_metric_scaled.notna(),
    df_scores.hydric_regime.eq("RAINFED"),
]
water_choices = [df_scores.water_cons_hyb_dam3 * df_scores.ro_metric_scaled, 0]
df_scores["esi_water"] = np.select(water_conditions, water_choices, default=np.nan)

# Climate ESI scores
df_scores["esi_climate"] = (
    df_scores["net_ghg_emissions_tco2eq"] * df_scores["cc_metric_rate_scaled"]
)

# Total ESI scores
df_scores["esi_total"] = df_scores[["esi_land", "esi_water", "esi_climate"]].sum(
    axis=1, skipna=False, min_count=1
)


# ESI intensity variables
def compute_esi_intensity_vars(df):
    """
    Automates ESI intensity calculations.
    """
    numerators = {
        "total": "esi_total",
        "land": "esi_land",
        "water": "esi_water",
        "climate": "esi_climate",
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
# Write as feather file
df_scores.to_feather(PATH_PROCESSED_TABLES / r"table_12_esi_scores.feather")

# Write to SQL database
con = sqlite3.connect(PATH_SQL_DB)

df_scores.to_sql(name="table_12_esi_scores", con=con, if_exists="replace", index=False)

# END
exit("Execution complete.")
