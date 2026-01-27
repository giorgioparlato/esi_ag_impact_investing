"""
Computation of ESI scores.

Authors:
    Dario Marone
    Giorgio Parlato

Date:
    2026-01-26
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Paths and constants
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
# Import data
# -----------------------------------------------------------------------------
# Connection
conn = sqlite3.connect(PATH_SQL_DB)

# Multi-table query
query = """
SELECT 
    reference_table.farmer_id, 
    reference_table.plot_id, 
    reference_table.logbook_id, 
    reference_table.practice_id, 
    reference_table.year,
    reference_table.practices_type,

    reference_table_enriched.luc_metric_scaled,
    reference_table_enriched.ro_metric_scaled,
    reference_table_enriched.cc_metric_rate_scaled,

    agronomic_features.area_km2,
    water_features.hydric_regime,
    water_features.water_cons_hyb_dam3,
    emissions_net_ghg.net_ghg_emissions_tco2eq,
    economic_features.total_revenue_mxn

FROM reference_table

LEFT JOIN reference_table_enriched USING (practice_id)
LEFT JOIN agronomic_features USING (practice_id)
LEFT JOIN water_features USING (practice_id)
LEFT JOIN emissions_net_ghg USING (practice_id)
LEFT JOIN economic_features USING (practice_id)
"""

df_in = pd.read_sql_query(query, conn)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Compute ESI scores
df_scores = df_in.copy()

# Land ESI scores
df_scores["esi_land"] = df_scores["area_km2"] * df_scores["luc_metric_scaled"]

# Water ESI scores
# Conditions
water_conditions = [
    df_scores.water_cons_hyb_dam3.notna() & df_scores.ro_metric_scaled.notna(),
    df_scores.hydric_regime.eq("RAINFED"),
]

# Choices
water_values = [df_scores.water_cons_hyb_dam3 * df_scores.ro_metric_scaled, 0]

# Compute water ESI
df_scores["esi_water"] = np.select(water_conditions, water_values, default=np.nan)

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
