"""
Computation of net GHG emissions based on conditional logic.
Main operations:
    - query input data from the SQL database
    - calculate the net GHG emissions
    - export results as feather file and to SQL database

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

    table_03_agronomic_features.area_ha,

    table_07_emissions_ag_inputs.product_types,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_app,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_prod,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_tran,
    table_07_emissions_ag_inputs.flag_missing_emissions_org_app,
    table_07_emissions_ag_inputs.flag_imputed_amount,
    
    table_07_emissions_ag_inputs.synth_input_prod_t_co2eq,
    table_07_emissions_ag_inputs.synth_input_tran_dist_t_co2eq,
    table_07_emissions_ag_inputs.synth_input_app_t_co2eq,
    table_07_emissions_ag_inputs.org_fert_app_t_co2eq,

    table_08_emissions_residues.flag_total_pct,
    table_08_emissions_residues.maize_residues_returned_tco2eq,
    table_08_emissions_residues.maize_grain_residues_burnt_tco2eq,

    table_06_emissions_fuel.fuel_cons_status,
    table_06_emissions_fuel.fuel_cons_t_co2eq,
    table_05_water_features.irrigation_types,
    table_05_water_features.irr_equip_ml_tco2eq,
    table_05_water_features.irr_app_ml_tco2eq,

    table_09_emissions_soc.soc_delta_t_co2eq,
    table_10_emissions_albedo.albedo_emissions_tco2eq_ha,
    table_10_emissions_albedo.albedo_emissions_tco2eq

FROM table_01_reference

LEFT JOIN table_03_agronomic_features USING (practice_id)
LEFT JOIN table_05_water_features USING (practice_id)
LEFT JOIN table_06_emissions_fuel USING (practice_id)
LEFT JOIN table_07_emissions_ag_inputs USING (practice_id)
LEFT JOIN table_08_emissions_residues USING (practice_id)
LEFT JOIN table_09_emissions_soc USING (practice_id)
LEFT JOIN table_10_emissions_albedo USING (practice_id)
"""

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Connection
conn = sqlite3.connect(PATH_SQL_DB)

df = pd.read_sql_query(QUERY, conn)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Compute net GHG emissions based on conditions
# GHG fluxes to include
ghg_fluxes = [
    "fuel_cons_t_co2eq",
    "irr_equip_ml_tco2eq",
    "irr_app_ml_tco2eq",
    "synth_input_prod_t_co2eq",
    "synth_input_tran_dist_t_co2eq",
    "synth_input_app_t_co2eq",
    "org_fert_app_t_co2eq",
    "maize_residues_returned_tco2eq",
    "maize_grain_residues_burnt_tco2eq",
    "soc_delta_t_co2eq",
]

# Rows must have at least 1 non-null GHG flux
valid_rows = df[ghg_fluxes].notna().any(axis=1)

# Default formulas for net GHG emissions
net_ghg_emissions_formula = df[ghg_fluxes].sum(axis=1, skipna=True).where(valid_rows)

SYN_OK_TOKENS = {"NOT APPLICABLE", "NOT MISSING"}
## for application we allow NO INPUTS MISSING (ESTIMATE ERROR), because this occurs for all and only pesticides, where there are no emissions from application
SYN_APP_OK_TOKENS = {"NOT APPLICABLE", "NOT MISSING", "NO INPUTS MISSING (ESTIMATE ERROR)"}
ORG_OK_TOKENS = {"NOT APPLICABLE", "AVAILABLE"}


def na_flag(series: pd.Series, ok_tokens: set) -> pd.Series:
    """True if aggregated flag contains any token outside `ok_tokens`. NaN treated as na."""
    return series.fillna("__MISSING__").map(
        lambda s: not set(s.split(" | ")).issubset(ok_tokens)
    )

na_syn_prod = na_flag(df.flag_missing_emissions_syn_prod, SYN_OK_TOKENS)
na_syn_tran = na_flag(df.flag_missing_emissions_syn_tran, SYN_OK_TOKENS)
na_syn_app = na_flag(df.flag_missing_emissions_syn_app, SYN_APP_OK_TOKENS)
na_org_app = na_flag(df.flag_missing_emissions_org_app, ORG_OK_TOKENS)
na_imputed_amount = ~df.flag_imputed_amount.fillna("").eq("REPORTED ONLY")

net_ghg_emissions_conditions = [
    # Fuel
    df.fuel_cons_status.str.contains("UNKNOWN|MISSING", na=False),
    # Irrigation
    df.irrigation_types.str.contains("DRIP|SPRINKLER", na=False)
    & df.irr_equip_ml_tco2eq.isna(),
    df.irrigation_types.str.contains("DRIP|SPRINKLER", na=False)
    & df.irr_app_ml_tco2eq.isna(),
    # Fertilizer flags — allow-list
    na_syn_prod,
    na_syn_tran,
    na_syn_app,
    na_org_app,
    na_imputed_amount,
    # Exceptions for crop residues
    df.flag_total_pct.isin(["MISSING ONLY", "PARTIALLY MISSING"]),
    # Exceptions for SOC
    df.soc_delta_t_co2eq.isna(),
]

# Choices for GHG emissions
net_ghg_emissions_choices = [
    # Choices for fuel consumption
    np.nan,
    # Choices for irrigation application and equipment
    np.nan,
    np.nan,
    # Choices for agricultural inputs
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    # Choices for crop residues
    np.nan,
    # Choices for SOC
    np.nan,
]

mask_full_invalid = np.logical_or.reduce(net_ghg_emissions_conditions)

mask_nofuel_invalid = np.logical_or.reduce(
    net_ghg_emissions_conditions[1:]  # drop fuel condition
)

df["net_ghg_emissions_wfuel_tco2eq"] = net_ghg_emissions_formula.mask(mask_full_invalid)

df["net_ghg_emissions_tco2eq"] = (
    net_ghg_emissions_formula - df.fuel_cons_t_co2eq.fillna(0)
).mask(mask_nofuel_invalid)

# Per hectare net GHG emissions
df["net_ghg_emissions_tco2eq_ha"] = df.net_ghg_emissions_tco2eq / df.area_ha
df["net_ghg_emissions_wfuel_tco2eq_ha"] = df.net_ghg_emissions_wfuel_tco2eq / df.area_ha

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
df.to_feather(PATH_PROCESSED_TABLES / "table_11_emissions_net_ghg.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    df.to_sql(
        name="table_11_emissions_net_ghg",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
