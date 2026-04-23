"""
Computation of net GHG emissions based on conditional logic.
Main operations:
    - query input data from the SQL database
    - define invalidation conditions
    - calculate the net GHG emissions
    - export results as feather file and to SQL database

Authors: Dario Marone, Giorgio Parlato
Date: 2026-04-23
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

    table_09_emissions_soc.soc_delta_t_co2eq

FROM table_01_reference

LEFT JOIN table_03_agronomic_features USING (practice_id)
LEFT JOIN table_05_water_features USING (practice_id)
LEFT JOIN table_06_emissions_fuel USING (practice_id)
LEFT JOIN table_07_emissions_ag_inputs USING (practice_id)
LEFT JOIN table_08_emissions_residues USING (practice_id)
LEFT JOIN table_09_emissions_soc USING (practice_id)
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

# GHG fluxes except SOC
ghg_fluxes_no_soc = [col for col in ghg_fluxes if col != "soc_delta_t_co2eq"]

# Rows must have at least 1 non-null GHG flux
valid_rows = df[ghg_fluxes].notna().any(axis=1)

# Default formula for net GHG emissions
net_ghg_emissions_formula = df[ghg_fluxes].sum(axis=1, skipna=True).where(valid_rows)

# Exceptions
# Exceptions for fuel consumption
is_fuel_missing = df["fuel_cons_status"].str.contains(
    r"UNKNOWN|MISSING", na=False, regex=True
)

# Exceptions for irrigation equipment and application
is_irrigation_pressurized = df["irrigation_types"].str.contains(
    r"DRIP|SPRINKLER", na=False, regex=True
)
is_irr_equip_missing = is_irrigation_pressurized & df["irr_equip_ml_tco2eq"].isna()
is_irr_app_missing = is_irrigation_pressurized & df["irr_app_ml_tco2eq"].isna()

# Exceptions for agricultural inputs
is_syn_prod_factor_missing = df["flag_missing_emissions_syn_prod"].str.contains(
    r"\bMISSING EMISSION FACTOR\b", na=False, regex=True
)
is_syn_tran_factor_missing = df["flag_missing_emissions_syn_tran"].str.contains(
    r"\bMISSING EMISSION FACTOR\b", na=False, regex=True
)
is_syn_app_factor_missing = df["flag_missing_emissions_syn_app"].str.contains(
    r"\bMISSING EMISSION FACTOR\b", na=False, regex=True
)

amount_reported_only = df["flag_imputed_amount"].str.contains(
    r"\bREPORTED ONLY\b", na=False, regex=True
)
is_amount_invalid = ~amount_reported_only

org_app_status_ok = df["flag_missing_emissions_org_app"].isin(
    ["AVAILABLE", "NOT APPLICABLE", "AVAILABLE | NOT APPLICABLE"]
)
org_app_missing_quantity = df["flag_missing_emissions_org_app"].str.contains(
    r"\bMISSING Q\b", na=False, regex=True
)
is_org_app_invalid = ~org_app_status_ok & org_app_missing_quantity

# Exceptions for crop residues
is_crop_residues_missing = df["flag_total_pct"].isin(
    ["MISSING ONLY", "PARTIALLY MISSING"]
)

# Exceptions for SOC
is_soc_missing = df["soc_delta_t_co2eq"].isna()
is_soc_only_nonnegative = df[ghg_fluxes_no_soc].isna().all(axis=1) & df[
    "soc_delta_t_co2eq"
].ge(0)
is_soc_invalid = is_soc_missing | is_soc_only_nonnegative

# Assemble invalidation masks
full_invalid_conditions = [
    is_fuel_missing,
    is_irr_equip_missing,
    is_irr_app_missing,
    is_syn_prod_factor_missing,
    is_syn_tran_factor_missing,
    is_syn_app_factor_missing,
    is_amount_invalid,
    is_org_app_invalid,
    is_crop_residues_missing,
    is_soc_invalid,
]

nofuel_invalid_conditions = [
    is_irr_equip_missing,
    is_irr_app_missing,
    is_syn_prod_factor_missing,
    is_syn_tran_factor_missing,
    is_syn_app_factor_missing,
    is_amount_invalid,
    is_org_app_invalid,
    is_crop_residues_missing,
    is_soc_invalid,
]

mask_full_invalid = np.logical_or.reduce(full_invalid_conditions)
mask_nofuel_invalid = np.logical_or.reduce(nofuel_invalid_conditions)

df["net_ghg_emissions_wfuel_tco2eq"] = net_ghg_emissions_formula.mask(mask_full_invalid)

df["net_ghg_emissions_tco2eq"] = (
    net_ghg_emissions_formula - df.fuel_cons_t_co2eq.fillna(0)
).mask(mask_nofuel_invalid)

# Per hectare net GHG emissions
valid_area = df["area_ha"].gt(0)

df["net_ghg_emissions_tco2eq_ha"] = (df.net_ghg_emissions_tco2eq / df.area_ha).where(
    valid_area
)

df["net_ghg_emissions_wfuel_tco2eq_ha"] = (
    df.net_ghg_emissions_wfuel_tco2eq / df.area_ha
).where(valid_area)

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
