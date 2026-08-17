"""
Computation of net GHG emissions based on conditional logic.
Main operations:
    - query input data from the SQL database
    - calculate the net GHG emissions
    - export results as feather file and to SQL database

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

    table_03_agronomic_features.area_ha,

    table_07_emissions_ag_inputs.product_types,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_app,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_prod,
    table_07_emissions_ag_inputs.flag_missing_emissions_syn_tran,
    table_07_emissions_ag_inputs.flag_missing_emissions_org_app,
    table_07_emissions_ag_inputs.flag_imputed_amount,
    
    table_07_emissions_ag_inputs.synth_input_prod_tco2eq,
    table_07_emissions_ag_inputs.synth_input_tran_dist_tco2eq,
    table_07_emissions_ag_inputs.synth_input_app_tco2eq,
    table_07_emissions_ag_inputs.org_fert_app_tco2eq,

    table_08_emissions_residues.flag_total_pct,
    table_08_emissions_residues.maize_residues_returned_tco2eq,
    table_08_emissions_residues.maize_grain_residues_burnt_tco2eq,

    table_06_emissions_fuel.fuel_cons_status,
    table_06_emissions_fuel.fuel_cons_tco2eq,
    table_05_water_features.irrigation_types,
    table_05_water_features.irr_equip_tco2eq,
    table_05_water_features.irr_app_tco2eq,

    table_09_emissions_soc.soc_delta_tco2eq,
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
    "fuel_cons_tco2eq",
    "irr_equip_tco2eq",
    "irr_app_tco2eq",
    "synth_input_prod_tco2eq",
    "synth_input_tran_dist_tco2eq",
    "synth_input_app_tco2eq",
    "org_fert_app_tco2eq",
    "maize_residues_returned_tco2eq",
    "maize_grain_residues_burnt_tco2eq",
    "soc_delta_tco2eq",
]

# Rows must have at least 1 non-null GHG flux
valid_rows = df[ghg_fluxes].notna().any(axis=1)

# Default formulas for net GHG emissions
net_ghg_emissions_formula = df[ghg_fluxes].sum(axis=1, skipna=True).where(valid_rows)

# Input-emission reasons that indicate an incomplete component estimate.
missing_synthetic_input_pattern = (
    r"\bMISSING EMISSION FACTOR\b"
    r"|MISSING OR UNSUPPORTED UNIT"
    r"|\bUNIT CONVERSION\b"
)
missing_organic_input_pattern = (
    r"\bMISSING Q\b"
    r"|\bMISSING UC\b"
    r"|MISSING OR UNSUPPORTED UNIT"
    r"|\bUNIT CONVERSION\b"
)

# Conditions defining exceptions where net GHG emissions should be missing
net_ghg_emissions_conditions = [
    # Exceptions for fuel consumption
    df.fuel_cons_status.str.contains("UNKNOWN|MISSING|IMPUTED", na=False),
    # Exceptions for irrigation equipment and application
    df.irrigation_types.str.contains("DRIP|SPRINKLER", na=False)
    & df.irr_equip_tco2eq.isna(),
    df.irrigation_types.str.contains("DRIP|SPRINKLER", na=False)
    & df.irr_app_tco2eq.isna(),
    # Exceptions for agricultural inputs
    df.flag_missing_emissions_syn_prod.str.contains(
        missing_synthetic_input_pattern, na=False, regex=True
    ),
    df.flag_missing_emissions_syn_tran.str.contains(
        missing_synthetic_input_pattern, na=False, regex=True
    ),
    df.flag_missing_emissions_syn_app.str.contains(
        missing_synthetic_input_pattern, na=False, regex=True
    ),
    ~df.flag_imputed_amount.str.contains(r"\bREPORTED ONLY\b", na=False, regex=True),
    ~df.flag_missing_emissions_org_app.isin(
        ["AVAILABLE", "NOT APPLICABLE", "AVAILABLE | NOT APPLICABLE"]
    )
    & df.flag_missing_emissions_org_app.str.contains(
        missing_organic_input_pattern, na=False, regex=True
    ),
    # Exceptions for crop residues
    df.flag_total_pct.isin(["MISSING ONLY", "PARTIALLY MISSING"]),
    # Exceptions for SOC
    df.soc_delta_tco2eq.isna(),
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
    net_ghg_emissions_formula - df.fuel_cons_tco2eq.fillna(0)
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
