"""
Preprocessing and emission estimation for soil organic carbon.
Main operations:
    - standard cleaning steps
    - creation of tillage level column
    - assignment of IPCC coefficients for estimating SOC
    - estimation of SOC change and corresponding (negative) emissions
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
import rasterio 

from paths import (
    PATH_REF_TABLE_ENRICHED,
    PATH_AGRONOMIC_FEATURES,
    PATH_AG_INPUTS,
    PATH_CROP_RESIDUES,
    PATH_CIMMYT_ACTIVITIES,
    SHEET_CIMMYT_ACTIVITIES,
    PATH_PROCESSED_TABLES,
    PATH_SQL_DB,
    PATH_SOIL_CLASS_IPCC,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KEY_IDS = ["farmer_id", "plot_id", "logbook_id", "practice_id", "activity_id"]

KEY_VARS = ["year", "tillage_bem", "tillage_land"]

# Map IPCC coefficients based on all the variables
# Relative stock change factors for different management activities on cropland
# Source: table 5.5 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch05_Cropland.pdf
# Land use coefficients (FLU) for long-term cultivated class
FLU_MAPPING = {
    # Temperate/Boreal
    "Warm Temperate Moist": 0.69,
    "Warm Temperate Dry": 0.76,
    "Cool Temperate Moist": 0.70,
    "Cool Temperate Dry": 0.77,
    # Tropical
    "Tropical Wet": 0.83,
    "Tropical Moist": 0.83,
    "Tropical Dry": 0.92,
    "Tropical Montane": 0.87, # value missing in table, we approximate with the average of the other tropical classes
}

# Tillage coefficients (FMG)
# BAU coefficient
FTILL_BAU = 1
FTILL_MAPPING = {
    # Traditional tillage = Full tillage
    "TRADITIONAL TILLAGE": {"default": 1.00},
    # Minimum tillage = Reduced
    "MINIMUM-REDUCED TILLAGE": {
        "Cool Temperate Dry": 0.98,
        "Cool Temperate Moist": 1.04,
        "Warm Temperate Dry": 0.99,
        "Warm Temperate Moist": 1.05,
        "Tropical Dry": 0.99,
        "Tropical Moist": 1.04,
        "Tropical Wet": 1.04,
        "Tropical Montane": 1.015, # value missing in table, we approximate with the average of the other tropical classes
    },

    # Zero tillage = No-till
    "ZERO TILLAGE": {
        "Cool Temperate Dry": 1.03,
        "Cool Temperate Moist": 1.09,
        "Warm Temperate Dry": 1.04,
        "Warm Temperate Moist": 1.10,
        "Tropical Dry": 1.04,
        "Tropical Moist": 1.10,
        "Tropical Wet": 1.10,
        "Tropical Montane": 1.07, # value missing in table, we approximate with the average of the other tropical classes
    },
   # Averaging for combinations
    "MINIMUM-REDUCED TILLAGE | TRADITIONAL TILLAGE": {
        "Cool Temperate Dry": (0.98 + 1) / 2,
        "Cool Temperate Moist": (1.04 + 1) / 2,
        "Warm Temperate Dry": (0.99 + 1) / 2,
        "Warm Temperate Moist": (1.05 + 1) / 2,
        "Tropical Dry": (0.99 + 1) / 2,
        "Tropical Moist": (1.04 + 1) / 2,
        "Tropical Wet": (1.04 + 1) / 2,
        "Tropical Montane": (1.015 + 1) / 2,
    },
    "MINIMUM-REDUCED TILLAGE | ZERO TILLAGE": {
        "Cool Temperate Dry": (0.98 + 1.03) / 2,
        "Cool Temperate Moist": (1.04 + 1.09) / 2,
        "Warm Temperate Dry": (0.99 + 1.04) / 2,
        "Warm Temperate Moist": (1.05 + 1.10) / 2,
        "Tropical Dry": (0.99 + 1.04) / 2,
        "Tropical Moist": (1.04 + 1.10) / 2,
        "Tropical Wet": (1.04 + 1.10) / 2,
        "Tropical Montane": (1.015 + 1.07) / 2,
    },
    "MINIMUM-REDUCED TILLAGE | TRADITIONAL TILLAGE | ZERO TILLAGE": {
        "Cool Temperate Dry": (0.98 + 1 + 1.03) / 3,
        "Cool Temperate Moist": (1.04 + 1 + 1.09) / 3,
        "Warm Temperate Dry": (0.99 + 1 + 1.04) / 3,
        "Warm Temperate Moist": (1.05 + 1 + 1.10) / 3,
        "Tropical Dry": (0.99 + 1 + 1.04) / 3,
        "Tropical Moist": (1.04 + 1 + 1.10) / 3,
    },
}

# Input coefficients (FI)
FI_MAPPING = {
    # LOW
    ("Low", "Cool Temperate Dry"): 0.95,
    ("Low", "Cool Temperate Moist"): 0.92,
    ("Low", "Tropical Dry"): 0.95,
    ("Low", "Tropical Moist"): 0.92,
    ("Low", "Tropical Wet"): 0.92,
    ("Low", "Tropical Montane"): 0.94,
    # MEDIUM — all get 1.00
    ("Medium", "any"): 1.00,
    # HIGH without manure
    ("High_no_manure", "Cool Temperate Dry"): 1.04,
    ("High_no_manure", "Cool Temperate Moist"): 1.11,
    ("High_no_manure", "Warm Temperate Dry"): 1.04,
    ("High_no_manure", "Warm Temperate Moist"): 1.11,
    ("High_no_manure", "Tropical Dry"): 1.04,
    ("High_no_manure", "Tropical Moist"): 1.11,
    ("High_no_manure", "Tropical Wet"): 1.11,
    ("High_no_manure", "Tropical Montane"): 1.08,
    # HIGH with manure
    ("High_with_manure", "Cool Temperate Dry"): 1.37,
    ("High_with_manure", "Cool Temperate Moist"): 1.44,
    ("High_with_manure", "Tropical Dry"): 1.37,
    ("High_with_manure", "Tropical Moist"): 1.44,
    ("High_with_manure", "Tropical Wet"): 1.44,
    ("High_with_manure", "Tropical Montane"): 1.41,
}

## the exact-match MANURE_TYPES set is replaced by a robust token match
## (the original set missed real combinations such as "Cattle Manure | Liquid Biofertilizer").
ANIMAL_MANURE = (
    "Cattle Manure",
    "Sheep Manure",
    "Poultry Manure",
    "Swine Manure",
    "Horses/Mules/Asses Manure",
)

## CHANGES: floor (kg/ha of product) below which a synthetic-fertilizer amount is NOT counted as
## "mineral fertilization" (applications run as low as 0.5 kg/ha)
SYNTH_FLOOR_KG_HA = 50

## assign the IPCC default reference SOC stock (SOCref) by climate x soil class 
## Legend of the IPCC soil-types raster (codes -> categories);
SOIL_CODE_MAP = {
    0: None,    # masked (oceans)
    1: "GG",    # glaciers / land ice
    2: "HAC",   # High activity clay
    3: "LAC",   # Low activity clay
    4: "ND",    # no data
    5: "ORG",   # organic soils -> method N/A
    6: "POD",   # spodic
    7: "RK",    # rock outcrops
    8: "SAN",   # sandy
    9: "ST",    # salt flats
    10: "VOL",  # volcanic
    11: "WET",  # wetland
    12: "WR",   # water
    13: "IS",   # islands
}

# Mineral soil categories that have an IPCC Table 2.3 reference SOC stock
SOCREF_SOILS = {"HAC", "LAC", "SAN", "POD", "VOL", "WET"}

# IPCC 2019 Vol 4, Ch 2, Table 2.3 — default reference SOC stocks under native vegetation
# (tonnes C ha^-1, 0-30 cm). Keyed on (original climate class, soil category). Missing combos are
# simply absent -> lookup returns NaN (never silently contributes a zero).
SOCREF_IPCC = {
    ("Warm Temperate Dry", "HAC"): 24, ("Warm Temperate Dry", "LAC"): 19,
    ("Warm Temperate Dry", "SAN"): 10, ("Warm Temperate Dry", "VOL"): 84,
    ("Warm Temperate Dry", "WET"): 74,
    # POD = NO for Warm Temperate Dry -> omitted
 
    ("Warm Temperate Moist", "HAC"): 64, ("Warm Temperate Moist", "LAC"): 55,
    ("Warm Temperate Moist", "SAN"): 36, ("Warm Temperate Moist", "POD"): 143,
    ("Warm Temperate Moist", "VOL"): 138, ("Warm Temperate Moist", "WET"): 135,
 
    ("Cool Temperate Dry", "HAC"): 43, ("Cool Temperate Dry", "LAC"): 33,
    ("Cool Temperate Dry", "SAN"): 13, ("Cool Temperate Dry", "VOL"): 20,
    ("Cool Temperate Dry", "WET"): 87,
    # POD = NO for Cool Temperate Dry -> omitted
 
    ("Cool Temperate Moist", "HAC"): 81, ("Cool Temperate Moist", "LAC"): 76,
    ("Cool Temperate Moist", "SAN"): 51, ("Cool Temperate Moist", "POD"): 128,
    ("Cool Temperate Moist", "VOL"): 136, ("Cool Temperate Moist", "WET"): 128,
 
    ("Tropical Dry", "HAC"): 21, ("Tropical Dry", "LAC"): 19,
    ("Tropical Dry", "SAN"): 9, ("Tropical Dry", "VOL"): 50,
    ("Tropical Dry", "WET"): 22,
    # POD = NA for Tropical Dry -> omitted
 
    ("Tropical Moist", "HAC"): 40, ("Tropical Moist", "LAC"): 38,
    ("Tropical Moist", "SAN"): 27, ("Tropical Moist", "VOL"): 70,
    ("Tropical Moist", "WET"): 68,
    # POD = NA for Tropical Moist -> omitted
 
    ("Tropical Wet", "HAC"): 60, ("Tropical Wet", "LAC"): 52,
    ("Tropical Wet", "SAN"): 46, ("Tropical Wet", "VOL"): 77,
    ("Tropical Wet", "WET"): 49,
    # POD = NA for Tropical Wet -> omitted
 
    ("Tropical Montane", "HAC"): 51, ("Tropical Montane", "LAC"): 44,
    ("Tropical Montane", "SAN"): 52, ("Tropical Montane", "VOL"): 96,
    ("Tropical Montane", "WET"): 82,
    # POD = NA for Tropical Montane -> omitted
 
    ("Boreal", "HAC"): 63, ("Boreal", "SAN"): 10, ("Boreal", "POD"): 117,
    ("Boreal", "VOL"): 20, ("Boreal", "WET"): 116,
    # LAC = NA for Boreal -> omitted
}
# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Activities dataset
activities_df_raw = pd.read_excel(
    io=PATH_CIMMYT_ACTIVITIES,
    sheet_name=SHEET_CIMMYT_ACTIVITIES,
    header=1,
)

# Climate zones and SOC reference values
soc_df_ref = pd.read_feather(
    PATH_REF_TABLE_ENRICHED,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "latitude_n",  
        "longitude_w", 
        "climate_class",  
        "climate_class_corrected",
        "soc_ref",
    ],
)

# sample the IPCC soil-types raster at each farm and assign the IPCC default reference SOC  stock (climate x soil, Table 2.3)
# The GSOCseq value sampled in script 02 is kept as`soc_ref_gsocseq` for comparison, and `soc_ref` refers to the IPCC value 
_coords = list(zip(soc_df_ref["longitude_w"].astype(float), soc_df_ref["latitude_n"].astype(float)))
with rasterio.open(PATH_SOIL_CLASS_IPCC) as _src:
    soc_df_ref["soil_code_ipcc"] = [v[0] for v in _src.sample(_coords)]
soc_df_ref["soil_category_ipcc"] = soc_df_ref["soil_code_ipcc"].map(SOIL_CODE_MAP)


def lookup_socref_ipcc(row):
    """IPCC Table 2.3 reference SOC stock for the (climate, soil) pair; NaN where undefined
    (organic/water/non-soil categories, or combos absent from the table)."""
    category = row["soil_category_ipcc"]
    if category not in SOCREF_SOILS:
        return np.nan
    return SOCREF_IPCC.get((row["climate_class"], category), np.nan)


soc_df_ref["soc_ref_ipcc"] = soc_df_ref.apply(lookup_socref_ipcc, axis=1).astype(float)
# Keep GSOCseq value for reference; use the IPCC value as the active SOCref
soc_df_ref = soc_df_ref.rename(columns={"soc_ref": "soc_ref_gsocseq"})
soc_df_ref["soc_ref"] = soc_df_ref["soc_ref_ipcc"]


# Areas
soc_agronomic_df = pd.read_feather(
    PATH_AGRONOMIC_FEATURES,
    columns=["farmer_id", "plot_id", "logbook_id", "practice_id", "area_ha"],
)

# Organic fertilizers categories and nitrogen amounts
soc_ag_inputs_df = pd.read_feather(
    PATH_AG_INPUTS,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "categories_organic",
        "nitrogen_org_t_ha",
    ],
)

# Crop residues
soc_crop_res_df = pd.read_feather(
    PATH_CROP_RESIDUES,
    columns=[
        "farmer_id",
        "plot_id",
        "logbook_id",
        "practice_id",
        "residues_returned_perc_harvest",
        "maize_residues_returned_dm_t_ha",
    ],
)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Clean column names
activities_df_clean_columns = activities_df_raw.copy()
activities_df_clean_columns = activities_df_clean_columns.clean_names(
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
activities_df_clean_columns.rename(
    columns={
        "innovation_conventional_practices_id": "practice_id",
    },
    inplace=True,
)

# Select variables of interest
activities_df_variables = activities_df_clean_columns[KEY_IDS + KEY_VARS].copy()

# Convert id columns to string type
activities_df_dtypes = activities_df_variables.copy()
activities_df_dtypes[KEY_IDS] = activities_df_dtypes[KEY_IDS].astype("string")

# Clean year column
activities_df_year = activities_df_dtypes.copy()
activities_df_year["year"] = activities_df_year["year"].apply(
    lambda x: str(x)[0:4] if pd.notna(x) and isinstance(x, str) else x
)
activities_df_year["year"] = pd.to_numeric(activities_df_year["year"], errors="coerce")
activities_df_year["year"] = activities_df_year["year"].astype("Int64")


# Merge tillage_bem and tillage_land columns
soc_tillage_df = activities_df_year.copy()
soc_tillage_df["tillage"] = np.where(
    soc_tillage_df["tillage_bem"].notna(),
    soc_tillage_df["tillage_bem"],
    soc_tillage_df["tillage_land"],
)

# Drop the original tillage columns
soc_tillage_df.drop(columns=["tillage_bem", "tillage_land"], inplace=True)


# Aggregate at the practices id level
def join_unique(x):
    return " | ".join(sorted(x.dropna().unique()))


soc_tillage_df_agg = soc_tillage_df.copy()
soc_tillage_df_agg = soc_tillage_df_agg.groupby(
    ["farmer_id", "plot_id", "logbook_id", "practice_id", "year"], as_index=False
).agg(tillage=("tillage", join_unique), tillage_count=("tillage", "nunique"))

# Merge
labels = ["farmer_id", "plot_id", "logbook_id", "practice_id"]
soc_df_merged = soc_df_ref.copy()
soc_df_merged = soc_df_merged.merge(soc_agronomic_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_crop_res_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_ag_inputs_df, on=labels, how="left")
soc_df_merged = soc_df_merged.merge(soc_tillage_df_agg, on=labels, how="left")


# Assigning input coefficients
# Classify FI level
def classify_fi_level(row):
    """Classify FI level based on organic inputs and crop residues."""
    n_input = row.get("nitrogen_org_t_ha", 0)
    category = row.get("categories_organic", "")
    residues = row.get("residues_returned_perc_harvest", 0)

    # Treat missing values as 0 or empty string
    n_input = 0 if pd.isna(n_input) else n_input
    residues = 0 if pd.isna(residues) else residues
    category = "" if pd.isna(category) else category

    # Check conditions in order of specificity
    if category in MANURE_TYPES and n_input > 0 and residues >= 80:
        return "High_with_manure"
    elif n_input == 0 and residues >= 80:
        return "High_no_manure"
    elif residues > 40 or (n_input > 0 and residues > 20):
        return "Medium"

    return "Low"


soc_df_mapping = soc_df_merged.copy()

soc_df_mapping["fi_level"] = soc_df_mapping.apply(classify_fi_level, axis=1)


# Assign FI value
def map_fi(row):
    key = (row["fi_level"], row["climate_class_corrected"])
    if row["fi_level"] == "Medium":
        return 1.00
    return FI_MAPPING.get(key, None)


soc_df_mapping["fi"] = soc_df_mapping.apply(map_fi, axis=1)


# Assign FI values for business as usual - Low inputs
def map_fi_bau(row):
    climate = row.get("climate_class_corrected")
    return FI_MAPPING.get(("Low", climate), None)


soc_df_mapping["fi_bau"] = soc_df_mapping.apply(map_fi_bau, axis=1)

# Assign FLU directly from climate_class
soc_df_mapping["flu"] = soc_df_mapping["climate_class_corrected"].map(FLU_MAPPING)

# Fill missing values for tillage with 'TRADITIONAL TILLAGE'
soc_df_mapping["tillage"] = soc_df_mapping["tillage"].fillna("TRADITIONAL TILLAGE")


def map_ftill(row):
    till_type = row.get("tillage")
    climate = row.get("climate_class_corrected", "")

    if pd.isna(till_type) or till_type == "TRADITIONAL TILLAGE":
        return 1.00

    return FTILL_MAPPING.get(till_type, {}).get(climate, 1.00)


# Apply the mapping
soc_df_mapping["ftill"] = soc_df_mapping.apply(map_ftill, axis=1)

# SOC estimation
soc_df_estimation = soc_df_mapping.copy()

# SOC under BAU scenario
soc_df_estimation["soc_0"] = (
    soc_df_estimation["soc_ref"]
    * soc_df_estimation["flu"]
    * FTILL_BAU
    * soc_df_estimation["fi_bau"]
    * soc_df_estimation["area_ha"]
)

# SOC under current scenario
soc_df_estimation["soc_1"] = (
    soc_df_estimation["soc_ref"]
    * soc_df_estimation["flu"]
    * soc_df_estimation["ftill"]
    * soc_df_estimation["fi"]
    * soc_df_estimation["area_ha"]
)

# SOC change (annual, over a 20 years period)
soc_df_estimation["soc_delta"] = (
    soc_df_estimation["soc_1"] - soc_df_estimation["soc_0"]
) / 20

# Conversion to (negative) carbon dioxide equivalents, to reflect carbon sequestration
# Source: https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator#:~:text=To%20convert%20a%20quantity%20of,carbon%20dioxide%2C%20multiply%20by%203.67.
soc_df_estimation["soc_delta_t_co2eq"] = soc_df_estimation["soc_delta"] * 3.67 * -1

# Calculate on a per hectare basis
soc_df_estimation["soc_delta_t_co2eq_ha"] = (
    soc_df_estimation["soc_delta"] * 3.67 * -1 / soc_df_estimation["area_ha"]
)

# Extra tillage variable for conservation agriculture dummy
soc_df_estimation["tillage_ca_dummy"] = np.where(
    soc_df_estimation["tillage"].str.contains("MINIMUM-REDUCED TILLAGE|ZERO TILLAGE"),
    1,
    0,
)

# Conservation agriculture dummy based on FAO definition
# Source: FAO CA principles https://www.fao.org/conservation-agriculture/en/
# Source: threshold for crop residues in accordance with FAO https://doi.org/10.1007/s13593-017-0432-z
soc_df_estimation["ca_fao_dummy"] = np.where(
    (soc_df_estimation["tillage_ca_dummy"] == 1)
    & (soc_df_estimation["maize_residues_returned_dm_t_ha"] >= 2),
    1,
    0,
)

# Final DataFrame
soc_df_final = soc_df_estimation.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
).copy()

new_index = [
    "farmer_id",
    "plot_id",
    "logbook_id",
    "practice_id",
    "year",
    "area_ha",
    "climate_class_corrected",
    "flu",
    "tillage",
    "tillage_count",
    "tillage_ca_dummy",
    "ftill",
    "categories_organic",
    "nitrogen_org_t_ha",
    "residues_returned_perc_harvest",
    "maize_residues_returned_dm_t_ha",
    "ca_fao_dummy",
    "fi_bau",
    "fi",
    "soc_ref",
    "soc_0",
    "soc_1",
    "soc_delta",
    "soc_delta_t_co2eq_ha",
    "soc_delta_t_co2eq",
]

soc_df_final = soc_df_final.reindex(columns=new_index)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write feather file
soc_df_final.to_feather(PATH_PROCESSED_TABLES / "table_09_emissions_soc.feather")

# Write to SQL database
with sqlite3.connect(PATH_SQL_DB) as con:
    soc_df_final.to_sql(
        name="table_09_emissions_soc",
        con=con,
        if_exists="replace",
        index=False,
    )

# END
print("Execution complete.")
