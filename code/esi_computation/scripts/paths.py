# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
import os
from pathlib import Path

# Optional project root for code-relative operations
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
# Uses environment variable if available; otherwise falls back to local default
PATH_DATA = Path(
    os.environ.get("ESI_CIMMYT_DATA", r"C:\Users\darmar07.KVA\esi_cimmyt\data")
).expanduser()

# -----------------------------------------------------------------------------
# Raw data
# -----------------------------------------------------------------------------
# Path to CIMMYT folder with raw data
PATH_CIMMYT = PATH_DATA / "raw" / "cimmyt" / "dataverse_files"

PATH_CIMMYT_LOGBOOK = PATH_CIMMYT / "1.-Farmer_Plot_Logbook_2012-2022_03.xlsx"
PATH_CIMMYT_AGRONOMIC_FEATURES = (
    PATH_CIMMYT / "2.-Sowing_harvest_yields_2012-2022_02.xlsx"
)
PATH_CIMMYT_ACTIVITIES = PATH_CIMMYT / "3._Labor_harvest_activities_2012-2022_02.xlsx"
PATH_CIMMYT_INPUTS = PATH_CIMMYT / "4.-Agricultural supplies_2012-2022_01.xlsx"
PATH_CIMMYT_ECONOMICS = PATH_CIMMYT / "6.-Costs and revenues_2012-2022_02.xlsx"
PATH_CIMMYT_WATER = PATH_CIMMYT / "5.-Irrigacion_2012-2022_01.xlsx"

# Sheet names
SHEET_CIMMYT_LOGBOOK = "Data"
SHEET_CIMMYT_AGRONOMIC_FEATURES = "Data"
SHEET_CIMMYT_ACTIVITIES = "Data"
SHEET_CIMMYT_ECONOMICS = "Data"
SHEET_CIMMYT_SYNTHETIC_INPUTS = "Agricultural supplies"
SHEET_CIMMYT_ORGANIC_INPUTS = "Organic Fertilization"
SHEET_CIMMYT_WATER = "Data"

# -----------------------------------------------------------------------------
# Geospatial layers
# -----------------------------------------------------------------------------
# Path to geospatial layer for continents
# Source: https://data.niaid.nih.gov/resources?id=zenodo_10778078
PATH_CONTINENTS = (
    PATH_DATA
    / "raw"
    / "geospatial"
    / "vector"
    / "world_geographic_regions"
    / "World_Geographic_Regionst.shp"
)

# Path to vegetation types raster
# Source: model output (LPJmL)
PATH_VEGETATION_TYPES = (
    PATH_DATA
    / "raw"
    / "geospatial"
    / "raster"
    / "vegetation_types"
    / "vegmap_nolu_late_20251117.asc"
)

# Path to climatic zones raster
# Reference: European Soil Data Centre (ESDAC)
PATH_CLIMATE_ZONES = PATH_DATA / "raw" / "geospatial" / "raster" / "climate_zones"

# Path to SOC reference values raster
# Reference: FAO and ITPS. 2018. Global Soil Organic Carbon Map
PATH_SOC_REF = PATH_DATA / "raw" / "geospatial" / "raster" / "soil_organic_carbon"

PATH_SOIL_CLASS = PATH_DATA / "raw" / "geospatial" / "raster" / "soil_classes.tif"

# -----------------------------------------------------------------------------
# Model inputs
# -----------------------------------------------------------------------------
# Path to ESI coefficients
# Source: model output (LPJmL, processed in Excel)
PATH_ESI_COEFFICIENTS = (
    PATH_DATA / "raw" / "esi_coefficients" / "esi_coefficients_table_20260127.xlsx"
)

# Path to water tariffs
# Source: https://www.gob.mx/conagua/documentos/ley-federal-de-derechos-386262
PATH_WATER_TARIFFS = PATH_DATA / "raw" / "water" / "water_tariffs_conagua.xlsx"

# -----------------------------------------------------------------------------
# Processed tables
# -----------------------------------------------------------------------------
PATH_PROCESSED_TABLES = PATH_DATA / "processed" / "tables"
PATH_REFERENCE_TABLE = PATH_PROCESSED_TABLES / "table_01_reference.feather"
PATH_REF_TABLE_ENRICHED = PATH_PROCESSED_TABLES / "table_02_reference_enriched.feather"
PATH_AGRONOMIC_FEATURES = PATH_PROCESSED_TABLES / "table_03_agronomic_features.feather"
PATH_WATER_RANDOM_FOREST = PATH_PROCESSED_TABLES / "irrigation_df_ml_imputed.feather"
PATH_AG_INPUTS = PATH_PROCESSED_TABLES / "table_07_emissions_ag_inputs.feather"
PATH_CROP_RESIDUES = PATH_PROCESSED_TABLES / "table_08_emissions_residues.feather"

# -----------------------------------------------------------------------------
# SQL Database
# -----------------------------------------------------------------------------
PATH_SQL_DB = PATH_DATA / "processed" / "sql_db" / "cimmyt.db"
PATH_SQL_DB_EXT = PATH_DATA / "processed" / "sql_db" / "cimmytExt.db"
# -----------------------------------------------------------------------------
# Ensure required directories exist
# -----------------------------------------------------------------------------
PATH_PROCESSED_TABLES.mkdir(parents=True, exist_ok=True)
PATH_SQL_DB.parent.mkdir(parents=True, exist_ok=True)
