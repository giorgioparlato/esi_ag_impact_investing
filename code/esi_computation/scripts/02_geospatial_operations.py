"""
Enrichment of the reference table with additional features.
Main operations:
    - assign a continent to each farm based on coordinates
    - assign a vegetation type to each farm based on coordinates
    - assign ESI coefficients based on continent and vegetation type
    - assign a climate zone to each farm based on coordinates
    - assign a soil organic carbon reference value to each farm based on coordinates
    - export the enriched reference table as a feather file and to an SQL database

Authors: Dario Marone, Giorgio Parlato
Date: 2026-02-13
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import geopandas as gpd
import janitor
import numpy.ma as ma
import pandas as pd
import rasterio
from shapely.geometry import Point

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Home directory
PATH_HOME = Path.home()

# Data directory
PATH_DATA = r"esi_cimmyt"

# Path to geospatial layer for continents
# Source: https://data.niaid.nih.gov/resources?id=zenodo_10778078
PATH_CONTINENTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\geospatial\vector\world_geographic_regions\World_Geographic_Regionst.shp"
)

# Path to vegetation types raster
# Source: model output (LPJmL)
PATH_VEGETATION_TYPES = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\geospatial\raster\vegetation_types\vegmap_nolu_late_20251117.asc"
)

# Path to ESI coefficients
# Source: model output (LPJmL, processed in Excel)
PATH_ESI_COEFFICIENTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\esi_coefficients\esi_coefficients_table_20260127.xlsx"
)

# Path to climatic zones raster
# Reference: European Soil Data Centre (ESDAC), esdac.jrc.ec.europa.eu, European Commission, Joint Research Centre
# Source: https://esdac.jrc.ec.europa.eu/content/support-renewable-energy-directive#tabs-0-description=1
PATH_CLIMATE_ZONES = PATH_HOME / PATH_DATA / r"data\raw\geospatial\raster\climate_zones"

# Path to SOC reference values raster
# Reference: FAO and ITPS. 2018. Global Soil Organic Carbon Map (GSOCmap) Technical Report. Rome. 162 pp.
# Source: https://console.cloud.google.com/storage/browser/fao-gismgr-gsocseq-data/DATA/GSOCSEQ/MAP;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
PATH_SOC_REF = PATH_HOME / PATH_DATA / r"data\raw\geospatial\raster\soil_organic_carbon"

# Path to reference table
PATH_REF_TABLE = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\table_01_reference_table.feather"
)

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Class names to identify the climatic zone of each farm
# Reference: 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 4, Agriculture, Forestry and Other Land Use
# Source: Figure 3A.5.1 in https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch03_Land%20Representation.pdf
CLIMATE_ZONE_NAMES = {
    1: "Warm Temperate Moist",
    2: "Warm Temperate Dry",
    3: "Cool Temperate Moist",
    4: "Cool Temperate Dry",
    5: "Polar Moist",
    6: "Polar Dry",
    7: "Boreal Moist",
    8: "Boreal Dry",
    9: "Tropical Montane",
    10: "Tropical Wet",
    11: "Tropical Moist",
    12: "Tropical Dry",
}

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the reference table
ref_tab_df = pd.read_feather(PATH_REF_TABLE)

# Load continents shape file
continents_shp = gpd.read_file(PATH_CONTINENTS)

# Load vegetation types raster
vegetation_types_raster = rasterio.open(PATH_VEGETATION_TYPES, mode="r+")

# Load ESI coefficients
esi_coefficients_df = pd.read_excel(PATH_ESI_COEFFICIENTS)

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
# Regions
# Load the polygon for North America
north_america_shp = continents_shp.query('Region==["North America"]').copy()

# Select the region and geometry columns
north_america_shp = north_america_shp[["Region", "geometry"]]

# Create point geometries from lon-lat coordinates for spatial join
geometries = [
    Point(lon, lat)
    for lon, lat in zip(ref_tab_df["longitude_w"], ref_tab_df["latitude_n"])
]

# Create a GeoDataFrame using the reference table
ref_gdf_geometries = gpd.GeoDataFrame(
    ref_tab_df, geometry=geometries, crs=continents_shp.crs
)

# Spatial join with the continents GeoDataFrame
# This operation assigns a region to coordinates based on the overlap between geometries
ref_gdf_regions = gpd.sjoin(
    ref_gdf_geometries,
    north_america_shp,
    how="left",  # Use index from the reference table and retain only the coordinates
    predicate="within",  # The coordinates must fall within the region
)

# Drop the right index
ref_gdf_regions.drop(columns=["index_right"], inplace=True)

# Rename column
ref_gdf_regions.rename(columns={"Region": "region"}, inplace=True)

# Vegetation types
# Copy previous GeoDataFrame for further enrichment
ref_gdf_vtypes = ref_gdf_regions.copy()

# Ensure the vegetation types raster and the GeoDataFrame have the same CRS
vegetation_types_raster.crs = rasterio.CRS.from_epsg(4326)


# Function to extract raster values for points
def get_raster_value(raster, point):
    # Get coordinates of the point
    coords = (point.x, point.y)

    # Extract the value at the point coordinates
    row, col = raster.index(*coords)
    value = raster.read(1)[row, col]
    return value


# Apply the function to each point in the plots gdf
ref_gdf_vtypes["raster_value"] = ref_gdf_vtypes["geometry"].apply(
    lambda geom: get_raster_value(vegetation_types_raster, geom)
)

# Mapping raster values to vegetation types
# Source: based on the metric input tab in the ESI Excel tool
raster_value_vtype_label_map = {
    1: "boreal forest",
    2: "cool climate grassland",
    3: "warm climate grassland",
    4: "temperate forest",
    5: "tropical forest",
    6: "barren land",
    7: "others",
}

ref_gdf_vtypes["veg_type"] = ref_gdf_vtypes["raster_value"].map(
    raster_value_vtype_label_map
)

# ESI coefficients
ref_gdf_esi = ref_gdf_vtypes.merge(
    esi_coefficients_df, on=["region", "veg_type"], how="left"
)

# Climate zones
# Coordinates
coords = [(geom.x, geom.y) for geom in ref_gdf_esi.geometry]

# Sample values
with rasterio.open(PATH_CLIMATE_ZONES / "CLIMATE_ZONE.rst") as src:
    sampled_values = [val[0] for val in src.sample(coords)]

# Create the GeoDataFrame and Map values
ref_gdf_zones = ref_gdf_esi.copy()
ref_gdf_zones["climate_class"] = pd.Series(sampled_values).map(CLIMATE_ZONE_NAMES)

# Apply climate alias correction
climate_alias = {
    "Warm Temperate Dry": "Cool Temperate Dry",
    "Warm Temperate Moist": "Cool Temperate Moist",
}

ref_gdf_zones["climate_class_corrected"] = ref_gdf_zones["climate_class"].replace(
    climate_alias
)

# SOC reference values
ref_gdf_soc_ref = ref_gdf_zones.copy()

coords = [(point.x, point.y) for point in ref_gdf_soc_ref.geometry]

with rasterio.open(PATH_SOC_REF / r"DATA_GSOCSEQ_MAP_GSOCSEQ.GSOCMAP1-5-0.tif") as src:
    sampled_values = [val[0] for val in src.sample(coords)]
    ref_gdf_soc_ref["soc_ref"] = ma.masked_equal(sampled_values, src.nodata)

# Sort values
ref_gdf_final = ref_gdf_soc_ref.copy()
ref_gdf_final = ref_gdf_final.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write the final GeoDataFrame as a feather file
ref_gdf_final.to_feather(
    PATH_EXPORT_TABLES / r"table_02_reference_table_enriched.feather"
)

# Write the final DataFrame to an SQL database
# Connect to the database
con = sqlite3.connect(PATH_SQL_DB)

# Write the DataFrame to SQL
ref_gdf_final.drop(columns="geometry").to_sql(
    name="table_02_reference_table_enriched", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
