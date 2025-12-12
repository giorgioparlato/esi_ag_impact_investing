"""
Module Name: geospatial_operations
Authors: Dario Marone, Giorgio Parlato
Date: 2025-11-04
Description:
    This script performs a series of geospatial operations
    The objective is to enrich the reference table with default attributes required in the ESI computation.
    These attributes are the region, vegetation type, and the associated land, water, and carbon/GHG coefficients.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import geopandas as gpd
import janitor
import pandas as pd
import rasterio
from shapely.geometry import Point

# -----------------------------------------------------------------------------
# Paths and other constants
# -----------------------------------------------------------------------------
# File paths and sheet names
# Get home directory (useful if you are working on different machines)
PATH_HOME = Path.home()

# Define where the data is located
PATH_DATA = r"esi_cimmyt"

# Path to geospatial layer for continents
# Source: https://data.niaid.nih.gov/resources?id=zenodo_10778078
PATH_CONTINENTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\geospatial\vector\world_geographic_regions\World_Geographic_Regionst.shp"
)

# Path to vegetation types raster
PATH_VEGETATION_TYPES = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\geospatial\raster\vegetation_types\vegmap.NoLU.late.asc"
)

# Path to ESI coefficients
PATH_ESI_COEFFICIENTS = (
    PATH_HOME
    / PATH_DATA
    / r"data\raw\esi_coefficients\esi_coefficients_table_updated_20251212.xlsx"
)

# Path to reference table
PATH_REF_TABLE = (
    PATH_HOME / PATH_DATA / r"data\processed\tables\reference_table.feather"
)

# Export path for processed tables
PATH_EXPORT_TABLES = PATH_HOME / PATH_DATA / r"data\processed\tables"

# SQL DB connection
PATH_SQL_DB = PATH_HOME / PATH_DATA / r"data\processed\sql_db\cimmyt.db"

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
# Load the reference table to enrich
ref_tab_df = pd.read_feather(PATH_REF_TABLE)

# Load the continents shapefile into a GeoDataFrame
continents_shp = gpd.read_file(PATH_CONTINENTS)

# Open the raster in read/write mode (to allow setting the missing CRS)
vegetation_types_raster = rasterio.open(PATH_VEGETATION_TYPES, mode="r+")

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

# Mapping raster values to vegetation types, based on the metric input tab in the ESI Excel
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
ref_gdf_final = ref_gdf_vtypes.merge(
    esi_coefficients_df, on=["region", "veg_type"], how="left"
)

# Sort values
ref_gdf_final = ref_gdf_final.sort_values(
    by=["farmer_id", "plot_id", "logbook_id", "practice_id", "year"]
)

# -----------------------------------------------------------------------------
# Export data
# -----------------------------------------------------------------------------
# Write the final GeoDataFrame as a feather file
ref_gdf_final.to_feather(PATH_EXPORT_TABLES / r"reference_table_enriched.feather")

# Write the final DataFrame to an SQL database
# Connect to the database
con = sqlite3.connect(PATH_SQL_DB)

# Write the DataFrame to SQL
ref_gdf_final.drop(columns="geometry").to_sql(
    name="reference_table_enriched", con=con, if_exists="replace", index=False
)

# END
exit("Execution complete.")
