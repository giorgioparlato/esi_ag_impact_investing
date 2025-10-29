#!/usr/bin/env python3      # Shebang line telling the OS which interpreter should run the script
# -*- coding: utf-8 -*-     # Coding declaration telling the Python interpreter which character encoding the source file is written in

"""
Module Name: geospatial_operations
Authors: Dario Marone, Giorgio Parlato
Date: 2025-10-03
Description:
    This script performs a series of geospatial operations.
    The objective is to enrich the reference table with default attributes required in the ESI computation.
    These attributes are the region, vegetation type, and the associated land, water, and carbon/GHG coefficients.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os # Miscellaneous operating system interfaces
import sys # System-specific parameters and functions
from pathlib import Path # Object-oriented filesystem paths

# Detect env base path dynamically
ENV_PATH = Path(sys.prefix)

# Force GDAL and PROJ paths
os.environ["GDAL_DATA"] = str(ENV_PATH / "Library" / "share" / "gdal")
os.environ["PROJ_LIB"] = str(ENV_PATH / "Library" / "share" / "proj")

import pandas as pd # Data analysis toolkit
import geopandas as gpd # Extends pandas datatypes to allow spatial operations on geometric types
from shapely.geometry import Point # Geometry type representing a single coordinate
import rasterio # Reads and writes geospatial raster data
from rasterio.crs import CRS # Coordinate reference systems, the CRS class and supporting functions
from fiona.crs import from_epsg # A geographic or projected coordinate reference system


#-----------------------------------------------------------------------------
# Paths and other constants
#-----------------------------------------------------------------------------

# Home directory
HOME = Path.home()

# Path to geospatial layer for continents
# Source: https://data.niaid.nih.gov/resources?id=zenodo_10778078 
CONTINENTS = (
    HOME / 
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\Geospatial\World_Geographic_Regions\World_Geographic_Regionst.shp"
)

# Path to vegetation types raster
VEGETATION_TYPES = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\Geospatial\Raster\Vegetation Types\NLu_2022-Oct-14.asc"
)

# Path to land ESI coefficients
LAND_ESI_K = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data\Input"
    r"\ESI Coefficients\Land_ESI_coefficients.csv"
)

# Path to water ESI coefficients
WATER_ESI_K = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data\Input"
    r"\ESI Coefficients\Water_ESI_coefficients.csv"
)

# Global carbon ESI coefficient (expecting emissions in C equivalents)
CARBON_ESI_K = 2.80E-12

# Global GHG ESI coefficient (expecting GHG emissions in CO2 equivalents)
GHG_ESI_K = 2.80E-12 / 3.67

# Path to reference table
REF_TABLE = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\Output\Intermediate Tables\out_reference_table.feather"
)

# Output path
OUTPUT_PATH = (
    HOME /
    r"OneDrive - Kungl. Vetenskapsakademien\ESI\Data"
    r"\Output\Intermediate Tables"
)

#-----------------------------------------------------------------------------
# Regions
#-----------------------------------------------------------------------------

# Load the continents shapefile into a GeoDataFrame
continents_shp = gpd.read_file(
    CONTINENTS
)

# Load the polygon for North America
north_america_shp = continents_shp.query(
    'Region==["North America"]'
).copy()

# Select the region and geometry columns
north_america_shp = north_america_shp[
    [
        "Region",
        "geometry"
    ]
]

# Load the reference table to enrich
ref_tab_df = pd.read_feather(
    REF_TABLE
)

# Create point geometries from lon-lat coordinates for spatial join
geometries = [
    Point(lon, lat) for lon, lat in zip(
        ref_tab_df['longitude_w'], ref_tab_df['latitude_n']
    )
]

# Create a GeoDataFrame using the reference table
ref_gdf_geometries = gpd.GeoDataFrame(
    ref_tab_df,
    geometry=geometries,
    crs=continents_shp.crs
)

# Data validation check ensuring there are no missing coordinates
assert ref_gdf_geometries.geometry.notna().all()

# Spatial join with the continents GeoDataFrame
# This operation assigns a region to coordinates based on the overlap between geometries
ref_gdf_regions = gpd.sjoin(
    ref_gdf_geometries,
    north_america_shp,
    how="left", # Use index from the reference table and retain only the coordinates
    predicate='within' # The coordinates must fall within the region
)

# Drop the right index
ref_gdf_regions.drop(
    columns=["index_right"],
    inplace=True
)

# Rename region column 
ref_gdf_regions.rename(
    columns={"Region":"region"},
    inplace=True
)

# Data validation check ensuring there are no missing regions
assert ref_gdf_regions.region.notna().all()

#-----------------------------------------------------------------------------
# Vegetation types
#-----------------------------------------------------------------------------

# Copy previous GeoDataFrame for further enrichment
ref_gdf_vtypes = ref_gdf_regions.copy()

# Open the raster in read/write mode
vegetation_types_raster = rasterio.open(
    VEGETATION_TYPES,
    mode="r+"
)

# We confirmed the CRS for the vegetation types raster should be EPSG 4326
# Ensure the vegetation types raster and the GeoDataFrame have the same CRS
vegetation_types_raster.crs = CRS.from_epsg(4326)

# Function to extract raster values
def get_raster_value(raster, point):
    # Get coordinates of the point
    coords = (point.x, point.y)
    
    # Extract the value at the point coordinates
    row, col = raster.index(*coords)
    value = raster.read(1)[row, col]
    return value

# Apply the function to each point in the reference table
ref_gdf_vtypes['raster_value'] = ref_gdf_vtypes['geometry'].apply(
    lambda geom: get_raster_value(vegetation_types_raster, geom)
)

# Map raster values to vegetation types, based on the metric input tab in the ESI Excel
raster_value_vtype_label_map = {
    1:"agriculture",
    2:"tropical forest",
    3:"temperate forest",
    4:"boreal forest",
    5:"cool climate grassland",
    6:"warm climate grassland",
    7:"bare land"
}

ref_gdf_vtypes['veg_type'] = ref_gdf_vtypes['raster_value'].map(
    raster_value_vtype_label_map
)

#-----------------------------------------------------------------------------
# ESI coefficients
#-----------------------------------------------------------------------------

# Copy previous GeoDataFrame for further enrichment
ref_gdf_coefficients = ref_gdf_vtypes.copy()

# Import land ESI coefficients
land_esi_k_df = pd.read_csv(
    LAND_ESI_K
)

# Rename unnamed column
land_esi_k_df = land_esi_k_df.rename(
    columns={land_esi_k_df.columns[0]: "region"}
)

# Unpivot
land_esi_k_df = land_esi_k_df.melt(
    id_vars="region",
    var_name="veg_type",
    value_name="land_esi_k"
)

# Sort values
land_esi_k_df.sort_values(
    by=["region", "veg_type"],
    inplace=True
)

# Import water ESI coefficients
water_esi_k_df = pd.read_csv(
    WATER_ESI_K
)

# Rename unnamed column
water_esi_k_df = water_esi_k_df.rename(
    columns={water_esi_k_df.columns[0]: "region"}
)

# Unpivot
water_esi_k_df = water_esi_k_df.melt(
    id_vars="region",
    var_name="veg_type",
    value_name="water_esi_k"
)

# Sort values
water_esi_k_df.sort_values(
    by=["region", "veg_type"],
    inplace=True
)

# Merge all ESI coefficients on region and vegetation types
esi_k_df = land_esi_k_df.merge(water_esi_k_df, on=["region", "veg_type"], how="outer")

# Create columns storing the carbon and GHG ESI coefficients
esi_k_df['carbon_esi_k'] = CARBON_ESI_K
esi_k_df['ghg_esi_k'] = GHG_ESI_K

# Merge the ESI coefficients DataFrame with the reference table
ref_gdf_enriched = ref_gdf_coefficients.merge(
    esi_k_df,
    on=["region", "veg_type"],
    how='left'
)

# Write to an intermediate table
ref_gdf_enriched.to_feather(
    OUTPUT_PATH / "reference_table_enriched.feather"
)