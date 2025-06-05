import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

# INPUT: path to your single-band raster
raster_path = r"Reclassified_Image.tif"

# OUTPUT: path to save the GeoJSON
geojson_output_path = r"ras_to_vec/output.geojson"  # Updated to include a filename

# Step 1: Read the raster
with rasterio.open(raster_path) as src:
    raster = src.read(1)  # Read first (and only) band
    transform = src.transform

# Step 2: Create mask of valid pixels (optional: skip nodata or zero if you want)
mask = raster != src.nodata if src.nodata is not None else np.ones_like(raster, dtype=bool)

# Step 3: Raster to Vector
# 'features.shapes' generates (geometry, value) pairs
shapes = features.shapes(raster, mask=mask, transform=transform)

# Step 4: Build GeoDataFrame
records = []
for geom, value in shapes:
    records.append({
        'geometry': shape(geom),
        'value': value
    })

gdf = gpd.GeoDataFrame(records, crs=src.crs)

# Step 5: Save to GeoJSON
gdf.to_file(geojson_output_path, driver="GeoJSON")

print(f"Vector saved to {geojson_output_path}")