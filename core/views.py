from django.shortcuts import render
import tempfile, requests, rasterio, json
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np

import os
import uuid
import tempfile
import rasterio
from rasterio import features
from shapely.geometry import shape
import numpy as np
import geopandas as gpd
from django.http import JsonResponse, FileResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response

# @api_view(["POST"])
# def raster_to_vector_view(request):
#     parser_classes = [MultiPartParser]

#     uploaded_file = request.FILES.get("raster")
#     if not uploaded_file:
#         return Response({"error": "No file uploaded"}, status=400)

#     # Save uploaded raster temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_raster_file:
#         for chunk in uploaded_file.chunks():
#             temp_raster_file.write(chunk)
#         raster_path = temp_raster_file.name

#     try:
#         with rasterio.open(raster_path) as src:
#             raster = src.read(1)
#             transform = src.transform
#             mask = raster != src.nodata if src.nodata is not None else np.ones_like(raster, dtype=bool)
#             shapes = features.shapes(raster, mask=mask, transform=transform)
#             # records = [{'geometry': shape(geom), 'value': value} for geom, value in shapes] # ! This is for full quality images
#             records = [{'geometry': shape(geom).simplify(0.1), 'value': value} for geom, value in shapes]
#             gdf = gpd.GeoDataFrame(records, crs=src.crs)

#         # Save GeoJSON to a temp file
#         output_path = os.path.join(tempfile.gettempdir(), f"vector_{uuid.uuid4().hex}.geojson")
#         gdf.to_file(output_path, driver="GeoJSON")

#         # Return the file as a download
#         return FileResponse(open(output_path, 'rb'), as_attachment=True, filename="convertedVector.geojson")

#     except Exception as e:
#         return Response({"error": str(e)}, status=500)
#     finally:
#         os.remove(raster_path)  # Clean up uploaded file


# ! time reduce by 8x but still very very slow for size 500MB
# import os
# import uuid
# import tempfile
# import numpy as np
# import geopandas as gpd
# from shapely.geometry import shape
# from rest_framework.decorators import api_view
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from django.http import FileResponse
# import rasterio
# from rasterio import features
# from rasterio.enums import Resampling

# @api_view(["POST"])
# def raster_to_vector_view(request):
#     parser_classes = [MultiPartParser]

#     uploaded_file = request.FILES.get("raster")
#     if not uploaded_file:
#         return Response({"error": "No file uploaded"}, status=400)

#     # Save uploaded raster temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_raster_file:
#         for chunk in uploaded_file.chunks():
#             temp_raster_file.write(chunk)
#         raster_path = temp_raster_file.name

#     try:
#         # Open and down_sample raster
#         with rasterio.open(raster_path) as src:
#             scale = 4  # Change to 8 or 16 for more speed
#             new_height = src.height // scale
#             new_width = src.width // scale

#             raster = src.read(
#                 1,
#                 out_shape=(1, new_height, new_width),
#                 resampling=Resampling.nearest
#             )

#             # Adjust transform
#             transform = src.transform * src.transform.scale(
#                 (src.width / new_width),
#                 (src.height / new_height)
#             )

#             mask = raster != src.nodata if src.nodata is not None else np.ones_like(raster, dtype=bool)

#             # Generate simplified polygons
#             records = []
#             for geom, value in features.shapes(raster, mask=mask, transform=transform):
#                 if value == 0:  # Skip background
#                     continue
#                 simplified = shape(geom).simplify(0.5, preserve_topology=True)
#                 if not simplified.is_valid or simplified.is_empty:
#                     continue
#                 records.append({
#                     'geometry': simplified,
#                     'value': value
#                 })

#             if not records:
#                 return Response({"error": "No valid geometries found"}, status=400)

#             gdf = gpd.GeoDataFrame(records, crs=src.crs)

#         # Save GeoJSON to temp file
#         output_path = os.path.join(tempfile.gettempdir(), f"vector_{uuid.uuid4().hex}.geojson")
#         gdf.to_file(output_path, driver="GeoJSON")

#         # Return GeoJSON file for download
#         return FileResponse(open(output_path, 'rb'), as_attachment=True, filename="convertedVector.geojson")

#     except Exception as e:
#         return Response({"error": str(e)}, status=500)
#     finally:
#         os.remove(raster_path)


# ! trying gdal
import os
import uuid
import subprocess
import tempfile
from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response

@api_view(["POST"])
def raster_to_vector_view(request):
    parser_classes = [MultiPartParser]

    uploaded_file = request.FILES.get("raster")
    if not uploaded_file:
        return Response({"error": "No file uploaded"}, status=400)

    # Save uploaded raster temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_raster_file:
        for chunk in uploaded_file.chunks():
            temp_raster_file.write(chunk)
        raster_path = temp_raster_file.name

    try:
        # Generate output GeoJSON path
        output_path = os.path.join(tempfile.gettempdir(), f"vector_{uuid.uuid4().hex}.geojson")

        # Run GDAL's polygonize command (subprocess)
        subprocess.run([
            "gdal_polygonize.py", raster_path, 
            "-f", "GeoJSON", output_path
        ], check=True)

        # Return the GeoJSON file for download
        return FileResponse(open(output_path, 'rb'), as_attachment=True, filename="convertedVector.geojson")

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    finally:
        os.remove(raster_path)  # Clean up uploaded file

# @csrf_exempt
# def convert_tif_to_geojson(request):
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

#     try:
#         body = json.loads(request.body)
#         tif_url = body.get('url')
#         if not tif_url:
#             return JsonResponse({'error': 'No URL provided'}, status=400)

#         # Download TIF
#         response = requests.get(tif_url)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
#             tmp.write(response.content)
#             tmp_path = tmp.name

#         # Convert TIF to GeoJSON
#         with rasterio.open(tmp_path) as src:
#             image = src.read(1)
#             mask = image != src.nodata
#             results = (
#                 {"properties": {"value": v}, "geometry": s}
#                 for s, v in shapes(image, mask=mask, transform=src.transform)
#             )
#             gdf = gpd.GeoDataFrame.from_features(results)
#             gdf.crs = src.crs

#         geojson_data = gdf.to_json()

#         return JsonResponse(json.loads(geojson_data), safe=False)

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)