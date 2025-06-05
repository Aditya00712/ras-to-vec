# # import time
# # import numpy as np
# # from django.http import JsonResponse
# # from rest_framework.decorators import api_view
# # from rest_framework.response import Response
# # from rest_framework import status
# # from shapely.geometry import Polygon, mapping
# # from skimage import measure
# # from PIL import Image
# # from rasterio.io import MemoryFile
# # import rasterio
# # import matplotlib.pyplot as plt
# # from rasterio.warp import transform
# # from pyproj import CRS, Transformer

# # @api_view(['POST'])
# # def raster_to_vector_view(request):
# #     try:
# #         start = time.time()

# #         # Check if file is uploaded
# #         if 'raster' not in request.FILES:
# #             return Response({'error': 'No raster file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

# #         raster_file = request.FILES['raster']

# #         # Get parameters with defaults
# #         scale = float(request.data.get('scale', 1))  # scale > 1 will downsample
# #         simplify_tol = float(request.data.get('simplify_tol', 0.0))  # simplification tolerance
# #         threshold = float(request.data.get('threshold', 10))  # threshold for binarization
# #         use_contour = request.data.get('use_contour', 'false').lower() == 'true'

# #         print(f"Contour level: {threshold} Use contour: {use_contour} Scale: {scale} Simplify tolerance: {simplify_tol}")

# #         # Read raster using rasterio
# #         with MemoryFile(raster_file.read()) as memfile:
# #             with memfile.open() as src:
# #                 # Use only the first band for processing
# #                 data = src.read(1)
# #                 print(f"Raster shape (bands, height, width): {src.count} {src.height} {src.width}")
# #                 print(f"Read data shape: {data.shape}")

# #                 # Visualize the raw data before thresholding
# #                 plt.imshow(data, cmap='viridis')
# #                 plt.title('Raw Raster Data')
# #                 plt.show()

# #                 # Resample image if scaling is required
# #                 if scale > 1:
# #                     h, w = data.shape
# #                     new_h, new_w = int(h / scale), int(w / scale)
# #                     data = np.array(Image.fromarray(data).resize((new_w, new_h)))
# #                     print(f"Resized raster to: {data.shape}")

# #                 # Optionally, calculate a dynamic threshold based on the raster data
# #                 dynamic_threshold = np.percentile(data, 90)  # Use the 90th percentile threshold
# #                 print(f"90th percentile threshold: {dynamic_threshold}")
# #                 mask = data > dynamic_threshold

# #                 # Visualize the mask after thresholding for debugging
# #                 plt.imshow(mask, cmap='gray')
# #                 plt.title('Mask Visualization')
# #                 plt.show()

# #                 # Only find contours if the mask is not uniform (i.e., has both True and False values)
# #                 contours = []
# #                 if np.sum(mask) > 0 and np.sum(mask) < mask.size:  # Mask has mixed values
# #                     contours = measure.find_contours(mask.astype(np.uint8), 0.5)

# #                 print(f"Contours found: {len(contours)}")

# #                 polygons = []
# #                 for contour in contours:
# #                     # Convert row, col to x, y
# #                     contour = np.flip(contour, axis=1)
                    
# #                     # Only process contours with at least 4 points
# #                     if len(contour) >= 4:
# #                         geo_coords = [src.transform * tuple(pt) for pt in contour]
# #                         poly = Polygon(geo_coords)
# #                         if poly.is_valid and not poly.is_empty:
# #                             if simplify_tol > 0:
# #                                 poly = poly.simplify(simplify_tol)
# #                             polygons.append(mapping(poly))
# #                     else:
# #                         print(f"Skipping contour with less than 4 points: {contour}")

# #                 print(f"Generated {len(polygons)} polygons in {time.time() - start:.2f}s")

# #                 # Re-project to WGS84 (EPSG:4326) if source CRS is not WGS84
# #                 src_crs = src.crs
# #                 print(f"Source CRS: {src_crs}")
                
# #                 if src_crs.to_epsg() != 4326:  # Check if the source CRS is not WGS84
# #                     transformer = Transformer.from_crs(src_crs, CRS('EPSG:4326'), always_xy=True)

# #                     # Re-project each polygon to WGS84
# #                     reprojected_polygons = []
# #                     for poly in polygons:
# #                         try:
# #                             reprojected_coords = []
# #                             for ring in poly['coordinates']:
# #                                 reprojected_ring = [transformer.transform(x, y) for x, y in ring]
# #                                 reprojected_coords.append(reprojected_ring)

# #                             # Validate the reprojected polygon
# #                             reprojected_poly = Polygon(reprojected_coords[0], reprojected_coords[1:])
# #                             if reprojected_poly.is_valid and not reprojected_poly.is_empty:
# #                                 reprojected_polygons.append(mapping(reprojected_poly))
# #                         except Exception as e:
# #                             print(f"Skipping invalid polygon during reprojection: {e}")

# #                     polygons = reprojected_polygons

# #                 # Construct GeoJSON FeatureCollection
# #                 feature_collection = {
# #                     "type": "FeatureCollection",
# #                     "features": [
# #                         {"type": "Feature", "geometry": geom, "properties": {}}
# #                         for geom in polygons
# #                     ]
# #                 }

# #                 return Response(feature_collection, status=status.HTTP_200_OK)

# #     except Exception as e:
# #         print("Error:", str(e))
# #         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# import time
# import numpy as np
# from django.http import JsonResponse
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status
# from shapely.geometry import Polygon, mapping
# from skimage import measure
# from PIL import Image
# from rasterio.io import MemoryFile
# import rasterio
# import matplotlib.pyplot as plt
# from rasterio.warp import transform
# from pyproj import CRS, Transformer

# @api_view(['POST'])
# def raster_to_vector_view(request):
#     try:
#         start = time.time()

#         # Check if file is uploaded
#         if 'raster' not in request.FILES:
#             return Response({'error': 'No raster file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

#         raster_file = request.FILES['raster']

#         # Get parameters with defaults
#         scale = float(request.data.get('scale', 1))  # scale > 1 will downsample
#         simplify_tol = float(request.data.get('simplify_tol', 0.0))  # simplification tolerance
#         threshold = float(request.data.get('threshold', 10))  # threshold for binarization
#         use_contour = request.data.get('use_contour', 'false').lower() == 'true'

#         print(f"Contour level: {threshold} Use contour: {use_contour} Scale: {scale} Simplify tolerance: {simplify_tol}")

#         # Read raster using rasterio
#         with MemoryFile(raster_file.read()) as memfile:
#             with memfile.open() as src:
#                 # Use only the first band for processing
#                 data = src.read(1)
#                 print(f"Raster shape (bands, height, width): {src.count} {src.height} {src.width}")
#                 print(f"Read data shape: {data.shape}")

#                 # Visualize the raw data before thresholding
#                 plt.imshow(data, cmap='viridis')
#                 plt.title('Raw Raster Data')
#                 plt.show()

#                 # Resample image if scaling is required
#                 if scale > 1:
#                     h, w = data.shape
#                     new_h, new_w = int(h / scale), int(w / scale)
#                     data = np.array(Image.fromarray(data).resize((new_w, new_h)))
#                     print(f"Resized raster to: {data.shape}")

#                     # Adjust the transform to account for scaling
#                     new_transform = src.transform * src.transform.scale(
#                         (data.shape[1] / new_w), (data.shape[0] / new_h)
#                     )
#                 else:
#                     new_transform = src.transform

#                 # Optionally, calculate a dynamic threshold based on the raster data
#                 dynamic_threshold = np.percentile(data, 90)  # Use the 90th percentile threshold
#                 print(f"90th percentile threshold: {dynamic_threshold}")
#                 mask = data > dynamic_threshold

#                 # Visualize the mask after thresholding for debugging
#                 plt.imshow(mask, cmap='gray')
#                 plt.title('Mask Visualization')
#                 plt.show()

#                 # Only find contours if the mask is not uniform (i.e., has both True and False values)
#                 contours = []
#                 if np.sum(mask) > 0 and np.sum(mask) < mask.size:  # Mask has mixed values
#                     contours = measure.find_contours(mask.astype(np.uint8), 0.5)

#                 print(f"Contours found: {len(contours)}")

#                 polygons = []
#                 for contour in contours:
#                     # Convert row, col to x, y using the new transform
#                     contour = np.flip(contour, axis=1)  # Flip to match x, y order

#                     # Only process contours with at least 4 points
#                     if len(contour) >= 4:
#                         geo_coords = [new_transform * tuple(pt) for pt in contour]
#                         poly = Polygon(geo_coords)
#                         if poly.is_valid and not poly.is_empty:
#                             if simplify_tol > 0:
#                                 poly = poly.simplify(simplify_tol)
#                             polygons.append(mapping(poly))
#                         else:
#                             print(f"Skipping invalid polygon: {poly}")
#                     else:
#                         print(f"Skipping contour with less than 4 points: {contour}")

#                 print(f"Generated {len(polygons)} polygons in {time.time() - start:.2f}s")

#                 # Re-project to WGS84 (EPSG:4326) if source CRS is not WGS84
#                 src_crs = src.crs
#                 print(f"Source CRS: {src_crs}")
                
#                 if src_crs.to_epsg() != 4326:  # Check if the source CRS is not WGS84
#                     transformer = Transformer.from_crs(src_crs, CRS('EPSG:4326'), always_xy=True)

#                     # Re-project each polygon to WGS84
#                     reprojected_polygons = []
#                     for poly in polygons:
#                         try:
#                             reprojected_coords = []
#                             for ring in poly['coordinates']:
#                                 reprojected_ring = [transformer.transform(x, y) for x, y in ring]
#                                 reprojected_coords.append(reprojected_ring)

#                             # Validate the reprojected polygon
#                             reprojected_poly = Polygon(reprojected_coords[0], reprojected_coords[1:])
#                             if reprojected_poly.is_valid and not reprojected_poly.is_empty:
#                                 reprojected_polygons.append(mapping(reprojected_poly))
#                         except Exception as e:
#                             print(f"Skipping invalid polygon during reprojection: {e}")

#                     polygons = reprojected_polygons

#                 # Construct GeoJSON FeatureCollection
#                 feature_collection = {
#                     "type": "FeatureCollection",
#                     "features": [
#                         {"type": "Feature", "geometry": geom, "properties": {}}
#                         for geom in polygons
#                     ]
#                 }

#                 return Response(feature_collection, status=status.HTTP_200_OK)

#     except Exception as e:
#         print("Error:", str(e))
#         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ! working with detection and location of contours
import time
import numpy as np
from rest_framework.response import Response  # Correct import for Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from affine import Affine
from skimage import measure
from shapely.geometry import Polygon
from pyproj import Transformer
import matplotlib.pyplot as plt
from django.http import JsonResponse

@api_view(["POST"])
@parser_classes([MultiPartParser])
def raster_to_vector_view(request):
    start = time.time()

    # 1. Validate upload
    uploaded = request.FILES.get("raster")
    if not uploaded:
        return Response({"error": "No raster file uploaded"}, status=400)

    # 2. Params
    scale        = float(request.data.get("scale",        1))
    simplify_tol = float(request.data.get("simplify_tol", 0.1))
    threshold    = float(request.data.get("threshold",   100))
    print(f"Contour level: {threshold} \nScale: {scale} \nSimplify tolerance: {simplify_tol}")

    # 3. Read raster into memory
    with MemoryFile(uploaded.read()) as mem:
        with mem.open() as src:
            # 4. Read & optionally down‑sample the first band
            if scale > 1:
                new_h = int(src.height // scale)
                new_w = int(src.width  // scale)
                arr = src.read(
                    1,
                    out_shape=(1, new_h, new_w),
                    resampling=Resampling.bilinear
                )
                # ensure 2D
                data = arr[0] if arr.ndim == 3 else arr
                # Adjust transform for new resolution
                transform = src.transform * Affine.scale(
                    src.width  / new_w,
                    src.height / new_h
                )
            else:
                data = src.read(1)
                transform = src.transform

            # 5. Build binary mask with fallback logic
            mask = data > threshold

            # Visualize the mask after thresholding for debugging
            plt.imshow(mask, cmap='gray')
            plt.title('Mask Visualization')
            plt.show()

            if not mask.any() or mask.all():
                # fallback = np.percentile(data, 90)
                # mask = data > fallback
                # print(f"Fallback threshold → {fallback}")
                for fallback_percentile in [90, 85, 80, 75]:
                    fallback = np.percentile(data, fallback_percentile)
                    mask = data > fallback
                    print(f"Fallback threshold → {fallback}")
                    if mask.any() and not mask.all():
                        break

            if not mask.any():
                return Response({"error": "No features detected even after fallback"}, status=400)

            # 6. Extract contours (pixel space)
            raw_contours = measure.find_contours(mask.astype(np.uint8), 0.3)
            
            for contour in raw_contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1)
            plt.title('Detected Contours')
            plt.show()

            # 7. Build polygons in source CRS
            polygons = []
            for contour in raw_contours:
                if len(contour) < 3:
                    continue
                # skimage gives (row, col) → flip to (x=col, y=row)
                xy = np.flip(contour, axis=1)
                # Apply affine: pixel → map coords
                coords = [transform * tuple(pt) for pt in xy]
                poly = Polygon(coords)

                if simplify_tol > 0:
                    poly = poly.simplify(simplify_tol, preserve_topology=True)
                if poly.is_valid and not poly.is_empty: #and poly.area > 1e-4
                    polygons.append(poly)

            if not polygons:
                return Response({"error": "No valid polygons generated"}, status=400)

            # 8. Reproject to WGS84
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            features = []
            
            # Add polygons to GeoJSON features
            for poly in polygons:
                ext = [transformer.transform(x, y) for x, y in poly.exterior.coords]
                ints = [[transformer.transform(x, y) for x, y in ring.coords]
                        for ring in poly.interiors]
                features.append({
                    "type":       "Feature",
                    "properties": {},
                    "geometry": {
                        "type":        "Polygon",
                        "coordinates": [ext] + ints
                    }
                })

    # 9. Return GeoJSON
    fc = {
        "type":     "FeatureCollection",
        "features": features
    }
    elapsed = time.time() - start
    print(f"Generated {len(features)} polygons in {elapsed:.2f}s")
    return Response(fc, content_type="application/geo+json")


# ! test for getting the highest and the lowest latitude from the clipped part
# https://geoserver.vasundharaa.in/geoserver/useruploads/wms?service=WMS&version=1.1.0&request=GetMap&layers=useruploads%3Aupload_178d6d&bbox=677791.8125%2C3144899.75%2C758404.3125%2C3216037.25&width=768&height=677&srs=EPSG%3A32643&styles=&format=application/openlayers

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from pyproj import Transformer
# import json

# ? only returns the lat and lon of the bounding box
# @csrf_exempt
# def get_highest_lowest_latitude(request):
#     """
#     Accepts a clipped bounding box in EPSG:32643 (UTM zone 43N),
#     reprojects it to EPSG:4326 (WGS84), and returns the highest and lowest latitudes.
#     """
#     if request.method != 'POST':
#         return JsonResponse({"error": "Only POST method is allowed"}, status=405)

#     try:
#         data = json.loads(request.body)
#         bbox = data.get("bbox")

#         if not bbox or len(bbox) != 4:
#             return JsonResponse({
#                 "error": "Invalid or missing 'bbox'. Expected format: [minx, miny, maxx, maxy]"
#             }, status=400)

#         minx, miny, maxx, maxy = map(float, bbox)

#         # Create transformer from EPSG:32643 to EPSG:4326
#         transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

#         # Convert all 4 corner points to lat/lon
#         lon1, lat1 = transformer.transform(minx, miny)
#         lon2, lat2 = transformer.transform(minx, maxy)
#         lon3, lat3 = transformer.transform(maxx, miny)
#         lon4, lat4 = transformer.transform(maxx, maxy)

#         # Get all latitudes
#         latitudes = [lat1, lat2, lat3, lat4]

#         return JsonResponse({
#             "highest_latitude": max(latitudes),
#             "lowest_latitude": min(latitudes)
#         })

#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON payload"}, status=400)
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)


import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from pyproj import Transformer
import requests
import json
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
import numpy as np

logger = logging.getLogger(__name__)

@csrf_exempt
def get_lat_elevation(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        full_url = data.get("wms_url")  # Full WMS GetMap URL
        bbox = data.get("bbox")         # [minx, miny, maxx, maxy]

        if not full_url or not bbox or len(bbox) != 4:
            return JsonResponse({"error": "Missing or invalid 'wms_url' or 'bbox'"}, status=400)

        minx, miny, maxx, maxy = map(float, bbox)

        # Parse and extract base WMS URL and parameters
        parsed = urlparse(full_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query_params = parse_qs(parsed.query)

        layer = query_params.get("layers", [None])[0]
        srs = query_params.get("srs", [None])[0]

        if not layer or not srs:
            return JsonResponse({"error": "Invalid WMS URL: missing layers or srs"}, status=400)

        # Construct proper GeoTIFF WMS URL
        geotiff_params = {
            "service": "WMS",
            "version": "1.1.0",
            "request": "GetMap",
            "layers": layer,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "width": "512",
            "height": "512",
            "srs": srs,
            "format": "image/geotiff"
        }
        wms_geotiff_url = f"{base_url}?{urlencode(geotiff_params)}"

        # Fetch raster image
        response = requests.get(wms_geotiff_url, timeout=10)
        if response.status_code != 200:
            return JsonResponse({"error": "Failed to fetch WMS raster"}, status=500)

        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                with WarpedVRT(dataset, crs="EPSG:4326") as vrt:
                    data = vrt.read(1, masked=True)
                    if data.mask.all():
                        return JsonResponse({"error": "No valid elevation data in raster"}, status=500)

                    min_elevation = float(data.min())
                    max_elevation = float(data.max())

                    max_idx = np.unravel_index(np.argmax(data.data), data.shape)
                    min_idx = np.unravel_index(np.argmin(data.data), data.shape)

                    max_lon, max_lat = vrt.xy(*max_idx)
                    min_lon, min_lat = vrt.xy(*min_idx)

        return JsonResponse({
            "highest_latitude": max_lat,
            "highest_longitude": max_lon,
            "lowest_latitude": min_lat,
            "lowest_longitude": min_lon,
            "highest_elevation": max_elevation,
            "lowest_elevation": min_elevation,
        })

    except Exception as e:
        logger.error(f"Exception in get_lat_elevation: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)


