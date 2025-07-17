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
from PIL import Image as PILImage
import io

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


# ! function to get min and max from geoserver link of tif file uploaded on it, that will be passed by the frontend to the backend
# import requests
# @csrf_exempt
# def get_min_max_from_geoserver(request):
#     """
#     Fetches min and max values from a GeoServer raster layer (TIFF only).
#     Expects a POST request with WMS URL and optional bbox parameters.
#     """
#     if request.method != 'POST':
#         return JsonResponse({"error": "Only POST method allowed"}, status=405)

#     try:
#         data = json.loads(request.body)
#         wms_url = data.get("wms_url")
#         bbox = data.get("bbox")  # Required: [minx, miny, maxx, maxy]
#         bbox_srs = data.get("bbox_srs")  # Optional: SRS for the bbox if different from layer SRS
        
#         if not wms_url:
#             return JsonResponse({"error": "Missing 'wms_url' parameter"}, status=400)

#         # Parse the WMS URL to extract parameters
#         parsed = urlparse(wms_url)
#         base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
#         query_params = parse_qs(parsed.query)

#         layer = query_params.get("layers", [None])[0]
#         srs = query_params.get("srs", [None])[0] or query_params.get("crs", [None])[0]

#         if not layer:
#             return JsonResponse({"error": "Invalid WMS URL: missing layers parameter"}, status=400)

#         # If bbox is provided, use it; otherwise extract from original URL
#         if bbox and len(bbox) == 4:
#             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
#             # Use bbox_srs if provided, otherwise use the layer's SRS
#             effective_srs = bbox_srs or srs or "EPSG:4326"
#         else:
#             bbox_str = query_params.get("bbox", [None])[0]
#             effective_srs = srs or "EPSG:4326"
#             if not bbox_str:
#                 return JsonResponse({"error": "No bbox found in URL or request. Please provide bbox parameter."}, status=400)

#         # Construct GeoTIFF request - only try TIFF formats
#         tiff_formats = [
#             "image/geotiff",
#             "image/tiff", 
#             "application/geotiff"
#         ]
        
#         successful_response = None
#         last_error = None
#         used_format = None
        
#         for format_type in tiff_formats:
#             geotiff_params = {
#                 "service": "WMS",
#                 "version": "1.1.0",
#                 "request": "GetMap",
#                 "layers": layer,
#                 "bbox": bbox_str,
#                 "width": "256",
#                 "height": "256",
#                 "srs": effective_srs,
#                 "format": format_type
#             }

#             geotiff_url = f"{base_url}?{urlencode(geotiff_params)}"
            
#             try:
#                 # Fetch the raster data
#                 response = requests.get(geotiff_url, timeout=30)
                
#                 if response.status_code == 200:
#                     content_type = response.headers.get('content-type', '')
#                     content_length = len(response.content)
                    
#                     # Skip if empty response
#                     if content_length == 0:
#                         last_error = f"Empty response for format {format_type}"
#                         continue
                        
#                     # Skip if we got an XML error
#                     if 'xml' in content_type.lower() or 'text' in content_type.lower():
#                         try:
#                             error_text = response.content.decode('utf-8')[:500]
#                             last_error = f"GeoServer error: {error_text}"
#                             continue
#                         except:
#                             last_error = f"GeoServer returned XML error for format {format_type}"
#                             continue
                    
#                     # Check if the response is actually a TIFF
#                     if 'tiff' in content_type.lower() or 'geotiff' in content_type.lower():
#                         successful_response = response
#                         used_format = format_type
#                         break
#                     else:
#                         last_error = f"Response is not TIFF format. Got: {content_type}"
#                         continue
#                 else:
#                     last_error = f"HTTP {response.status_code} for format {format_type}"
                    
#             except Exception as e:
#                 last_error = f"Request failed for {format_type}: {str(e)}"
#                 continue
        
#         if not successful_response:
#             return JsonResponse({
#                 "error": "Failed to fetch TIFF raster data from GeoServer",
#                 "details": "GeoServer may not support TIFF output for this layer or the layer may not be a raster",
#                 "last_error": last_error,
#                 "tiff_formats_tried": tiff_formats,
#                 "final_url": geotiff_url if 'geotiff_url' in locals() else None,
#                 "suggestion": "Ensure the layer is a raster layer and GeoServer supports TIFF output"
#             }, status=500)

#         # Process the TIFF raster to get min/max values
#         try:
#             with MemoryFile(successful_response.content) as memfile:
#                 with memfile.open() as dataset:
#                     data = dataset.read(1, masked=True)
                    
#                     if data.mask.all():
#                         return JsonResponse({"error": "No valid data in TIFF raster"}, status=400)

#                     # Calculate comprehensive statistics
#                     valid_data = data.compressed()
#                     min_value = float(np.min(valid_data))
#                     max_value = float(np.max(valid_data))
#                     mean_value = float(np.mean(valid_data))
#                     std_value = float(np.std(valid_data))
#                     percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])

#                     return JsonResponse({
#                         "min_value": min_value,
#                         "max_value": max_value,
#                         "mean_value": mean_value,
#                         "std_value": std_value,
#                         "percentiles": {
#                             "10th": float(percentiles[0]),
#                             "25th": float(percentiles[1]),
#                             "50th": float(percentiles[2]),
#                             "75th": float(percentiles[3]),
#                             "90th": float(percentiles[4])
#                         },
#                         "data_points": len(valid_data),
#                         "layer": layer,
#                         "bbox": bbox_str,
#                         "format_used": used_format,
#                         "content_type": successful_response.headers.get('content-type', ''),
#                         "raster_info": {
#                             "width": dataset.width,
#                             "height": dataset.height,
#                             "crs": str(dataset.crs),
#                             "data_type": str(dataset.dtypes[0])
#                         }
#                     })
                
#         except Exception as raster_error:
#             return JsonResponse({
#                 "error": f"Failed to process TIFF raster data: {str(raster_error)}",
#                 "details": "The response may not be a valid TIFF file",
#                 "url_used": geotiff_url,
#                 "content_type": successful_response.headers.get('content-type', ''),
#                 "content_length": len(successful_response.content),
#                 "suggestion": "Check if the GeoServer layer is properly configured as a raster layer"
#             }, status=500)

#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON payload"}, status=400)
#     except Exception as e:
#         logger.error(f"Exception in get_min_max_from_geoserver: {e}", exc_info=True)
#         return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def get_min_max_from_geoserver(request):
    """
    Fetches min and max values from a GeoServer raster layer (TIFF only).
    Expects a POST request with WMS URL and optional bbox parameters.
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        wms_url = data.get("wms_url")
        bbox = data.get("bbox")  # Required: [minx, miny, maxx, maxy]
        bbox_srs = data.get("bbox_srs")  # Optional: SRS for the bbox if different from layer SRS
        
        if not wms_url:
            return JsonResponse({"error": "Missing 'wms_url' parameter"}, status=400)

        # Parse the WMS URL to extract parameters
        parsed = urlparse(wms_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query_params = parse_qs(parsed.query)

        layer = query_params.get("layers", [None])[0]
        srs = query_params.get("srs", [None])[0] or query_params.get("crs", [None])[0]

        if not layer:
            return JsonResponse({"error": "Invalid WMS URL: missing layers parameter"}, status=400)

        # If bbox is provided, use it; otherwise extract from original URL
        if bbox and len(bbox) == 4:
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            # Use bbox_srs if provided, otherwise use the layer's SRS
            effective_srs = bbox_srs or srs or "EPSG:4326"
        else:
            bbox_str = query_params.get("bbox", [None])[0]
            effective_srs = srs or "EPSG:4326"
            if not bbox_str:
                return JsonResponse({"error": "No bbox found in URL or request. Please provide bbox parameter."}, status=400)

        # Construct GeoTIFF request - only try TIFF formats
        tiff_formats = [
            "image/geotiff",
            "image/tiff", 
            "application/geotiff"
        ]
        
        successful_response = None
        
        for format_type in tiff_formats:
            geotiff_params = {
                "service": "WMS",
                "version": "1.1.0",
                "request": "GetMap",
                "layers": layer,
                "bbox": bbox_str,
                "width": "256",
                "height": "256",
                "srs": effective_srs,
                "format": format_type
            }

            geotiff_url = f"{base_url}?{urlencode(geotiff_params)}"
            
            try:
                # Fetch the raster data
                response = requests.get(geotiff_url, timeout=30)
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                        
                    # Check if the response is actually a TIFF
                    if 'tiff' in content_type.lower() or 'geotiff' in content_type.lower():
                        successful_response = response
                        break
            
            except Exception as e:
                logger.error(f"Request failed for {format_type}: {str(e)}")
                continue

        # Process the TIFF raster to get min/max values
        try:
            with MemoryFile(successful_response.content) as memfile:
                with memfile.open() as dataset:
                    data = dataset.read(1, masked=True)
                    
                    if data.mask.all():
                        return JsonResponse({"error": "No valid data in TIFF raster"}, status=400)

                    # Calculate comprehensive statistics
                    valid_data = data.compressed()
                    min_value = float(np.min(valid_data))
                    max_value = float(np.max(valid_data))
                    mean_value = float(np.mean(valid_data))

                    return JsonResponse({
                        "min_value": min_value,
                        "max_value": max_value,
                        "mean_value": mean_value,
                        "data_points": len(valid_data),
                        "layer": layer,
                        "bbox": bbox_str,
                        "content_type": successful_response.headers.get('content-type', ''),
                    })
                
        except Exception as raster_error:
            return JsonResponse({
                "error": f"Failed to process TIFF raster data: {str(raster_error)}",
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)
    except Exception as e:
        logger.error(f"Exception in get_min_max_from_geoserver: {e}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)



#! this is for tif inages uploadad from frontend, this will take min max values for each band and give them to the frontend 
# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status

# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def get_min_max_from_uploaded_tiff(request):
#     """
#     Fetches min and max values from an uploaded TIFF file.
#     Expects a POST request with a TIFF file upload.
#     Returns statistics for all bands in the TIFF.
#     """
#     try:
#         # Check if file is uploaded
#         if 'tiff_file' not in request.FILES:
#             return Response({'error': 'No TIFF file uploaded. Use "tiff_file" as the field name.'}, status=status.HTTP_400_BAD_REQUEST)

#         tiff_file = request.FILES['tiff_file']
        
#         # Validate file extension
#         if not tiff_file.name.lower().endswith(('.tif', '.tiff')):
#             return Response({'error': 'File must be a TIFF (.tif or .tiff) format'}, status=status.HTTP_400_BAD_REQUEST)

#         # Optional parameters
#         band_number = request.data.get('band', None)  # Specific band to analyze (1-based index)
        
#         # Read TIFF file using rasterio
#         with MemoryFile(tiff_file.read()) as memfile:
#             with memfile.open() as dataset:
                
#                 # Get basic raster information
#                 raster_info = {
#                     "filename": tiff_file.name,
#                     "width": dataset.width,
#                     "height": dataset.height,
#                     "band_count": dataset.count,
#                     "crs": str(dataset.crs) if dataset.crs else "No CRS",
#                     "data_type": str(dataset.dtypes[0]) if dataset.dtypes else "Unknown",
#                     "bounds": list(dataset.bounds),
#                     "transform": list(dataset.transform)[:6] if dataset.transform else None
#                 }
                
#                 bands_statistics = {}
                
#                 # If specific band is requested
#                 if band_number:
#                     try:
#                         band_num = int(band_number)
#                         if band_num < 1 or band_num > dataset.count:
#                             return Response({
#                                 'error': f'Invalid band number. File has {dataset.count} bands (1-{dataset.count})'
#                             }, status=status.HTTP_400_BAD_REQUEST)
                        
#                         # Read specific band
#                         data = dataset.read(band_num, masked=True)
                        
#                         if data.mask.all():
#                             bands_statistics[f"band_{band_num}"] = {
#                                 "error": "No valid data in this band"
#                             }
#                         else:
#                             valid_data = data.compressed()
#                             percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])
                            
#                             bands_statistics[f"band_{band_num}"] = {
#                                 "min_value": float(np.min(valid_data)),
#                                 "max_value": float(np.max(valid_data)),
#                                 "mean_value": float(np.mean(valid_data)),
#                                 "std_value": float(np.std(valid_data)),
#                                 "percentiles": {
#                                     "10th": float(percentiles[0]),
#                                     "25th": float(percentiles[1]),
#                                     "50th": float(percentiles[2]),
#                                     "75th": float(percentiles[3]),
#                                     "90th": float(percentiles[4])
#                                 },
#                                 "data_points": len(valid_data),
#                                 "nodata_value": dataset.nodata
#                             }
#                     except Exception as band_error:
#                         bands_statistics[f"band_{band_num}"] = {
#                             "error": f"Failed to process band {band_num}: {str(band_error)}"
#                         }
                            
#                 else:
#                     # Process all bands
#                     for band_idx in range(1, dataset.count + 1):
#                         try:
#                             data = dataset.read(band_idx, masked=True)
                            
#                             if data.mask.all():
#                                 bands_statistics[f"band_{band_idx}"] = {
#                                     "error": "No valid data in this band"
#                                 }
#                             else:
#                                 valid_data = data.compressed()
                                
#                                 if len(valid_data) == 0:
#                                     bands_statistics[f"band_{band_idx}"] = {
#                                         "error": "No valid data points"
#                                     }
#                                 else:
#                                     percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])
                                    
#                                     bands_statistics[f"band_{band_idx}"] = {
#                                         "min_value": float(np.min(valid_data)),
#                                         "max_value": float(np.max(valid_data)),
#                                         "mean_value": float(np.mean(valid_data)),
#                                         "std_value": float(np.std(valid_data)),
#                                         "percentiles": {
#                                             "10th": float(percentiles[0]),
#                                             "25th": float(percentiles[1]),
#                                             "50th": float(percentiles[2]),
#                                             "75th": float(percentiles[3]),
#                                             "90th": float(percentiles[4])
#                                         },
#                                         "data_points": len(valid_data),
#                                         "nodata_value": dataset.nodata
#                                     }
                                    
#                         except Exception as band_error:
#                             bands_statistics[f"band_{band_idx}"] = {
#                                 "error": f"Failed to process band {band_idx}: {str(band_error)}"
#                             }

#                 # Create response
#                 response_data = {
#                     "file_info": raster_info,
#                     "bands_statistics": bands_statistics,
#                     "total_bands_processed": len(bands_statistics),
#                     "success": True
#                 }
                
#                 return Response(response_data, status=status.HTTP_200_OK)
                
#     except Exception as e:
#         logger.error(f"Exception in get_min_max_from_uploaded_tiff: {e}", exc_info=True)
#         return Response({
#             'error': f'Failed to process TIFF file: {str(e)}',
#             'details': 'Make sure the file is a valid TIFF raster'
#         }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status

# import numpy as np
# import rasterio
# from rasterio.io import MemoryFile
# import logging

# logger = logging.getLogger(__name__)

# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def get_min_max_from_uploaded_tiff(request):
#     """
#     Fetches min and max values from an uploaded TIFF file.
#     Expects a POST request with a TIFF file upload.
#     Returns statistics for all bands in the TIFF.
#     """

#     try:
#         if 'tiff_file' not in request.FILES:
#             return Response({'error': 'No TIFF file uploaded. Use "tiff_file" as the field name.'}, status=status.HTTP_400_BAD_REQUEST)

#         tiff_file = request.FILES['tiff_file']

#         if not tiff_file.name.lower().endswith(('.tif', '.tiff')):
#             return Response({'error': 'File must be a TIFF (.tif or .tiff) format'}, status=status.HTTP_400_BAD_REQUEST)

#         band_number = request.data.get('band', None)

#         with MemoryFile(tiff_file.read()) as memfile:
#             with memfile.open() as dataset:
#                 raster_info = {
#                     "filename": tiff_file.name,
#                     "width": dataset.width,
#                     "height": dataset.height,
#                     "band_count": dataset.count,
#                     "crs": str(dataset.crs) if dataset.crs else "No CRS",
#                     "data_type": str(dataset.dtypes[0]) if dataset.dtypes else "Unknown",
#                     "bounds": list(dataset.bounds),
#                     "transform": list(dataset.transform)[:6] if dataset.transform else None
#                 }

#                 bands_statistics = {}

#                 import math  # Make sure this is at the top of your file

#                 def safe_float(val):
#                     try:
#                         f = float(val)
#                         if math.isnan(f) or math.isinf(f):
#                             return None
#                         return f
#                     except:
#                         return None

#                 def compute_band_statistics(band_idx):
#                     nodata = dataset.nodatavals[band_idx - 1]

#                     try:
#                         if nodata is not None:
#                             data = dataset.read(band_idx, masked=True)
#                         else:
#                             raw_data = dataset.read(band_idx)
#                             placeholder = 0
#                             data = np.ma.masked_equal(raw_data, placeholder)

#                         if data.mask.all():
#                             return {"error": "No valid data in this band"}

#                         valid_data = data.compressed()
#                         percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])

#                         # Precomputed GDAL stats (safely parsed)
#                         tags = dataset.tags(band_idx)
#                         gdal_stats = {
#                             "min": safe_float(tags.get("STATISTICS_MINIMUM")),
#                             "max": safe_float(tags.get("STATISTICS_MAXIMUM")),
#                             "mean": safe_float(tags.get("STATISTICS_MEAN")),
#                             "stddev": safe_float(tags.get("STATISTICS_STDDEV")),
#                             "approximate": tags.get("STATISTICS_APPROXIMATE", "NO")
#                         }

#                         return {
#                             "min_value": float(np.min(valid_data)),
#                             "max_value": float(np.max(valid_data)),
#                             "mean_value": float(np.mean(valid_data)),
#                             "std_value": float(np.std(valid_data)),
#                             "percentiles": {
#                                 "10th": float(percentiles[0]),
#                                 "25th": float(percentiles[1]),
#                                 "50th": float(percentiles[2]),
#                                 "75th": float(percentiles[3]),
#                                 "90th": float(percentiles[4])
#                             },
#                             "data_points": len(valid_data),
#                             "nodata_value": nodata,
#                             "gdal_precomputed_stats": gdal_stats
#                         }

#                     except Exception as e:
#                         return {"error": f"Failed to process band {band_idx}: {str(e)}"}

#                 # If specific band requested
#                 if band_number:
#                     try:
#                         band_idx = int(band_number)
#                         if not 1 <= band_idx <= dataset.count:
#                             return Response({'error': f'Invalid band number. TIFF has {dataset.count} bands.'}, status=status.HTTP_400_BAD_REQUEST)

#                         bands_statistics[f"band_{band_idx}"] = compute_band_statistics(band_idx)

#                     except ValueError:
#                         return Response({'error': 'Band number must be an integer.'}, status=status.HTTP_400_BAD_REQUEST)

#                 else:
#                     # Process all bands
#                     for band_idx in range(1, dataset.count + 1):
#                         bands_statistics[f"band_{band_idx}"] = compute_band_statistics(band_idx)

#                 response_data = {
#                     "file_info": raster_info,
#                     "bands_statistics": bands_statistics,
#                     "total_bands_processed": len(bands_statistics),
#                     "success": True
#                 }

#                 return Response(response_data, status=status.HTTP_200_OK)

#     except Exception as e:
#         logger.error(f"Exception in get_min_max_from_uploaded_tiff: {e}", exc_info=True)
#         return Response({
#             'error': f'Failed to process TIFF file: {str(e)}',
#             'details': 'Make sure the file is a valid TIFF raster'
#         }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#! this is for tif inages uploadad from frontend, this will take min max values for each band and give them to the frontend this one is striped down version
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status

@api_view(['POST'])
@parser_classes([MultiPartParser])
def get_min_max_from_uploaded_tiff(request):
    """
    Fetches min and max values from an uploaded TIFF file.
    Expects a POST request with a TIFF file upload.
    Returns statistics for all bands in the TIFF.
    """
    try:
        # Check if file is uploaded
        if 'tiff_file' not in request.FILES:
            return Response({'error': 'No TIFF file uploaded. Use "tiff_file" as the field name.'}, status=status.HTTP_400_BAD_REQUEST)

        tiff_file = request.FILES['tiff_file']
        
        # Validate file extension
        if not tiff_file.name.lower().endswith(('.tif', '.tiff')):
            return Response({'error': 'File must be a TIFF (.tif or .tiff) format'}, status=status.HTTP_400_BAD_REQUEST)

        # Optional parameters
        band_number = request.data.get('band', None)  # Specific band to analyze (1-based index)
        
        # Read TIFF file using rasterio
        with MemoryFile(tiff_file.read()) as memfile:
            with memfile.open() as dataset:
                
                # Get basic raster information
                raster_info = {
                    "filename": tiff_file.name,
                    "width": dataset.width,
                    "height": dataset.height,
                    "band_count": dataset.count,
                    "crs": str(dataset.crs) if dataset.crs else "No CRS",
                    "data_type": str(dataset.dtypes[0]) if dataset.dtypes else "Unknown",
                    "bounds": list(dataset.bounds),
                }
                
                bands_statistics = {}
                
                # If specific band is requested
                if band_number:
                    try:
                        band_num = int(band_number)
                        if band_num < 1 or band_num > dataset.count:
                            return Response({
                                'error': f'Invalid band number. File has {dataset.count} bands (1-{dataset.count})'
                            }, status=status.HTTP_400_BAD_REQUEST)
                        
                        # Read specific band
                        data = dataset.read(band_num, masked=True)
                        
                        if data.mask.all():
                            bands_statistics[f"band_{band_num}"] = {
                                "error": "No valid data in this band"
                            }
                        else:
                            valid_data = data.compressed()                            
                            bands_statistics[f"band_{band_num}"] = {
                                "min_value": float(np.min(valid_data)),
                                "max_value": float(np.max(valid_data)),
                            }
                    except Exception as band_error:
                        bands_statistics[f"band_{band_num}"] = {
                            "error": f"Failed to process band {band_num}: {str(band_error)}"
                        }
                            
                else:
                    # Process all bands
                    for band_idx in range(1, dataset.count + 1):
                        try:
                            data = dataset.read(band_idx, masked=True)
                            
                            if data.mask.all():
                                bands_statistics[f"band_{band_idx}"] = {
                                    "error": "No valid data in this band"
                                }
                            else:
                                valid_data = data.compressed()
                                
                                if len(valid_data) == 0:
                                    bands_statistics[f"band_{band_idx}"] = {
                                        "error": "No valid data points"
                                    }
                                else:                                    
                                    bands_statistics[f"band_{band_idx}"] = {
                                        "min_value": float(np.min(valid_data)),
                                        "max_value": float(np.max(valid_data)),
                                    }
                                    
                        except Exception as band_error:
                            bands_statistics[f"band_{band_idx}"] = {
                                "error": f"Failed to process band {band_idx}: {str(band_error)}"
                            }

                # Create response
                response_data = {
                    "file_info": raster_info,
                    "bands_statistics": bands_statistics,
                    "total_bands_processed": len(bands_statistics),
                    "success": True
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
                
    except Exception as e:
        logger.error(f"Exception in get_min_max_from_uploaded_tiff: {e}", exc_info=True)
        return Response({
            'error': f'Failed to process TIFF file: {str(e)}',
            'details': 'Make sure the file is a valid TIFF raster'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status

# import numpy as np
# import rasterio
# from rasterio.io import MemoryFile
# import math
# import logging

# logger = logging.getLogger(__name__)

# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def get_min_max_from_uploaded_tiff(request):
#     try:
#         if 'tiff_file' not in request.FILES:
#             return Response({'error': 'No TIFF file uploaded. Use "tiff_file" as the field name.'}, status=status.HTTP_400_BAD_REQUEST)

#         tiff_file = request.FILES['tiff_file']

#         if not tiff_file.name.lower().endswith(('.tif', '.tiff')):
#             return Response({'error': 'File must be a TIFF (.tif or .tiff) format'}, status=status.HTTP_400_BAD_REQUEST)

#         band_number = request.data.get('band', None)

#         with MemoryFile(tiff_file.read()) as memfile:
#             with memfile.open() as dataset:
#                 raster_info = {
#                     "filename": tiff_file.name,
#                     "path": f"Uploaded file: {tiff_file.name}",
#                     "size": f"{tiff_file.size / (1024*1024):.2f} MB",
#                     "last_modified": "Current upload",
#                     "provider": "gdal",
                    
#                     # Extent information (matching QGIS format)
#                     "extent": f"{dataset.bounds[0]:.14f},{dataset.bounds[1]:.14f} : {dataset.bounds[2]:.14f},{dataset.bounds[3]:.14f}",
#                     "width": dataset.width,
#                     "height": dataset.height,
#                     "band_count": dataset.count,
                    
#                     # Data type information
#                     "data_type": str(dataset.dtypes[0]) if dataset.dtypes else "Unknown",
                    
#                     # Coordinate system
#                     "crs": str(dataset.crs) if dataset.crs else "No CRS",
#                     "gdal_driver_description": "GeoTIFF",
#                     "gdal_driver_metadata": "GeoTIFF",
                    
#                     # Transform and resolution
#                     "bounds": list(dataset.bounds),
#                     "transform": list(dataset.transform)[:6] if dataset.transform else None,
#                     "pixel_size_x": abs(dataset.transform[0]) if dataset.transform else None,
#                     "pixel_size_y": abs(dataset.transform[4]) if dataset.transform else None,
                    
#                     # Compression and other info
#                     "compression": dataset.profile.get('compress', 'None'),
#                     "dataset_description": dataset.descriptions[0] if dataset.descriptions and dataset.descriptions[0] else "No description",
#                 }

#                 dataset_metadata = {
#                     "dataset_tags": dataset.tags(),
#                     "bands_tags": {f"band_{i}": dataset.tags(i) for i in range(1, dataset.count + 1)}
#                 }

#                 def safe_float(val):
#                     try:
#                         f = float(val)
#                         if math.isnan(f) or math.isinf(f):
#                             return None
#                         return f
#                     except:
#                         return None

#                 def compute_band_statistics(band_idx):
#                     nodata = dataset.nodatavals[band_idx - 1]
#                     try:
#                         # Get band tags first to check for pre-computed GDAL statistics
#                         tags = dataset.tags(band_idx)
                        
#                         # Check if GDAL statistics are already computed and stored in metadata
#                         has_gdal_stats = all(key in tags for key in [
#                             "STATISTICS_MINIMUM", "STATISTICS_MAXIMUM", 
#                             "STATISTICS_MEAN", "STATISTICS_STDDEV"
#                         ])
                        
#                         if has_gdal_stats:
#                             # Use pre-computed GDAL statistics (like QGIS does)
#                             min_val = float(tags["STATISTICS_MINIMUM"])
#                             max_val = float(tags["STATISTICS_MAXIMUM"])
#                             mean_val = float(tags["STATISTICS_MEAN"])
#                             std_val = float(tags["STATISTICS_STDDEV"])
#                             valid_percent = int(tags.get("STATISTICS_VALID_PERCENT", "100"))
#                             is_approximate = tags.get("STATISTICS_APPROXIMATE", "YES")
                            
#                             # Still read data for percentiles
#                             if nodata is not None:
#                                 data = dataset.read(band_idx, masked=True)
#                             else:
#                                 raw_data = dataset.read(band_idx)
#                                 # Use the actual nodata value if specified
#                                 if dataset.profile.get('nodata') is not None:
#                                     data = np.ma.masked_equal(raw_data, dataset.profile['nodata'])
#                                 else:
#                                     data = np.ma.array(raw_data, mask=False)
                            
#                             if not data.mask.all():
#                                 valid_data = data.compressed().astype(np.float64)
#                                 percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])
#                                 total_pixels = data.size
#                                 valid_pixels = len(valid_data)
#                             else:
#                                 return {"error": "No valid data in this band"}
                                
#                         else:
#                             # Compute statistics ourselves using QGIS-compatible method
#                             if nodata is not None:
#                                 data = dataset.read(band_idx, masked=True)
#                             else:
#                                 raw_data = dataset.read(band_idx)
#                                 # Use the actual nodata value if specified in the dataset profile
#                                 if dataset.profile.get('nodata') is not None:
#                                     data = np.ma.masked_equal(raw_data, dataset.profile['nodata'])
#                                 else:
#                                     # Don't mask any values if no nodata is specified
#                                     data = np.ma.array(raw_data, mask=False)

#                             if data.mask.all():
#                                 return {"error": "No valid data in this band"}

#                             valid_data = data.compressed()
                            
#                             # QGIS-style statistics calculation
#                             # Use double precision for calculations (like QGIS)
#                             valid_data_float64 = valid_data.astype(np.float64)
                            
#                             min_val = float(np.min(valid_data_float64))
#                             max_val = float(np.max(valid_data_float64))
                            
#                             # QGIS uses population mean and standard deviation
#                             mean_val = float(np.mean(valid_data_float64))
                            
#                             # QGIS uses population standard deviation (ddof=0, which is NumPy's default)
#                             std_val = float(np.std(valid_data_float64, ddof=0))
                            
#                             # Alternative: Use rasterio's native statistics (closer to GDAL)
#                             try:
#                                 from rasterio.enums import MaskFlags
#                                 # Try to get statistics using rasterio's method
#                                 rasterio_stats = {}
                                
#                                 # Read data without masking first to get raw statistics
#                                 raw_data = dataset.read(band_idx)
                                
#                                 # Apply proper nodata masking
#                                 if nodata is not None:
#                                     mask = raw_data != nodata
#                                     masked_data = raw_data[mask]
#                                 else:
#                                     masked_data = raw_data.flatten()
                                
#                                 if len(masked_data) > 0:
#                                     # Convert to float64 for precision (like GDAL)
#                                     masked_data = masked_data.astype(np.float64)
                                    
#                                     rasterio_stats = {
#                                         'min': float(np.min(masked_data)),
#                                         'max': float(np.max(masked_data)),
#                                         'mean': float(np.mean(masked_data)),
#                                         'std': float(np.std(masked_data, ddof=0)),  # Population std like GDAL
#                                         'count': len(masked_data)
#                                     }
                                    
#                                     # Use rasterio stats if they seem more accurate
#                                     min_val = rasterio_stats['min']
#                                     max_val = rasterio_stats['max'] 
#                                     mean_val = rasterio_stats['mean']
#                                     std_val = rasterio_stats['std']
#                                     valid_pixels = rasterio_stats['count']
                                    
#                             except Exception as e:
#                                 print(f"Rasterio stats failed: {e}")
#                                 # Fall back to the previous method
#                                 pass
                            
#                             # Calculate percentiles for additional info
#                             percentiles = np.percentile(valid_data_float64, [10, 25, 50, 75, 90])
                            
#                             # Calculate additional statistics shown in QGIS
#                             total_pixels = data.size
#                             valid_percent = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
#                             is_approximate = "NO"
                        
#                         # Add scale and offset information (commonly 1 and 0)
#                         scale = tags.get("SCALE", "1")
#                         offset = tags.get("OFFSET", "0")

#                         return {
#                             # Main statistics
#                             "min_value": min_val,
#                             "max_value": max_val,
#                             "mean_value": mean_val,
#                             "std_value": std_val,
#                             "percentiles": {
#                                 "10th": float(percentiles[0]),
#                                 "25th": float(percentiles[1]),
#                                 "50th": float(percentiles[2]),
#                                 "75th": float(percentiles[3]),
#                                 "90th": float(percentiles[4])
#                             },
                            
#                             # # QGIS-style detailed statistics (using GDAL precomputed if available)
#                             # "qgis_style_statistics": {
#                             #     "STATISTICS_APPROXIMATE": is_approximate,
#                             #     "STATISTICS_MINIMUM": int(min_val),
#                             #     "STATISTICS_MAXIMUM": int(max_val),
#                             #     "STATISTICS_MEAN": round(mean_val, 13),
#                             #     "STATISTICS_STDDEV": round(std_val, 11),
#                             #     "STATISTICS_VALID_PERCENT": int(round(valid_percent)) if isinstance(valid_percent, float) else valid_percent,
#                             #     "Scale": float(scale) if scale != "1" else 1,
#                             #     "Offset": float(offset) if offset != "0" else 0
#                             # },
                            
#                             # Data information
#                             "data_points": valid_pixels,
#                             "total_pixels": total_pixels,
#                             "valid_percent": valid_percent if isinstance(valid_percent, float) else float(valid_percent),
#                             "nodata_value": nodata,
#                             "uses_precomputed_stats": has_gdal_stats,
                            
#                             # Debug information
#                             "debug_info": {
#                                 "actual_nodata_used": nodata,
#                                 "profile_nodata": dataset.profile.get('nodata'),
#                                 "data_type": str(dataset.dtypes[band_idx - 1]),
#                                 "data_range": f"{min_val} to {max_val}",
#                                 "computation_method": "precomputed_gdal" if has_gdal_stats else "calculated",
#                                 "sample_values": list(map(float, valid_data[:10])) if not has_gdal_stats else "N/A"
#                             },
                            
#                             # Band metadata
#                             "band_metadata": tags if tags else {}
#                         }

#                     except Exception as e:
#                         return {"error": f"Failed to process band {band_idx}: {str(e)}"}

#                 bands_statistics = {}

#                 if band_number:
#                     try:
#                         band_idx = int(band_number)
#                         if not 1 <= band_idx <= dataset.count:
#                             return Response({'error': f'Invalid band number. TIFF has {dataset.count} bands.'}, status=status.HTTP_400_BAD_REQUEST)

#                         bands_statistics[f"band_{band_idx}"] = compute_band_statistics(band_idx)

#                     except ValueError:
#                         return Response({'error': 'Band number must be an integer.'}, status=status.HTTP_400_BAD_REQUEST)

#                 else:
#                     for band_idx in range(1, dataset.count + 1):
#                         bands_statistics[f"band_{band_idx}"] = compute_band_statistics(band_idx)

#                 response_data = {
#                     "file_info": raster_info,
#                     "bands_statistics": bands_statistics,
#                     "dataset_metadata": dataset_metadata,
#                     "total_bands_processed": len(bands_statistics),
#                     "success": True
#                 }

#                 return Response(response_data, status=status.HTTP_200_OK)

#     except Exception as e:
#         logger.error(f"Exception in get_min_max_from_uploaded_tiff: {e}", exc_info=True)
#         return Response({
#             'error': f'Failed to process TIFF file: {str(e)}',
#             'details': 'Make sure the file is a valid TIFF raster'
#         }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
