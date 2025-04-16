from django.shortcuts import render
import tempfile, requests, rasterio, json
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def convert_tif_to_geojson(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    try:
        body = json.loads(request.body)
        tif_url = body.get('url')
        if not tif_url:
            return JsonResponse({'error': 'No URL provided'}, status=400)

        # Download TIF
        response = requests.get(tif_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Convert TIF to GeoJSON
        with rasterio.open(tmp_path) as src:
            image = src.read(1)
            mask = image != src.nodata
            results = (
                {"properties": {"value": v}, "geometry": s}
                for s, v in shapes(image, mask=mask, transform=src.transform)
            )
            gdf = gpd.GeoDataFrame.from_features(results)
            gdf.crs = src.crs

        geojson_data = gdf.to_json()

        return JsonResponse(json.loads(geojson_data), safe=False)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
