from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    # path('convert/', convert_tif_to_geojson, name='convert_tif_to_geojson'),
    path('convert-raster/', raster_to_vector_view, name='raster_to_vector_view'),
    # path('get-highest-lowest-latitude/', get_highest_lowest_latitude, name='get_highest_lowest_latitude'),
    path('get-latitude-and-elevation-from-raster/', get_lat_elevation, name='get_latitude_and_elevation_from_raster'),
    path('get-min-max-from-geoserver/', views.get_min_max_from_geoserver, name='get_min_max_from_geoserver'),

    path('get-min-max-from-uploaded-tiff/', views.get_min_max_from_uploaded_tiff, name='get_min_max_from_uploaded_tiff'),

]
