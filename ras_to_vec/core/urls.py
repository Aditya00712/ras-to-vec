from django.urls import path, include
from .views import *

urlpatterns = [
    path('convert/', convert_tif_to_geojson, name='convert_tif_to_geojson'),
]
