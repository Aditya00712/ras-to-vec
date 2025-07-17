import rasterio
import os
from collections import OrderedDict

file_path = "C:\\Users\\adity\\OneDrive\\Desktop\\ras to vec\\Image.tif"

response = {
    "filename": os.path.basename(file_path),
    "path": file_path,
    "size": f"{os.path.getsize(file_path) / (1024 * 1024):.2f} MB",
    "bands": OrderedDict()
}

with rasterio.open(file_path) as dataset:
    for band_index in range(1, dataset.count + 1):
        tags = dataset.tags(band_index)  # <--- read GDAL-style precomputed metadata

        band_name = f"Band {band_index}"
        band_stats = OrderedDict()

        for stat_key in [
            "STATISTICS_APPROXIMATE",
            "STATISTICS_MINIMUM",
            "STATISTICS_MAXIMUM",
            "STATISTICS_MEAN",
            "STATISTICS_STDDEV",
            "STATISTICS_VALID_PERCENT",
            "Scale",
            "Offset"
        ]:
            if stat_key in tags:
                val = tags[stat_key]
                try:
                    # convert to number if possible
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except:
                    pass
                band_stats[stat_key] = val
                print(stat_key)

        response["bands"][band_name] = band_stats

print(response)
