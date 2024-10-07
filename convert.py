import rasterio
from rasterio.enums import Resampling

# Open the GeoTIFF file
with rasterio.open('C:/Users/251874/Desktop/EMIT_MaskRCNN/data/gt_path/EMIT_L2B_CH4ENH_001_20220815T042838_2222703_003.tif') as src:
    # Read the data
    data = src.read()

    # Get metadata
    metadata = src.meta

# Write to ENVI format
metadata.update(driver='ENVI')
with rasterio.open('EMIT_L2B_CH4ENH_001_20220815T042838_2222703_003.img', 'w', **metadata) as dst:
    dst.write(data)
