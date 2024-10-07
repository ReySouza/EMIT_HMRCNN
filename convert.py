#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Criado em 30 de setembro de 2024

@autor: Reynaldo Souza de Carvalho
https://github.com/nasa/EMIT-Data-Resources.git
Email: LPDAAC@usgs.gov
Voice: +1-866-573-3222
Organization: Land Processes Distributed Active Archive Center (LP DAAC)¹
Website: https://lpdaac.usgs.gov/
"""

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
