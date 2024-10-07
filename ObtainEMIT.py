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
import earthaccess
import os
import warnings
import argparse
from osgeo import gdal
import numpy as np
import rasterio as rio
import xarray as xr
import getpass
import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emit_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser(description="Instala e converte imagens do sensor EMIT para .envi")
    parser.add_argument('url', type=str, help='URL para a imagem EMIT')
    
    args = parser.parse_args()
    earthaccess.login(persist=True)
    url = args.url

    fs = earthaccess.get_fsspec_https_session()
    granule_asset_id = url.split('/')[-1]
    fp = f'C:/Users/251874/Desktop/EMIT_MaskRCNN/data/{granule_asset_id}'
    if not os.path.isfile(fp):
        print(f"Instalando arquivo de {url}")
        fs.download(url, fp)

    outpath = 'C:/Users/251874/Desktop/EMIT_MaskRCNN/data/envi'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    import sys
    sys.path.append('C:/Users/251874/Desktop/EMIT_MaskRCNN/modules')
    import emit_tools as et

    ds = et.emit_xarray(fp, ortho=True)
    et.write_envi(ds, outpath, overwrite=False, extension='.img', interleave='BIL', glt_file=False)

if __name__ == "__main__":
    main()
