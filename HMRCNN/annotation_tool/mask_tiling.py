#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#file usage : python mask_tiling.py ../data
"""
Created on Tue Apr  2 13:27:27 2019

@author: satish
"""

import sys
import os
import cv2
import math
import numpy as np
import logging
import coloredlogs

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviris_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

cur_dir = os.getcwd()
print(cur_dir)
DIRECTORY = "{}/".format(sys.argv[1])
print(DIRECTORY)
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

import os
import shutil
import cv2
import math

for fname in FILES:
    # Construct the mask file name directly from fname
    mask_file_name = f'{fname.split(".")[0]}_mask.png'  # Get base name and append _img_mask.png
    file_check = os.path.join(DIRECTORY, mask_file_name)

    print(f"Checking for mask file: {file_check}")

    # Skip if the mask file does not exist
    if not os.path.isfile(file_check):
        print(f"Mask file not found: {file_check}, continuing to next file.")
        continue

    print(f"Processing file: {mask_file_name}")
    
    img = cv2.imread(file_check)
    if img is None:
        print(f"Error loading image: {file_check}")
        continue

    img_shape = img.shape
    print("Shape of mask:", img_shape)

    tiles_folder = os.path.join(DIRECTORY, f'{fname.split(".")[0]}_mask_tiles')

    if not os.path.isdir(tiles_folder):
        os.mkdir(tiles_folder)
        print("\nDirectory", tiles_folder, "created.")
    else:
        print("\nDirectory", tiles_folder, "already exists..deleting it")
        shutil.rmtree(tiles_folder)
        os.mkdir(tiles_folder)
        print("\nNew Directory", tiles_folder, "created.")

    tile_size = (1024, 1024)
    offset = (512, 512)

    for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            print(f"Creating tile at: ({i}, {j})")
            cropped_img = img[offset[1]*i:min(offset[1]*i + tile_size[1], img_shape[0]),
                              offset[0]*j:min(offset[0]*j + tile_size[0], img_shape[1])]
            tile_name = f'{fname.split(".")[0]}_radiance'
            cv2.imwrite(os.path.join(tiles_folder, f"{tile_name}_{i}_{j}.png"), cropped_img)

    print("Done Tiling mask file", mask_file_name)
