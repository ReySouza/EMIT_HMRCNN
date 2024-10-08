#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#file usage : python json_creator.py ../data
"""
Created on Tue Apr  2 17:19:37 2019

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

FP_REMOVAL = True
DIRECTORY = "{}".format(sys.argv[1])
FILES = [x for x in os.listdir(DIRECTORY)]
print(FILES)

## supplimentary functions for DFS
def is_safe(row,col, count ,row_size, col_size, channel):
    return(row>=0 and col>=0 and row<row_size and col<col_size)

def num_of_sources(row, col, count, row_size, col_size, channel, x_points, y_points):
    itr_DFS(row, col, count, row_size, col_size, channel, x_points, y_points)
    return

# iterative way of doing the DFS search to get the number of sources
# Facing the problem of stack overflow in recursive implementation, iterative approach.
    
def itr_DFS(row, col, count, row_size, col_size, channel, x_points, y_points):
    if((is_safe(row, col, count, row_size, col_size, channel)) == False):
        return
    if(np_image[row][col][channel] >= 200):
        plume_is = [(row,col,channel)]
        while plume_is:
            row,col,channel = plume_is.pop()
            np_image[row][col][channel] = count
            
            #checking neighbours now row,col +- 1
            for delta_row in range(-1,2):
                row_dr = row + delta_row
                if(row_dr>=0 and row_dr<row_size):
                    for delta_col in range(-1,2):
                        col_dc = col + delta_col
                        if (delta_row == 0) ^ (delta_col == 0):
                            if(col_dc>=0 and col_dc<col_size):
                                if(np_image[row_dr][col_dc][channel] >= 200):
                                    plume_is.append((row_dr,col_dc,channel))
                                    x_points.append((int(row_dr)))
                                    y_points.append((int(col_dc)))

#%% creating json file and dumping data(the mask information) in json format
import json
import copy

sample = {
  "default_name.png":
    {
      "filename":"default_name.png",
      "size":-1,
      "regions":[
              {
                "shape_attributes":
                  {
                    "name":"polygon",
                    "all_points_x":[],
                    "all_points_y":[]
                  },
              "region_attributes":
                {
                  "name":"default"
                }
            }
      ]
    }
}                  
            
#%% creating the dictionary for the json file
def new_image_add_json(img_name, coordinates):
    if img_name not in coordinates:
        value = dict(copy.deepcopy(sample["default_name.png"]))
        coordinates[img_name] = value
        coordinates[img_name]["filename"] = img_name
    else:
        print("do something")
    return
    
def new_region_add_json(img_name, coordinates, x_points, y_points, channel):
    if img_name in coordinates:
        region = dict(copy.deepcopy(sample["default_name.png"]["regions"][0]))
        region['shape_attributes']['all_points_x'] = x_points
        region['shape_attributes']['all_points_y'] = y_points
        if (channel == 2):
            region['region_attributes']['name'] = 'point_source'
        if (channel == 0):
            region['region_attributes']['name'] = 'diffused_source'
        if(len(x_points) == 0):
            region['region_attributes']['name'] = 'no_methane'
        coordinates[img_name]["regions"].append(region)
        if (coordinates[img_name]['regions'][0]['region_attributes']['name'] == 'default'):
            print("deleting default")
            del coordinates[img_name]['regions'][0]
    return

#%% creating json file to dump the data
def json_generate(img_name, coordinates, json_file):
    if not os.path.isfile(json_file):
        print("file does not exists", json_file)
        with open(json_file, mode='w') as f:    
		#if file do not exists it will create one and write data to it
            f.write(json.dumps(coordinates, indent=4))
    else:
        print("file exists", json_file)
        with open(json_file) as feedsjson:  #if json file is already there
            feeds = json.load(feedsjson)
    
        feeds.update(coordinates)
        with open(json_file, mode='w') as f:
            f.write(json.dumps(feeds, indent=4))
    return

#%% calculating the number of point and diffused sources
json_file = "../src/custom-mask-rcnn-detector/ch4_data/annotation_plumes.json"

for fname in FILES:
    logger.info(f"Processing file: {fname}")

    file_check = os.path.join(DIRECTORY, fname, f"{fname}_img_mask.png").replace("\\","/")
    if not os.path.isfile(file_check):
        logger.warning(f"Mask file does not exist: {file_check}")
        continue

    png_read = cv2.imread(file_check)
    if png_read is None:
        logger.error(f"Failed to read mask image: {file_check}")
        continue

    # Constructing paths for other required files
    sname = f"{fname}_img"
    hname = f"{sname}.hdr"
    mname = f"{sname}_mask.png"
    
    png_img = os.path.join(DIRECTORY, fname, mname).replace("\\","/")
    print(f"Attempting to read: {png_img}")  # Debugging line

    if not os.path.isfile(png_img):
        logger.error(f"Image file does not exist: {png_img}")
        continue

    png_read = cv2.imread(png_img)
    if png_read is None:
        logger.error(f"Failed to read mask image: {png_img}")
        continue
    img_img = os.path.join(DIRECTORY, fname, sname)
    hdr_file = os.path.join(DIRECTORY, fname, hname)
    png_read = cv2.imread(png_img)
    img_shape = png_read.shape
    tile_size = (1024, 1024)
    offset = (512, 512)

    mask_tiles_dir = os.path.join(DIRECTORY, fname, f"{fname}_img_mask_tiles").replace("\\","/")

for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
    for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
        png_mask_file = os.path.join(mask_tiles_dir, f"{sname}_radiance_{i}_{j}.png").replace("\\", "/")
        png_mask_tile_name = os.path.basename(png_mask_file) 
        logger.debug(f"Processing mask tile: {png_mask_file}")
        print("Files in mask_tiles_dir:", os.listdir(mask_tiles_dir))

        if os.path.isfile(png_mask_file):
            png_mask_read = cv2.imread(png_mask_file)
            if png_mask_read is None:
                print(f"Failed to read image: {png_mask_file}")
                continue  # Skip to the next file if reading fails

            try:
                np_image = np.array(png_mask_read)
                row_size, col_size, bands = np_image.shape
            except ValueError as e:
                print(f"Error getting shape of image: {e}")
                continue  # Skip to the next file if there's an error with the shape

            logger.info(f"Image shape: {row_size}, {col_size}, {bands}")
            print(row_size, col_size, bands)

            point_source = 0  # Number of point sources
            diffused_source = 0  # Number of diffused sources
            source_count = 0
            coordinates = {}  # Dictionary for the JSON file

            new_image_add_json(png_mask_tile_name, coordinates)

            for channel in range((bands - 1), -1, -1):
                source_count = 0
                done = False
                while not done:
                    x_points = []
                    y_points = []
                    colored_spots = np.transpose(np.argmax(np_image[:, :, channel] > 200))  # [:,:,channel] BGR
                    if colored_spots == 0:
                        done = True
                        break

                    source_count += 1
                    row = colored_spots // col_size
                    col = colored_spots % col_size
                    if is_safe(row, col, source_count, row_size, col_size, channel):
                        num_of_sources(row, col, source_count, row_size, col_size, channel, x_points, y_points)

                    if len(x_points) > 0:
                        new_region_add_json(png_mask_tile_name, coordinates, x_points, y_points, channel)
                        json_generate(png_mask_tile_name, coordinates, json_file)  # Dumping everything to the JSON file

                if channel == 2:
                    point_source = source_count
                elif channel == 0:
                    diffused_source = source_count

            print("\npoint_source = ", point_source)
            print("\ndiffused_source = ", diffused_source)

        else:
            print(f"Mask tile file not found: {png_mask_file}")
