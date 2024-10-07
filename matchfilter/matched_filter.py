#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Criado em 04 de outubro de 2024

@autor: Reynaldo Souza de Carvalho
https://github.com/nasa/EMIT-Data-Resources.git
Email: LPDAAC@usgs.gov
Voice: +1-866-573-3222
Organization: Land Processes Distributed Active Archive Center (LP DAAC)¹
Website: https://lpdaac.usgs.gov/
"""

import os
import cv2
import numpy as np
import spectral as spy
import spectral.io.envi as envi
import spectral.algorithms as algo
from spectral.algorithms.detectors import MatchedFilter, matched_filter

import logging
import coloredlogs

import shutil
import statistics

# set the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emit_data_loader")
coloredlogs.install(level='DEBUG', logger=logger)

DIRECTORY = "C:/Users/251874/Desktop/EMIT_MaskRCNN/data/envi"
FILES = []
for x in os.listdir(DIRECTORY):        
    if(os.path.isdir(os.path.join(DIRECTORY, x))):
        FILES.append(x)
print(FILES)

#load gas signature
t_sig_path = "C:/Users/251874/Desktop/EMIT_MaskRCNN/data/gas_signature"
t_mean = np.loadtxt(os.path.join(t_sig_path, os.listdir(t_sig_path)[0]))[:,-1]
print(t_mean.shape)

gt_image_hdr_path = 'C:/Users/251874/Desktop/EMIT_MaskRCNN/data/gt_path/EMIT_L2B_CH4ENH_001_20220815T042838_2222703_003.hdr'
data_file_path = gt_image_hdr_path.replace('.hdr', '.img')  # or replace with .dat if that is the correct extension

# Check if the header and data files exist
if not os.path.exists(gt_image_hdr_path):
    logger.error(f"Header file does not exist: {gt_image_hdr_path}")
    raise FileNotFoundError("Header file not found.")

if not os.path.exists(data_file_path):
    logger.error(f"Data file does not exist: {data_file_path}")
    raise FileNotFoundError("Data file not found.")

# Open the ENVI file
try:
    gt_image = envi.open(gt_image_hdr_path, data_file_path)
    gt_image_data = gt_image.load()  # Load the image data into a numpy array
except Exception as e:
    logger.error(f"Failed to load the ENVI image. Error: {e}")
    raise FileNotFoundError("ENVI image file not found or cannot be read.")

# Check the shape of the image data
if len(gt_image_data.shape) == 2:
    gt_image_data = np.expand_dims(gt_image_data, axis=-1)


#If target signature is not available then we can compute from the image itself to get rough estimate.
class target_sig:
    def __init__(self):
        print("[DEBUG] Initialized target_sig class.")

    def is_safe(self, gt_image, row, col, count, row_size, col_size, channel):
        result = row >= 0 and col >= 0 and row < row_size and col < col_size
        print(f"[DEBUG] Checking is_safe for row: {row}, col: {col} -> {result}")
        return result

    def itr_DFS(self, gt_image, row, col, count, row_size, col_size, channel, x_points, y_points):
        print(f"[DEBUG] Starting itr_DFS at row: {row}, col: {col}, channel: {channel}")

    if not self.is_safe(gt_image, row, col, count, row_size, col_size, channel):
        print("[DEBUG] itr_DFS: Position is not safe.")
        return
    
    # Convert gt_image to a NumPy array
    gt_image_data = gt_image.read_bands()  # Read all bands or specific ones as needed

    # Print shape and type of gt_image for debugging
    print(f"[DEBUG] gt_image shape: {gt_image_data.shape}, dtype: {gt_image_data.dtype}")

    # Check if gt_image has 3 dimensions
    if len(gt_image_data.shape) == 3:
        # Properly access the pixel value with NumPy indexing
        if gt_image_data[row, col, channel] >= 200:
            print(f"[DEBUG] Pixel value {gt_image_data[row, col, channel]} >= 200, exploring neighbors...")
            plume_is = [(row, col, channel)]
            while plume_is:
                row, col, channel = plume_is.pop()
                gt_image_data[row, col, channel] = count  # Assign the count to the NumPy array
                print(f"[DEBUG] Marked pixel at row: {row}, col: {col}, channel: {channel} with count: {count}")
                for delta_row in range(-1, 2):
                    row_dr = row + delta_row
                    if 0 <= row_dr < row_size:
                        for delta_col in range(-1, 2):
                            col_dc = col + delta_col
                            if (delta_row == 0) ^ (delta_col == 0):
                                if 0 <= col_dc < col_size:
                                    # Correctly access the pixel value
                                    if gt_image_data[row_dr, col_dc, channel] >= 200:
                                        plume_is.append((row_dr, col_dc, channel))
                                        x_points.append(int(row_dr))
                                        y_points.append(int(col_dc))
                                        print(f"[DEBUG] Added neighbor row: {row_dr}, col: {col_dc} to plume.")
    else:
        print("[DEBUG] Warning: gt_image is not a 3D array.")


    def target_sign(self, GTfile_path, img_data_obj, channel, good_bands):
        print("[DEBUG] Calculating target signature from ground truth.")
        source_count = 0
        norm_value = 0
        t_mean = np.zeros((channel,), dtype=np.float64)
        r_size, c_size, bands = gt_image.shape

        for channel in range((bands-1), -1, -1):
            print(f"[DEBUG] Processing channel: {channel}")
            done = False
            while not done:
                x_points = []
                y_points = []
                clrd_spot = np.transpose(np.argmax(gt_image[:, :, channel] > 200))
                if clrd_spot == 0:
                    done = True
                    print("[DEBUG] No more target spots found in this channel.")
                    break

                r = clrd_spot // c_size
                c = clrd_spot % c_size
                print(f"[DEBUG] Found target spot at row: {r}, col: {c}")
                if self.is_safe(gt_image, r, c, source_count, r_size, c_size, channel):
                    self.itr_DFS(gt_image, r, c, source_count, r_size, c_size, channel, x_points, y_points)
                    norm_value += len(x_points)
                    print(f"[DEBUG] Found {len(x_points)} points for target in this region.")
                    rect_mean = self.enclosing_rect_mean(x_points, y_points, img_data_obj, good_bands)
                    t_mean += np.array(rect_mean) * len(x_points)

        t_mean = np.array(t_mean) / norm_value if norm_value != 0 else t_mean
        print(f"[DEBUG] Final target mean shape: {t_mean.shape}")
        print(f"[DEBUG] Target mean values: {t_mean}")
        return t_mean
    
    def sort_coordinates(self, x_points, y_points):
        print("[DEBUG] Sorting coordinates.")
        if len(x_points) > 1:
            mid = len(x_points) // 2
            L_x, L_y = x_points[:mid], y_points[:mid]
            R_x, R_y = x_points[mid:], y_points[mid:]
            self.sort_coordinates(L_x, L_y)
            self.sort_coordinates(R_x, R_y)

            i = j = k = 0
            while i < len(L_x) and j < len(R_x):
                if L_x[i] < R_x[j]:
                    x_points[k], y_points[k] = L_x[i], L_y[i]
                    i += 1
                else:
                    x_points[k], y_points[k] = R_x[j], R_y[j]
                    j += 1
                k += 1

            while i < len(L_x):
                x_points[k], y_points[k] = L_x[i], L_y[i]
                i += 1
                k += 1

            while j < len(R_x):
                x_points[k], y_points[k] = R_x[j], R_y[j]
                j += 1
                k += 1

        print(f"[DEBUG] Sorted coordinates: {x_points[:5]}... (showing first 5)")
        return x_points, y_points

    def enclosing_rect_mean(self, x_points, y_points, img_data_obj, good_bands):
        print("[DEBUG] Calculating enclosing rectangle mean.")
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)
        print(f"[DEBUG] Bounding box: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        
        x_points, y_points = np.array(x_points), np.array(y_points)
        mask_x = x_points - min_x
        mask_y = y_points - min_y

        t_img_data = img_data_obj.read_subregion((min_x, max_x), (min_y, max_y), good_bands)
        mask = np.zeros((max(mask_x + 1), max(mask_y + 1)))
        mask[mask_x, mask_y] = 1
        print(f"[DEBUG] Size of mask: {mask.shape}")

        print("[DEBUG] Done reading target image. Calculating target mean...")
        t_mean, t_cov, t_S = algo.mean_cov(t_img_data, mask=None, index=None)
        print(f"[DEBUG] Calculated target mean: {t_mean}")

        return t_mean

sig_calculator = target_sig()

def image_obj(hdr, img):
    head = envi.read_envi_header(hdr)
    param = envi.gen_params(head)
    param.filename = img   # spectral data file corresponding to .hdr file
    
    # Extract good_bands from header or use all bands
    if 'bbl' in head:
        head['bbl'] = [float(b) for b in head['bbl']]
        print(len(head['bbl']))
        good_bands = np.nonzero(head['bbl'])[0]  # get good bands only for computation
        bad_bands = np.nonzero(np.array(head['bbl']) == 0)[0]
        print(f"Good bands: {good_bands}")
        print(f"Bad bands: {bad_bands}")
    else:
        # Ensure 'bands' is converted to an integer before using it
        num_bands = int(head['bands'])
        good_bands = np.arange(num_bands)  # If 'bbl' is not available, use all bands
        print(f"No BBL found in header, using all {num_bands} bands as good bands.")

    interleave = head['interleave']
    if interleave.lower() == 'bip':
        print("It is a BIP file")
        from spectral.io.bipfile import BipFile
        img_obj = BipFile(param, head)
        
    elif interleave.lower() == 'bil':
        print("It is a BIL file")
        from spectral.io.bilfile import BilFile
        img_obj = BilFile(param, head)
        
    # Return both the image object and good_bands
    return img_obj, good_bands

#%% creates a list of bands to read everytime    
def set_bands_toread(channel):
    overlap = 100; reset_pos = 0 ; num_channel = 200; start_pos=0
    overall_set = []
    band_set = []
    while(start_pos<channel):
        band_set.append(start_pos)
        start_pos+=1 ; reset_pos+=1
        if((reset_pos%num_channel)==0):
            reset_pos = 0
            overall_set.append(band_set)
            band_set = []
            start_pos = start_pos-overlap
    overall_set.append(band_set)
    return overall_set

#%% Matched Filter
for fname in FILES:
    sname   = f'{fname}'  # spectral data
    hname   = f'{sname}.hdr'  # header 
    dir_path= f'{DIRECTORY}/{fname}'
    img_img = f'{DIRECTORY}/{fname}/{sname}.img'
    hdr_img = f'{DIRECTORY}/{fname}/{hname}'

    print(f"Processing file: {fname}")
    print(f"Spectral data path: {img_img}")
    print(f"Header file path: {hdr_img}")

    # Load image dimensions
    row_size, col_size, channel = spy.envi.open(hdr_img).shape
    print(f"Number of bands (channels): {channel}")
    
    # Get the image object and good_bands from image_obj()
    print("[DEBUG] Calling image_obj to retrieve image object and good bands.")
    img_data_obj, good_bands = image_obj(hdr_img, img_img)

    print(f"[DEBUG] good_bands: {good_bands}")  # Check if good_bands is populated
    
    # Now call target_sign and pass good_bands
    print("[DEBUG] Calling target_sign to calculate the target signature.")
    t_mean = sig_calculator.target_sign(hdr_img, img_data_obj, channel, good_bands)  # <- Pass good_bands correctly
    
    # Handle the case where t_mean's size does not match the channel size
    if (channel - t_mean.size) > 0:
        target_mean = np.append(t_mean, np.zeros((channel - t_mean.size)))
    elif (channel - t_mean.size) < 0:
        target_mean = t_mean[0:channel]
        
    mf_output_folder = f'{dir_path}/{sname}_mfout'
    if not(os.path.isdir(mf_output_folder)):
        os.mkdir(mf_output_folder)
        print("\nDirectory", mf_output_folder ," created.")
    else:
        print("\nDirectory", mf_output_folder ," already exists..deleting it")
        shutil.rmtree(mf_output_folder)
        os.mkdir(mf_output_folder)
        print("\nNew Directory", mf_output_folder ," created.")
    
    #reading the whole image at once as we have a 128GB of RAM size
    print("Reading image.. ", fname)
    big_img_data = img_data_obj.read_subregion((0,row_size), (0,col_size))
    
    start_pos = 0; overlap = 0; bands_in_set = 285; reset_pos = 0
while start_pos < channel:
    alpha = np.zeros((row_size, 0), dtype=np.float)

    end_pos = min(start_pos + bands_in_set, channel + 1)
    print("taking sub-section from the big_img_data background", start_pos, "to", end_pos)
    b_img_data = big_img_data[:, :, start_pos:end_pos].copy()

    for columns in range(0, b_img_data.shape[1],3000):
        print("Calculating gaussian stats, mean, cov of background")
        col_range = min(columns + 3000, b_img_data.shape[1])
        b_mean_cov_obj = algo.calc_stats(b_img_data[:, columns:col_range, :], mask=None, index=None)
        cov_matrix = b_mean_cov_obj.cov
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        print("Calculating stats of matchFilter...")

        print("b_img_data", b_img_data[:, columns:col_range, :].shape, "target_mean : ", \
              target_mean[start_pos:start_pos + bands_in_set].shape)

        try:
            alpha = np.concatenate((alpha, matched_filter(b_img_data[:, columns:col_range, :], \
                    target_mean[start_pos:start_pos + bands_in_set], b_mean_cov_obj)), axis=1)
            print("Shape of Alpha : ", alpha.shape)
        except np.linalg.LinAlgError as e:
            logger.error(f"LinAlgError during matched_filter: {e}")
            continue

    b_img_data = None
    del b_img_data
    bands_set_end = start_pos + bands_in_set
    processed_file = f'{mf_output_folder}/{sname}_{start_pos}_{end_pos}.npy'
    print("Saving the alpha output", processed_file)
    np.save(processed_file, alpha)

    start_pos += bands_in_set
    reset_pos += bands_in_set

    if start_pos > channel:
        break

    if (reset_pos % bands_in_set) == 0:
        start_pos = start_pos - overlap
        reset_pos = 0

    alpha_data = np.load(processed_file)

    # Define the ENVI file paths
    envi_output_file = f'{mf_output_folder}/{sname}_{start_pos}_{end_pos}.img'
    envi_hdr_file = f'{envi_output_file}.hdr'

    # Save the array data as an ENVI image
    metadata = {
        'description': f'Processed output from matched_filter from {start_pos} to {end_pos}',
        'samples': alpha_data.shape[1],  # number of columns
        'lines': alpha_data.shape[0],    # number of rows
        'bands': 1,                      # one band (because matched filter outputs a single layer)
        'interleave': 'bil',             # interleave format (could be 'bil' or 'bip' depending on your needs)
        'data type': 5,                  # 5 corresponds to 32-bit float in ENVI
        'byte order': 0                  # byte order (0 for little-endian, 1 for big-endian)
    }

    # Save the alpha data in ENVI format
    spy.envi.save_image(envi_hdr_file, alpha_data, dtype=np.float32, interleave='bil', metadata=metadata)

    print(f"Alpha data saved as ENVI file: {envi_output_file}")
