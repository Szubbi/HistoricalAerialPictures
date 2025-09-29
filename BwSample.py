#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:24:59 2025

@author: pszubert
"""
import os
import numpy as np
import geopandas as gpd
import shelve

from ImageConverter import ImageConverter
from GenerateTrainingSmple import *
from util import *
from rasterio.crs import CRS
from DisplayAnnotations import *
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import KBinsDiscretizer
from shutil import copy
from datetime import datetime
from PIL import Image
from random import choices
from shutil import copy
from Mask_RCNN.convert_annotations import create_mask_rcnn_labels

if __name__ == "__main__":
    working_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec'
    db_dir = os.path.join(working_dir, '05_Data/Data.gpkg')
    buildings_db_dir = '/mnt/96729E38729E1D55/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    rasters_dir = os.path.join(working_dir, '01_InData/08_OrtoRGB')
    converted_rasters_dir = os.path.join(working_dir, '05_Data/05_BW_Dataset/01_ConvertedImages')
    patches_dir = os.path.join(working_dir, '05_Data/05_BW_Dataset/02_Patches')
    yolo_labels_dir = os.path.join(working_dir, '05_Data/05_BW_Dataset/03_YoloLabels')   
    mask_labels_dir = os.path.join(working_dir, '05_Data/05_BW_Dataset/04_MaskRCNNLabels')  
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    rasters = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir)]
    log_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Logger_files'
    PATCH_SIZE = 640
    cache_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Processing_caches'
    #cache_file = os.path.join(cache_dir, f'cache_{datetime.now().strftime("%Y%m%dT%H%M%S%f")}')
    cache_file = os.path.join(cache_dir, 'cache_2bw_20250911')
    

    # 1. Cache and dirs
    if not os.path.exists(cache_file):
        os.makedirs(cache_dir, exist_ok=True)
        cache = shelve.open(cache_file)
        cache['processed_rasters'] = []
        cache['created_patches'] = []
        cache['created_labels'] = []
        cache.close()

    with shelve.open(cache_file) as cache:
        processed_rasters = cache['processed_rasters']  
        created_patches = cache['created_patches']
        created_labels = cache['created_labels']   

    for dir_ in [converted_rasters_dir, patches_dir, yolo_labels_dir, mask_labels_dir]:
        os.makedirs(dir_, exist_ok=True)


    # 2. Convert rasters
        for raster in rasters:

            if raster in processed_rasters:
                print(f'Skipping {os.path.basename(raster)} - already processed')
                continue

            print(f'Processing {os.path.basename(raster)}')
            IC = ImageConverter(raster, log_dir)
            out_dir = os.path.join(converted_rasters_dir, 
                                   raster.split('/')[-1].replace('.tif', '_BW.tif'))
            conv_img, transform = IC.save(out_dir)

            # update cache
            processed_rasters.append(raster)
            cache['processed_rasters'] = processed_rasters

    # 3. Generate patches and labels
            print(f'Generating patches and labels for {os.path.basename(IC.img_nme)}')
            overlapping_layers = get_layers_extent((conv_img, transform), hash_table)
            buildings_gdf = get_geometries(buildings_db_dir, 
                                           overlapping_layers['file_name'].to_list())
            patches = split_geotiff_to_patches((conv_img, transform),
                                               PATCH_SIZE,
                                               overlap=0.25)
            
            for idx, patch in enumerate(patches):
                print_progress_bar(
                    idx + 1, len(patches), 
                    prefix = 'Progress:', 
                    suffix = IC.img_nme, 
                    length = 50)

                patch_dir = os.path.join(patches_dir, IC.img_nme.replace('.tif', f'_{idx}.png'))
                yolo_label_dir = os.path.join(yolo_labels_dir, IC.img_nme.replace('.tif', f'_{idx}.txt'))
                mask_label_dir = os.path.join(mask_labels_dir, IC.img_nme.replace('.tif', f'_{idx}.png'))
                
                # get building geometries in the patch
                patch_extent = get_raster_extent(patch)
                bld_masks_gdf = gpd.clip(buildings_gdf, patch_extent)
                bld_masks_gdf = bld_masks_gdf.explode(
                    index_parts=False).reset_index(drop=True)
                
                # define raster metadata for rasterio 
                raster_meta = {
                    'driver': 'PNG',
                    'height': patch[0].shape[0],
                    'width': patch[0].shape[1],
                    'count': 1,
                    'dtype': 'uint8',
                    'crs': CRS.from_epsg(2180),
                    'transform': patch[1],
                    'nodata': None
                }

                # save patch
                with rasterio.open(patch_dir, 'w', **raster_meta) as dst:
                    dst.write(patch[0], 1)

                # generate yolo label
                yolo_txt = generate_yolo_labels(PATCH_SIZE, patch[1], bld_masks_gdf)
                yolo_txt = yolo_txt if yolo_txt else ''  # handle empty labels
                with open(yolo_label_dir, 'w') as f:
                    f.write(yolo_txt)

                # create mask R-CNN labels using yolo txt
                create_mask_rcnn_labels(yolo_label_dir, PATCH_SIZE, mask_label_dir)

            # update cache
            created_patches.append(raster)
            cache['created_patches'] = created_patches
            created_labels.append(raster)
            cache['created_labels'] = created_labels



            







    
        


          
        
      
        
        
