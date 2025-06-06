#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:24:59 2025

@author: pszubert
"""
import os
import numpy as np
import geopandas as gpd

from ImageConverter import ImageConverter
from GenerateTrainingSmple import *
from util import load_sqllite_dataframe
from rasterio.crs import CRS
from DisplayAnnotations import *

if __name__ == "__main__":
    working_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec'
    db_dir = os.path.join(working_dir, '05_Data/Data.gpkg')
    buildings_db_dir = '/media/pszubert/DANE/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    rasters_dir = os.path.join(working_dir, '01_InData/08_OrtoRGB')
    converted_rasters_dir = os.path.join(working_dir, '05_Data/01_ConvertedImages')
    patches_dir = os.path.join(working_dir, '05_Data/02_Patches')
    yolo_labels_dir = os.path.join(working_dir, '05_Data/03_YoloLabels')
    sam_labels_dir = os.path.join(working_dir, '05_Data/04_SamLabels')       
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    blur_sharp_table_BW = load_sqllite_dataframe(db_dir, 'img_BlurSharpTable_04')
    blur_sharp_table_RGB = load_sqllite_dataframe(db_dir, 'img_BlurSharpRGBTable_05')
    rasters = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir)]
    log_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Logger_files'
    PATCH_SIZE = 640
    
    # histograms are bytes, we need to convert them back to numpy first
    blur_sharp_table_BW['histogram'] = blur_sharp_table_BW['histogram'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32))    
    
    # calculated target conversion values
    target_values = generate_stratified_samples(blur_sharp_table_BW, len(rasters))

    # generate and save patches and labels
    for raster, (trg_blur, trg_noise, trg_contrast, trg_hist) in zip(rasters, target_values):
        print(raster, trg_noise, trg_blur)
        
        # convert orto to simulate historical BW images
        IC = ImageConverter(raster, log_dir)
        IC.noise_lvl_trg = trg_noise
        IC.blur_lvl_trg = trg_blur
        IC.contrast_lvl_trg = trg_contrast
        IC.hist_trg = trg_hist
        IC.find_convertion_values(45, 0.8)
        IC.convert_image()
        conv_img, transform = IC.save(
            os.path.join(converted_rasters_dir, IC.img_nme.replace('.tif', '_conv.tif')))
        
        # generate patches and labels - first get pice of of buildings database
        # for faster processing in patches 
        overlaping_layers = get_layers_extent((conv_img, transform), hash_table)
        buildings_gdf = get_geometries(buildings_db_dir, overlaping_layers['file_name'].to_list(), raster)
        
        print(f'Generating Patches for: {IC.img_nme}')
        patches = split_geotiff_to_patches((conv_img, transform), PATCH_SIZE, 0.25)
        
        for index, patch in enumerate(patches):           
            patch_dir = os.path.join(patches_dir, IC.img_nme.replace('.tif', f'_{index}.jpg'))
            yolo_label_dir = os.path.join(yolo_labels_dir, IC.img_nme.replace('.tif', f'_{index}.txt'))
            sam_label_dir = os.path.join(sam_labels_dir, IC.img_nme.replace('.tif', f'_{index}.jpg'))
            
            patch_extent = get_raster_extent(patch)
            bld_masks_gdf = gpd.clip(buildings_gdf, patch_extent)
            # removing multipart polygons
            bld_masks_gdf = bld_masks_gdf.explode(index_parts=False).reset_index(drop=True)
            
            # save patch 
            raster_meta = {'driver': 'JPEG',
                           'dtype': 'uint8', 
                           'nodata': None,
                           'height': patch[0].shape[0], 
                           'width': patch[0].shape[1], 
                           'count': 1, 
                           'crs': CRS.from_epsg(2180), 
                           'transform': patch[1]}
            
            with rasterio.open(patch_dir, 'w', **raster_meta) as dst:
                dst.write(patch[0], 1)
                
            # generate rasterized masks
            binary_mask = rasterize_geometries(src_geom_gdf = bld_masks_gdf, 
                                        raster_path = patch_dir,
                                        mode = 'Binary', 
                                        out_path = sam_label_dir) 
            
            # generate YOLO Txt labels
            yolo_txt = generate_yolo_labels(PATCH_SIZE, patch[1], bld_masks_gdf)
            with open(yolo_label_dir, 'w') as f:
                f.write(yolo_txt)
                
        print(f'Work on {IC.img_nme} completed')
        
    
    print('Process ENDED')
            
          
       
        
      
        
        
        