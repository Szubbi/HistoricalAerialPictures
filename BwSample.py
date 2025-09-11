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

if __name__ == "__main__":
    working_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec'
    db_dir = os.path.join(working_dir, '05_Data/Data.gpkg')
    buildings_db_dir = '/mnt/96729E38729E1D55/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    rasters_dir = os.path.join(working_dir, '01_InData/08_OrtoRGB')
    converted_rasters_dir = os.path.join(working_dir, '05_Data/01_ConvertedImages')
    patches_dir = os.path.join(working_dir, '05_Data/02_Patches')
    yolo_labels_dir = os.path.join(working_dir, '05_Data/03_YoloLabels')      
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    rasters = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir)]
    log_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Logger_files'
    PATCH_SIZE = 640
    cache_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Processing_caches'
    #cache_file = os.path.join(cache_dir, f'cache_{datetime.now().strftime("%Y%m%dT%H%M%S%f")}')
    cache_file = os.path.join(cache_dir, 'cache_2bw_20250911')
    

    # generate and save patches and labels
    for raster, (trg_blur, trg_noise, trg_contrast, trg_hist) in zip(rasters, target_values):
        print(raster, trg_noise, trg_blur)
        out_dir = os.path.join(converted_rasters_dir, raster.split('/')[-1].replace('.tif', '_conv.tif'))

        if not os.path.isfile(out_dir):
        
            # convert orto to simulate historical BW images
            IC = ImageConverter(raster, log_dir)
            IC.noise_lvl_trg = trg_noise
            IC.blur_lvl_trg = trg_blur
            IC.contrast_lvl_trg = trg_contrast
            IC.hist_trg = trg_hist
            IC.find_convertion_values(45, 0.4)
            IC.convert_image()
            conv_img, transform = IC.save(
                out_dir)
            
            
            # generate patches and labels - first get pice of of buildings database
            # for faster processing in patches 
            overlaping_layers = get_layers_extent((conv_img, transform), hash_table)
            buildings_gdf = get_geometries(buildings_db_dir, overlaping_layers['file_name'].to_list(), raster)
            
            print(f'Generating Patches for: {IC.img_nme}')
            patches = split_geotiff_to_patches((conv_img, transform), PATCH_SIZE, 0.25)
            
            for index, patch in enumerate(patches):           
                patch_dir = os.path.join(patches_dir, IC.img_nme.replace('.tif', f'_{index}.png'))
                yolo_label_dir = os.path.join(yolo_labels_dir, IC.img_nme.replace('.tif', f'_{index}.txt'))
                sam_label_dir = os.path.join(sam_labels_dir, IC.img_nme.replace('.tif', f'_{index}.png'))
                
                patch_extent = get_raster_extent(patch)
                bld_masks_gdf = gpd.clip(buildings_gdf, patch_extent)
                # removing multipart polygons
                bld_masks_gdf = bld_masks_gdf.explode(index_parts=False).reset_index(drop=True)
                
                # save patch 
                raster_meta = {'driver': 'PNG',
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
                yolo_txt = yolo_txt if yolo_txt is not None else ""
                with open(yolo_label_dir, 'w') as f:
                    f.write(yolo_txt)
                    
                
            print(f'Work on {IC.img_nme} completed')
        else:
            print(f'{raster} Already converted')
    
    print('Process ENDED')
      
    
    """
    Split into train/val/test
    Some of the labels are empty - we would like to considier empty areas in 
    in split to achive stratified samples 
    """

      
    #C heck if number of patches and labels is the same
    # Load from cache if available
    with shelve.open(cache_file) as cache:
        if "yolo_labels" not in cache:
            yolo_labels = [_ for _ in os.listdir(yolo_dir) if _.endswith('.txt')]
            cache["yolo_labels"] = yolo_labels
        else: 
            yolo_labels = cache["yolo_labels"]

        if "imgs" not in cache:
            imgs = [_ for _ in os.listdir(patches_dir) if _.endswith('.png')]
            cache["imgs"] = imgs
        else:
            imgs = cache["imgs"]
            
        if "binary_labels" not in cache:
            binary_labels = [_ for _ in os.listdir(labels_dir) if _.endswith('.png')]
            cache["binary_labels"] = binary_labels
        else: 
            binary_labels = cache["binary_labels"]

        if "cat" not in cache:
            print('Calculating categories')
            cat = [count_rows_in_file(os.path.join(yolo_dir, _)) for _ in cache["yolo_labels"]]
            cache["cat"] = cat
        else:
            cat = cache["cat"]

    assert len(imgs) == len(yolo_labels), 'Number of patches and yolo labels is different!'
    assert len(imgs) == len(binary_labels), 'Number of patches and binary labels is different!'
    
    # Using number of buildings as classes fro stratified sample split 
    with shelve.open(cache_file) as cache:
        if "classes" not in cache:
            print('Classifying')

            classifier = KBinsDiscretizer(
                n_bins=6, encode='ordinal', strategy='kmeans')
            classifier.fit(np.array(cat).reshape(-1,1))
            
            bins = classifier.bin_edges_[0].tolist()
            print(bins)
            classes = classifier.transform(np.array(cat).reshape(-1,1)).reshape(-1)

            cache["classes"] = classes
            cache["bins"] = bins
        
        else:
            classes = cache["classes"]
            bins = cache["bins"]

        if "X_train" not in cache and "X_val" not in cache:
            print('Splitting into test/val/test') 

            X_train, X_other, y_train, y_other = train_test_split(
                yolo_labels, classes, stratify=classes, test_size=0.3)            
            X_test, X_val, y_test, y_val = train_test_split(
                X_other, y_other, stratify=y_other, test_size=0.3)
            
            cache["X_train"] = X_train
            cache["X_test"] = X_test
            cache["X_val"] = X_val

        else:
            X_train = cache["X_train"]
            X_test = cache["X_test"]
            X_val = cache["X_val"]


    print(f'Train dataset len: {len(X_train)}')
    print(f'Test dataset len: {len(X_test)}')
    print(f'Val dataset len: {len(X_val)}')    
    
    # Move files to destination                
    DATASET_DIR = '/home/pszubert/Dokumenty/04_ConvDataset'
    
    copy_dict = {
        'train': X_train,
        'test': X_test,
        'val': X_val
    }
    
    
    for split, dataset in copy_dict.items():
        move_patches(yolo_dir, labels_dir, patches_dir, DATASET_DIR, dataset, split)
                  
        
    for split in ['train', 'test', 'val']:
        check_imgs_labels(DATASET_DIR, split)

    # labels needs replaceing commas with spaces
    labels_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/02_YOLO_dataset/labels'
    for root, dirs, files in os.walk(labels_dir):
        files_num = len(files)
        for i, file in enumerate(files):
            file_dir = os.path.join(root, file)
            replace_commas_with_spaces_in_file(file_dir)
            print(f"\rUpdated {i} out of {files_num}", end='', flush=True)
             



    # random tests of dataset
    dataset_dir = '/home/pszubert/Dokumenty/04_ConvDataset'
    split = 'val'
    
    def show_eight_random(dataset_dir, split):
        img_dir = os.path.join(dataset_dir, 'images', split)
        binary_dir = os.path.join(dataset_dir, 'binary_labels', split)
        yolo_dir = os.path.join(dataset_dir, 'labels', split)
        mask_dir = os.path.join(dataset_dir, 'maskrcnn_labels', split)
        
        imgs = [_ for _ in os.listdir(img_dir) if _.endswith('.png')]
        imgs = choices(population=imgs, k=8)
        binary_labels = imgs
        mask_labels = imgs
        yolo_labels = [_.replace('.png', '.txt') for _ in imgs]
    
        imgs = [os.path.join(img_dir, _) for _ in imgs]  
        binary_labels = [os.path.join(binary_dir, _) for _ in binary_labels]
        mask_labels = [os.path.join(mask_dir, _) for _ in mask_labels]
        yolo_labels = [os.path.join(yolo_dir, _) for _ in yolo_labels]
        
        imgs = [Image.open(_) for _ in imgs]
        binary_labels = [Image.open(_) for _ in binary_labels]
        mask_labels = [Image.open(_) for _ in mask_labels]
        
        display_labels(imgs, binary_labels, yolo_labels, mask_labels)
        
    show_eight_random(dataset_dir, split)
        


          
        
      
        
        
