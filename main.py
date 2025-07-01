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

if __name__ == "__main__":
    working_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec'
    db_dir = os.path.join(working_dir, '05_Data/Data.gpkg')
    buildings_db_dir = '/mnt/96729E38729E1D55/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    rasters_dir = os.path.join(working_dir, '01_InData/08_OrtoRGB')
    converted_rasters_dir = os.path.join(working_dir, '05_Data/01_ConvertedImages')
    patches_dir = os.path.join(working_dir, '05_Data/02_Patches')
    yolo_labels_dir = os.path.join(working_dir, '05_Data/03_YoloLabels')
    sam_labels_dir = os.path.join(working_dir, '05_Data/04_SamLabels')       
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    blur_sharp_table_BW = load_sqllite_dataframe(db_dir, 'img_BlurSharpTable_04')
    blur_sharp_table_RGB = load_sqllite_dataframe(db_dir, 'img_BlurSharpRGBTable_05')
    rasters = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir)]
    log_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Logger_files'
    PATCH_SIZE = 640
    
    # # histograms are bytes, we need to convert them back to numpy first
    # blur_sharp_table_BW['histogram'] = blur_sharp_table_BW['histogram'].apply(
    #     lambda x: np.frombuffer(x, dtype=np.float32))    
    
    # # calculated target conversion values
    # target_values = generate_stratified_samples(blur_sharp_table_BW, len(rasters))

    # # generate and save patches and labels
    # for raster, (trg_blur, trg_noise, trg_contrast, trg_hist) in zip(rasters, target_values):
    #     print(raster, trg_noise, trg_blur)
    #     out_dir = os.path.join(converted_rasters_dir, raster.split('/')[-1].replace('.tif', '_conv.tif'))

    #     if not os.path.isfile(out_dir):
        
    #         # convert orto to simulate historical BW images
    #         IC = ImageConverter(raster, log_dir)
    #         IC.noise_lvl_trg = trg_noise
    #         IC.blur_lvl_trg = trg_blur
    #         IC.contrast_lvl_trg = trg_contrast
    #         IC.hist_trg = trg_hist
    #         IC.find_convertion_values(45, 0.4)
    #         IC.convert_image()
    #         conv_img, transform = IC.save(
    #             out_dir)
            
            
    #         # generate patches and labels - first get pice of of buildings database
    #         # for faster processing in patches 
    #         overlaping_layers = get_layers_extent((conv_img, transform), hash_table)
    #         buildings_gdf = get_geometries(buildings_db_dir, overlaping_layers['file_name'].to_list(), raster)
            
    #         print(f'Generating Patches for: {IC.img_nme}')
    #         patches = split_geotiff_to_patches((conv_img, transform), PATCH_SIZE, 0.25)
            
    #         for index, patch in enumerate(patches):           
    #             patch_dir = os.path.join(patches_dir, IC.img_nme.replace('.tif', f'_{index}.png'))
    #             yolo_label_dir = os.path.join(yolo_labels_dir, IC.img_nme.replace('.tif', f'_{index}.txt'))
    #             sam_label_dir = os.path.join(sam_labels_dir, IC.img_nme.replace('.tif', f'_{index}.png'))
                
    #             patch_extent = get_raster_extent(patch)
    #             bld_masks_gdf = gpd.clip(buildings_gdf, patch_extent)
    #             # removing multipart polygons
    #             bld_masks_gdf = bld_masks_gdf.explode(index_parts=False).reset_index(drop=True)
                
    #             # save patch 
    #             raster_meta = {'driver': 'PNG',
    #                         'dtype': 'uint8', 
    #                         'nodata': None,
    #                         'height': patch[0].shape[0], 
    #                         'width': patch[0].shape[1], 
    #                         'count': 1, 
    #                         'crs': CRS.from_epsg(2180), 
    #                         'transform': patch[1]}
                
    #             with rasterio.open(patch_dir, 'w', **raster_meta) as dst:
    #                 dst.write(patch[0], 1)
                    
    #             # generate rasterized masks
    #             binary_mask = rasterize_geometries(src_geom_gdf = bld_masks_gdf, 
    #                                         raster_path = patch_dir,
    #                                         mode = 'Binary', 
    #                                         out_path = sam_label_dir) 
                
    #             # generate YOLO Txt labels
    #             yolo_txt = generate_yolo_labels(PATCH_SIZE, patch[1], bld_masks_gdf)
    #             yolo_txt = yolo_txt if yolo_txt is not None else ""
    #             with open(yolo_label_dir, 'w') as f:
    #                 f.write(yolo_txt)
                    
                
    #         print(f'Work on {IC.img_nme} completed')
    #     else:
    #         print(f'{raster} Already converted')
    
    # print('Process ENDED')
    
    """
    Split into train/val/test
    Some of the labels are empty - we would like to considier empty areas in 
    in split to achive stratified samples 
    """

      
    # Check if number of patches and labels is the same
    # Load from cache if available
    with shelve.open("my_cache_shelf") as cache:
        if "patches" not in cache:
            patches = [_ for _ in os.listdir(yolo_labels_dir)]
            cache["patches"] = patches
        else: 
            patches = cache["patches"]

        if "imgs" not in cache:
            imgs = [_ for _ in os.listdir(patches_dir) if _.endswith('.jpg')]
            cache["imgs"] = imgs
        else:
            imgs = cache["imgs"]

        if "cat" not in cache:
            print('Calculating categories')
            cat = [count_rows_in_file(os.path.join(yolo_labels_dir, _)) for _ in cache["patches"]]
            cache["cat"] = cat
        else:
            cat = cache["cat"]

    assert len(imgs) == len(patches), 'Number of patches and labels is different!'
    
    # Using number of buildings as classes fro stratified sample split 
    with shelve.open("my_cache_shelf") as cache:
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
                patches, classes, stratify=classes, test_size=0.3)            
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
    DATASET_DIR = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/02_YOLO_dataset'
    
    copy_dict = {
        'train': X_train,
        'test': X_test,
        'val': X_val
    }
    
    for split, dataset in copy_dict.items():
        move_patches(yolo_labels_dir, patches_dir, DATASET_DIR, dataset, split)
        
        
    # Check if the labels and images fits 
    def check_imgs_labels(dataset_dir, split):
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        
        imgs = [_.split('.')[0] for _ in os.listdir(images_dir)]
        labels = [_.split('.')[0] for _ in os.listdir(labels_dir)]
        
        diff_imgs = list(set(imgs) - set(labels))
        diff_labs = list(set(labels) - set(imgs))
        
        if len(diff_imgs) > 0 or len(diff_labs) > 0:
            if len(diff_imgs) > 0:
                print('Not every img has label: ')
                for _ in diff_imgs:
                    print(_)
            if len(diff_labs) > 0:
                print('Not every label has img: ')
                for _ in diff_labs:
                    print(_)
        else:
            print('All imgs has matching labels')
            
        
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
             



        
    
    
    
    
    
    
    
    


          
        
      
        
        
