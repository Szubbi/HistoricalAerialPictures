#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:31:51 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import os
import pandas as pd
import ImageConverter as ic

from util import save_datarame_sqllite, load_sqllite_dataframe
from GenerateTrainingSmple import display_histograms

if __name__ == "__main__":
    rasters_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/07_Orto_bw'
    db_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg'
    rasters_list = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir) if _.endswith('.tif')]
    log_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/Logger_files'
    
    results = pd.DataFrame(columns=['name', 'blur', 'noise', 'contrast', 'histogram'])
    
    print('Meassuring BW Images')
    
    for raster in rasters_list:
        img = ic.ImageConverter(raster, log_dir)
        print(f'Working on: {img.img_nme}', end='\r', flush=True)
        results.loc[len(results)] = [img.img_nme, 
                                     img.estimate_blur(img()),
                                     img.estimate_noise(img()),
                                     img.estimate_contrast(img()),
                                     img.measure_histogram(img())]
        
    save_datarame_sqllite(results, db_dir, 'img_BlurSharpTable_04')
    
    rasters_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/08_OrtoRGB'
    rasters_list = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir) if _.endswith('.tif')]
    
    results = pd.DataFrame(columns=['name', 'blur', 'noise', 'contrast', 'histogram'])
    
    print('\nMeassuring RGB Images')
    
    for raster in rasters_list:
        img = ic.ImageConverter(raster, log_dir)
        print(f'Working on: {img.img_nme}', end='\r', flush = True)
        results.loc[len(results)] = [img.img_nme, 
                                     img.estimate_blur(img()),
                                     img.estimate_noise(img()),
                                     img.estimate_contrast(img()),
                                     img.measure_histogram(img())]

  
      
    save_datarame_sqllite(results, db_dir, 'img_BlurSharpRGBTable_05')
    
    BW_df = load_sqllite_dataframe(db_dir, 'img_BlurSharpTable_04')
    RGB_df = load_sqllite_dataframe(db_dir, 'img_BlurSharpRGBTable_05')
    
    BWs = list(zip(BW_df['blur'], BW_df['noise'])) 
    RGBs = list(zip(RGB_df['blur'], RGB_df['noise'])) 
    
    display_histograms(BWs, RGBs, ['BW Orto', 'RGB Orto'])
    
