#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:31:51 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import os
import pandas as pd
import ImageConverter as ic

from util import save_datarame_sqllite

if __name__ == "__main__":
    rasters_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/07_Orto_bw'
    db_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg'
    rasters_list = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir) if _.endswith('.tif')]
    
    results = pd.DataFrame(columns=['name', 'blur', 'noise'])
    
    for raster in rasters_list:
        img = ic.ImageConverter(raster)
        results.loc[len(results)] = [img.img_nme, 
                                     img.estimate_blur(img()),
                                     img.estimate_noise(img())]
        
    save_datarame_sqllite(results, db_dir, 'img_BlurSharpTable_01')
        
    
    
    
    
