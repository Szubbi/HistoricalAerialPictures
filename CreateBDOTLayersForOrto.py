#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:26:24 2025

@author: pszubert

"""
import os
import geopandas as gpd
import pandas as pd

from GenerateTrainingSmple import *


if __name__ == "__main__":
    working_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec'
    db_dir = os.path.join(working_dir, '05_Data/Data.gpkg')
    buildings_db_dir = '/mnt/96729E38729E1D55/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    ortos_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/12_testOrtoBW'
    ortos = [_ for _ in os.listdir(ortos_dir) if _.endswith('.tif')]
    
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    
    bdot_gdf = gpd.GeoDataFrame(columns = ['file_name', 'geometry'], crs='ETRS_1989_Poland_CS92')
    
    for orto in ortos:
        print(f'Working on {orto}')
        orto_dir = os.path.join(ortos_dir, orto)
        overlaping_layers = get_layers_extent(orto_dir, hash_table)
        buildings_gdf = get_geometries(buildings_db_dir, overlaping_layers['file_name'].to_list(), orto_dir)

        bdot_gdf  = pd.concat([bdot_gdf , buildings_gdf])
        

    
    bdot_gdf.to_file(db_dir, layer = 'obszaryTestoweOrtoBW_BDOT_00')
    
    
    