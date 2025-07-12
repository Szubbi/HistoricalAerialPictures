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
    
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    orto_extent = gpd.read_file(db_dir, layer = 'obszaryTreningoweOrtoBW_00')
    
    overlap_layers = hash_table[
        hash_table.intersects(orto_extent.union_all())
        ]['file_name'].to_list()
    
    bdot_gdf = gpd.GeoDataFrame(columns = ['file_name', 'geometry'], crs='ETRS_1989_Poland_CS92')
    
    for layer in overlap_layers:
        print(f'Working on {layer}')
        layer_gdf = gpd.read_file(buildings_db_dir, layer=layer).to_crs('ETRS_1989_Poland_CS92')
        # Fix invalid geometries
        layer_gdf = layer_gdf[layer_gdf.geometry.notnull()]
        layer_gdf['geometry'] = layer_gdf['geometry'].buffer(0)
        layer_gdf = layer_gdf[
            layer_gdf.intersects(orto_extent.union_all())]
        bdot_gdf  = pd.concat([bdot_gdf , layer_gdf])
        

    
    bdot_gdf.to_file(db_dir, layer = 'obszaryTreningoweOrtoBW_BDOT_00')
    
    
    