#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:25:42 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import os
import fiona
import geopandas as gpd


if __name__ == "__main__":
    dst_db_dir = '/media/pszubert/DANE/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'
    src_dir = '/media/pszubert/DANE/03_Dane/04_BDOT_2023'
    
    existing_layers = fiona.listlayers(dst_db_dir)
    existing_layers = [_.replace('xml', '') for _ in existing_layers]

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_name = file.split('.')[0]
            if 'BUBD' in file_name and file.endswith('.shp') and file_name not in existing_layers:
                print(f'Appending: {file} to GPKG')
                
                file_path = os.path.join(root, file)
                
                gdf = gpd.read_file(file_path)
                gdf.to_file(dst_db_dir, layer=file_name)
                
                            
    print('Done')
                            
                
                    
                
                
