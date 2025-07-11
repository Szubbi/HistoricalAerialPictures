#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:45:00 2024

@author: piotr.szubert@doctoral.uj.edu.pl

Script generating list of images for given study area

"""

import geopandas as gpd
import pandas as pd
import os



if __name__ == '__main__':
    
    # punkty wraz z nazwami plików z plików shp
    files_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/02_Skorowidz'
    
    dataframe_schema = gpd.read_file(
        os.path.join(files_dir, 'srd_51_55.shp')
        ).columns.to_list()

    pnts_gdf = gpd.GeoDataFrame(columns = dataframe_schema)
    
    for shp in os.listdir(files_dir):
        if shp.endswith('.shp'):
            temp_gpd = gpd.read_file(
                os.path.join(files_dir, shp))
            
            pnts_gdf = pd.concat([pnts_gdf, temp_gpd])
            
    pnts_gdf['nazwy_plikow'] = pnts_gdf["numer_szer"] + '_' + pnts_gdf["numer_zdje"] + '.tif'
    pnts_gdf['label'] = pnts_gdf["numer_szer"] + '_' + pnts_gdf["numer_zdje"]
    
    # dodajemy powiaty 
    powiaty_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/03_Powiaty/Powiaty/Powiaty.shp'
    powiaty_gdf = gpd.read_file(
        powiaty_dir)
    
    
    pnts_powiaty_gdf = gpd.overlay(pnts_gdf, powiaty_gdf, how='intersection')
    
    # oraz gminy
    gminy_dir = '/media/pszubert/DANE/Uniwersytet Jagielloński/PhD Seminar - Piotrs_work - Piotrs_work/02_Budynki_MapJournal/02_Data/BudynkiPolskaDB.gpkg'
    gminy_gdf = gpd.read_file(
        gminy_dir, layer = 'ad_gminy_00')
    # dodajemy bufor do gmin - chcemy mieć punkty też dalej od granic 
    gminy_gdf['geometry'] = gminy_gdf['geometry'].buffer(5000)
    
    
    pnts_powiaty_gdf = gpd.overlay(pnts_powiaty_gdf, gminy_gdf, how='intersection')
    
    pnts_powiaty_gdf.columns
    # pliki do stworzenia listy
    imgs_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/08_Ostrodzki_Probka'
    
    imgs = [_ for _ in os.listdir(imgs_dir) if _.endswith('.tif')]   


    # tabela ze współrzędnymi dla zdjęć
    # chcemy tylko powait ostrodzki
    pnts_imgs = pnts_powiaty_gdf[pnts_powiaty_gdf['JPT_KOD_JE_2'] == '2815073']
    
    pnts_imgs = pnts_imgs[pnts_imgs['rok_wykona'] < 1972]
    pnts_imgs = pnts_imgs[pnts_imgs['rok_wykona'] > 1968]
    pnts_imgs = pnts_imgs[pnts_imgs['nazwy_plikow'].isin(imgs)]
    pnts_imgs = pnts_imgs.to_crs(epsg=4326)
    
    pnts_imgs['x'] = pnts_imgs.geometry.x
    pnts_imgs['y'] = pnts_imgs.geometry.y
    pnts_imgs = pnts_imgs.drop_duplicates(['nazwy_plikow', 'x', 'y'])
    
    pnts_imgs[['nazwy_plikow', 'x', 'y']].to_csv(os.path.join(imgs_dir, 'coordinates.csv'))
    
    pnts_imgs.to_file(os.path.join(imgs_dir, 'test.shp'))

    
    
    print(list(set(pnts_imgs['nazwy_plikow'].to_list())))
