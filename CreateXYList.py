#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:45:00 2024

@author: piotr.szubert@doctoral.uj.edu.pl

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
    
    #dodajemy powiaty 
    powiaty_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/03_Powiaty/Powiaty/Powiaty.shp'
    powiaty_gdf = gpd.read_file(
        powiaty_dir)
    
    
    pnts_powiaty_gdf = gpd.overlay(pnts_gdf, powiaty_gdf, how='intersection')
    
    #oraz gminy
    gminy_dir = '/media/pszubert/DANE/Uniwersytet Jagielloński/PhD Seminar - Piotrs_work - Piotrs_work/02_Budynki_MapJournal/02_Data/BudynkiPolskaDB.gpkg'
    gminy_gdf = gpd.read_file(
        gminy_dir, layer = 'ad_gminy_00')
    #dodajemy bufor do gmin - chcemy mieć punkty też dalej od granic 
    gminy_gdf['geometry'] = gminy_gdf['geometry'].buffer(5000)
    
    
    pnts_powiaty_gdf = gpd.overlay(pnts_powiaty_gdf, gminy_gdf, how='intersection')
    
    pnts_powiaty_gdf.columns
    # pliki do stworzenia listy
    imgs_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/08_Ostrodzki_Probka'
    
    imgs = [_ for _ in os.listdir(imgs_dir) if _.endswith('.tif')]   


    # tabela ze współrzędnymi dla zdjęć
    #chcemy tylko powait ostrodzki
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

dst_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/08_Ostrodzki_Probka'

imgs = ['7_4049.tif', '8_4097.tif', '15_4406.tif', '7_4046.tif', '15_4413.tif', '12_4280.tif', '17_4496.tif', '12_4279.tif', '17_4494.tif', '12_4264.tif', '11_4213.tif', '13_4317.tif', '12_4274.tif', '10_4182.tif', '16_4456.tif', '13_4320.tif', '10_4200.tif', '15_4408.tif', '10_4180.tif', '13_4309.tif', '15_4410.tif', '17_4495.tif', '15_4405.tif', '9_4136.tif', '9_4141.tif', '8_4104.tif', '9_4142.tif', '15_4403.tif', '11_4230.tif', '15_4407.tif', '10_4184.tif', '12_4278.tif', '6_3331.tif', '9_4144.tif', '13_4328.tif', '12_4267.tif', '8_4099.tif', '10_4197.tif', '10_4196.tif', '13_4331.tif', '7_4042.tif', '10_4185.tif', '6_3323.tif', '8_4111.tif', '8_4095.tif', '8_4107.tif', '12_4273.tif', '13_4324.tif', '13_4322.tif', '11_4222.tif', '6_3325.tif', '10_4194.tif', '11_4215.tif', '14_4382.tif', '10_4191.tif', '9_4130.tif', '10_4187.tif', '9_4137.tif', '12_4268.tif', '13_4315.tif', '12_4272.tif', '10_4190.tif', '11_4217.tif', '12_4285.tif', '8_4096.tif', '9_4143.tif', '11_4234.tif', '7_4041.tif', '13_4319.tif', '14_4373.tif', '8_4094.tif', '16_4453.tif', '11_4227.tif', '13_4311.tif', '15_4399.tif', '14_4369.tif', '14_4362.tif', '7_4047.tif', '6_3326.tif', '17_4499.tif', '13_4310.tif', '9_4132.tif', '15_4412.tif', '12_4284.tif', '7_4055.tif', '17_4493.tif', '8_4108.tif', '8_4102.tif', '11_4232.tif', '15_4396.tif', '6_3324.tif', '12_4287.tif', '6_3330.tif', '12_4283.tif', '16_4457.tif', '8_4112.tif', '10_4199.tif', '14_4370.tif', '13_4326.tif', '10_4183.tif', '8_4110.tif', '11_4235.tif', '16_4455.tif', '13_4316.tif', '11_4233.tif', '12_4286.tif', '11_4237.tif', '13_4329.tif', '11_4226.tif', '9_4124.tif', '11_4228.tif', '12_4277.tif', '15_4409.tif', '15_4402.tif', '8_4098.tif', '12_4265.tif', '12_4288.tif', '11_4224.tif', '13_4325.tif', '6_3327.tif', '8_4093.tif', '16_4451.tif', '10_4178.tif', '7_4053.tif', '15_4411.tif', '9_4128.tif', '7_4054.tif', '16_4454.tif', '6_3322.tif', '10_4201.tif', '9_4125.tif', '12_4281.tif', '17_4500.tif', '9_4133.tif', '9_4129.tif', '9_4140.tif', '10_4181.tif', '14_4365.tif', '11_4236.tif', '9_4145.tif', '13_4323.tif', '17_4501.tif', '14_4372.tif', '9_4146.tif', '11_4229.tif', '12_4275.tif', '10_4186.tif', '8_4103.tif', '16_4459.tif', '17_4497.tif', '9_4127.tif', '7_4048.tif', '11_4216.tif', '10_4177.tif', '8_4105.tif', '10_4192.tif', '10_4188.tif', '9_4126.tif', '8_4100.tif', '7_4052.tif', '10_4195.tif', '11_4221.tif', '11_4220.tif', '12_4266.tif', '14_4368.tif', '10_4202.tif', '12_4271.tif', '6_3329.tif', '11_4218.tif', '12_4270.tif', '15_4404.tif', '6_3328.tif', '12_4282.tif', '9_4135.tif', '8_4101.tif', '7_4045.tif', '15_4397.tif', '8_4109.tif', '13_4313.tif', '8_4106.tif', '17_4498.tif', '9_4131.tif']

[os.path.join(dst_dir, img) for img in imgs if os.path.isfile(os.path.join(dst_dir, img))]
