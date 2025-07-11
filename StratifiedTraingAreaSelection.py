#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:28:45 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer



if __name__ == '__main__':
    sql_dir = '/mnt/96729E38729E1D55/03_Dane/BudynkiPolskaDB.gpkg'
    table_name = 'bd_siatka1970_02'
    dst_db = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg'
    
    #buildings_grid_df = load_sqllite_dataframe(sql_dir, table_name)
    buildings_grid_df = gpd.read_file(sql_dir, layer = table_name)
    buildings_grid_df = buildings_grid_df[buildings_grid_df['Point_Coun'] > 200]
    buildings_grid_df = buildings_grid_df.sample(frac=1).reset_index(drop=1)
    
    #limiting samples only to areas we have data for 
    orto_area = gpd.read_file(dst_db, layer='orto_BW_grid_00')
    buildings_grid_df = buildings_grid_df[
        buildings_grid_df.intersects(orto_area.union_all())]
        
    buildings_number = buildings_grid_df['Point_Coun'].to_numpy().reshape(-1,1)
    
    # putting records into classes for stratified sampling
    classifier = KBinsDiscretizer(
        n_bins=6, encode='ordinal', strategy='kmeans')
    classifier.fit(buildings_number)
    
    bins = classifier.bin_edges_[0].tolist()
    classes = classifier.transform(buildings_number).reshape(-1)
    buildings_grid_df['class'] = classes
    
    
    plt.hist(buildings_grid_df['Point_Coun'], bins=120, log=True)
    for bin in bins:
        plt.axvline(bin, linestyle = '--', color='gray')
    plt.grid(False)
    plt.show()
    
    # we do not want test from class 0 - they are almost empty
    bd_df = buildings_grid_df[buildings_grid_df['class'] != 0]
    
    
    # getting the training data location indexes
    bd_idx = bd_df['OBJECTID_1'].to_list()
    classes = bd_df['class']
    test_size = 1 - 12/len(bd_df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        bd_idx, classes, test_size=test_size, stratify=classes, random_state=42,
        shuffle=True)
    
    train_df = buildings_grid_df[buildings_grid_df['OBJECTID_1'].isin(X_train)]
    
    # saving choosen areas back to the geopackage
    #save_datarame_sqllite(train_df, sql_dir, 'bd_trainingAreas_12')
    train_df.to_file(dst_db, layer = 'obszaryTreningoweOrtoBW_00')
    
    
    
    pass