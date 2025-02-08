#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:28:45 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def load_sqllite_dataframe(sql_dir:str, table_name:str):
    connection = sqlite3.connect(sql_dir)
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', connection)
    
    return df

def save_datarame_sqllite(df:pd.DataFrame, sql_dir:str, table_name:str):
    connection = sqlite3.connect(sql_dir)
    df.to_sql(table_name, connection)
    connection.close()


if __name__ == '__main__':
    sql_dir = '/media/pszubert/DANE/Uniwersytet Jagielloński/PhD Seminar - Piotrs_work - Piotrs_work/02_Budynki_MapJournal/02_Data/BudynkiPolskaDB.gpkg'
    table_name = 'bd_siatka1970_2km_02'
    
    buildings_grid_df = load_sqllite_dataframe(sql_dir, table_name)
    buildings_number = buildings_grid_df['Point_Count'].to_numpy().reshape(-1,1)
    
    # putting records into classes for stratified sampling
    classifier = KBinsDiscretizer(
        n_bins=6, encode='ordinal', strategy='kmeans')
    classifier.fit(buildings_number)
    
    bins = classifier.bin_edges_[0].tolist()
    classes = classifier.transform(buildings_number).reshape(-1)
    buildings_grid_df['class'] = classes
    
    
    plt.hist(buildings_grid_df['Point_Count'], bins=120, log=True)
    for bin in bins:
        plt.axvline(bin, linestyle = '--', color='gray')
    plt.grid(False)
    plt.show()
    
    # we do not want test from class 0 - they are almost empty
    bd_df = buildings_grid_df[buildings_grid_df['class'] != 0]
    
    
    # getting the training data location indexes
    bd_idx = bd_df['OBJECTID'].to_list()
    classes = bd_df['class']
    test_size = 1 - 200/len(bd_df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        bd_idx, classes, test_size=test_size, random_state=42, stratify=classes)
    
    train_df = buildings_grid_df[buildings_grid_df['OBJECTID'].isin(X_train)]
    
    # saving choosen areas back to the geopackage
    save_datarame_sqllite(train_df, sql_dir, 'bd_trainingAreas_00')
    
    
    
    pass