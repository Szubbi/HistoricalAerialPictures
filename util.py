#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:56:00 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import numpy as np
import xmltodict
import sqlite3
import pandas as pd


def read_annotations_xml(xml_dir:str) -> np.array:
    with open(xml_dir, 'r') as xml:
        xml_data = xml.read()
        
    xml_dict = xmltodict.parse(xml_data)
    out_points = []
    
    for points_dict in xml_dict['annotations']['image']['points']:
        out_points.extend(
            [[float(pnt) for pnt in _.split(',')] 
             for _ in points_dict['@points'].split(';')])
    
    return np.array(out_points)


def load_sqllite_dataframe(sql_dir:str, table_name:str):
    connection = sqlite3.connect(sql_dir)
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', connection)
    
    return df


def save_datarame_sqllite(df:pd.DataFrame, sql_dir:str, table_name:str):
    connection = sqlite3.connect(sql_dir)
    df.to_sql(table_name, connection)
    connection.close()