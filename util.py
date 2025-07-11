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
import os

from shutil import copy


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
    
def count_rows_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def is_file_valid(filepath):
    try:
        with open(filepath, 'rb') as f:
            while f.read(1024*1024):  # Try reading a small part
                pass
        return True
    except (IOError, FileNotFoundError) as e:
        print(f"Error with file '{filepath}': {e}")
        return False
    
def move_patches(yolo_lables_src, binary_labels_src, imgs_src, dataset_dir, dataset, split):
    images_dir = os.path.join(dataset_dir, 'images', split)
    labels_dir = os.path.join(dataset_dir, 'binary_labels', split)
    yolo_dir = os.path.join(dataset_dir, 'yolo_labels', split)
    
    for dir in [images_dir, labels_dir, yolo_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    for i, patch in enumerate(dataset):
        print(f"\rCopping {i} out of {len(dataset)}", end='', flush=True)
        
        img_nme = patch.replace('.txt', '.png')

        img_src_dir = os.path.join(imgs_src, img_nme)
        lbl_src_dir = os.path.join(binary_labels_src, img_nme)
        yolo_src_dir = os.path.join(yolo_lables_src, patch)

        if is_file_valid(img_src_dir) and is_file_valid(yolo_src_dir):
            # move binary label
            if not os.path.isfile(os.path.join(labels_dir, img_nme)):
                copy(lbl_src_dir, labels_dir)
            # move yolo label
            if not os.path.isfile(os.path.join(yolo_dir, patch)):
                copy(yolo_src_dir, yolo_dir)
            if not os.path.isfile(os.path.join(images_dir, img_nme)):
                copy(img_src_dir, images_dir) 
    
    print('Done') 
    
    
def replace_commas_with_spaces_in_file(file_path):
    try:
        # Try reading the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Fallback to Latin-1 encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()

    # Replace commas with spaces
    updated_content = content.replace(',', ' ')

    # Overwrite the file with updated content using the same encoding used for reading
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
    except UnicodeEncodeError:
        with open(file_path, 'w', encoding='latin-1') as file:
            file.write(updated_content)

