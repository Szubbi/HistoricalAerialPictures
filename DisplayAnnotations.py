#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:57:11 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import os
import matplotlib.pyplot as plt

from cv2 import imread

def get_coordinates(coords_str_list:str) -> list:
    points_tupples = []
    coords_str_list = [_.strip()[1:] for _ in coords_str_list.split('\n')]
    
    for str_point_list in coords_str_list:
        for str_coordinates in str_point_list.split(' '):
            if len(str_coordinates) > 1:
                coords = tuple(float(_)*640 for _ in str_coordinates.split(','))
                points_tupples.append(coords)
           
            
    return points_tupples

def display_points_image(img_dir:str, points_tupples:list):
    x_crds, y_crds = zip(*points_tupples)
    
    plt.figure(figsize=(16,9))
    
    plt.subplot(121)
    plt.imshow(imread(img_dir), cmap='gray')
    plt.title('Raw Image')
    
    plt.subplot(122)
    plt.imshow(imread(img_dir), cmap='gray')
    plt.scatter(x_crds, y_crds, color = 'red', marker='x')    
    plt.title('Image with Labels')
      
    
    plt.show()
    

def display_yolo_annotations(img_dir:str, polygons_txt:str, patch_size=int):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    polygons = polygons_txt.strip().split('\n')

    axes[0].imshow(imread(img_dir), cmap='gray')
    axes[0].set_title('Raw Image')
    
    axes[1].imshow(imread(img_dir), cmap='gray')
    # Iterate over each polygon
    for polygon in polygons:
        coords = polygon.split(' ')[1:]  # Skip the leading '0'
        if len(coords) > 1:
            x_coords = [float(pair.split(',')[0]) * patch_size for pair in coords]
            y_coords = [float(pair.split(',')[1]) * patch_size for pair in coords]
            axes[1].fill(x_coords, y_coords, color='blue', alpha=0.5)
    axes[1].set_title('Image with Labels')
    

    plt.tight_layout()
    plt.show()

    
    
if __name__ == '__main__':
    pass
