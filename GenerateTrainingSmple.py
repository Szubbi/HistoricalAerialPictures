#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:28:49 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import geopandas as gpd
import pandas as pd
import rasterio
import sqlite3
import fiona 
import matplotlib.pyplot as plt
import numpy
import os

from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.windows import Window


def spatial_hash_table(src_db_dir:str):
    bbox_gdf = gpd.GeoDataFrame(columns=['file_name', 'geometry'], 
                                crs = 'EPSG:2180')
    
    # get all the alyers of gpkg
    layers = fiona.listlayers(src_db_dir)
    
    print(f'{len(layers)} tables to process')
    
    for number, layer in enumerate(layers):
        if 'BUBD' in layer:
            print('\x1b[2K' + f'Working on {number + 1} of {len(layers)}: {layer}', 
                  end="\r")
            
            with fiona.open(src_db_dir, layer=layer) as layer_src:
                bounds = layer_src.bounds
                minx, miny, maxx, maxy = bounds
                bounding_box = box(minx, miny, maxx, maxy)
                
                row = gpd.GeoDataFrame({
                    'file_name': [layer],
                    'geometry': [bounding_box]
                }, crs='EPSG:2180')
                
                bbox_gdf = gpd.GeoDataFrame(pd.concat([bbox_gdf, row], 
                                                      ignore_index=True))
            
    
    print("\nDone")           
    return bbox_gdf
            

def get_raster_extent(input) -> box:  
    if isinstance(input, str):
        if os.path.isfile(input):
            with rasterio.open(raster_path, mode='r') as src:
                bounds = src.bounds
                raster_extent = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        else:
            print(f'File does not exists: {input}')
            
    if isinstance(input, tuple):
        raster, transform = input
        height, width = raster.shape
        
        minx, miny = transform * (0, height)
        maxx, maxy = transform * (width, 0)
        raster_extent = box(minx, miny, maxx, maxy)

    return raster_extent


def get_layers_extent(raster, hash_table) -> list:
    """
    Function returning GeoDataFrame of hash table layers intersected by given
    raster file. It has two parameters:
        raster_path - path to the raster dataset readible by rasterio
        hash_table - GeoDataFrame with extents of layers  
    """
    raster_extent = get_raster_extent(raster)
        
    intersecting_layers = hash_table[hash_table.intersects(raster_extent)]
        
    return intersecting_layers


def get_geometries(src_db_dir:str, layers:list, raster_path:str) -> gpd.GeoDataFrame:
    """
    Function to get actual geometries from GPKG. Parameters:
        - src_db_dir - path to the database with 
        - layers - layers to get from db
    """
    gdf = gpd.GeoDataFrame(columns = ['file_name', 'geometry'])
    
    for layer in layers:
        layer_gdf = gpd.read_file(src_db_dir, layer=layer)
        gdf = pd.concat([gdf, layer_gdf])
        
    gdf = gpd.clip(gdf, get_raster_extent(raster_path))
        
    return gdf
    

def rasterize_geometries(src_geom_gdf:gpd.GeoDataFrame, raster_path:str, 
                         out_path:str = None) -> numpy.ndarray:
    
    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

    
    out_shape = (height, width)
    
    raster = rasterize(
        [(geom, 1) for geom in src_geom_gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
        )
    
    if out_path:
        with rasterio.open(out_path, 'w', driver='GTiff', height=raster.shape[0],
                           width=raster.shape[1], count=1, dtype=raster.dtype,
                           crs=src_geom_gdf.crs, transform=transform) as dst:
            dst.write(raster, 1)
    
    return (raster, transform)


def plot_rasters_and_geometries(rgb_raster, binary_raster, gdf):
    """ 
    Generate Plot visualizing source raster, geodataframe and binary mask
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot RGB raster
    if type(rgb_raster) is not numpy.ndarray:
        with rasterio.open(rgb_raster) as src:
            rgb_image = src.read([1, 2, 3])
    else:
        rgb_image = rgb_raster
    
    axes[0].imshow(rgb_image.transpose(1, 2, 0))
    axes[0].set_title('RGB Raster')
    
    
    # Plot binary raster
    if type(binary_raster) is not numpy.ndarray:
        with rasterio.open(binary_raster) as src:
            binary_image = src.read(1)
    else:
        binary_image = binary_mask
    
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title('Binary Raster')
    
    
    # Plot GeoDataFrame geometries
    gdf.plot(ax=axes[2], color='blue', edgecolor='black')
    axes[2].set_title('GeoDataFrame Geometries')

    plt.tight_layout()
    plt.show()
    
    
def extract_central_25_percent(raster_path, output_path):
    with rasterio.open(raster_path) as src:
        # Get the dimensions of the raster
        width = src.width
        height = src.height

        # Calculate the coordinates for the central 25%
        central_width = width // 5
        central_height = height // 5
        left = (width - central_width) // 2
        top = (height - central_height) // 2

        # Define the window for the central 25%
        window = Window(left, top, central_width, central_height)

        # Read the data within the window
        central_data = src.read(window=window)

        # Update the metadata to reflect the new dimensions
        out_meta = src.meta.copy()
        out_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window)
        })

        # Write the central 25% to a new file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(central_data)

 

def split_geotiff_to_patches(input, patch_size, overlap_ratio):
    if isinstance(input, str):
        with rasterio.open(input) as src:
            img = src.read(1)
            height, width = img.shape
            transform = src.transform
            
    if isinstance(input, tuple):
        img, transform = input
        height, width = img.shape
    
    patches = []
    step = int(patch_size * (1 - overlap_ratio))
    
    for i in range(0, height, step):
        for j in range(0, width, step):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                # Handle the last row/column patches
                patch = img[max(0, height-patch_size):height, max(0, width-patch_size):width]
            
            # Calculate the transform for the patch
            patch_transform = rasterio.transform.Affine(
                transform.a, transform.b, transform.c + j * transform.a,
                transform.d, transform.e, transform.f + i * transform.e
            )
            
            patches.append((patch, patch_transform))

    return patches


# Example usage
tiff_file = 'path_to_your_geotiff_file.tif'
patch_size = 256
overlap_ratio = 0.2
patches = split_geotiff_to_patches(tiff_file, patch_size, overlap_ratio)
print(f"Generated {len(patches)} patches.")


        
if __name__ == "__main__":
    hash_table = gpd.read_file('/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg',
                     layer = 'hash_table_01')
    rasters_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/08_OrtoRGB'
    dst_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/09_training'
    
    raster_path = [os.path.join(rasters_dir, _) for _ in os.listdir(rasters_dir)[14:15]][0]
    
    layers = get_layers_extent(raster_path, hash_table)
    buildings_gdf = get_geometries('/media/pszubert/DANE/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg',
                                   layers['file_name'].to_list(), raster_path)
    
    binary_mask = rasterize_geometries(buildings_gdf, raster_path,
                                       '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/09_training/asd.tif')
    
    patches = split_geotiff_to_patches(raster_path, 640, 0.25)
    
    for number, patch in enumerate(patches[:50]):
        array, transform = patch
        raster_dir = os.path.join(dst_dir, f'raster_{number}.tif')
        
        metadata = {
            'driver': 'GTiff',
            'dtype': array.dtype,
            'nodata': None,
            'width': array.shape[1],
            'height': array.shape[0],
            'count': 1,
            'crs': 'EPSG:2180',
            'transform': transform
        }
        
        with rasterio.open(raster_dir, 'w', **metadata) as dst:
            dst.write(array, 1)

    
    

