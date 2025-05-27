#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:28:49 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import geopandas as gpd
import pandas as pd
import rasterio
import fiona 
import matplotlib.pyplot as plt
import numpy
import os
import numpy as np
import cv2


from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.windows import Window
from enum import Enum
from scipy.stats import rv_discrete
from scipy.fftpack import dct
from ImageConverter import ImageConverter
from util import save_datarame_sqllite, load_sqllite_dataframe


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
            with rasterio.open(input, mode='r') as src:
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
    

class RasterizeModes(Enum):
    opt1 = 'Binary'
    opt2 = 'MultiClass'

def rasterize_geometries(src_geom_gdf:gpd.GeoDataFrame, raster_path:str,
                         mode:RasterizeModes = 'Binary', out_path:str = None) -> numpy.ndarray:
    
    if not RasterizeModes(mode):
        raise ValueError(f'Mode value is not one of the {list(RasterizeModes)}')
    
    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform

    
    out_shape = (height, width)
    
    if RasterizeModes == 'MultiClass':
        raster = rasterize(
            [(geom, num) for num, geom in enumerate(src_geom_gdf.geometry)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='int',
            all_touched=True
            )
    else:
        raster = rasterize(
            [(geom, 1) for  geom in src_geom_gdf.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True
            )
        
    
    if out_path:
        with rasterio.open(out_path, 'w', driver='JPEG', height=raster.shape[0],
                           width=raster.shape[1], count=1, dtype=raster.dtype,
                           crs=src_geom_gdf.crs, transform=transform) as dst:
            dst.write(raster, 1)
    
    return (raster, transform)


def generate_yolo_labels(size, rasterio_transform, buildings_gdf):
    """
    Generates YOLO-compatible label text from a rasterio patch and a clipped GeoDataFrame.

    Parameters:
    rasterio_src: rasterio source object
    rasterio_transform: rasterio transform object
    buildings_gdf (GeoDataFrame): Clipped GeoDataFrame containing building geometries.

    Returns:
    str: YOLO-compatible label text.
    """
    label_text = ""

    for _, row in buildings_gdf.iterrows():
        polygon = row.geometry

        # Convert world coordinates to pixel coordinates
        pixel_coords = [
            rasterio.transform.rowcol(rasterio_transform, x, y)[::-1]  # (col, row)
            for x, y in polygon.exterior.coords
        ]

        # Normalize coordinates
        normalized_coords = [
            f"{x / size:.6f},{y / size:.6f}" for x, y in pixel_coords
        ]

        # Format as YOLO label (assuming class 0 for buildings)
        label_text += "0 " + " ".join(normalized_coords) + "\n"

    return label_text


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
        binary_image = binary_raster
    
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title('Binary Raster')
    
    
    # Plot GeoDataFrame geometries
    gdf.plot(ax=axes[2], color='blue', edgecolor='black')
    axes[2].set_title('GeoDataFrame Geometries')

    plt.tight_layout()
    plt.show()
    
    
def extract_central_25_percent(raster_path, output_path=None):
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
        if output_path:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(central_data)
        
        return central_data
 

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


def generate_stratified_samples(df, num_samples):
    # Create probability distribution for noise
    noise_values = df['noise'].values
    noise_probs = np.ones_like(noise_values) / len(noise_values)
    noise_dist = rv_discrete(values=(noise_values, noise_probs))
    
    # Create probability distribution for blur
    blur_values = df['blur'].values
    blur_probs = np.ones_like(blur_values) / len(blur_values)
    blur_dist = rv_discrete(values=(blur_values, blur_probs))
    
    # Generate stratified samples
    noise_samples = noise_dist.rvs(size=num_samples)
    blur_samples = blur_dist.rvs(size=num_samples)
    
    # Find matching histograms
    selected_histograms = []
    for blur, noise in zip(blur_samples, noise_samples):
        # Find rows that match or are closest to the sampled blur and noise
        match = df.loc[(df['blur'] == blur) & (df['noise'] == noise)]
        
        if match.empty:
            # If no exact match, find the closest one
            df['distance'] = (df['blur'] - blur)**2 + (df['noise'] - noise)**2
            closest = df.loc[df['distance'].idxmin()]
            selected_histograms.append(closest['histogram'])
            df = df.drop(columns='distance')  # Clean up
        else:
            selected_histograms.append(match.sample(1)['histogram'].values[0])
    
    # Create list of tuples with stratified values
    stratified_tuples = list(zip(blur_samples, noise_samples, selected_histograms))
    
    return stratified_tuples


def display_histograms(data1, data2, titles:list, num_bins=10):
    # Unpack the tuples into separate lists for blur and noise
    blur1, noise1 = zip(*data1)
    blur2, noise2 = zip(*data2)
    
    blur_max = max(max(blur1), max(blur2))
    noise_max = max(max(noise1), max(noise2))
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot histograms for blur and noise in data1
    axs[0, 0].hist(blur1, bins=num_bins, color='blue', alpha=0.7, range=[0, blur_max])
    axs[0, 0].set_title(f'Blur Distribution of {titles[0]}')
    axs[0, 0].set_xlabel('Blur')
    axs[0, 0].set_ylabel('Frequency')
    
    axs[0, 1].hist(noise1, bins=num_bins, color='green', alpha=0.7, range=[0, noise_max])
    axs[0, 1].set_title(f'Noise Distribution of {titles[0]}')
    axs[0, 1].set_xlabel('Noise')
    axs[0, 1].set_ylabel('Frequency')
    
    # Plot histograms for blur and noise in data2
    axs[1, 0].hist(blur2, bins=num_bins, color='blue', alpha=0.7, range=[0, blur_max])
    axs[1, 0].set_title(f'Blur Distribution of {titles[1]}')
    axs[1, 0].set_xlabel('Blur')
    axs[1, 0].set_ylabel('Frequency')
    
    axs[1, 1].hist(noise2, bins=num_bins, color='green', alpha=0.7, range=[0, noise_max])
    axs[1, 1].set_title(f'Noise Distribution of {titles[1]}')
    axs[1, 1].set_xlabel('Noise')
    axs[1, 1].set_ylabel('Frequency')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the histograms
    plt.show()


def sharpen_image(image, strength = 0):
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + strength, -1],
                       [0, -1, 0]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    sharpened_image = cv2.normalize(sharpened_image, None,
                                    0, 255, cv2.NORM_MINMAX)
    
    return sharpened_image


def compare_convertion_side_by_side(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    img1_blur = round(ImageConverter.estimate_blur(img1), 2)
    img1_noise = round(ImageConverter.estimate_noise(img1), 2)
    img2_blur = round(ImageConverter.estimate_blur(img2), 2)
    img2_noise = round(ImageConverter.estimate_noise(img2), 2)

    # Display the first image
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_xlabel(f'Sharpness: {img1_blur}; Noise: {img1_noise}')


    # Display the second image
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_xlabel(f'Sharpness: {img2_blur}; Noise: {img2_noise}')
    
    fig.suptitle('Comparison of raw and processed image')

    # Show the plot
    plt.show()

       
if __name__ == "__main__":
    pass
        
        
        
        
        
        
        
        
        
        


