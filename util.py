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
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import torch
import geopandas as gpd

from rasterio.plot import show as rio_show
from shutil import copy
from PIL import Image, ImageDraw
from Mask_RCNN.train_maskrcnn import get_model_instance_segmentation
from torchvision.transforms import functional as F


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


def draw_yolo_polygons_on_pil(image, yolo_text_path):
    """
    Draw YOLO polygon annotations on a PIL image.

    Parameters:
    - image: PIL.Image object
    - yolo_text_path: path to a .txt file containing YOLO polygon annotations (one per line)

    Returns:
    - PIL.Image object with polygons drawn
    """
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(yolo_text_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            coords = list(map(float, parts[1:]))
            points = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
            draw.polygon(points, outline="red", width=4)

    return img


def split_geotiff_to_patches(input, patch_size, overlap_ratio):
    if isinstance(input, str):
        with rasterio.open(input) as src:
            img = src.read(1)
            height, width = img.shape
            transform = src.transform
    else:  # tuple
        img, transform = input
        height, width = img.shape

    patches = []
    step = int(patch_size * (1 - overlap_ratio))

    for i in range(0, height, step):
        for j in range(0, width, step):
            # Adjust start indices if near edge
            i_start = min(i, height - patch_size)
            j_start = min(j, width - patch_size)

            patch = img[i_start:i_start + patch_size, j_start:j_start + patch_size]

            patch_transform = rasterio.transform.Affine(
                transform.a, transform.b, transform.c + j_start * transform.a,
                transform.d, transform.e, transform.f + i_start * transform.e
            )

            patches.append((patch, patch_transform))

    return patches



def draw_patch_grid_on_geotiff(input, patch_size, overlap_ratio, show_labels=True):
    """
    Draws a grid overlay on a GeoTIFF image to visualize patch splitting and optionally labels each patch.

    Parameters:
    - file_path: str, path to the GeoTIFF file
    - patch_size: int, size of each patch (in pixels)
    - overlap_ratio: float, overlap ratio between patches (0 to <1)
    - show_labels: bool, whether to display patch index labels
    """
    if isinstance(input, str):
        with rasterio.open(input) as src:
            img = src.read(1)
            height, width = img.shape
            transform = src.transform
            
    if isinstance(input, tuple):
        img, transform = input
        height, width = img.shape

    step = int(patch_size * (1 - overlap_ratio))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    
    print('Calculating grid postitions')
    # Calculate grid positions
    x_positions = []
    for j in range(0, width, step):
        if j + patch_size > width:
            j = width - patch_size
            x_positions.append(j)
            x_positions.append(width)
        else:
            x_positions.append(j)
            x_positions.append(j + patch_size)
            
    x_positions = sorted(set(x_positions))
    
    y_positions = []
    for i in range(0, height, step):
        if i + patch_size > height:
            i = height - patch_size
            y_positions.append(i)
            y_positions.append(height)
        else:
            y_positions.append(i)
            y_positions.append(i + patch_size)
            
    y_positions = sorted(set(y_positions))


    # Draw grid lines
    for x in x_positions:
        ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
    ax.axvline(x=width, color='red', linestyle='--', linewidth=0.5)

    for y in y_positions:
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(y=height, color='red', linestyle='--', linewidth=0.5)

    # Add labels if requested
    print("drawing labels")
    if show_labels:
        patch_index = 0
        for i in range(0, height, step):
            for j in range(0, width, step):
                center_x = j + patch_size / 2
                center_y = i + patch_size / 2
                if center_x < width and center_y < height:
                    ax.text(center_x, center_y, str(patch_index),
                            color='yellow', fontsize=6, weight='bold',
                            ha='center', va='center')
                elif center_x < width and center_y > height:
                    ax.text(center_x, i, str(patch_index),
                            color='yellow', fontsize=6, weight='bold',
                            ha='center', va='center')
                elif center_x > width and center_y < height:
                    ax.text(j, center_y, str(patch_index),
                            color='yellow', fontsize=6, weight='bold',
                            ha='center', va='center')
                else:
                    ax.text(j, i, str(patch_index),
                            color='yellow', fontsize=6, weight='bold',
                            ha='center', va='center')
                patch_index += 1

    ax.set_title("Patch Grid" + (" with Labels" if show_labels else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    plt.tight_layout()
    plt.show()


def load_model(weights_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(
        torch.load(weights_dir, 
                   map_location=torch.device(device)))
    
    return model


def load_image(pil_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert to tensor
    img = F.to_tensor(pil_img)
    # add batch dimension
    img = img.unsqueeze(0)
    img.to(device)
    
    return img

# Progress bar
def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush = True)
    if iteration == total:
        print()



def plot_gpkg_on_geotiff(
    tif_path: str,
    gdf: gpd.GeoDataFrame,
    figsize: tuple[int, int] = (10, 10),
    title: str = "GeoPackage polygons over GeoTIFF",
):
    """
    Display a GeoTIFF and overlay polygons from a GeoPackage using matplotlib/rasterio.

    Parameters
    ----------
    tif_path : path to the GeoTIFF
    gpkg_path : path to the GeoPackage
    layer : optional layer name inside the GeoPackage (None = default/first layer)
    facecolor, edgecolor, linewidth, alpha : polygon styling
    figsize : matplotlib figure size
    """
    # --- open raster ---
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        fig, ax = plt.subplots(figsize=figsize)
        rio_show(src, ax=ax, cmap = 'gray')  # draws the GeoTIFF in its native CRS

        # keep only valid polygonal geometries (quietly skip empties)
        gdf = gdf[gdf.geometry.notnull()].copy()
        if gdf.empty:
            print("No geometries found in the provided layer.")
            return ax

        # --- reproject to raster CRS if needed ---
        if gdf.crs is None:
            # If the GPKG layer has no CRS defined, you can set it here manually instead of raising.
            raise ValueError("GeoPackage layer has undefined CRS. Set gdf.set_crs(<epsg>, inplace=True) first.")
        if raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # --- plot polygons on top ---
        gdf.plot(ax=ax, edgecolor = 'red', alpha = 0.5)

        # --- tighten view to raster extent (helpful when layers are larger) ---
        left, bottom, right, top = src.bounds
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")  # cleaner map-like look
        ax.set_ylabel("")
        ax.set_aspect("equal")

    plt.tight_layout()

    return ax


if __name__ == "__main__":
    for i in range(0, 101):
        print_progress_bar(i, 100, prefix='Progress:', suffix='Complete')
