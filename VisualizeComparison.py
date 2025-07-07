#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:25:56 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import os
import matplotlib.pyplot as plt

from ImageConverter import ImageConverter
from util import load_sqllite_dataframe

def visualize_comparison(rgb_image, bw_image1, bw_image2, bw_image3):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Modern RGB")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(bw_image1, cmap='gray')
    axes[0, 1].set_title("Archival BW")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(bw_image2, cmap='gray')
    axes[1, 0].set_title("Modern RGB in BW")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(bw_image3, cmap='gray')
    axes[1, 1].set_title("Simulated old BW")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_image_metrics_histograms(contrast_data, blur_data, noise_data,
                                  metric_names, collection_names):
    """
    Plots a 3x3 grid of histograms using matplotlib.
    Each column represents an image collection.
    Each row represents a metric: contrast, blur, and sharpness.

    Parameters:
    - contrast_data: List of 3 lists, each containing contrast values for an image collection.
    - blur_data: List of 3 lists, each containing blur values for an image collection.
    - sharpness_data: List of 3 lists, each containing sharpness values for an image collection.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    metric_data = [contrast_data, blur_data, noise_data]

    # Compute x-axis limits for each metric row
    x_limits = []
    for metric in metric_data:
        all_values = [value for collection in metric for value in collection]
        x_limits.append((min(all_values), max(all_values)))

    for row in range(3):
        for col in range(3):
            axes[row, col].hist(metric_data[row][col], bins=20, color='skyblue', edgecolor='black')
            axes[row, col].set_xlim(x_limits[row])  # Set consistent x-axis range per row
            if row == 0:
                axes[row, col].set_title(collection_names[col])
            if col == 0:
                axes[row, col].set_ylabel(metric_names[row])
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # creating sample image
    wrk_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial'
    log_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial'
    rgb_img_dir = os.path.join(wrk_dir, 'modern_rgb.tif')
    bw_img_dir = os.path.join(wrk_dir, 'archival_bw.tif')
    
    src_img = ImageConverter(bw_img_dir, log_dir)
    trg_img = ImageConverter(rgb_img_dir, log_dir)
    
    bw_img = trg_img()
    
    # calculate target metrics 
    trg_img.blur_lvl_trg = src_img.estimate_blur(src_img())
    trg_img.noise_lvl_trg = src_img.estimate_noise(src_img())
    trg_img.contrast_lvl_trg =  src_img.estimate_contrast(src_img())
    trg_img.hist_trg = src_img.measure_histogram(src_img())
    
    # convert image
    trg_img.find_convertion_values(45, 0.4)
    trg_img.convert_image()
    conv_img, transform = trg_img.save(os.path.join(wrk_dir, 'conv_00.tif'))
    
    
    visualize_comparison(trg_img.src_img, src_img(), trg_img(), trg_img())
    
    
    # histograms comparison
    db_dir = '/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/05_Data/Data.gpkg'
    
    BW_df = load_sqllite_dataframe(db_dir, 'img_BlurSharpTable_04')
    RGB_df = load_sqllite_dataframe(db_dir, 'img_BlurSharpRGBTable_05')
    Conv_df = load_sqllite_dataframe(db_dir, 'img_BlurSharpConvTable_04')
    
    plot_image_metrics_histograms(
        contrast_data= [BW_df['contrast'], RGB_df['contrast'], Conv_df['contrast']],
        blur_data=[BW_df['blur'], RGB_df['blur'], Conv_df['blur']],
        noise_data=[BW_df['noise'], RGB_df['noise'], Conv_df['noise']],
        metric_names = ['Contrast', 'Blur', 'Noise'],
        collection_names = ['Historical BW Imagery', 'Modern RGB Imagery', 'Simulated BW Imagery']
        )

    