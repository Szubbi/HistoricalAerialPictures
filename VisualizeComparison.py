#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:25:56 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""
import os
import matplotlib.pyplot as plt

from ImageConverter import ImageConverter

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

    