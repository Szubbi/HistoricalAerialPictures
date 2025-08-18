# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:15:21 2025

@author: pzu
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np 
import pandas as pd
import torch

from util import split_geotiff_to_patches, draw_patch_grid_on_geotiff, load_image, load_model
from ultralytics import YOLO
from PIL import Image
from Mask_RCNN.detect_objects import visualize_predictions



if __name__ == "__main__":
    model = YOLO(r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\best.pt')
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\17_25981_M-34-52-D-b-1-1.tif'
    
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    
    Image.fromarray(patches[50][0])
    results = model(Image.fromarray(patches[50][0]))
    
    for result in results:
        mask = result.masks
        
    mask
    result.show()
    
    df = results[0].to_df()
    txt = results[0].to_csv()

    dir(result)


    # test pytorch mask r-cnn
    weight_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\exp1_best_model.pth"
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\30_26667_M-34-77-B-b-1-4.tif'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load model
    model = load_model(weight_dir)
    model.eval()
    model.to(device)

    # test image
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    draw_patch_grid_on_geotiff(test_img_dir, 640, 0.25)
    
    img_idx = 193
    plt.imshow(Image.fromarray(patches[img_idx][0]), cmap='gray')
    plt.title(f'Patch {img_idx}')
    plt.show()
    
    test_img = Image.fromarray(patches[img_idx][0])
    test_img = load_image(test_img)
    
    # predict
    with torch.no_grad():
        prediction = model(test_img)
        
    # display predict    
    visualize_predictions(test_img, prediction, threshold=0.5)
        
    
    
    
    

