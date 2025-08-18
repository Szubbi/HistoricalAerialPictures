# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:15:21 2025

@author: pzu
"""
import os
import matplotlib.pyplot as plt
import numpy as np 

from util import split_geotiff_to_patches
from ultralytics import YOLO
from PIL import Image

if __name__ == "__main__":
    model = YOLO(r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\best.pt')
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\34_34736_M-34-66-D-d-3-4.tif'
    
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    
    Image.fromarray(patches[150][0])
    results = model(Image.fromarray(patches[150][0]))
    
    test_img = Image.open(r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\test_patch.png')
    
    results = model(test_img)
    for result in results:
        mask = result.masks
        
    mask
    result.show()

