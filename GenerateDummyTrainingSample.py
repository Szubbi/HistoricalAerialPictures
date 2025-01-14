#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:30:04 2024

@author: piotr.szubert@doctoral.uj.edu.pl

"""

import cv2
import rasterio as rio
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski


orto_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/06_Orto'
dst_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing'
img_dir = os.path.join(orto_dir, '79045_1296512_M-34-74-A-b-3-4.tif')

with rio.open(img_dir) as src:
    transform = src.transform
    crs = src.crs

with rio.open(os.path.join(orto_dir, 'bw_test.tif')) as src:
    profile = src.profile

img = cv2.imread(img_dir)
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.float32)/25
avg_img = cv2.filter2D(bw_img, -1, kernel)
blured_img = cv2.GaussianBlur(avg_img, (5,5), 0)
noise_img = ski.util.random_noise(blured_img, mode='gaussian') 

plt.figure(figsize=(22,22))
plt.imshow(noise_img, cmap='gray')
plt.show()

profile.update({ 'count': 1, 
                'dtype': noise_img.dtype, 
                'height': noise_img.shape[0], 
                'width': noise_img.shape[1],
                'crs': crs,
                'transform': transform})

dst_img = os.path.join(dst_dir, 'test_02.tif')

with rio.open(dst_img, 'w', **profile) as dst:
    dst.write(noise_img, 1)
