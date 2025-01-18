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



class ImageConverter:
    
    def __init__(self, img_dir:str):
        self.img_dir = img_dir
        
        
    def convert_image(self, kernel_size:int):
        self.img = cv2.imread(self.img_dir)
        self.bw_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        self.kernel = np.ones((kernel_size, kernel_size),np.float32)/25
        self.avg_img = cv2.filter2D(self.bw_img, -1, self.kernel)
        self.blured_img = cv2.GaussianBlur(self.avg_img, (5,5), 0)
        self.noise_img = ski.util.random_noise(self.blured_img, mode='gaussian')
        
        return self.noise_img
        
    
    def save(self, dst_dir:str):

        with rio.open(self.img_dir) as src:
            transform = src.transform
            crs = src.crs
    
        raster_meta = {'driver': 'GTiff',
                       'dtype': self.noise_img.dtype, 
                       'nodata': None,
                       'height': self.noise_img.shape[0], 
                       'width': self.noise_img.shape[1], 
                       'count': 1, 
                       'crs': crs, 
                       'transform': transform}
        
        with rio.open(dst_img, 'w', **raster_meta) as dst:
            dst.write(self.noise_img, 1)
            
            
if __name__ == "__main__":
    orto_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/06_Orto'
    dst_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing'
    img_dir = os.path.join(orto_dir, '79045_1296512_M-34-74-A-b-3-4.tif')
    dst_img = os.path.join(dst_dir, 'test_03.tif')
    
    ImageConv = ImageConverter(img_dir)
    ImageConv.convert_image(5)
    ImageConv.save(dst_img)




