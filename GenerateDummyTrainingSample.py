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

from typing import Union

img_or_dir = Union[np.ndarray, str]


class ImageConverter:
    
    def __init__(self, img_dir:str):
        self.img_dir = img_dir
        self.src_img = cv2.imread(self.img_dir)
        self.out_img = = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        
    def blur(self, kernel_size:int):
        kernel = np.ones((kernel_size, kernel_size),np.float32)/25
        self.out_img = cv2.filter2D(self.out_img, -1, kernel)
        self.out_img = cv2.GaussianBlur(self.out_img, (5,5), 0)
        self.out_img = np.uint8(self.out_img * 255)
        
        
    def noise(self, 
        self.out_img = ski.util.random_noise(self.out_img, mode='gaussian')
        self.out_img = np.uint8(self.out_img * 255)
        
    
    #inspired by https://unimatrixz.com/topics/story-telling/analyzing-image-noise-using-opencv-and-python/
    @staticmethod
    def estimate_noise(img_dir:img_or_dir):
        if os.path.isfile(img_dir):
            img = cv2.imread(img_dir)
        elif type(img_dir) == np.ndarray:
            img = img_dir
        else:
            raise TypeError('Input must be image array or image dir')
        
        #image has to be grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blured_img = cv2.GaussianBlur(img, (5,5), 0)   
        
        noise = img - blured_img
                
        return np.mean(noise), np.std(noise)
        
    
    def save(self, dst_dir:str):

        with rio.open(self.img_dir) as src:
            transform = src.transform
            crs = src.crs
    
        raster_meta = {'driver': 'GTiff',
                       'dtype': 'uint8', 
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
    dst_img = os.path.join(dst_dir, 'test_04.tif')
    
    ImageConv = ImageConverter(img_dir)
    noise_img, bw_img = ImageConv.convert_image(5)
    
    ImageConverter.estimate_noise(img)
    ImageConverter.estimate_noise('/home/pszubert/Pobrane/17_37262_M-34-64-D-a-4-4.tif')
    
    img = cv2.imread(img_dir)
    ImageConv.save(dst_img)
    
    type(img)

    plt.figure(figsize=(24,24))
    plt.imshow(noise_img, cmap='gray')
    plt.show()

len(img.shape)
