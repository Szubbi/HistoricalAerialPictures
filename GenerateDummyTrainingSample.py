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

from skopt import gp_minimize
from skopt.space import Real
from typing import Union


img_or_dir = Union[np.ndarray, str]


class ImageConverter:
    
    def __init__(self, img_dir:str):
        self.img_dir = img_dir
        self.img_nme = img_dir.split('/')[-1]
        self.src_img = cv2.imread(self.img_dir)
        self.out_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        self.blur_lvl_trg = None
        self.noise_lvl_trg = None
        self.best_blur_lvl = None
        self.best_noise_lvl = None
        
        
    def __str__(self):
        return f'Image converter Class with {self.out_img.shape}'
        
    
    def __call__(self):
        return self.out_img


    def load_img(self, img_dir:img_or_dir):
        if os.path.isfile(img_dir):
            img = cv2.imread(img_dir)
        elif type(img_dir) == np.ndarray:
            img = img_dir
        else:
            raise TypeError('Input must be image array or image dir')
            
        return img
        
        
    def blur(self, img, kernel_size:int):
        kernel = np.ones((kernel_size, kernel_size),np.float32)/25
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = np.uint8(img * 255)
        
        return img
        
        
    def noise(self, img, std_dev): 
        img = ski.util.random_noise(img, mode='gaussian', var=std_dev)
        img = np.uint8(img * 255)
        
        return img
        
    
    #inspired by https://unimatrixz.com/topics/story-telling/analyzing-image-noise-using-opencv-and-python/
    @staticmethod
    def estimate_noise(img_dir:img_or_dir):
        img = ImageConverter.load_img(None, img_dir)      
        
        #image has to be grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        blured_img = cv2.GaussianBlur(img, (5,5), 0)   
        noise = img - blured_img
                
        return np.std(noise)
    
    
    @staticmethod
    def estimate_blur(img_dir:img_or_dir):
        img = ImageConverter.load_img(None, img_dir)
        
        #image has to be grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        
        return blur
    

    def meassure_trg_img(self, trg_img_dir:img_or_dir):
        trg_img = self.load_img(trg_img_dir)
        
        #image has to be grayscale
        if len(trg_img.shape) > 2:
            trg_img = cv2.cvtColor(trg_img, cv2.COLOR_BGR2GRAY)
        
        self.blur_lvl_trg = self.estimate_blur(trg_img)
        self.noise_lvl_trg = self.estimate_noise(trg_img)
        
                
    def asses_convertion_values(self, params):
        blur_kernel, noise_std_dev = params
        
        blured_img = self.blur(self.out_img, int(blur_kernel))
        noise_img = self.noise(blured_img, noise_std_dev)
        
        blur_lvl = self.estimate_blur(blured_img)
        noise_lvl = self.estimate_noise(noise_img)
        
        blur_difference = abs(blur_lvl - self.blur_lvl_trg)
        noise_difference = abs(noise_lvl - self.noise_lvl_trg)
        total_difference = blur_difference + noise_difference
        
        return total_difference
    
    
    def find_convertion_values(self, max_blur:float, max_noise:float, epochs:int):
        if self.blur_lvl_trg is None or self.noise_lvl_trg is None:
            print('Meassuring source image statistics first')
            self.meassure_src_img()
            
        assert max_noise <= 1.0, 'Noise std dev can not be larger than 1.0'
            
        print(f'Source Image {self.img_nme} statistics:',
              f' -Blur: {self.blur_lvl_trg}',
              f' -Noise: {self.noise_lvl_trg}',
              sep = os.linesep)
        
        param_space = [
            Real(1.0, max_blur, name = 'blur_kernel'),
            Real(0.1, max_noise, name = 'noise_std_dev')]
        
        result = gp_minimize(self.asses_convertion_values,
                             param_space, 
                             n_calls=epochs, 
                             random_state=40,
                             n_initial_points=10,
                             acq_func="EI")
        
        self.best_blur_lvl, self.best_noise_lvl = result.x
        
        print(f'Best blur param: {self.best_blur_lvl}',
              f'Best noise param: {self.best_noise_lvl}',
              sep = os.linesep)
        
        return self.best_blur_lvl, self.best_noise_lvl
    
    
    def convert_image(self):
        self.out_img = self.blur(self.out_img, self.best_blur_lvl)
        self.out_img = self.noise(self.out_img, self.best_noise_lvl)
        
            
    def save(self, dst_dir:str):
        with rio.open(self.img_dir) as src:
            transform = src.transform
            crs = src.crs
    
        raster_meta = {'driver': 'GTiff',
                       'dtype': 'uint8', 
                       'nodata': None,
                       'height': self.out_img.shape[0], 
                       'width': self.out_img.shape[1], 
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
    trg_img = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/07_Orto_bw/17_37262_M-34-64-D-a-4-4.tif'
    
    ImageConv = ImageConverter(img_dir)
    ImageConv.meassure_trg_img(trg_img)
    ImageConv.find_convertion_values(max_blur = 10, 
                                     max_noise = 1.0, 
                                     epochs = 75)   
    
    ImageConverter.estimate_noise()
    
    


    orto_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/08_Ostrodzki_Probka'

    for _ in os.listdir(orto_dir):
        if _.endswith('.tif'):
            img_dir = os.path.join(orto_dir, _)
            img = cv2.imread(img_dir)
            
            print(
                f'Img name: {_}',
                f'- noise: {ImageConverter.estimate_noise(img)}',
                f'- blur {ImageConverter.estimate_blur(img)}',
                sep = os.linesep)
