#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:30:04 2024

@author: piotr.szubert@doctoral.uj.edu.pl

ImageConverter class is designed to simulate historical black-and-white (BW) aerial photographs
using modern 3-band RGB aerial imagery as the source.

The class enables the analysis of historical image characteristics by measuring key statistics such as:
    - Sharpness
    - Noise levels
    - Histogram distribution

It then adjusts the modern image to match these historical characteristics using the
`gp_minimize` method from the `skopt` library for parameter optimization.

"""

import cv2
import rasterio as rio
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import geopandas as gpd

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from typing import Union
from scipy.fftpack import dct
from enum import Enum
from datetime import datetime
from skimage.exposure import match_histograms
from util import save_datarame_sqllite, load_sqllite_dataframe

img_or_dir = Union[np.ndarray, str]



def setup_logger(name='ImageConverterLogger', log_file='image_converter.log', level=logging.INFO):
    """Set up a logger for the ImageConverter class."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger



class ImageConverter:
    
    def __init__(self, img_dir:str, log_dir:str):
        self.img_dir = img_dir
        self.img_nme = img_dir.split('/')[-1]
        self.src_img = cv2.imread(self.img_dir)
        self.out_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        self.blurred_img = None
        self.noise_img = None
        self.blur_lvl_trg = None
        self.noise_lvl_trg = None
        self.hist_trg = None
        self.best_blur_lvl = None
        self.best_blur_alpha = None
        self.best_noise_alpha = None
        self.best_noise_lvl = None
        self.best_clip_limit = None
        self.best_unsharp_strength = None
        self.blur_sigma = None
        self.convertion_values = {}
        self.logger = setup_logger(name = f'ImageConverter_{self.img_nme}_logger',
                                   log_file = os.path.join(log_dir,
                                                           f'ImageConverter_{self.img_nme}_logger_{datetime.now()}.log'))
        
        
    def __str__(self):
        return f'Image converter Class with {self.out_img.shape}'
        
    
    def __call__(self):
        return self.out_img


    def load_img(self, img_dir:img_or_dir):
        if type(img_dir) == np.ndarray:
            img = img_dir
        elif os.path.isfile(img_dir):
            img = cv2.imread(img_dir)
        else:
            raise TypeError('Input must be image array or image dir')
            
        return img
        
    
    # IMAGE CONVERTION FUNCTIONS    
    def blur(self, kernel_size:int):
        # Ensure kernel_size is odd and greater than 0
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size <= 0:
            kernel_size = 1
            
        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(self.out_img, (kernel_size, kernel_size), 0)
        
        # Apply bilateral filter for edge-preserving smoothing
        blurred_image = cv2.bilateralFilter(blurred_image, kernel_size, 
                                            kernel_size * 2, kernel_size / 2)
        
        # Ensure the image values are correctly scaled before converting to uint8
        blurred_image = np.clip(blurred_image, 0, 255)
        self.blurred_img = np.uint8(blurred_image)
        
        return self.blurred_img
    
    
    def noise(self, sigma):
        gaussian_noise = np.random.normal(0, sigma, self.out_img.shape).astype(np.float32)
        noisy_image = cv2.add(self.out_img.astype(np.float32), gaussian_noise)
        self.noise_img = np.clip(noisy_image, 0, 255).astype(np.uint8)        
        return self.noise_img
    
    
    # histogram adjustment
    def compute_cdf(self, hist):
        cdf = hist.cumsum()
        return cdf / cdf[-1]  # Normalize to [0, 1]
    
    
    def match_histogram(self):
        source_cdf = self.compute_cdf(self.measure_histogram(self.out_img))
        reference_cdf = self.compute_cdf(self.hist_trg)
        
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            closest_value = np.argmin(np.abs(reference_cdf - source_cdf[i]))
            mapping[i] = closest_value
            
        self.out_img = mapping[self.out_img]
    
    
    class BlendModes(Enum):
        opt1 = 'blur'
        opt2 = 'noise'
        
        
    def blend_images(self, mode:BlendModes, alpha):
        if not ImageConverter.BlendModes(mode):
            raise ValueError(f'Mode value is not one of the {list(ImageConverter.BlendModes)}')      
        
        # Blend the original and blurred/noised images
        if mode == 'blur':
            blended_image = cv2.addWeighted(self.out_img, alpha, 
                                            self.blurred_img, 1 - alpha, 0)
        elif mode == 'noise':
            blended_image = cv2.addWeighted(self.out_img, alpha, 
                                            self.noise_img, 1 - alpha, 0)
        else:
            raise Exception('Modes DO NOT WORK')
            
        self.out_img = blended_image        
        return blended_image
        
    
    # IMAGE STATISTICS FUNCTIONS
    @staticmethod
    def estimate_noise(img_dir:img_or_dir):
        """
        Estimate noise based on:
            https://unimatrixz.com/topics/story-telling/analyzing-image-noise-using-opencv-and-python/
            
        Parameters:
        - image: Input image (grayscale).
        
        Returns:
        - noise metric: Estimated noise metric.             
        """
        img = ImageConverter.load_img(None, img_dir)      
        
        #image has to be grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        return np.std(img)
    
    
    @staticmethod
    def estimate_blur(img_dir:img_or_dir):
        """
        Estimate blur using sequency spectrum truncation.
        Based on: https://link.springer.com/article/10.1007/s40747-021-00592-7
        
        Parameters:
        - image: Input image (grayscale).
        
        Returns:
        - blur metric: Estimated blur metric.       
        """
        img = ImageConverter.load_img(None, img_dir)
        
        # Apply Discrete Cosine Transform (DCT)
        dct_image = dct(dct(img.T, norm='ortho').T, norm='ortho')
        
        # Calculate the sequency spectrum
        sequency_spectrum = np.abs(dct_image)
        
        # Truncate the sequency spectrum to reduce noise influence
        truncation_threshold = np.percentile(sequency_spectrum, 95)
        truncated_spectrum = sequency_spectrum[sequency_spectrum < truncation_threshold]
        
        # Calculate the blur metric
        blur_metric = np.mean(truncated_spectrum)
        
        return blur_metric
    
    
    @staticmethod
    def measure_histogram(img_dir:img_or_dir):
        img = ImageConverter.load_img(None, img_dir)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return hist.flatten()
    
    
    # Below methoods are used together to reduce the foggy appearance in images 
    # by enhancing contrast and sharpness.
    # Based on https://www.ijeat.org/wp-content/uploads/papers/v9i2/A9607109119.pdf
    # and https://link.springer.com/chapter/10.1007/978-3-319-95930-6_32
    
    def apply_clahe(self, clip_limit):
        """
        Contrast Limited Adaptive Histogram Equalization.
        It improves local contrast and enhances the definition of edges in areas with low contrast,
        which is especially useful for foggy or hazy images where global contrast enhancement might fail.
        
        Parameters:
        - clip_limit: Threshold for contrast limiting. Higher values give more contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        self.out_img = clahe.apply(self.out_img)


    def apply_unsharp_mask(self, blur_sigma, unsharp_strength):
        """
        This method enhances edges by subtracting a blurred version of the image from the original,
        which helps to restore detail lost due to fog or noise.
        
        Parameters:
        - blur_sigma: Standard deviation for Gaussian blur. Controls the amount of blurring.
        - unsharp_strength: Weight of the original image in the final blend. Higher values increase sharpness.
        """
        blurred = cv2.GaussianBlur(self.out_img, (9,9), blur_sigma)
        self.out_img = cv2.addWeighted(self.out_img, unsharp_strength, blurred, -0.5, 0)


    def enhance_image(self, clip_limit, blur_sigma, unsharp_strength):
        """
        Combines CLAHE and unsharp masking to enhance the image.
        This method is designed to remove the foggy effect by:
        1. Increasing local contrast (CLAHE),
        2. Sharpening edges and restoring fine details (unsharp mask).
        
        Parameters:
        - clip_limit: For CLAHE contrast enhancement.
        - blur_sigma: For Gaussian blur in unsharp masking.
        - unsharp_strength: For sharpening intensity.
        """
        self.apply_clahe(clip_limit)
        self.apply_unsharp_mask(blur_sigma, unsharp_strength)
    

    def meassure_trg_img(self, trg_img_dir:img_or_dir):
        trg_img = self.load_img(trg_img_dir)
        
        # Image has to be grayscale
        if len(trg_img.shape) > 2:
            trg_img = cv2.cvtColor(trg_img, cv2.COLOR_BGR2GRAY)
        
        self.blur_lvl_trg = self.estimate_blur(trg_img)
        self.noise_lvl_trg = self.estimate_noise(trg_img)
        self.hist_trg = self.measure_histogram(trg_img)
        
        
    def get_best_convertion_values(self):
        lowest_values = {}

        # Iterate over each dictionary in the dictionary of dictionaries
        for key, inner_dict in self.convertion_values.items():
            for param, value in inner_dict.items():
                # If the parameter is not in the lowest_values dictionary, add it
                if param not in lowest_values:
                    lowest_values[param] = {key: value}
                else:
                    # Update if the current value is lower
                    if value < list(lowest_values[param].values())[0]:
                        lowest_values[param] = {key: value}

        return lowest_values
    
                
    def asses_convertion_values(self, params):
        blur_kernel, noise_std_dev, blur_alpha, noise_alpha, clip_limit, \
        blur_sigma, unsharp_strength = params        
        
        # Ensure blur_kernel is an odd integer
        blur_kernel = int(np.round(blur_kernel))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel <= 0:
            blur_kernel = 1
        
        self.match_histogram()
        self.blur(int(blur_kernel))
        self.blend_images('blur', blur_alpha)        
        self.noise(noise_std_dev)
        self.blend_images('noise', noise_alpha)
        self.enhance_image(clip_limit, blur_sigma, unsharp_strength)
        
        blur_lvl = self.estimate_blur(self.out_img)
        noise_lvl = self.estimate_noise(self.out_img)
                
        blur_difference = abs(blur_lvl - self.blur_lvl_trg)
        noise_difference = abs(noise_lvl - self.noise_lvl_trg)
        
        # Normalize differences
        normalized_blur_difference = blur_difference / self.blur_lvl_trg if self.blur_lvl_trg != 0 else blur_difference
        normalized_noise_difference = noise_difference / self.noise_lvl_trg if self.noise_lvl_trg != 0 else noise_difference
        
        total_difference = normalized_blur_difference + normalized_noise_difference
        
        self.logger.info(
            "Achieved Error: %.5f, blur: %s, noise: %.5f\n"
            "Used Values: blur: %.2f (alpha: %.2f) | noise: %.2f (alpha: %.2f) |\n"
            "Image Stats: blur: %.2f (target: %.2f) | noise: %.2f (target: %.2f)",
            total_difference, blur_difference, noise_difference,
            blur_kernel, blur_alpha, noise_std_dev, noise_alpha,
            blur_lvl, self.blur_lvl_trg, noise_lvl, self.noise_lvl_trg)
 
        self.convertion_values[len(self.convertion_values) + 1] = {
            'total_difference' : total_difference,
            'blur_difference' : blur_difference,
            'blur_kernel' : blur_kernel,
            'blur_alpha' : blur_alpha,
            'noise_difference' : noise_difference,
            'noise_level' : noise_std_dev,
            'noise_alpha' : noise_alpha,
            'clip_limit' : clip_limit, 
            'blur_sigma' : blur_sigma, 
            'unsharp_strength' : unsharp_strength}
        
        # reset out image to orginal state
        self.out_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
              
        return total_difference
    

    # gp minimize early stopper
    def early_stop_callback(self, threshold):
        def callback(res):
            if res.fun < threshold:
                return True  # This stops the optimization
            return False
        return callback

    
    def find_convertion_values(self, epochs:int, treshold:float):
        if self.blur_lvl_trg is None or self.noise_lvl_trg is None:
            print('Meassuring source image statistics first')
            self.meassure_src_img()
            
        self.logger.info("Target: Blur: %.2f, Noise: %.2f | Source: Blur: %.2f, Noise: %.2f",
                 self.blur_lvl_trg, self.noise_lvl_trg, 
                 self.estimate_blur(self.out_img), self.estimate_noise(self.out_img))
        
        param_space = [
            Integer(1, 50, name = 'blur_kernel'),
            Integer(1, 75, name = 'noise_std_dev'),
            Categorical([0.7, 0.5, 0.3], name = 'blur_alpha'),
            Categorical([0.7, 0.5, 0.3], name = 'noise_alpha'),
            Real(1.0, 4.0, name = 'clip_limit'),
            Real(1, 20, name = 'blur_sigma'),
            Real(1.0, 2.5, name = 'unsharp_strength')]
        
        result = gp_minimize(self.asses_convertion_values,
                             param_space, 
                             n_calls=epochs, 
                             random_state=40,
                             n_initial_points=10,
                             acq_func="EI",
                             n_jobs=-1, 
                             callback=[self.early_stop_callback(threshold=treshold)])

        self.best_blur_lvl, self.best_noise_lvl, self.best_blur_alpha, \
        self.best_noise_alpha, self.clip_limit, self.blur_sigma, \
        self.unsharp_strength = result.x
        
        print(f'Best blur param: {self.best_blur_lvl}',
              f'Best noise param: {self.best_noise_lvl}',
              sep = os.linesep)
        
        return result
    
    
    def convert_image(self):
        self.match_histogram()
        self.blur(int(self.best_blur_lvl))
        self.blend_images('blur', self.best_blur_alpha)        
        self.noise(self.best_noise_lvl)
        self.blend_images('noise', self.best_noise_alpha) 
        self.enhance_image(self.clip_limit, self.blur_sigma, self.unsharp_strength)
        
            
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
        
        with rio.open(dst_dir, 'w', **raster_meta) as dst:
            dst.write(self.out_img, 1)
            
        return self.out_img, transform
        
            
    def compare_convertion_side_by_side(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        img1_blur = round(ImageConverter.estimate_blur(self.src_img), 2)
        img1_noise = round(ImageConverter.estimate_noise(self.src_img), 2)
        img2_blur = round(ImageConverter.estimate_blur(self.out_img), 2)
        img2_noise = round(ImageConverter.estimate_noise(self.out_img), 2)

        # Display the first image
        axes[0].imshow(self.src_img, cmap='gray')
        axes[0].set_title('Orginal Image')
        axes[0].set_xlabel(f'Sharpness: {img1_blur}; Noise: {img1_noise}')


        # Display the second image
        axes[1].imshow(self.out_img, cmap='gray')
        axes[1].set_title('Out Image')
        axes[1].set_xlabel(f'Sharpness: {img2_blur}; Noise: {img2_noise}')
        
        fig.suptitle('Comparison of raw and processed image')

        # Show the plot
        plt.show()

            
if __name__ == "__main__":    
    
    pass
