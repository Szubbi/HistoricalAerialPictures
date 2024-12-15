# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:45:15 2024

@author: szube
"""

from PIL import Image
from skimage import measure
from skimage.draw import polygon

import matplotlib.pyplot as plt
import os 
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = None

def clip_aerial_picture_cv2(src_dir : str, kernel_size : int):
    img = np.array(Image.open(src_dir))
    
    # czarny ma wartoci poniżej 10
    reclass = np.where(img > 25, 1, 0).astype(np.uint8)
    contours,hierarchy = cv2.findContours(
        reclass,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = (cv2.drawContours(
        np.zeros_like(img), [largest_contour], -1, color=255, thickness=cv2.FILLED))
    kernel = np.ones((150, 150), np.uint8)
    im_extend = cv2.dilate(mask, kernel, iterations=1) 
    im_erosion = cv2.erode(im_extend, kernel, iterations=1)
    masked_img = cv2.bitwise_and(img, img, mask = im_erosion)
    
    x, y, w, h = cv2.boundingRect(mask)
    masked_img = masked_img[y:y+h, x:x+w]

    return masked_img

def clip_aerial_picture_ski(src_dir : str):
    img = np.array(Image.open(src_dir))
    
    # czarny ma wartoci poniżej 10
    reclass = np.where(img > 20, 1, 0) 
    contours = measure.find_contours(reclass)
    biggest_part = max(contours, key=len)

    masked_img = img[
        int(min(biggest_part[:,1])) : int(max(biggest_part[:,1])),
        int(min(biggest_part[:,0])) : int(max(biggest_part[:,0]))]
    
    return masked_img

    

if __name__ == '__main__':
       
    src_dir = r''
    dst_dir = r''
    
    
    for img in os.listdir(src_dir):
        if img.endswit('.tif'):
            img_dir = os.path.join(src_dir, img)
            clipped = clip_aerial_picture_cv2(img, 25)
            
            plt.figure(figsize = (25,12))
            plt.subplot(121).set_title('Clipped')
            plt.imshow(clipped, cmap='gray')
            
            plt.subplot(122).set_title('Orginal')
            plt.imshow(Image.open(img_dir), cmap='gray')
            
            cv2.imwrite(
                os.path.join(dst_dir, img), 
                clipped, 
                params=(cv2.IMWRITE_TIFF_COMPRESSION,1))
            
            
    




