# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:45:15 2024

@author: piottr.szubert@doctoral.uj.edu.pl
"""

from PIL import Image
from skimage import measure
from skimage.draw import polygon

import matplotlib.pyplot as plt
import os 
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = None

def clip_aerial_picture_cv2(src_dir : str, kernel_size : int, threshold : int):
    
    if type(src_dir) == np.ndarray:
        img = src_dir
    else:
        img = np.array(Image.open(src_dir))
    
    # czarny ma wartoci poniżej 10
    reclass = np.where(img > threshold, 1, 0).astype(np.uint8)
    contours,hierarchy = cv2.findContours(
        reclass,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = (cv2.drawContours(
        np.zeros_like(img), [largest_contour], -1, color=255, thickness=cv2.FILLED))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    im_extend = cv2.dilate(mask, kernel, iterations=1) 
    im_erosion = cv2.erode(im_extend, kernel, iterations=1)
    masked_img = cv2.bitwise_and(img, img, mask = im_erosion)
    
    x, y, w, h = cv2.boundingRect(mask)
    masked_img = masked_img[y:y+h, x:x+w]

    return masked_img

def clip_aerial_picture_ski(src_dir : str):
    img = np.array(Image.open(src_dir))
    
    # czarny ma wartoci poniżej 10
    reclass = np.where(img > 5, 1, 0) 
    contours = measure.find_contours(reclass)
    contours = sorted(contours, key=len)
    biggest_part = max(contours, key=len)
    
    # czasem obraz może zostać podzielony na dwie części, 
    # wtedy dodajemy dwie największe do siebie części
    if len(biggest_part) * 0.5 > len(contours[-2]):
        masked_img = img[
            int(min(biggest_part[:,1])) : int(max(biggest_part[:,1])),
            int(min(biggest_part[:,0])) : int(max(biggest_part[:,0]))]
        
    else:
        x0_0 = int(min(biggest_part[:,1]))
        x1_0 = int(max(biggest_part[:,1]))
        y0_0 = int(min(biggest_part[:,0]))
        y1_0 = int(max(biggest_part[:,0]))
        
        x0_1 = int(min(contours[-2][:,1]))
        x1_1 = int(max(contours[-2][:,1]))
        y0_1 = int(min(contours[-2][:,0]))
        y1_1 = int(max(contours[-2][:,0]))
        
        masked_img = img[
            x0_0 if x0_0 < x0_1 else x0_1 : x1_0 if x1_0 > x1_1 else x1_1,
            y0_0 if y0_0 < y0_1 else y0_1 : y1_0 if y1_0 > y1_1 else y1_1]
        
    # jeśli mimo wszytko obciąliśmy więcej niż 40% zwracamy oryginał
    if masked_img.shape[0] < img.shape[0] * 0.4 or masked_img.shape[1] < img.shape[1] * 0.4:
            masked_img = img
    
    return masked_img

    

if __name__ == '__main__':
       
    src_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/01_InData/05_ZdjeciaGUGIK'
    dst_dir = '/media/pszubert/DANE/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/08_Ostrodzki_Probka'
    
    imgs = ['7_4049.tif', '8_4097.tif', '15_4406.tif', '7_4046.tif', '15_4413.tif', '12_4280.tif', '17_4496.tif', '12_4279.tif', '17_4494.tif', '12_4264.tif', '11_4213.tif', '13_4317.tif', '12_4274.tif', '10_4182.tif', '16_4456.tif', '13_4320.tif', '10_4200.tif', '15_4408.tif', '10_4180.tif', '13_4309.tif', '15_4410.tif', '17_4495.tif', '15_4405.tif', '9_4136.tif', '9_4141.tif', '8_4104.tif', '9_4142.tif', '15_4403.tif', '11_4230.tif', '15_4407.tif', '10_4184.tif', '12_4278.tif', '6_3331.tif', '9_4144.tif', '13_4328.tif', '12_4267.tif', '8_4099.tif', '10_4197.tif', '10_4196.tif', '13_4331.tif', '7_4042.tif', '10_4185.tif', '6_3323.tif', '8_4111.tif', '8_4095.tif', '8_4107.tif', '12_4273.tif', '13_4324.tif', '13_4322.tif', '11_4222.tif', '6_3325.tif', '10_4194.tif', '11_4215.tif', '14_4382.tif', '10_4191.tif', '9_4130.tif', '10_4187.tif', '9_4137.tif', '12_4268.tif', '13_4315.tif', '12_4272.tif', '10_4190.tif', '11_4217.tif', '12_4285.tif', '8_4096.tif', '9_4143.tif', '11_4234.tif', '7_4041.tif', '13_4319.tif', '14_4373.tif', '8_4094.tif', '16_4453.tif', '11_4227.tif', '13_4311.tif', '15_4399.tif', '14_4369.tif', '14_4362.tif', '7_4047.tif', '6_3326.tif', '17_4499.tif', '13_4310.tif', '9_4132.tif', '15_4412.tif', '12_4284.tif', '7_4055.tif', '17_4493.tif', '8_4108.tif', '8_4102.tif', '11_4232.tif', '15_4396.tif', '6_3324.tif', '12_4287.tif', '6_3330.tif', '12_4283.tif', '16_4457.tif', '8_4112.tif', '10_4199.tif', '14_4370.tif', '13_4326.tif', '10_4183.tif', '8_4110.tif', '11_4235.tif', '16_4455.tif', '13_4316.tif', '11_4233.tif', '12_4286.tif', '11_4237.tif', '13_4329.tif', '11_4226.tif', '9_4124.tif', '11_4228.tif', '12_4277.tif', '15_4409.tif', '15_4402.tif', '8_4098.tif', '12_4265.tif', '12_4288.tif', '11_4224.tif', '13_4325.tif', '6_3327.tif', '8_4093.tif', '16_4451.tif', '10_4178.tif', '7_4053.tif', '15_4411.tif', '9_4128.tif', '7_4054.tif', '16_4454.tif', '6_3322.tif', '10_4201.tif', '9_4125.tif', '12_4281.tif', '17_4500.tif', '9_4133.tif', '9_4129.tif', '9_4140.tif', '10_4181.tif', '14_4365.tif', '11_4236.tif', '9_4145.tif', '13_4323.tif', '17_4501.tif', '14_4372.tif', '9_4146.tif', '11_4229.tif', '12_4275.tif', '10_4186.tif', '8_4103.tif', '16_4459.tif', '17_4497.tif', '9_4127.tif', '7_4048.tif', '11_4216.tif', '10_4177.tif', '8_4105.tif', '10_4192.tif', '10_4188.tif', '9_4126.tif', '8_4100.tif', '7_4052.tif', '10_4195.tif', '11_4221.tif', '11_4220.tif', '12_4266.tif', '14_4368.tif', '10_4202.tif', '12_4271.tif', '6_3329.tif', '11_4218.tif', '12_4270.tif', '15_4404.tif', '6_3328.tif', '12_4282.tif', '9_4135.tif', '8_4101.tif', '7_4045.tif', '15_4397.tif', '8_4109.tif', '13_4313.tif', '8_4106.tif', '17_4498.tif', '9_4131.tif']
            
    for img in imgs:
        if img.endswith('.tif'):
            img_dir = os.path.join(src_dir, img)
            img_dst = os.path.join(dst_dir, img)
            
            if not os.path.isfile(img_dst):
                masked = clip_aerial_picture_cv2(img_dir, 250, 25)
                
                masked_tmb = Image.fromarray(masked)
                img_tmb = Image.open(img_dir)
                
                masked_tmb.thumbnail((1000,1000))
                img_tmb.thumbnail((1000,1000))
                
                plt.figure(figsize = (18,8))
                plt.title(img)
                plt.grid(False)
                plt.axis('off')
                plt.tight_layout()            
                
                plt.subplot(121).set_title('Masked')
                plt.imshow(masked_tmb, cmap='gray')
                    
                plt.subplot(122).set_title('Orginal')
                plt.imshow(img_tmb, cmap='gray')
                
                plt.show()
                
                cv2.imwrite(
                    img_dst, 
                    masked, 
                    params=(cv2.IMWRITE_TIFF_COMPRESSION,5))
            
            
            
    '''        
    for img in os.listdir(src_dir):
        if img.endswith('.tif'):
            img_dir = os.path.join(src_dir, img)
            clipped= clip_aerial_picture_ski(img_dir)
            masked = clip_aerial_picture_cv2(clipped, 25, 35)
            
            clipped_tmb = Image.fromarray(clipped)
            masked_tmb = Image.fromarray(masked)
            img_tmb = Image.open(img_dir)
            
            clipped_tmb.thumbnail((1000, 1000))
            masked_tmb.thumbnail((1000,1000))
            img_tmb.thumbnail((1000,1000))
            
            plt.figure(figsize = (36,12))
            plt.title(img)
            plt.grid(False)
            plt.axis('off')
            plt.tight_layout()            
            
            plt.subplot(131).set_title('Clipped')
            plt.imshow(clipped_tmb, cmap='gray')
            
            plt.subplot(132).set_title('Masked')
            plt.imshow(masked_tmb, cmap='gray')
                
            plt.subplot(133).set_title('Orginal')
            plt.imshow(img_tmb, cmap='gray')
            
            plt.show()
            
            
            cv2.imwrite(
                os.path.join(dst_dir, img), 
                masked, 
                params=(cv2.IMWRITE_TIFF_COMPRESSION,5))'''
                    
            
            
            



