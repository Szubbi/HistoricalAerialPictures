#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:31:23 2025

@author: pszubert
"""

import torch
import onnx
import numpy as np
import os
import random
import matplotlib.pyplot as plt


from ultralytics import YOLO
from util import count_rows_in_file
from PIL import Image, ImageDraw
from cv2 import imread


def get_valid_random_samples_lazy(file_list, sample_size, count_rows_in_file):
    remaining_files = file_list[:]
    random.shuffle(remaining_files)
    valid_samples = []

    while remaining_files and len(valid_samples) < sample_size:
        candidate = remaining_files.pop()
        if count_rows_in_file(candidate) > 0:
            valid_samples.append(candidate)

    return valid_samples


def draw_yolo_polygons_on_pil(image, yolo_text_path):
    """
    Draw YOLO polygon annotations on a PIL image.

    Parameters:
    - image: PIL.Image object
    - yolo_text_path: path to a .txt file containing YOLO polygon annotations (one per line)

    Returns:
    - PIL.Image object with polygons drawn
    """
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(yolo_text_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            coords = list(map(float, parts[1:]))
            points = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
            draw.polygon(points, outline="red", width=4)

    return img



def display_predictions(pil_images, yolo_results, yolo_texts):
    """
    Display 8 PIL images with YOLO predictions and ground truth polygon annotations.

    Parameters:
    - pil_images: list of 8 PIL.Image objects
    - yolo_results: list of 8 Ultralytics Results objects
    - yolo_texts: list of 8 strings, each containing YOLO polygon annotations
    """
    assert len(pil_images) == len(yolo_results) == len(yolo_texts) == 8, "All input lists must have 8 elements."

    fig, axes = plt.subplots(8, 3, figsize=(15, 30))
    for i in range(8):
        # Original image
        axes[i, 0].imshow(pil_images[i].convert('L'), cmap='gray')
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        # YOLO prediction overlay
        result_img = yolo_results[i][0].plot()
        axes[i, 1].imshow(result_img)
        axes[i, 1].set_title(f'YOLO Prediction {i+1}')
        axes[i, 1].axis('off')

        # Ground truth polygon overlay
        gt_img = draw_yolo_polygons_on_pil(pil_images[i], yolo_texts[i])
        axes[i, 2].imshow(gt_img)
        axes[i, 2].set_title(f'Ground Truth Polygons {i+1}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    model = YOLO('/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/12_YOLO_Training/runs/train/yolo-25e-005-012/weights/best.pt')
    
    model.export(format='onnx')
    
    
    test_immgs_dir = '/home/pszubert/Dokumenty/03_DeepLab_dataset/images/'
    test_labels_dir = '/home/pszubert/Dokumenty/03_DeepLab_dataset/yolo_labels/'
    
    test_labels = [os.path.join(test_labels_dir, _) for _ in os.listdir(test_labels_dir) if _.endswith('.txt')]

    sample_labels = get_valid_random_samples_lazy(test_labels, 8, count_rows_in_file)
    sample_imgs = [os.path.join(_.replace('.txt', '.png')) for _ in sample_labels]
    sample_imgs = [os.path.join(_.replace(test_labels_dir, test_immgs_dir)) for _ in sample_imgs]
    sample_imgs = [Image.open(_) for _ in sample_imgs]
    
    predictions = [model(_) for _ in sample_imgs]
    
    display_predictions(sample_imgs, predictions, sample_labels)

    plt.imshow(draw_yolo_polygons_on_pil(sample_imgs[2], sample_labels[2]))

    
