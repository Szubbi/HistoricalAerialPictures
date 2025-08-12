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

from util import split_geotiff_to_patches, draw_patch_grid_on_geotiff
from ultralytics import YOLO
from PIL import Image
from Mask_RCNN.train_maskrcnn import get_model_instance_segmentation
from torchvision.transforms import functional as F

def load_model(weights_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(
        torch.load(weights_dir, 
                   map_location=torch.device(device)))
    
    return model


def load_image(pil_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert to tensor
    img = F.to_tensor(pil_img)
    # add batch dimension
    img = img.unsqueeze(0)
    img.to(device)
    
    return img


def visualize_predictions(image, prediction, threshold=0.5):
    """
    Visualize predictions from a Mask R-CNN model using matplotlib.

    Parameters:
    - image: torch.Tensor of shape [1, H, W], [H, W], or [1, 1, H, W]
    - prediction: output from the model, a list of dicts with keys 'boxes', 'masks', 'scores'
    - threshold: float, score threshold to filter predictions
    """
    # Handle different image shapes
    if image.dim() == 4:  # [1, 1, H, W]
        image_np = image.squeeze().cpu().numpy()
    elif image.dim() == 3 and image.shape[0] == 1:  # [1, H, W]
        image_np = image.squeeze(0).cpu().numpy()
    elif image.dim() == 2:  # [H, W]
        image_np = image.cpu().numpy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np, cmap='gray')

    for i in range(len(prediction[0]['scores'])):
        score = prediction[0]['scores'][i].item()
        if score >= threshold:
            box = prediction[0]['boxes'][i].cpu().numpy()
            mask = prediction[0]['masks'][i, 0].cpu().numpy()

            # Create RGBA mask image: red where mask is True, transparent elsewhere
            rgba_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba_mask[mask > 0.5] = [1, 0, 0, 0.4]  # Red with alpha 0.4
            # Draw bounding box
            rect = mpatches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                      linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Overlay mask
            ax.imshow(rgba_mask)

            # Annotate score
            ax.text(box[0], box[1]-5, f"{score:.2f}", color='yellow', fontsize=12, weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.show()




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
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\17_25981_M-34-52-D-b-1-1.tif'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load model
    model = load_model(weight_dir)
    model.eval()
    model.to(device)

    # test image
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    draw_patch_grid_on_geotiff(test_img_dir, 640, 0.25)
    
    img_idx = 121
    plt.imshow(Image.fromarray(patches[img_idx][0]), cmap='gray')
    plt.title(f'Patch {img_idx}')
    plt.show()
    
    test_img = Image.fromarray(patches[img_idx][0])
    test_img = load_image(test_img)
    
    # predict
    with torch.no_grad():
        prediction = model(test_img)
        
    # display predict    
    visualize_predictions(test_img, prediction, threshold=0.1)
        
    
    
    
    

