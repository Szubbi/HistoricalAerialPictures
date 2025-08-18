# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import geopandas as gpd


from util import split_geotiff_to_patches, load_image, load_model
from rasterio.features import shapes
from shapely import geometry


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
    
    weight_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\exp1_best_model.pth"
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\30_26667_M-34-77-B-b-1-4.tif'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_THRESHOLD = 0.5
    
    #load model
    model = load_model(weight_dir)
    model.eval()
    model.to(device)

    # test image
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    predictions = {}

    for idx, (image, transform) in enumerate(patches):
        print(f'\rWorking on: {idx+1}/{len(patches)}', end='', flush=True)
        
        img = load_image(image)

        
        # detect objects
        with torch.no_grad():
            prediction = model(img)           
        
        for i in range(len(prediction[0]['scores'])):
            score = prediction[0]['scores'][i].item()
            mask = prediction[0]['masks'][i, 0].cpu().numpy()
            
            shape = shapes((mask > MASK_THRESHOLD).astype(np.uint8), 
                           mask=(mask > MASK_THRESHOLD).astype(np.uint8),
                           transform=transform)
            
            geoms = [geometry.shape(s) for s,v in shape]
            
            for geom in geoms:
                predictions[f'id_{idx}_{i}'] = {
                    "score" : score,
                    "geometry" : geom}

            
    results = gpd.GeoDataFrame.from_dict(predictions, orient='index', crs = 'EPSG:2180')
    results.to_file(r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\results_11.shp')
    #visualize_predictions(img, prediction, MASK_THRESHOLD)
    
    plt.imshow()
            
    print('Detection process completed')
            
        