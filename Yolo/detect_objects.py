# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from ultralytics import YOLO
from util import split_geotiff_to_patches, draw_patch_grid_on_geotiff
from PIL import Image
from rasterio.features import shapes
from shapely import geometry

   
# Progress bar
def print_progress_bar(iteration, total, prefix='', suffix='', iter_time = '', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()
        

def detect_georeferenced_buildings(src_img, model, patch_size, overlap_ratio):
    predictions = {}
    
    patches = split_geotiff_to_patches(src_img, patch_size, overlap_ratio)
    
    for idx, (image, transform) in enumerate(patches):
        print_progress_bar(idx+1, len(patches), prefix=f'Patch: {idx+1}')
        img = Image.fromarray(image)
        # detect objects
        results = model(img, verbose=False)
        
        for num, i in enumerate(results):
            if i:
                score = i.boxes.conf
                masks = i.masks
                
                for id, mask in enumerate(masks):
                    
                    shape = shapes(mask.data.numpy().astype(np.uint8), 
                                   mask=mask.data.numpy().astype(np.uint8),
                                   transform=transform)
                    
                    geoms = [geometry.shape(s) for s,v in shape]
                    
                    for geom in geoms:
                        predictions[f'id_{idx}_{num}_{id}'] = {
                            "score" : float(score[id]),
                            "geometry" : geom}

    results = gpd.GeoDataFrame.from_dict(predictions, orient='index', crs = 'EPSG:2180')
    
    return results


if __name__ == "__main__":
    
    test_img_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\30_26667_M-34-77-B-b-1-4.tif'
    model_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\best.pt'
    mask_threshold = 0.3
    
    patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
    draw_patch_grid_on_geotiff(test_img_dir, 640, 0.25)
    
    img_idx = 381
    plt.imshow(Image.fromarray(patches[img_idx][0]), cmap='gray')
    plt.title(f'Patch {img_idx}')
    plt.show()
    
    model = YOLO(model_dir)
    resuls = detect_georeferenced_buildings(test_img_dir, model, 640, 0.25)
    resuls.to_file(r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\results_yolo_02.shp')
                
 

    
    
