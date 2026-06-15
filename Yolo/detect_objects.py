# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os
import rasterio

from rasterio.crs import CRS
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
        

def yolo_detect_georeferenced_buildings(src_img, model, patch_size, overlap_ratio):
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
                    
                    shape = shapes(mask.data.cpu().numpy().astype(np.uint8), 
                                   mask=mask.data.cpu().numpy().astype(np.uint8),
                                   transform=transform)
                    
                    geoms = [geometry.shape(s) for s,v in shape]
                    
                    for geom in geoms:
                        predictions[f'id_{idx}_{num}_{id}'] = {
                            "score" : float(score[id]),
                            "geometry" : geom}

    results = gpd.GeoDataFrame.from_dict(predictions, orient='index', crs = 'EPSG:2180')
    
    return results


def yolo_patch_to_instance_mask_raster(
    patch: np.ndarray,
    transform,
    model,
    score_threshold: float = 0.25,
    dtype=np.uint16,
):
    """
    Run YOLO on a patch and return an instance mask raster.

    Returns
    -------
    mask : np.ndarray (H, W)
        0 = background, 1..N = instance IDs
    transform : Affine
        Georeferencing transform
    scores : dict[int, float]
        Instance ID -> confidence score
    """

    h, w = patch.shape[:2]
    mask_raster = np.zeros((h, w), dtype=dtype)
    scores = {}

    img = Image.fromarray(patch)
    result = model(img, verbose=False)[0]

    if result.masks is None:
        return mask_raster, transform, scores

    next_id = 1

    masks = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for i in range(masks.shape[0]):
        if confs[i] < score_threshold:
            continue

        binary_mask = masks[i].astype(bool)

        # overwrite only background pixels
        write_mask = (binary_mask & (mask_raster == 0))
        mask_raster[write_mask] = next_id
        scores[next_id] = float(confs[i])

        next_id += 1

    return mask_raster, transform, scores


def write_mask_geotiff(path, mask, transform, crs, nodata=0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        dst.write(mask, 1)

if __name__ == "__main__":
    models_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\05_Models'
    idxs = [295]
    
    for model in os.listdir(models_dir):
        if model.endswith('.pt'):   
                
            test_img_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\01_SampleBW\17_36541_M-34-63-B-b-1-3.tif"
            out_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\patches'
            mask_threshold = 0.3
            
            patches = split_geotiff_to_patches(test_img_dir, 640, 0.25)
            draw_patch_grid_on_geotiff(test_img_dir, 640, 0.25)
            
            img_idx = 381
            plt.imshow(Image.fromarray(patches[img_idx][0]), cmap='gray')
            plt.title(f'Patch {img_idx}')
            plt.show()
            
            yolo_model = YOLO(os.path.join(models_dir, model))
            resuls = yolo_detect_georeferenced_buildings(test_img_dir, yolo_model, 640, 0.25)
            resuls.to_file(os.path.join(out_dir, str(model).replace('.pt', '.shp')))
            
###############################################################################            
            yolo_model = YOLO(r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\05_Models\yolo-26m-bw.pt" )
            for idx, patch in enumerate(patches):
                if idx in idxs:
                    
                    mask_raster, transform, scores = yolo_patch_to_instance_mask_raster(
                        patch[0],
                        patch[1],
                        yolo_model                                         
                        )
                    
                    out_patch = os.path.join(out_dir, 'probka_patche', f'17_36541_M-34-63-B-b-1-3_{idx}.tif')
                    out_mask = os.path.join(out_dir, 'probka_patche', f'17_36541_M-34-63-B-b-1-3_{idx}_mask.tif')
                    
                    write_mask_geotiff(out_patch, patch[0], transform, CRS.from_epsg(2180))
                    write_mask_geotiff(out_mask, mask_raster, transform, CRS.from_epsg(2180))
                    
                    
            for idx, (patch, transform) in enumerate(patches):
                    out_patch = os.path.join(out_dir, f'17_36541_M-34-63-B-b-1-3_{idx}.tif')
                    write_mask_geotiff(out_patch, patch, transform, CRS.from_epsg(2180))
                
                    
                    
                
                
 

    
