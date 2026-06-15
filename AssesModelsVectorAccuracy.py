import os
import torch
import geopandas as gpd
import pandas as pd
import rasterio

from Mask_RCNN.detect_objects import detect_georeferenced_buildings
from Yolo.detect_objects import yolo_detect_georeferenced_buildings
from util import split_geotiff_to_patches, load_image, load_model, plot_gpkg_on_geotiff
from ultralytics import YOLO
from Mask_RCNN.train_maskrcnn import get_model_instance_segmentation

'''
Do Napisania:
    - na podstawie zakresu zdjęcia pozyskanie ground trouth 
    - pozyskanie detekcji z modelu
    - policzenie IoU i mAP
    - policzenie acuracy
    
'''
def get_ground_truth(src_img, gpkg_path, data_layer):
    raster_extent = rasterio.open(src_img).bounds
    gdf = gpd.read_file(gpkg_path, layer=data_layer)
    
    gdf = gdf.cx[raster_extent.left:raster_extent.right, raster_extent.bottom:raster_extent.top]
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf.dissolve()
    gdf = gdf.explode()
    gdf = gdf.reset_index()
    gdf['uid'] = gdf.index
    
    return gdf


def calculate_iou(pred_gdf, gt_gdf):
    pred_union = pred_gdf.union_all()
    gt_union = gt_gdf.union_all()
    
    intersection = pred_union.intersection(gt_union).area
    union = pred_union.union(gt_union).area
    
    return intersection / union if union != 0 else 0


def calculate_metrics(pred_gdf, gt_gdf):
    intersect_gdf = gpd.overlay(pred_gdf, gt_gdf, how='intersection', keep_geom_type=False)
    
    TP_gdf = treshold_mask[pred_gdf['index' ].isin(intersect_gdf['index_1'])]
    FP_gdf = treshold_mask[~pred_gdf['index' ].isin(intersect_gdf['index_1'])]
    FN_gdf = ground_truth_gdp[~gt_gdf['uid'].isin(intersect_gdf['uid'])]

    return len(TP_gdf), len(FP_gdf), len(FN_gdf)


if __name__ == "__main__":
    imgs_dir = "/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/02_TestBW"
    models_dir = "/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/05_Models"
    gpkg_path = os.path.join(imgs_dir, 'Data.gpkg')
    data_layer = 'obszarytestoweortobw_bdot_00'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_THRESHOLD = 0.85

    results = gpd.GeoDataFrame(columns=['model', 'IMG', 'TP', 'FP', 'FN', 'IoU', 'F1', 'Precision', 'Recall', 'Accuracy'])

    #for model in  os.listdir(models_dir):
    for model in ['yolo-11l-bw.pt']:
        if model.endswith('.pt'):
            print(f'Processing model: {model}')
            if 'rcnn' in model:
                weight_dir = os.path.join(models_dir, model)
                checkpoint = torch.load(weight_dir, map_location=device)

                model = get_model_instance_segmentation(num_classes=2)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                model.to(device)

                for img in [_ for _ in os.listdir(imgs_dir) if _.endswith('.tif')]:
                    print(f'Processing image: {img}')

                    ground_truth_gdp = get_ground_truth(os.path.join(imgs_dir, img), gpkg_path, data_layer)

                    img_path = os.path.join(imgs_dir, img)
                    detected_gdp = detect_georeferenced_buildings(
                        img_path, model, MASK_THRESHOLD, 640, 0.25)
                    treshold_mask = detected_gdp[detected_gdp['score'] > MASK_THRESHOLD]
                    print(treshold_mask.head())

                    TP, FP, FN = calculate_metrics(treshold_mask, ground_truth_gdp)
                    iou = calculate_iou(treshold_mask, ground_truth_gdp)

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

                    results = results.append({
                        'model': model,
                        'IMG': img,
                        'TP': TP,
                        'FP': FP,
                        'FN': FN,
                        'IoU': iou,
                        'F1': f1_score,
                        'Precision': precision,
                        'Recall': recall,
                        'Accuracy': accuracy
                    }, ignore_index=True)

                    
            
            if 'yolo' in model:
                yolo_model_dir = os.path.join(models_dir, model)
                yolo_model = YOLO(yolo_model_dir)

                for img in [_ for _ in os.listdir(imgs_dir) if _.endswith('.tif')]:
                    ground_truth_gdp = get_ground_truth(os.path.join(imgs_dir, img), gpkg_path, data_layer)

                    img_path = os.path.join(imgs_dir, img)
                    detected_gdp = yolo_detect_georeferenced_buildings(
                        img_path, yolo_model, 640, 0.25)
                    treshold_mask = detected_gdp[detected_gdp['score'] > MASK_THRESHOLD]

                    TP, FP, FN = calculate_metrics(treshold_mask, ground_truth_gdp)
                    iou = calculate_iou(treshold_mask, ground_truth_gdp)

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

                    results = results.append({
                        'model': model,
                        'IMG': img,
                        'TP': TP,
                        'FP': FP,
                        'FN': FN,
                        'IoU': iou,
                        'F1': f1_score,
                        'Precision': precision,
                        'Recall': recall,
                        'Accuracy': accuracy
                    }, ignore_index=True)

        results.to_file(
            gpkg_path,
            layer='models_accuracy_results',
            driver='GPKG',
            mode='w')    

        break    
    

    

    
    


        



    