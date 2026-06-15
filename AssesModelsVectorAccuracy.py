import os
import torch
import geopandas as gpd
import rasterio

from Mask_RCNN.detect_objects import detect_georeferenced_buildings
from Yolo.detect_objects import yolo_detect_georeferenced_buildings
from util import split_geotiff_to_patches, load_image, load_model, plot_gpkg_on_geotiff
from ultralytics import YOLO

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
    img_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\02_TestBW\13_24302_M-34-34-D-b-4.tif"
    weight_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\05_Models\conv\mask-rcnn-conv.pth"
    yolo_model_dir = r"C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\05_Models\conv\yolo-26l-conv.pt"
    gpkg_path = r'C:/Users/pzu/Documents/01_Projekty/03_HistoricalAerial/02_TestBW/Data.gpkg'
    data_layer = 'obszarytestoweortobw_bdot_00'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_THRESHOLD = 0.5
    
    ground_truth_gdp = get_ground_truth(img_dir, gpkg_path, data_layer)
    plot_gpkg_on_geotiff(img_dir, ground_truth_gdp)
    
    # Mask R-CNN model
    model = load_model(weight_dir)
    model.eval()
    model.to(device)

    detected_gdp = detect_georeferenced_buildings(
        img_dir, model, MASK_THRESHOLD,  640, 0.25)
    treshold_mask = detected_gdp[detected_gdp['score'] > 0.75]
    
    plot_gpkg_on_geotiff(img_dir, treshold_mask)
    iou = calculate_iou(treshold_mask, ground_truth_gdp)
    accuracy = calculate_metrics(treshold_mask, ground_truth_gdp)
    print(f'IoU: {iou}')
    print(f'Accuracy: {accuracy}')

    # YOLO model
    yolo_model = YOLO(yolo_model_dir)

    yolo_detected_gdp = yolo_detect_georeferenced_buildings(
        img_dir, yolo_model, 640, 0.25)
    
    plot_gpkg_on_geotiff(img_dir, yolo_detected_gdp)
    yolo_iou = calculate_iou(yolo_detected_gdp, ground_truth_gdp)
    yolo_accuracy = calculate_metrics(yolo_detected_gdp, ground_truth_gdp)
    print(f'YOLO IoU: {yolo_iou}')
    print(f'YOLO Accuracy: {yolo_accuracy}')
    

    
    


        



    