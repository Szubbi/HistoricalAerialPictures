import os
import numpy as np
import geopandas as gpd

from ImageConverter import ImageConverter
from GenerateTrainingSmple import *
from util import *
from rasterio.crs import CRS
from Mask_RCNN.convert_annotations import yolo_to_mask

'''
Script to create training samples without histogram matching. 
Script logic:
1. Calculate target values distribution 
2. For each image:
    a. find conversion value
    b. convert image
    c. get buildings footprints
    d. split image into tiles
    e. generate building masks
'''

if __name__ == "__main__":
    PATCH_SIZE = 640

    dst_dataset_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\07_no_hist'
    db_dir = r'C:\Users\pzu\OneDrive - Uniwersytet Jagielloński\Badania\04_ArchiwalneZdjecia\02_DataProcessing\Data.gpkg'
    src_imgs_dir = r'C:\Users\pzu\Documents\01_Projekty\03_HistoricalAerial\03_SampleRGB'
    buildings_db_dir = '/mnt/96729E38729E1D55/01_Badania/10_BudynkiPolska/02_DataProcessing/dataProcessing.gpkg'


    src_imgs = [os.path.join(src_imgs_dir, f) for f in os.listdir(src_imgs_dir) if f.endswith('.tif')]
    hash_table = gpd.read_file(db_dir, layer = 'hash_table_01')
    log_dir = dst_dataset_dir

    blur_sharp_tables_BW = load_sqllite_dataframe(db_dir, 'img_BlurSharpTable_04')

    # histogram values need converting from bites
    blur_sharp_tables_BW['histogram'] = blur_sharp_tables_BW['histogram'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32))
    
    # calculate target values distribution
    target_values = generate_stratified_samples(blur_sharp_tables_BW, len(src_imgs))

    # dst folder structure 
    for subdir in ['conv_images', 'patches', 'yolo_labels', 'masks']:
        os.makedirs(os.path.join(dst_dataset_dir, subdir), exist_ok=True)

    ##############################################################################
    for idx, src_img, (trg_blur, trg_noise, trg_contrast, trg_hist) in enumerate(zip(src_imgs, target_values)):
        progress_label = f'Processing {src_img}. Target values: {trg_blur}, {trg_noise}, {trg_contrast}'
        print_progress_bar(idx+1, len(src_imgs), sufix=progress_label, length=50)
        
        IC = ImageConverter(src_img)
        IC.noise_lvl_trg = trg_noise
        IC.blur_lvl_trg = trg_blur
        IC.contrast_lvl_trg = trg_contrast
        IC.histogram_trg = trg_hist
        IC.find_convertion_values(45, 0.4, histogram_matching=False)
        IC.convert_image()
        conv_img, transform = IC.save(
            os.path.join(dst_dataset_dir, 'conv_images', IC.img_nme + '_conv.tif'))
        
        # get building footprints
        overlaping_layers = get_layers_extent((conv_img, transform), hash_table)
        buildings_gdf = get_geometries(buildings_db_dir, overlaping_layers['file_name'].to_list(), src_img)
    
        print(f'Generating Patches for: {IC.img_nme}')
        patches = split_geotiff_to_patches((conv_img, transform), PATCH_SIZE, 0.25)

        for index, patch in enumerate(patches):
            patch_img, patch_transform = patch
            patch_name = f'{IC.img_nme}_{index}.png'

            patch_dir = os.path.join(dst_dataset_dir, 'patches', patch_name)
            yolo_label_dir = os.path.join(dst_dataset_dir, 'yolo_labels', patch_name.replace('.png', '.txt'))
            mask_dir = os.path.join(dst_dataset_dir, 'masks', patch_name)

            patch_extent = get_raster_extent(patch)
            bld_masks_gdf = gpd.clip(buildings_gdf, patch_extent)
            bld_masks_gdf = bld_masks_gdf.explode(index_parts=True).reset_index(drop=True)
            
            # generate yolo masks
            yolo_label_txt = generate_yolo_labels(PATCH_SIZE, patch_transform, bld_masks_gdf)
            yolo_label_txt = yolo_label_txt if yolo_label_txt else ''

            # Save patch and yolo labels
            raster_metadata = {
                'driver': 'PNG',
                'dtype': 'uint8', 
                'nodata': None,
                'height': patch[0].shape[0], 
                'width': patch[0].shape[1], 
                'count': 1, 
                'crs': CRS.from_epsg(2180), 
                'transform': patch[1]}
            
            with rasterio.open(patch_dir, 'w', **raster_metadata) as dst:
                dst.write(patch_img, 1)

            with open(yolo_label_dir, 'w') as f:
                f.write(yolo_label_txt)




 
    
