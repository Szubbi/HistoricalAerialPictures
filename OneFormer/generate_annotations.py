import json
import numpy as np
from PIL import Image, ImageDraw
import os

from util import print_progress_bar

def yolo_polygons_to_panoptic(image_path, txt_path, output_mask_path, json_dict, json_path, image_id=1):
    """
    Convert YOLO polygon annotations (single class) to COCO Panoptic format.
    
    Parameters:
    - image_path: path to original image
    - txt_path: path to YOLO polygon annotation file
    - output_mask_path: where to save the panoptic PNG mask
    - json_path: where to save the JSON annotation
    - image_id: numeric ID for the image
    """
    # Load image to get size
    img = Image.open(image_path)
    width, height = img.size
    
    # Read YOLO polygons
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    mask = Image.new("I", (width, height), 0)  # 32-bit integer mask
    draw = ImageDraw.Draw(mask)
    
    segments_info = []
    instance_id = 1
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        # YOLO polygon: class_id x1 y1 x2 y2 ...
        coords = list(map(float, parts[1:]))
        # Convert normalized coords to pixel coords
        polygon = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]
        
        # Draw polygon with unique ID
        draw.polygon(polygon, fill=instance_id)
        
        # Compute bbox and area
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
        area = int(abs(sum(xs[i] * ys[i+1] - xs[i+1] * ys[i] for i in range(-1, len(xs)-1)) / 2))
        
        segments_info.append({
            "id": instance_id,
            "category_id": 1,  # single class
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        })
        instance_id += 1
    
    # Save mask
    mask.save(output_mask_path)
    
    # Append to json_dict
    json_dict["images"].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": os.path.basename(image_path)
    })
    json_dict["annotations"].append({
        "image_id": image_id,
        "file_name": os.path.basename(output_mask_path),
        "segments_info": segments_info
    })

    # Ensure categories exist
    if "categories" not in json_dict or not json_dict["categories"]:
        json_dict["categories"] = [{"id": 1, "name": "object", "isthing": 1}]

    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


if __name__ == "__main__":
    images_dir = '/home/pszubert/Dokumenty/04_ConvDataset/images'
    labels_dir = '/home/pszubert/Dokumenty/04_ConvDataset/labels'
    one_former_dir = '/home/pszubert/Dokumenty/04_ConvDataset/OneFormer_labels'
    splits = ['train', 'val', 'test']

    for split in splits:
        images_path = os.path.join(images_dir, split)
        txts_path = os.path.join(labels_dir, split)
        one_former_path = os.path.join(one_former_dir, split)
        json_path = os.path.join(one_former_path, 'annotations.json')
        
        json_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        if not os.path.exists(one_former_path):
            os.makedirs(one_former_path)

        images = [f for f in os.listdir(images_path) if f.endswith('.png')]

        for idx, image in enumerate(images):
            print_progress_bar(idx + 1, len(images), prefix=f"Processing {split}:", suffix="Complete", length=50)
            
            image_path = os.path.join(images_path, image)
            txt_path = os.path.join(txts_path, image.replace('.png', '.txt'))
            output_mask_path = os.path.join(one_former_path, image)

            yolo_polygons_to_panoptic(image_path, txt_path, output_mask_path, 
                                      json_dict, json_path, image_id=idx)

