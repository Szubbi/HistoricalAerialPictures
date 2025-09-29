import os
import numpy as np
from PIL import Image, ImageDraw


def yolo_to_mask(txt_file_path, image_shape, output_path):

    """
    Converts YOLO normalized polygon annotations to a single labeled instance mask PNG,
    where each polygon has a unique integer ID (1, 2, 3, ...), background is 0.

    Args:
        txt_file_path (str): Path to YOLO polygon .txt file.
        image_shape (tuple): (height, width) of the target image.
        output_path (str): Full path to save the labeled PNG mask.
    """
    H, W = image_shape

    # Initialize empty numpy array for labeled mask
    labeled_mask = np.zeros((H, W), dtype=np.uint16)

   
    with open(txt_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]


    if not lines:
        # Save empty mask if no polygons
        Image.fromarray(labeled_mask).save(output_path)
        return

    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7:
            continue

        coords = list(map(float, parts[1:]))

        polygon = [
            (x * W, y * H)
            for x, y in zip(coords[::2], coords[1::2])
        ]

        # Create a temporary mask image for this polygon
        mask_img = Image.new('I', (W, H), 0)  # 32-bit int mode
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(polygon, outline=idx, fill=idx)

        mask_np = np.array(mask_img)

        # Assign instance id idx to pixels where polygon mask is set
        labeled_mask[mask_np == idx] = idx

    # Save labeled mask as 16-bit PNG
    Image.fromarray(labeled_mask).save(output_path)



if __name__ == "__main__":
    yolo_txts_path = "/home/pszubert/Dokumenty/04_ConvDataset/labels/test"
    image_size = (640, 640)  
    output_path = "/home/pszubert/Dokumenty/04_ConvDataset/maskrcnn_labels/test"

    yolo_txt_files = [f for f in os.listdir(yolo_txts_path) if f.endswith('.txt')]
    for idx, yolo_txt_file in enumerate(yolo_txt_files):
        print(f"\rProcessing {idx + 1}/{len(yolo_txt_files)}: {yolo_txt_file}", end="", flush=True)
        yolo_txt_path = os.path.join(yolo_txts_path, yolo_txt_file)
        output_mask_path = os.path.join(output_path, yolo_txt_file.replace('.txt', '.png'))
        yolo_to_mask(yolo_txt_path, image_size, output_mask_path)