#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:28:31 2025

@author: piotr.szubert@doctoral.uj.edu.pl

"""


import torch
import onnx
import numpy as np

from ultralytics import YOLO
from onnx import helper, numpy_helper, TensorProto


def keep_segmentation_output_slice_and_resize(model_path: str, output_path: str = "best_segmentation_resized.onnx") -> str:
    model = onnx.load(model_path)

    # Identify segmentation output
    segmentation_outputs = []
    for output in model.graph.output:
        if "proto" in output.name.lower() or "mask" in output.name.lower():
            segmentation_outputs.append(output)

    if not segmentation_outputs:
        for output in model.graph.output:
            shape_dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            if len(shape_dims) == 4 and shape_dims[1] in (32, 64) and shape_dims[2] > 100:
                segmentation_outputs.append(output)

    if not segmentation_outputs and len(model.graph.output) > 1:
        segmentation_outputs = [model.graph.output[1]]

    selected_output = segmentation_outputs[0]
    original_output_name = selected_output.name

    # Slice to keep only the first channel
    slice_output_name = original_output_name + "_sliced"
    starts_initializer = numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_starts")
    ends_initializer = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_ends")
    axes_initializer = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")

    model.graph.initializer.extend([starts_initializer, ends_initializer, axes_initializer])

    slice_node = helper.make_node(
        "Slice",
        inputs=[original_output_name, "slice_starts", "slice_ends", "slice_axes"],
        outputs=[slice_output_name],
        name="SliceFirstChannel"
    )
    model.graph.node.append(slice_node)

    # Resize to 640x640
    resize_output_name = slice_output_name + "_resized"
    roi_initializer = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales_initializer = numpy_helper.from_array(np.array([], dtype=np.float32), name="scales")
    sizes_initializer = numpy_helper.from_array(np.array([1, 1, 640, 640], dtype=np.int64), name="sizes")

    model.graph.initializer.extend([roi_initializer, scales_initializer, sizes_initializer])

    resize_node = helper.make_node(
        "Resize",
        inputs=[slice_output_name, "roi", "scales", "sizes"],
        outputs=[resize_output_name],
        mode="linear",
        coordinate_transformation_mode="asymmetric",
        name="ResizeTo640x640"
    )
    model.graph.node.append(resize_node)

    # Set final output
    model.graph.ClearField("output")
    output_tensor = helper.make_tensor_value_info(resize_output_name, TensorProto.FLOAT, [1, 1, 640, 640])
    model.graph.output.extend([output_tensor])

    onnx.save(model, output_path)
    print(f"Saved modified model to: {output_path}")
    return output_path





if __name__ == '__main__':

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("GPU is available for training.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU is not available. Training will use CPU.")

    torch.cuda.set_device(0)

    # Load a YOLO model
    model = YOLO('/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/yolo11n-seg.pt')

    # HPO tunning 
    search_space = {
        'lr0': (1e-5, 1e-1), # Initial learning rate
        'lrf': (0.01, 1.0), # Final learning rate multiplier
        'momentum': (0.6, 0.98), # Momentum for SGD
        'weight_decay': (0.0001, 0.01), # Regularization
        'warmup_epochs': (0, 5), # Warm-up period
        'warmup_bias_lr': (0.0, 0.2), # Bias learning rate during warm-up
        'box': (0.02, 0.2), # Box loss gain
        'cls': (0.2, 4.0), # Class loss gain
        'obj': (0.2, 4.0), #	Objectness loss gain
        "degrees": (0.0, 45.0)
    }

    model.tune(
        data='/home/pszubert/Dokumenty/04_ConvDataset/YOLO_HPO_Dataset.yaml', 
        imgsz=640,
        batch=40,
        epochs=15,
        iterations=50,
        optimizer='AdamW',
        plots=True,
        project='/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/12_YOLO_Training/HPO'
        )

    

    # Train the model
    # model.train(
    #     data='/home/pszubert/Dokumenty/04_ConvDataset/YOLO_Dataset.yaml',         # Path to dataset YAML
    #     imgsz=640,                # Image size
    #     epochs=50,               # Max epochs
    #     patience=10,              # Early stopping patience
    #     batch=32,                 # Batch size
    #     lr0=0.005,                 # Initial learning rate
    #     lrf=0.01,                 # Final learning rate fraction
    #     warmup_epochs=5,
    #     optimizer='SGD',          # Optimizer (SGD or Adam)
    #     cos_lr=True,              # Use cosine learning rate scheduler
    #     amp=True,                 # Mixed precision training
    #     cache=True,               # Cache images for faster training
    #     save=True,                # Save checkpoints
    #     save_period=5,            # Save every 5 epochs
    #     project='/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/12_YOLO_Training/02_YOLO_100e_20250718',     # Output directory
    #     name='yolo-50e-005-01' # Run name
    # )



    
