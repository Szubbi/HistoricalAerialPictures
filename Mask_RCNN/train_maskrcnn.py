import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
from PIL import Image
import logging
from datetime import datetime
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import random

# Progress bar function 
def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# Dataset class - loading all to RAM
class GrayscaleMaskRCNNDatasetRAM(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([_ for _ in os.listdir(image_dir) if _.endswith(('.png'))])
        self.mask_files = sorted([_ for _ in os.listdir(mask_dir) if _.endswith(('.png'))])
        self.images = []
        self.targets = []

        total = len(self.image_files)
        broken_images = []

        print("Loading dataset into RAM:")
        for idx in range(total):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            try:
                img = Image.open(img_path).convert("L")
                mask = Image.open(mask_path)
            except Exception as e:
                broken_images.append(self.image_files[idx])
                continue

            img_tensor = F.to_tensor(img)
            mask_np = np.array(mask)

            obj_ids = np.unique(mask_np)
            obj_ids = obj_ids[obj_ids != 0]

            if len(obj_ids) == 0:
                continue

            masks = mask_np == obj_ids[:, None, None]

            valid_boxes = []
            valid_masks = []

            for m in masks:
                pos = np.where(m)
                if pos[0].size == 0 or pos[1].size == 0:
                    continue
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                if xmax <= xmin or ymax <= ymin:
                    continue
                valid_boxes.append([xmin, ymin, xmax, ymax])
                valid_masks.append(m)

            if len(valid_boxes) == 0:
                continue
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels = torch.ones((len(valid_boxes),), dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks_tensor,
                "image_id": torch.tensor([idx])
            }

            self.images.append(img_tensor)
            self.targets.append(target)

            suffix = f'processed {idx + 1}/{total}, Broken images: {len(broken_images)}'
            print_progress_bar(idx + 1, total, prefix='Loading', suffix=suffix)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

    def __len__(self):
        return len(self.images)

# Dataset - loading from drive 
class GrayscaleMaskRCNNDatasetDrive(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.broken_images = []
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        try:
            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path)
        except Exception as e:
            self.broken_images.append(self.image_files[idx])
            # dummy matric to just keep the process going
            img = Image.new('L', (640, 640))
            mask = Image.new('L', (640, 640))

        img_tensor = F.to_tensor(img)  # uint8 -> float32 [0,1]
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]  # Exclude background

        masks = mask_np == obj_ids[:, None, None]  # shape: [num_objs, H, W]
        valid_boxes = []
        valid_masks = []
        for m in masks:
            pos = np.where(m)
            if pos[0].size == 0 or pos[1].size == 0:
                continue
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            if xmax <= xmin or ymax <= ymin:
                continue
            valid_boxes.append([xmin, ymin, xmax, ymax])
            valid_masks.append(m)

        if len(valid_boxes) == 0:
            # Return a dummy target if no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels = torch.ones((len(valid_boxes),), dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx])
        }
        return img_tensor, target

# Model setup
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(Wweights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.backbone.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    model.transform = GeneralizedRCNNTransform(
        min_size=640,
        max_size=640,
        image_mean=[0.5],
        image_std=[0.5]
    )

    return model

# Metrics
def compute_iou(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).sum().item()
    union = (pred_mask | true_mask).sum().item()
    return intersection / union if union != 0 else 1.0

def compute_dice(pred_mask, true_mask):
    intersection = (pred_mask & true_mask).sum().item()
    total = pred_mask.sum().item() + true_mask.sum().item()
    return 2 * intersection / total if total != 0 else 1.0

def compute_pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total


# Training loop 
def train_one_epoch_with_progress(model, optimizer, data_loader, device, epoch, logger):
    model.train()
    total = len(data_loader)
    start_epoch = time.time()
    for i, (images, targets) in enumerate(data_loader):
        start_batch = time.time()
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - start_batch
        logger.info(f"Epoch {epoch}, Batch {i+1}/{total}, Loss: {losses.item():.4f}, Time: {batch_time:.2f}s")
        print_progress_bar(i + 1, total, prefix=f"Epoch {epoch}", suffix="Complete")


    epoch_time = time.time() - start_epoch
    logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds.")

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    logger.info(f"Epoch {epoch}, Batch {i+1}/{total}, Loss: {losses.item():.4f}")
    print_progress_bar(i + 1, total, prefix=f"Epoch {epoch}", suffix="Complete")


# Validation with logging
def validate_with_logging(model, data_loader, device, epoch, logger):
    model.eval()
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                if len(output["masks"]) == 0:
                    continue

                pred_mask = output["masks"][0, 0] > 0.5
                true_mask = target["masks"][0] > 0.5

                iou = compute_iou(pred_mask, true_mask)
                dice = compute_dice(pred_mask, true_mask)
                acc = compute_pixel_accuracy(pred_mask, true_mask)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(acc)

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_accuracy = np.mean(pixel_accuracies) if pixel_accuracies else 0.0

    logger.info(f"Epoch {epoch}, Validation - IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}, Accuracy: {mean_accuracy:.4f}")
    return mean_iou, mean_dice, mean_accuracy

# data loader filter for corrupted files
def collate_fn(batch):
    # Remove None or empty targets
    batch = [b for b in batch if b is not None and b[0] is not None]
    return tuple(zip(*batch))


# Main training loop
def run_experiments(train_dataset, val_dataset, device, log_dir, num_epochs=50):
    os.makedirs(log_dir, exist_ok=True)
    hyperparams = {
        "Full_Dataset": {
            "lr": 0.00005,  # Lower learning rate
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "scheduler": "ReduceLROnPlateau",
            "patience": 25,  # Longer patience
            "factor": 0.5
        },
        # "exp4_2": {
        #     "lr": 0.00001,  # Even lower learning rate
        #     "momentum": 0.9,
        #     "weight_decay": 0.0005,
        #     "scheduler": "ReduceLROnPlateau",
        #     "patience": 25,
        #     "factor": 0.5
        # },
        # "exp4_3": {
        #     "lr": 0.0001,
        #     "momentum": 0.95,  # Higher momentum
        #     "weight_decay": 0.001,  # More regularization
        #     "scheduler": "ReduceLROnPlateau",
        #     "patience": 25,
        #     "factor": 0.5
        # },
        # "exp4_4": {
        #     "lr": 0.0005,
        #     "momentum": 0.85,  # Lower momentum
        #     "weight_decay": 0.0001,  # Less regularization
        #     "scheduler": "ReduceLROnPlateau",
        #     "patience": 25,
        #     "factor": 0.5
        # },
        # "exp3_5": {
        #     "lr": 0.00005,
        #     "momentum": 0.9,
        #     "weight_decay": 0.0005,
        #     "scheduler": "StepLR",
        #     "step_size": 25,  # Shorter step size
        #     "gamma": 0.2
        # },
        # "exp3_6": {
        #     "lr": 0.00005,
        #     "momentum": 0.9,
        #     "weight_decay": 0.0005,
        #     "scheduler": "StepLR",
        #     "step_size": 25,  # Longer step size
        #     "gamma": 0.1
        # }
    }

    for exp_name, params in hyperparams.items():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logger = logging.getLogger(exp_name)

        logger.info(f"Starting {exp_name} with params: {params}")
        model = get_model_instance_segmentation(num_classes=2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], 
                                    momentum=params["momentum"], weight_decay=params["weight_decay"])

        # Instantiate scheduler if specified
        scheduler = None
        if "scheduler" in params:
            if params["scheduler"] == "StepLR":
                scheduler = StepLR(optimizer, step_size=params.get("step_size", 10), 
                                   gmma=params.get("gamma", 0.1))
            elif params["scheduler"] == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                              patience=params.get("patience", 3), 
                                              factor=params.get("factor", 0.5))

        train_loader = DataLoader(train_dataset, batch_size=8, num_workers = 10, 
                                  shuffle=False, collate_fn=collate_fn,
                                  pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers = 10, 
                                shuffle=False, collate_fn=collate_fn,
                                pin_memory=True, persistent_workers=True)

        best_iou = 0.0
        patience = params['patience']
        patience_counter = 0
        for epoch in range(1, num_epochs + 1):
            train_one_epoch_with_progress(model, optimizer, train_loader, device, epoch, logger)
            iou, dice, acc = validate_with_logging(model, val_loader, device, epoch, logger)

            # Step the scheduler
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(iou)
                else:
                    scheduler.step()

            if iou > best_iou:
                best_iou = iou
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(log_dir, f"{exp_name}_best_model.pth"))
                logger.info(f"New best model saved with IoU: {best_iou:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break
                


def show_random_samples(dataset, num_samples=4):
    """
    Display random samples from a GrayscaleMaskRCNNDataset instance.
    Each sample shows the grayscale image with all its masks overlaid in different colors.
    All samples are shown together in one plot.
    """
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))

    if num_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]
        img_np = image.squeeze().numpy()
        masks = target['masks'].numpy()
        labels = target['labels'].numpy()

        # Create a color mask
        color_mask = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        num_masks = masks.shape[0]
        colors = plt.cm.get_cmap('hsv', num_masks + 1)

        for i in range(num_masks):
            mask = masks[i]
            color = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
            for c in range(3):
                color_mask[:, :, c] = np.where(mask == 1, color[c], color_mask[:, :, c])

        # Display the image and overlay the color mask
        ax.imshow(img_np, cmap='gray')
        ax.imshow(color_mask, alpha=0.5)
        ax.set_title(f"Sample {idx} with {num_masks} masks")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_image_dir = "/home/pszubert/Dokumenty/05_FullDatasetSplit/images/train"
    train_mask_dir = "/home/pszubert/Dokumenty/05_FullDatasetSplit/MaskRCNN_labels/train"
    val_image_dir = "/home/pszubert/Dokumenty/05_FullDatasetSplit/images/test"
    val_mask_dir = "/home/pszubert/Dokumenty/05_FullDatasetSplit/MaskRCNN_labels/test"
    log_dir = "/mnt/96729E38729E1D55/07_OneDriveBackup/05_PrzetwarzanieDawnychZdjec/03_DataProcessing/13_MaskRcnn_Training"

    if torch.cuda.is_available():
        print("GPU is available for training.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU is not available")
        
    #use ram option for smaller datasets
    train_dataset = GrayscaleMaskRCNNDatasetDrive(train_image_dir, train_mask_dir)
    val_dataset = GrayscaleMaskRCNNDatasetDrive(val_image_dir, val_mask_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiments(train_dataset, val_dataset, device, log_dir, num_epochs=100)


#show_random_samples(val_dataset)

