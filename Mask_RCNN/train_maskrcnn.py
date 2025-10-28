import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import logging
from datetime import datetime
from collections import defaultdict
import time


# Progress bar function 
def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# Dataset class
class GrayscaleMaskRCNNDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.images = []
        self.targets = []

        total = len(self.image_files)
        print("Loading dataset into RAM:")
        for idx in range(total):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path)

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
            masks_tensor = torch.as_tensor(valid_masks, dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks_tensor,
                "image_id": torch.tensor([idx])
            }

            self.images.append(img_tensor)
            self.targets.append(target)

            print_progress_bar(idx + 1, total, prefix='Loading', suffix='Complete')

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

    def __len__(self):
        return len(self.images)

# Model setup
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
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

        percent = f"{100 * (i + 1) / float(total):.1f}"
        filled_length = int(50 * (i + 1) // total)
        bar = '=' * filled_length + '-' * (50 - filled_length)
        print(f"Epoch {epoch} |{bar}| {percent}% Loss: {losses.item():.4f} Time: {batch_time:.2f}s", end='')

    print()
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

# Main training loop
def run_experiments(train_dataset, val_dataset, device, log_dir, num_epochs=50):
    os.makedirs(log_dir, exist_ok=True)
    hyperparams = {
        "exp1": {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
        "exp2": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001},
        "exp3": {"lr": 0.01, "momentum": 0.85, "weight_decay": 0.0005},
        "exp4": {"lr": 0.003, "momentum": 0.95, "weight_decay": 0.001}
    }

    for exp_name, params in hyperparams.items():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        logger = logging.getLogger(exp_name)

        logger.info(f"Starting {exp_name} with params: {params}")
        model = get_model_instance_segmentation(num_classes=2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"], weight_decay=params["weight_decay"])

        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        best_iou = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_one_epoch_with_progress(model, optimizer, train_loader, device, epoch, logger)
            iou, dice, acc = validate_with_logging(model, val_loader, device, epoch, logger)

            if iou > best_iou:
                best_iou = iou
                patience_counter = 0
                torch.save(model.state_dict(), f"{exp_name}_best_model.pth")
                logger.info(f"New best model saved with IoU: {best_iou:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break


if __name__ == "__main__":
    train_image_dir = "/path/to/train/images"
    train_mask_dir = "/path/to/train/masks"
    val_image_dir = "/path/to/val/images"
    val_mask_dir = "/path/to/val/masks"

    if torch.cuda.is_available():
        print("GPU is available for training.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU is not available")
        

    train_dataset = GrayscaleMaskRCNNDataset(train_image_dir, train_mask_dir)
    val_dataset = GrayscaleMaskRCNNDataset(val_image_dir, val_mask_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiments(train_dataset, val_dataset, device)
