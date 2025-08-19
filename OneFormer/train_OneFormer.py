import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import random
import logging

from PIL import Image
from torch.utils.data import Dataset
from transformers import OneFormerConfig, OneFormerForUniversalSegmentation
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from util import print_progress_bar


class OneFormerPanopticDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        # Load annotations
        annotation_path = os.path.join(root_dir, 'OneFormer_labels', split, 'annotations.json')
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.images_info = self.annotations['images']
        self.annotations_info = {ann['image_id']: ann for ann in self.annotations['annotations']}

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.root_dir, 'images', self.split, image_info['file_name'])

        annotation = self.annotations_info[image_id]
        mask_path = os.path.join(self.root_dir, 'OneFormer_labels', self.split, annotation['file_name'])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'segments_info': annotation['segments_info']
        }
    
    
def visualize_dataset_samples(dataset, num_samples=4):
    """
    Display a figure with num_samples rows, each showing an image and its corresponding mask.
    
    Parameters:
    - dataset: OneFormerPanopticDataset object
    - num_samples: number of samples to display (default is 4)
    """
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # 2 columns: image and mask

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask']
        image_id = sample['image_id']

        # Convert mask to color
        mask_np = np.array(mask)
        unique_ids = np.unique(mask_np)
        color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

        for uid in unique_ids:
            if uid == 0:
                continue  # background
            color = np.random.randint(0, 255, size=3)
            color_mask[mask_np == uid] = color

        axs[i, 0].imshow(image.permute(1, 2, 0), cmap='gray')
        axs[i, 0].set_title(f'Image {image_id}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(color_mask)
        axs[i, 1].set_title(f'Mask {image_id}')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


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

# Training function
def train_model(hparams, train_dataset, val_dataset, device):
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    config = OneFormerConfig(
        image_size=640,
        num_labels=1,
        num_queries=100,
        num_transformer_layers=6,
        encoder_layers=6,
        decoder_layers=6,
        hidden_size=256,
        use_pretrained_backbone=False
    )
    model = OneFormerForUniversalSegmentation(config).to(device)

    # prepare model to accept one channel input
    model.model.encoder.patch_embed.proj = torch.nn.Conv2d(
        in_channels=1,  # change from 3 to 1
        out_channels=model.model.encoder.patch_embed.proj.out_channels,
        kernel_size=model.model.encoder.patch_embed.proj.kernel_size,
        stride=model.model.encoder.patch_embed.proj.stride,
        padding=model.model.encoder.patch_embed.proj.padding
    )

    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f"best_model_lr{hparams['lr']}_bs{hparams['batch_size']}.pt"

    for epoch in range(hparams['epochs']):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        iou_scores, dice_scores, acc_scores = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                iou_scores.append(compute_iou(preds[0], masks[0]))
                dice_scores.append(compute_dice(preds[0], masks[0]))
                acc_scores.append(compute_pixel_accuracy(preds[0], masks[0]))

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.mean(iou_scores)
        avg_dice = np.mean(dice_scores)
        avg_acc = np.mean(acc_scores)

        scheduler.step(avg_val_loss)

        suffix = f"Epoch {epoch+1}/{hparams['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, Acc: {avg_acc:.4f}"

        print_progress_bar(epoch + 1, hparams['epochs'], prefix='Training Progress:', suffix=suffix, length=50)
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, Accuracy: {avg_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hparams['early_stopping']:
                logging.info("Early stopping triggered.")
                break

    return best_model_path

# Hyperparameter search
def run_hpo(param_grid, train_dataset, val_dataset, project_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for hparams in ParameterGrid(param_grid):
        # Logging setup
        print(f"Running experiment with params: {hparams}")
        log_file = os.path.join(project_path, f"training_log_lr{hparams['lr']}_bs{hparams['batch_size']}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info(f"Starting experiment with params: {hparams}")
        best_model = train_model(hparams, train_dataset, val_dataset, device)
        logging.info(f"Best model saved to: {best_model}")


if __name__ == "__main__":
    project_path = 'path_to_project'

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("GPU is available for training.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU is not available. Training will use CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    train_dataset = OneFormerPanopticDataset(root_dir='path_to_data', split='test', transforms=transform)
    val_dataset = OneFormerPanopticDataset(root_dir='path_to_data', split='val', transforms=transform)

    param_grid = {
        'lr': [1e-3, 1e-4],                     # Learning rate
        'batch_size': [2, 4],                   # Batch size
        'epochs': [10, 20],                     # Number of epochs
        'early_stopping': [3, 5],               # Early stopping patience
        'optimizer': ['Adam', 'SGD'],           # Optimizer type
        'hidden_size': [128, 256, 512],         # Model hidden size
        'num_transformer_layers': [4, 6, 8]     # Transformer depth
    }

    run_hpo(param_grid, train_dataset, val_dataset, project_path='path_to_project')






