import os
import json
import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from transformers import (
    OneFormerProcessor,
    OneFormerForUniversalSegmentation
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import print_progress_bar

# ---------------- Dataset ----------------
class BuildingPanopticDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=640):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        ann_path = os.path.join(root_dir, "OneFormer_labels", split, "annotations.json")
        with open(ann_path, "r") as f:
            meta = json.load(f)

        self.images_info = meta["images"]
        self.annotations = {ann["image_id"]: ann for ann in meta["annotations"]}
        self.resize = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]
        image_id = info["id"]
        img_path = os.path.join(self.root_dir, "images", self.split, info["file_name"])
        pan_ann = self.annotations[image_id]
        pan_path = os.path.join(self.root_dir, "OneFormer_labels", self.split, pan_ann["file_name"])

        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("L")
            img = to_pil_image(np.repeat(np.array(img)[..., None], 3, axis=2))
        img = self.resize(img)

        pan_map = Image.open(pan_path)
        pan_map = self.resize(pan_map)
        pan_map_np = np.array(pan_map).astype(np.int32)

        segments_info = pan_ann["segments_info"]

        return {
            "image": img,
            "segmentation_map": pan_map_np,
            "segments_info": segments_info,
            "image_id": image_id
        }

def collate_panoptic(batch):
    images = [b["image"] for b in batch]
    seg_maps = [b["segmentation_map"] for b in batch]
    seg_info = [b["segments_info"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    return images, seg_maps, seg_info, image_ids

# ---------------- Model & Processor ----------------
def build_model_and_processor(backbone_ckpt="shi-labs/oneformer_ade20k_swin_tiny"):
    processor = OneFormerProcessor.from_pretrained(backbone_ckpt)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        backbone_ckpt,
        num_labels=1,
        id2label={0: "building"},
        label2id={"building": 0},
        ignore_mismatched_sizes=True
    )
    return model, processor

# ---------------- Training & Validation ----------------
def train_one_epoch(model, processor, loader, device, optimizer, scaler, task_token="panoptic"):
    model.train()
    total_loss = 0.0
    for images, seg_maps, seg_info, _ in loader:
        inputs = processor(
            images=images,
            task_inputs=[task_token] * len(images),
            segmentation_maps=seg_maps,
            segments_info=seg_info,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(**inputs)
            loss = out.loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))

# metrics
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


@torch.no_grad()
def validate_instance(model, processor, loader, device, image_size=640):
    model.eval()
    iou_scores, dice_scores, pixel_acc_scores = [], [], []
    for images, seg_maps, seg_info, _ in loader:
        inputs = processor(
            images=images,
            task_inputs=["instance"] * len(images),
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([[image_size, image_size]] * len(images), device=device)
        processed = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)

        gt_bin = (torch.from_numpy(seg_maps[0]) > 0).to(torch.int32)
        pred_bin = (processed[0]["segments"].sum(dim=0) > 0).to(torch.int32)

        iou = compute_iou(pred_bin, gt_bin)
        dice = compute_dice(pred_bin, gt_bin)
        pixel_acc = compute_pixel_accuracy(pred_bin, gt_bin)

        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_acc_scores.append(pixel_acc)


    return float(np.mean(iou_scores)), float(np.mean(dice_scores)), float(np.mean(pixel_acc_scores))

# ---------------- Main Training Function ----------------
def train_model(hparams, train_dataset, val_dataset, device, log_path="training_log.txt"):
    model, processor = build_model_and_processor()
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collate_panoptic)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_panoptic)

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"]) if hparams["optimizer"].lower() == "adam" else                 optim.SGD(model.parameters(), lr=hparams["lr"], momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler() if hparams.get("use_amp", False) else None

    best_val = float("inf")
    patience = 0
    best_path = "best_model.pt"

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("Resumed from last checkpoint.")

    with open(log_path, "a") as log_file:
        for epoch in range(hparams["epochs"]):
            train_loss = train_one_epoch(model, processor, train_loader, device, optimizer, scaler)
            val_loss = 0.0
            model.eval()
            for images, seg_maps, seg_info, _ in val_loader:
                val_inputs = processor(
                    images=images,
                    task_inputs=["panoptic"] * len(images),
                    segmentation_maps=seg_maps,
                    segments_info=seg_info,
                    return_tensors="pt"
                )
                val_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in val_inputs.items()}
                out = model(**val_inputs)
                val_loss += float(out.loss.detach().cpu())
            val_loss /= max(1, len(val_loader))
            iou, dice, pixel_acc = validate_instance(model, processor, val_loader, device)
            log_line = f"Epoch {epoch+1}/{hparams['epochs']} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} IoU={iou:.4f} Dice={dice:.4f} PixelAcc={pixel_acc:.4f}\n"

            scheduler.step(val_loss)

            suffix = f"Val loss: {val_loss:.4f} | IoU: {iou:.4f} | Dice: {dice:.4f} | PixelAcc: {pixel_acc:.4f}"
            print_progress_bar(epoch + 1, hparams['epochs'], prefix='Training Progress:', suffix=suffix, length=50)
            log_file.write(log_line)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= hparams["early_stopping"]:
                    print("Early stopping.")
                    break
    return best_path

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("GPU is available for training.")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU is not available. Training will use CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "./"  # Change to your dataset root
    train_dataset = BuildingPanopticDataset(root_dir, split="train")
    val_dataset = BuildingPanopticDataset(root_dir, split="val")

    hparams = {
        "lr": 1e-4,
        "batch_size": 2,
        "optimizer": "adam",
        "epochs": 20,
        "early_stopping": 3,
        "use_amp": True
    }

    train_model(hparams, train_dataset, val_dataset, device)


