import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

# Dataset class
class GrayscaleMaskRCNNDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("L")  # Grayscale
        mask = Image.open(mask_path)

        img = F.to_tensor(img)  # [1, H, W]
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        masks = mask == obj_ids[:, None, None]
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.image_files)

# Modify Mask R-CNN to accept 1-channel input
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model

# Training loop
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Loss: {losses.item():.4f}")

# Metrics functions
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

def validate(model, data_loader, device):
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

    print(f"Validation - IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}, Accuracy: {mean_accuracy:.4f}")
    return mean_iou, mean_dice, mean_accuracy

# Ray Tune training function
def train_maskrcnn(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(10):
        train_one_epoch(model, optimizer, train_loader, device)
        iou, dice, acc = validate(model, val_loader, device)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, path)

        tune.report(iou=iou, dice=dice, accuracy=acc)

if __name__ == "__main__":
    train_image_dir = "s"
    train_mask_dir = ""
    test_image_dir = ""
    test_mask_dir_= ""
    NUM_EPOCHS = 0
    

    train_dataset = GrayscaleMaskRCNNDataset(train_image_dir, train_mask_dir)
    val_dataset = GrayscaleMaskRCNNDataset(train_image_dir, train_mask_dir)

    search_space = {
        "lr": tune.uniform(0.001, 0.01),
        "momentum": tune.uniform(0.8, 0.95),
        "weight_decay": tune.uniform(0.0001, 0.001)
    }

    scheduler = ASHAScheduler(
        metric="iou",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    bayesopt = BayesOptSearch(metric="iou", mode="max")

    tuner = tune.Tuner(
        train_maskrcnn,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=bayesopt,
            scheduler=scheduler,
            num_samples=10
        ),
        run_config=air.RunConfig(
            name="maskrcnn_bayesopt",
            stop={"iou": 0.95},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="iou",
                checkpoint_score_order="max",
                num_to_keep=1
            )
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="iou", mode="max")
    best_checkpoint = best_result.checkpoint
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load(os.path.join(best_checkpoint.path, "checkpoint.pt"))["model_state_dict"])
    print("Best model restored.")

    

