import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
CLASS_TO_ID = {cls: i + 1 for i, cls in enumerate(CLASSES)}  # 0 is background


class BDD100KDetectionDataset(Dataset):
    """
    PyTorch Dataset for BDD100K object detection.
    """

    def __init__(self, image_dir: Path, label_json: Path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(label_json, "r") as f:
            self.annotations = json.load(f)

        # Keep only images with valid detection labels
        self.annotations = [
            ann for ann in self.annotations if "labels" in ann
        ]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]

        # Load image
        img_path = self.image_dir / ann["name"]
        image = Image.open(img_path).convert("RGB")

        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in ann["labels"]:
            if obj["category"] not in CLASS_TO_ID:
                continue

            box = obj["box2d"]
            boxes.append([
                box["x1"], box["y1"],
                box["x2"], box["y2"]
            ])
            labels.append(CLASS_TO_ID[obj["category"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
