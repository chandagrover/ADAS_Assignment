"""Parser for BDD100K object detection labels (JSON format)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DETECTION_CLASSES = {
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
    # Note: assignment says "light, signs, person, car" → but use the 10 standard ones
}


class BDDParser:
    """Parser for BDD100K detection labels."""

    def __init__(self, json_path: str | Path):
        """
        Initialize parser with path to bdd100k_labels_images_*.json.

        Args:
            json_path: Path to the JSON label file (train or val)
        """
        self.json_path = Path(json_path)
        self.data: list[dict[str, Any]] = self._load_json()
        self.split: str = "train" if "train" in self.json_path.name else "val"

    def _load_json(self) -> list[dict[str, Any]]:
        """Load and return the list of frame dictionaries."""
        with self.json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_image_count(self) -> int:
        """Return number of images in this split."""
        return len(self.data)

    def extract_detection_records(self) -> list[dict]:
        """
        Extract flat list of object records (one per bounding box).

        Returns:
            List of dicts with keys: image_name, category, occluded, truncated,
            area_px, width_px, height_px, x1, y1, x2, y2
        """
        records = []

        for frame in self.data:
            img_name = frame["name"]
            for label in frame.get("labels", []):
                cat = label.get("category")
                if cat not in DETECTION_CLASSES:
                    continue

                box = label.get("box2d", {})
                if not box:
                    continue

                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]
                w = x2 - x1
                h = y2 - y1
                area = w * h

                attr = label.get("attributes", {})

                records.append({
                    "image_name": img_name,
                    "split": self.split,
                    "category": cat,
                    "occluded": attr.get("occluded", False),
                    "truncated": attr.get("truncated", False),
                    "area_px": area,
                    "width_px": w,
                    "height_px": h,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

        return records

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all detection records to a pandas DataFrame."""
        records = self.extract_detection_records()
        return pd.DataFrame(records)


def combine_parsers(train_json: str | Path, val_json: str | Path) -> pd.DataFrame:
    """
    Load both train and val JSONs and combine into one DataFrame.

    Args:
        train_json: path to train labels json
        val_json: path to val labels json

    Returns:
        Combined pandas DataFrame of all object annotations
    """
    parser_train = BDDParser(train_json)
    parser_val = BDDParser(val_json)

    df_train = parser_train.to_dataframe()
    df_val = parser_val.to_dataframe()

    return pd.concat([df_train, df_val], ignore_index=True)


# ────────────────────────────────────────────────
# Quick usage example (can be in analyze.py)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Example paths — adjust to your folder structure
    train_path = "/home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    val_path   = "/home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

    df_all = combine_parsers(train_path, val_path)

    print("Total objects:", len(df_all))
    print("\nClass distribution:\n", df_all["category"].value_counts())

    # Save for later analysis / visualization
    df_all.to_parquet("bdd_objects.parquet", index=False)
    # or df_all.to_csv("bdd_objects.csv", index=False)