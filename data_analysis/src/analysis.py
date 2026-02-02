"""Main script for BDD100K object detection data analysis.

Performs:
- Train/val split statistics
- Class distribution & imbalance analysis
- Per-class patterns (size, occlusion, truncation)
- Anomaly detection & interesting sample identification
- Visualizations (bar plots, histograms, box plots)
- Simple findings report generation

Run with:
    python data_analysis/src/analysis.py \
    --labels-train /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir ./analysis_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parse_bdd import BDDParser, DETECTION_CLASSES, combine_parsers  # parser

# ────────────────────────────────────────────────
# Configurable constants
# ────────────────────────────────────────────────

OUTPUT_SUBDIRS = {
    "plots": "plots",
    "stats": "stats",
    "interesting": "interesting_samples"  # for listing / copying later
}

# For interesting samples — thresholds (tune as needed)
CROWD_THRESHOLD = 30          # images with many objects
SMALL_AREA_PX_THRESHOLD = 500  # very small objects (e.g. distant signs/lights)
HIGH_OCCLUDED_FRAC = 0.7      # classes with high occlusion rate

# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────

def setup_output_dirs(base_dir: Path) -> dict[str, Path]:
    """Create output subdirectories."""
    dirs = {}
    for name, sub in OUTPUT_SUBDIRS.items():
        p = base_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def savefig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    """Save figure with tight layout and high dpi."""
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_split(df: pd.DataFrame, output_dirs: dict) -> None:
    """Analyze train/val split: image count, object count, class balance."""
    split_stats = df.groupby("split").agg(
        n_images=("image_name", "nunique"),
        n_objects=("category", "count"),
    ).reset_index()

    class_by_split = (
        df.groupby(["split", "category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .melt(id_vars="split", var_name="category", value_name="count")
    )

    # Plot 1: Images & objects per split
    fig, ax = plt.subplots(figsize=(9, 5))
    split_stats.plot(kind="bar", x="split", ax=ax)
    ax.set_title("Train vs Val: Images & Object Instances")
    ax.set_ylabel("Count")
    savefig(fig, "split_overview", output_dirs["plots"])

    # Plot 2: Class distribution by split (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=class_by_split,
        x="category",
        y="count",
        hue="split",
        order=sorted(DETECTION_CLASSES),
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Class Distribution — Train vs Val (log scale)")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, "class_dist_by_split", output_dirs["plots"])

    # Save csv stats
    split_stats.to_csv(output_dirs["stats"] / "split_stats.csv", index=False)
    class_by_split.to_csv(output_dirs["stats"] / "class_by_split.csv", index=False)


def analyze_patterns_and_anomalies(df: pd.DataFrame, output_dirs: dict) -> dict:
    """Per-class statistics + patterns/anomalies."""
    # Basic per-class stats
    class_stats = (
        df.groupby("category")
        .agg(
            n_instances=("category", "count"),
            mean_area_px=("area_px", "mean"),
            median_area_px=("area_px", "median"),
            frac_occluded=("occluded", "mean"),
            frac_truncated=("truncated", "mean"),
            n_images_with_class=("image_name", "nunique"),
        )
        .reset_index()
    )

    # Images per class (how many images contain at least one instance)
    class_stats["frac_images_with_class"] = class_stats["n_images_with_class"] / df["image_name"].nunique()

    class_stats = class_stats.sort_values("n_instances", ascending=False)

    # ── Patterns & Anomalies ───────────────────────────────────────
    findings = {}

    # Extreme imbalance
    car_count = class_stats.loc[class_stats["category"] == "car", "n_instances"].values[0]
    rarest = class_stats.nsmallest(1, "n_instances")
    findings["imbalance"] = (
        f"Severe class imbalance: 'car' has ~{car_count:,} instances, "
        f"while rarest class ('{rarest['category'].values[0]}') has only {rarest['n_instances'].values[0]:,} "
        f"→ ratio > 100:1 in many cases."
    )

    # Small objects challenge
    small_classes = class_stats[class_stats["median_area_px"] < SMALL_AREA_PX_THRESHOLD]
    if not small_classes.empty:
        findings["small_objects"] = (
            f"Small/medium objects dominant in: {', '.join(small_classes['category'])} "
            f"(median area < {SMALL_AREA_PX_THRESHOLD} px²) → distant traffic lights/signs, persons are hard."
        )

    # High occlusion classes
    high_occl = class_stats[class_stats["frac_occluded"] > HIGH_OCCLUDED_FRAC]
    if not high_occl.empty:
        findings["occlusion"] = (
            f"High occlusion (> {HIGH_OCCLUDED_FRAC*100:.0f}% instances occluded): "
            f"{', '.join(high_occl['category'])} → expect failures in crowded scenes."
        )

    # Save stats
    class_stats.to_csv(output_dirs["stats"] / "per_class_stats.csv", index=False)

    # Plot: instance count (log)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=class_stats, x="category", y="n_instances", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Number of Instances per Class (log scale)")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, "class_instance_count", output_dirs["plots"])

    # Plot: boxplot of area per class (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="category", y="area_px", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Bounding Box Area Distribution per Class (log scale)")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, "bbox_area_per_class", output_dirs["plots"])

    # Plot: occlusion rate
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=class_stats, x="category", y="frac_occluded", ax=ax)
    ax.set_title("Fraction of Occluded Instances per Class")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    savefig(fig, "occlusion_rate_per_class", output_dirs["plots"])

    return findings


def find_interesting_samples(df: pd.DataFrame, output_dirs: dict) -> None:
    """Identify and list interesting/unique samples."""
    interesting = []

    # 1. Very crowded images
    img_counts = df.groupby("image_name").size().reset_index(name="n_objects")
    crowded = img_counts[img_counts["n_objects"] >= CROWD_THRESHOLD].sort_values("n_objects", ascending=False)
    if not crowded.empty:
        interesting.append(f"Most crowded images ({CROWD_THRESHOLD}+ objects):")
        for _, row in crowded.head(8).iterrows():
            interesting.append(f"  • {row['image_name']} — {row['n_objects']} objects")

    # 2. Images with very small traffic lights / signs
    small_lights = (
        df[(df["category"] == "traffic light") & (df["area_px"] < SMALL_AREA_PX_THRESHOLD)]
        .groupby("image_name")
        .size()
        .nlargest(6)
    )
    if not small_lights.empty:
        interesting.append("\nImages with many tiny traffic lights:")
        for img, cnt in small_lights.items():
            interesting.append(f"  • {img} — {cnt} tiny lights")

    # Similar for traffic signs, persons, etc. — add if desired

    # Save to text file
    with (output_dirs["stats"] / "interesting_samples.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(interesting))


def generate_report(findings: dict, output_dir: Path) -> None:
    """Generate simple markdown report with key findings."""
    lines = [
        "# BDD100K Object Detection Data Analysis Summary\n",
        "## Train / Val Split",
        "- ~87% train / ~13% val images (typical 70k / 10k split)",
        "- Class distribution roughly preserved across splits (minor sampling variation)\n",
        "## Key Patterns & Anomalies\n",
    ]

    for key, text in findings.items():
        lines.append(f"### {key.replace('_', ' ').title()}")
        lines.append(text + "\n")

    lines.extend([
        "## Recommendations for Modeling",
        "- Address severe **class imbalance** (car >> others) → oversampling rare classes, class-weighted loss, or focal loss.",
        "- Small objects (traffic lights, signs, distant persons) → use models good at multi-scale (e.g. FPN, YOLO with P2/P3 layers).",
        "- High occlusion in persons/riders → consider occlusion-aware augmentations or models robust to partial views.",
        "- Consider weather/time-of-day filtering for stratified sampling.\n",
    ])

    report_path = output_dir / "analysis_summary.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze BDD100K object detection data")
    parser.add_argument("--labels-train", required=True, help="Path to train labels JSON")
    parser.add_argument("--labels-val", required=True, help="Path to val labels JSON")
    parser.add_argument("--output-dir", default="./analysis_output", help="Output directory")
    args = parser.parse_args()

    out_base = Path(args.output_dir).resolve()
    out_dirs = setup_output_dirs(out_base)

    print("Loading and parsing labels...")
    df = combine_parsers(args.labels_train, args.labels_val)

    print(f"Total images: {df['image_name'].nunique():,}")
    print(f"Total objects: {len(df):,}")

    # ── Analyses ─────────────────────────────────────────────────
    analyze_split(df, out_dirs)
    findings = analyze_patterns_and_anomalies(df, out_dirs)
    find_interesting_samples(df, out_dirs)
    generate_report(findings, out_base)

    print(f"\nAnalysis complete. Results saved to: {out_base}")
    print(f"See {out_base}/analysis_summary.md for key findings.")
    print(f"Plots → {out_dirs['plots']}")
    print(f"CSV stats → {out_dirs['stats']}")


if __name__ == "__main__":
    main()