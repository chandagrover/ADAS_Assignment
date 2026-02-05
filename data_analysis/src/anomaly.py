"""BDD100K Object Detection Analysis – concise version with anomaly detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from parse_bdd import BDDParser, DETECTION_CLASSES, combine_parsers

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────

SMALL_AREA_PX = 400
CROWD_THRESHOLD = 35

OUTPUT_SUBDIRS = {"plots": "plots", "stats": "stats", "objs": "objs"}


"""Run with:
    python data_analysis/src/anomaly.py \
    --labels-train /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir ./anomaly_output
"""

def setup_dirs(base: Path) -> dict:
    dirs = {}
    for k, v in OUTPUT_SUBDIRS.items():
        p = base / v
        p.mkdir(parents=True, exist_ok=True)
        dirs[k] = p
    return dirs


def savefig(fig, name, out_dir):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def detect_anomalies(df: pd.DataFrame, out_dirs: dict) -> pd.DataFrame:
    """Isolation Forest on bounding box features + occlusion/truncation."""
    # Add derived features directly to df
    df = df.copy()  # safety (optional)
    df["aspect_ratio"] = df["width_px"] / (df["height_px"] + 1e-6)
    features = df[["area_px", "width_px", "height_px", "occluded", "truncated"]].copy()
    features["aspect_ratio"] = features["width_px"] / (features["height_px"] + 1e-6)
    features["occluded"] = features["occluded"].astype(int)
    features["truncated"] = features["truncated"].astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    iso = IsolationForest(contamination=0.015, random_state=42, n_estimators=120)
    df["anomaly_score"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    df["is_anomaly"] = df["anomaly_score"] == -1
    df.to_parquet(out_dirs['objs']/"processed_objects.parquet", index=False)
    print(f"objs → {out_dirs['objs']}")
    

    return df

def analyze_and_visualize(df: pd.DataFrame, out_dirs: dict) -> dict:
    findings = {}

    # Split stats
    split_stats = df.groupby("split").agg(
        images=("image_name", "nunique"),
        objects=("category", "count")
    )
    findings["split"] = f"Images: train {split_stats.loc['train','images']:,} | val {split_stats.loc['val','images']:,}"

    # Class distribution
    class_counts = df["category"].value_counts()
    findings["imbalance"] = f"Severe imbalance: car {class_counts.get('car',0):,} | rarest {class_counts.idxmin()} {class_counts.min():,}"

    # Per-class stats
    class_stats = df.groupby("category").agg(
        count=("category", "size"),
        med_area=("area_px", "median"),
        occ_rate=("occluded", "mean"),
        trunc_rate=("truncated", "mean")
    ).sort_values("count", ascending=False)

    # Small objects
    small = class_stats[class_stats["med_area"] < SMALL_AREA_PX].index.tolist()
    if small:
        findings["small"] = f"Small objects (med area < {SMALL_AREA_PX}px²): {', '.join(small)}"

    # High occlusion
    high_occ = class_stats[class_stats["occ_rate"] > 0.6].index.tolist()
    if high_occ:
        findings["occlusion"] = f"High occlusion (>60%): {', '.join(high_occ)}"

    # Anomaly detection summary
    anomalies = df[df["is_anomaly"]]
    if not anomalies.empty:
        top_anom_classes = anomalies["category"].value_counts().head(4)
        findings["anomalies"] = (
            f"Isolation Forest found {len(anomalies):,} potential outliers. "
            f"Most affected: {', '.join(f'{c} ({n})' for c,n in top_anom_classes.items())}"
        )

    # Plots ───────────────────────────────────────
    # 1. Class dist (log)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title("Object Count per Class (log)")
    savefig(fig, "class_distribution", out_dirs["plots"])

    # 2. Area boxplot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=df, x="category", y="area_px", ax=ax)
    ax.set_yscale("log")
    ax.set_title("BBox Area per Class (log)")
    savefig(fig, "area_per_class", out_dirs["plots"])

    # 3. Anomalies scatter (area vs aspect)
    if not anomalies.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=df, x="aspect_ratio", y="area_px", hue="is_anomaly",
                        alpha=0.6, size="is_anomaly", sizes=(20, 120), ax=ax)
        ax.set_yscale("log")
        ax.set_title("Anomaly Detection (Isolation Forest) – area vs aspect ratio")
        savefig(fig, "anomalies_scatter", out_dirs["plots"])

    class_stats.to_csv(out_dirs["stats"] / "class_stats.csv")
    anomalies.to_csv(out_dirs["stats"] / "anomalies.csv", index=False)

    return findings


def generate_concise_report(findings: dict, base_dir: Path):
    lines = [
        "# BDD100K Object Detection – Data Analysis Summary\n",
        "## Key Statistics",
        f"- {findings.get('split', 'N/A')}",
        f"- {findings.get('imbalance', 'N/A')}\n",
        "## Main Patterns & Anomalies",
    ]

    for k, v in findings.items():
        if k not in ["split", "imbalance"]:
            lines.append(f"- **{k.title()}**: {v}")

    lines.extend([
        "\n## Modeling Recommendations",
        "- Handle imbalance: class weights / oversampling rare classes",
        "- Focus on small objects: multi-scale features / augmentations",
        "- Occlusion handling: attention mechanisms or occlusion-aware loss",
        "- Anomalies → inspect extremes (tiny/dense/occluded boxes) for labeling issues",
    ])

    (base_dir / "analysis_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-train", required=True)
    parser.add_argument("--labels-val", required=True)
    parser.add_argument("--output-dir", default="./anomaly_output")
    args = parser.parse_args()

    out_base = Path(args.output_dir).resolve()
    out_dirs = setup_dirs(out_base)

    df = combine_parsers(args.labels_train, args.labels_val)

    df = detect_anomalies(df, out_dirs)               # ← added

    findings = analyze_and_visualize(df, out_dirs)

    generate_concise_report(findings, out_base)

    print(f"Done. Summary → {out_base}/analysis_summary.md")
    print(f"Plots → {out_dirs['plots']}")
    print(f"Stats → {out_dirs['stats']}")


if __name__ == "__main__":
    main()