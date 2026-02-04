from ultralytics import YOLO
from pathlib import Path
import json

# Paths
MODEL_PATH = "yolo26n.pt"
DATA_YAML = "yolo_converted/bdd100k_ultralytics.yaml"
PROJECT_DIR = "runs/"
NAME = "yolo11m_eval"

def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        device=0,
        project=PROJECT_DIR,
        name=NAME,
        save_json=True,
        plots=True
    )

    # Save summary metrics
    results = {
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "per_class_AP": metrics.box.maps.tolist()
    }

    output_path = Path(PROJECT_DIR) / NAME / "metrics_summary.json"
    print("output path=", output_path)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
