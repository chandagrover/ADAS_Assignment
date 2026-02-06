import fiftyone as fo
import fiftyone.utils.yolo as fouy
from ultralytics import YOLO

DATASET_NAME = "/app/yolo11m_eval"
YOLO_DATASET_DIR = "/app/yolo_converted"
MODEL_PATH = "/app/yolo11m.pt"

def main():
    # # Load dataset
    # dataset = fouy.load_yolo_dataset(
    #     dataset_dir=YOLO_DATASET_DIR,
    #     split="val",
    #     name=DATASET_NAME
    # )

    # Load YOLO-format dataset (images + labels)
    dataset = fo.Dataset.from_dir(
    dataset_dir=YOLO_DATASET_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    split="val",
    name=DATASET_NAME,
    )

        # Load pre-trained YOLO11m
    model = YOLO(MODEL_PATH)


    # Run inference and store predictions
    model = YOLO(MODEL_PATH)
    dataset.apply_model(
        model,
        label_field="predictions",
        confidence_thresh=0.25
    )

        # Optional: Tag false positives / false negatives
    dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
    )

    # Launch FiftyOne UI
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
