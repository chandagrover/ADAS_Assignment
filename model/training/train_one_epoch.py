from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")  # lightweight for demo

    model.train(
        data="data/bdd100k_yolo.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        project="runs",
        name="bdd100k_yolo11_1epoch",
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
