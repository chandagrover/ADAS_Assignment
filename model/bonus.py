from ultralytics import YOLO

model = YOLO("yolo11m.pt")

# Train 1 epoch on first 5000 images (modify data.yaml train: images/train[:5000])
results = model.train(
    data="/home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/model/yolo_converted/bdd100k_ultralytics.yaml",
    epochs=1,
    batch=16,
    imgsz=640,
    device=0,
    project="runs/bonus",
    name="one_epoch_subset"
)