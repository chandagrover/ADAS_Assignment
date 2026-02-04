# 1. Install ultralytics (2026 version supports YOLOv11 natively)
pip install -U ultralytics

# 2. Train (CLI – easiest to document & reproduce)
yolo task=detect \
     mode=train \
     model=yolo11m.pt \
     data=/home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/model/yolo_converted/bdd100k_ultralytics.yaml \
     epochs=5 \
     imgsz=640 \
     batch=16 \
     workers=8 \
     device=0 \
     project=runs/bdd100k \
     name=yolo11m_finetune \
     patience=15 \
     optimizer=AdamW \
     lr0=0.001 \
     cos_lr=True \
     mosaic=1.0 \
     mixup=0.2 \
     hsv_h=0.015 \
     hsv_s=0.7 \
     hsv_v=0.4 \
     degrees=10.0 \
     translate=0.1 \
     scale=0.5 \
     shear=2.0 \
     perspective=0.0001 \
     flipud=0.5 \
     fliplr=0.5 \
     copy_paste=0.2 \
     auto_augment=randaugment \
     erasing=0.4