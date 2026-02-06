# BDD100K Object Detection – Bosch Assignment

## 1. Training and  (Containerized)

All data analysis code (parser, statistics, anomaly detection, visualizations, report generation, interactive dashboard) is fully containerized.

### Folder structure

### Build the container

```bash
cd model
docker build -t bdd-eval .



# Example paths – adjust to your machine
DATA_DIR=$(pwd)/../data_bdd/
OUTPUT_DIR=$(pwd)/../

sudo docker run --rm -it \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python evaluate/config.py 
 
mkdir -p evaluate_output
sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python evaluate/convert.py


sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python evaluate/evaluate_yolo11m.py

sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python evaluate/error_analysis.py

sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python evaluate/visualize_fiftyone.py


sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  -e FIFTYONE_DATABASE_DIR=/workspace/fiftyone_db
  bdd-eval \
  python evaluate/visualize_fiftyone.py


sudo docker run --rm -it \
  -p 8501:8501 \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  streamlit Dashboard/app.py --server.port=8501 --server.address=0.0.0.0

sudo docker run --rm -it \
  -v $DATA_DIR:/data:ro \
  -v $OUTPUT_DIR:/workspace \
  bdd-eval \
  python training/train_one_epoch.py

