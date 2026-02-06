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

# Example – adjust paths to match your host machine
sudo docker run --rm \
  -v $DATA_DIR:/data:ro \
  -v $ANALYSIS_DIR:/analysis_output \
  bdd-analysis \
  python src/analysis.py \
    --labels-train /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val   /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir   /analysis_output

sudo docker run --rm \
  -v $DATA_DIR:/data:ro \
  -v $ANOMALY_DIR:/anomaly_output \
  bdd-analysis \
    python src/anomaly.py \
    --labels-train /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir /anomaly_output

sudo docker run --rm -it -p 8501:8501 \
  -v $(pwd)/../anomaly_output/objs:/objs \
  bdd-analysis \
  streamlit run src/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    -- \
    --data /objs/processed_objects.parquet

