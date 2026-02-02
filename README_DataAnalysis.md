# BDD100K Object Detection – Bosch Assignment

## 1. Data Analysis (Containerized)

All data analysis code (parser, statistics, anomaly detection, visualizations, report generation, interactive dashboard) is fully containerized.

### Folder structure
data_analysis/
├── Dockerfile
├── requirements.txt
└── src/
├── parse_bdd.py
├── analyze.py
└── dashboard.py

### Build the container

```bash
cd data_analysis
docker build -t bdd-analysis .


# Example paths – adjust to your machine
DATA_DIR=$(pwd)/../data_bdd/
ANALYSIS_DIR=$(pwd)/../analysis_output/
ANOMALY_DIR=$(pwd)/../anomaly_output/

sudo docker run --rm -it \
  -v "$DATA_DIR":/data:ro \
  -v "$OUTPUT_DIR":/output \
  bdd-analysis \
  python src/parse_bdd.py \
    --labels-train data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val   /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output /output/processed_objects.parquet

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

