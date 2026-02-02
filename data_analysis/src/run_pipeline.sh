#!/bin/bash
set -e

echo "🚀 Starting ADAS Pipeline..."

# Activate virtual environment
source /opt/venv/bin/activate

echo "📊 Parsing Data..."
python data_analysis/src/parse_bdd.py

echo "📈 Running Data Analysis..."
python data_analysis/src/analysis.py \
  --labels-train data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --labels-val data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --output-dir analysis_output

echo "🚨 Running Anomaly Analysis..."
python data_analysis/src/anomaly.py \
  --labels-train data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --labels-val data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --output-dir anomaly_output

echo "📊 Launching Streamlit Dashboard..."
streamlit run data_analysis/src/dashboard.py \
  --server.port=8501 \
  --server.address=0.0.0.0
