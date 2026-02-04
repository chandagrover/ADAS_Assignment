# ADAS_Assignment

### 1. Data Analysis
#### Parsing Data
python data_analysis/src/parse_bdd.py

#### Analyze Data
    python data_analysis/src/analysis.py \
    --labels-train /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir ./analysis_output

#### Analyze Anomaly
    python data_analysis/src/anomaly.py \
    --labels-train /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --labels-val /home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-dir ./anomaly_output

#### Dashboard    
streamlit run data_analysis/src/dashboard.py




