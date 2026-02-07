import streamlit as st
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("BDD100K – YOLO11m Evaluation Dashboard")

RUN_DIR = Path("/workspace/outputs/evaluate_output/runs/yolo11m_eval")

# ---- Metrics ----
st.header("Quantitative Metrics")
with open(RUN_DIR / "metrics_summary.json") as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("mAP@0.5", round(metrics["mAP50"], 3))
col2.metric("mAP@0.5:0.95", round(metrics["mAP50_95"], 3))
col3.metric("Precision", round(metrics["precision"], 3))
col4.metric("Recall", round(metrics["recall"], 3))

# ---- Per-class AP ----
st.header("Per-Class Average Precision")
df_ap = pd.DataFrame({
    "class_id": range(len(metrics["per_class_AP"])),
    "AP": metrics["per_class_AP"]
})

fig, ax = plt.subplots()
ax.bar(df_ap["class_id"], df_ap["AP"])
ax.set_xlabel("Class ID")
ax.set_ylabel("AP")
st.pyplot(fig)

# ---- Failure Analysis ----
st.header("Failure Analysis (Small Objects)")
df_preds = pd.read_csv(RUN_DIR / "predictions_analysis.csv")
df_preds["area"] = df_preds["bbox"].apply(
    lambda b: eval(b)[2] * eval(b)[3]
)

small = df_preds[df_preds["area"] < 32 * 32]
st.write("Small object prediction counts:")
st.dataframe(small["category_id"].value_counts())
