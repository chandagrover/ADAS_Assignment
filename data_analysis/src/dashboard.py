"""Streamlit dashboard for BDD100K object detection analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="BDD100K Object Detection Dashboard")
    parser.add_argument("--data", default="/output/processed_objects.parquet",
                        help="Path to the processed Parquet file (inside container)")
    return parser.parse_args()

args = parse_args()

st.set_page_config(page_title="BDD100K Data Explorer", layout="wide")

# @st.cache_data
# def load_data(parquet_path: str):
#     return pd.read_parquet(parquet_path)

@st.cache_data
def load_data(path: str):
    if not Path(path).exists():
        st.error(f"Cannot load data file: {path}\nMake sure you ran the parser/analyze step first and the file exists in the mounted output folder.")
        st.stop()
    return pd.read_parquet(path)

# ── Paths (adjust if needed or use st.file_uploader)
# PARQUET = "bdd_objects.parquet"   # assume you saved it from parser earlier
# PARQUET = "/home/phdcs2/Hard_Disk/Projects/Challenges/Bosch/ADAS_Assignment/anomaly_output/objs/processed_objects.parquet"   # assume you saved it from parser earlier

try:
    df = load_data(args.data)
except Exception:
    st.error(f"Cannot load {PARQUET}. Run parser first and save to parquet.")
    st.stop()

# ── Sidebar
st.sidebar.title("BDD100K Explorer")
view = st.sidebar.radio("View", ["Overview", "Class Distribution", "Box Size & Anomalies", "Interesting Samples"])

st.title("BDD100K Object Detection Dashboard")

# ── Overview
if view == "Overview":
    st.markdown("### Dataset at a glance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", df["image_name"].nunique())
    col2.metric("Total Objects", len(df))
    col3.metric("Classes", len(df["category"].unique()))

    st.markdown("#### Split")
    split_df = df["split"].value_counts().reset_index()
    fig = px.bar(split_df, x="split", y="count", title="Train vs Val Images")
    st.plotly_chart(fig, use_container_width=True)

# ── Class Distribution
elif view == "Class Distribution":
    st.markdown("### Class Distribution")
    counts = df["category"].value_counts().reset_index()
    fig = px.bar(counts, x="category", y="count", log_y=True,
                 title="Object Count per Class (log scale)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Per-split comparison")
    split_class = df.groupby(["split", "category"]).size().reset_index(name="count")
    fig = px.bar(split_class, x="category", y="count", color="split",
                 barmode="group", log_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ── Box Size & Anomalies
elif view == "Box Size & Anomalies":
    st.markdown("### Bounding Box Statistics & Anomalies")
    fig_area = px.box(df, x="category", y="area_px", log_y=True,
                      title="BBox Area per Class (log scale)")
    st.plotly_chart(fig_area, use_container_width=True)

    if "is_anomaly" in df.columns:
        anom_rate = df.groupby("category")["is_anomaly"].mean().reset_index(name="anomaly_rate")
        fig_anom = px.bar(anom_rate, x="category", y="anomaly_rate",
                            title="Anomaly Rate per Class (Isolation Forest)")
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("#### Sample Anomalous Boxes")
        anomalies = df[df["is_anomaly"]].sort_values("area_px").head(12)
        st.dataframe(anomalies[["image_name", "category", "area_px", "aspect_ratio"]])

# ── Interesting Samples
elif view == "Interesting Samples":
    st.markdown("### Interesting / Extreme Samples")
    img_counts = df.groupby("image_name").size().reset_index(name="n_objects")
    crowded = img_counts.nlargest(10, "n_objects")
    st.markdown("**Most crowded images**")
    st.dataframe(crowded)

    small_lights = df[(df["category"] == "traffic light") & (df["area_px"] < 600)]
    small_lights_top = small_lights.groupby("image_name").size().nlargest(8).reset_index(name="count")
    st.markdown("**Images with many tiny traffic lights**")
    st.dataframe(small_lights_top)

st.markdown("---")
st.caption("Dashboard for BDD100K object detection analysis – Bosch Assignment 2026")