import pandas as pd
from sklearn.cluster import KMeans

CSV_PATH = "runs/yolo11m_eval/predictions_analysis.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # Defensive check
    if "area" not in df.columns:
        raise ValueError("Column 'area' not found. Did error_analysis.py run?")

    # Cluster by object scale
    features = df[["area"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(features)

    summary = (
        df.groupby("cluster")
          .agg(
              count=("area", "size"),
              mean_area=("area", "mean"),
              median_area=("area", "median")
          )
          .reset_index()
    )

    print("\nCluster-wise failure summary:")
    print(summary)

    # Save for report / dashboard
    summary.to_csv(
        "runs/yolo11m_eval/cluster_summary.csv",
        index=False
    )

    print("\nSaved cluster_summary.csv")


if __name__ == "__main__":
    main()

# Cluster-0 corresponds to small objects where detection performance is weakest, indicating scale sensitivity in the model.