import json
import pandas as pd
from pathlib import Path

PRED_JSON = "runs/yolo11m_eval/predictions.json"

def load_predictions(pred_file):
    with open(pred_file, "r") as f:
        preds = json.load(f)
    return pd.DataFrame(preds)

def analyze_box_scale(df):
    df["area"] = df["bbox"].apply(lambda b: b[2] * b[3])
    return df

def main():
    df = load_predictions(PRED_JSON)
    df = analyze_box_scale(df)

    print("Predictions by category:")
    print(df["category_id"].value_counts())

    print("\nSmall object stats (area < 32^2):")
    small = df[df["area"] < 32 * 32]
    print(small["category_id"].value_counts())

    df.to_csv("runs/yolo11m_eval/predictions_analysis.csv", index=False)
    print("Saved predictions_analysis.csv")


if __name__ == "__main__":
    main()
