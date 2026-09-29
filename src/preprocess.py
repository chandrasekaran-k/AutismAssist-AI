"""
Data preprocessing utilities for AutismAssist AI.
"""
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    if "Child_ID" in df.columns:
        df["Child_ID"] = df["Child_ID"].astype(str)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = clean_data(load_data(args.input))
    data.to_csv(args.output, index=False)
    print(f"Saved cleaned data to {args.output}")
