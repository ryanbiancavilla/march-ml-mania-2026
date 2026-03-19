"""Load and inspect all competition datasets."""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_all():
    """Load every CSV in data/ into a dict of DataFrames keyed by filename (no extension)."""
    datasets = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            name = f.replace(".csv", "")
            datasets[name] = pd.read_csv(os.path.join(DATA_DIR, f))
    return datasets


def summarize(datasets: dict[str, pd.DataFrame]):
    """Print shape and columns for each dataset."""
    for name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"{name}  —  {df.shape[0]:,} rows x {df.shape[1]} cols")
        print(f"Columns: {', '.join(df.columns)}")
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No data found. Run download_data.py first.")
    else:
        summarize(data)
