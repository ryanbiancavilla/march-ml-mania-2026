"""Download all competition data from Kaggle for March ML Mania 2026."""

import os
import zipfile

import requests

COMPETITION = "march-machine-learning-mania-2026"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KAGGLE_TOKEN = os.environ["KAGGLE_TOKEN"]  # set via: export KAGGLE_TOKEN=KGAT_your_token
HEADERS = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}


def get_file_list():
    """Get list of all data files in the competition."""
    url = f"https://www.kaggle.com/api/v1/competitions/data/list/{COMPETITION}"
    files = []
    page_token = None
    while True:
        params = {}
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageTokenNullable")
        if not page_token:
            break
    return files


def download_file(filename):
    """Download a single competition file."""
    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{filename}"
    resp = requests.get(url, headers=HEADERS, stream=True)
    resp.raise_for_status()

    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return filepath


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching file list for '{COMPETITION}'...")
    files = get_file_list()
    print(f"Found {len(files)} files.\n")

    for i, file_info in enumerate(files, 1):
        name = file_info["name"]
        size_mb = file_info.get("totalBytes", 0) / 1e6
        print(f"  [{i}/{len(files)}] {name} ({size_mb:.1f} MB)...", end=" ", flush=True)
        filepath = download_file(name)

        # If it's a zip, extract it
        if filepath.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(DATA_DIR)
            os.remove(filepath)
            print("extracted.")
        else:
            print("done.")

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"\nTotal CSV files ready: {len(csv_files)}")
    for f in csv_files:
        print(f"  {f}")


if __name__ == "__main__":
    download()
