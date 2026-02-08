"""
Download and organize ASL Alphabet dataset
Run this first: python scripts/download_data.py
"""
import os
import sys
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def main():
    """
    Download ASL Alphabet dataset from Kaggle
    
    Prerequisites:
    1. Install Kaggle CLI: pip install kaggle
    2. Set up Kaggle API credentials: https://github.com/Kaggle/kaggle-api
    3. Place kaggle.json in ~/.kaggle/
    """
    print("=" * 60)
    print("ASL Alphabet Dataset Download")
    print("=" * 60)
    
    # Check if data already exists
    if (RAW_DATA_DIR / "asl_alphabet_train").exists():
        print("Dataset already downloaded!")
        return
    
    print("\nDownloading via Kaggle API...")
    print("Make sure you have kaggle.json configured!")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials\n")
    
    # Download using Kaggle API
    os.system(f"kaggle datasets download -d grassknoted/asl-alphabet -p {RAW_DATA_DIR}")
    
    # Unzip
    zip_path = RAW_DATA_DIR / "asl-alphabet.zip"
    if zip_path.exists():
        print("\nExtracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        zip_path.unlink()  # Remove zip file
        print("Extraction complete!")
    
    print("\n" + "=" * 60)
    print("Download complete! Dataset ready in:", RAW_DATA_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()