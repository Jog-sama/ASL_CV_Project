"""
Preprocess ASL dataset: split, organize, extract metadata
"""
import sys
from pathlib import Path
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ASL_CLASSES
from scripts.utils import get_skin_tone_estimate


def organize_dataset():
    """
    Organize dataset into train/val/test splits with metadata
    """
    print("=" * 60)
    print("Preprocessing ASL Dataset")
    print("=" * 60)
    
    # Source directory
    train_dir = RAW_DATA_DIR / "asl_alphabet_train" / "asl_alphabet_train"
    
    if not train_dir.exists():
        print(f"Error: {train_dir} not found!")
        print("Run download_data.py first!")
        return
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ASL_CLASSES:
            (PROCESSED_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Collect all image paths with metadata
    print("\nCollecting image metadata...")
    metadata = []
    
    for class_name in tqdm(ASL_CLASSES):
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob("*.jpg"))
        
        for img_path in images:
            # Load image and estimate skin tone
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            skin_tone = get_skin_tone_estimate(img_rgb)
            
            metadata.append({
                'path': str(img_path),
                'class': class_name,
                'skin_tone': skin_tone,
                'brightness': np.mean(img_rgb)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata)
    
    # Stratified split
    print(f"\nTotal images: {len(df)}")
    print(f"Splitting into train/val/test...")
    
    # First split: train+val vs test (80-20)
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df['class'], random_state=42
    )
    
    # Second split: train vs val (80-20 of remaining)
    train, val = train_test_split(
        train_val, test_size=0.2, stratify=train_val['class'], random_state=42
    )
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Copy files to organized structure
    print("\nOrganizing files...")
    
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            src = Path(row['path'])
            dst = PROCESSED_DATA_DIR / split_name / row['class'] / src.name
            shutil.copy(src, dst)
    
    # Save metadata
    train.to_csv(PROCESSED_DATA_DIR / 'train_metadata.csv', index=False)
    val.to_csv(PROCESSED_DATA_DIR / 'val_metadata.csv', index=False)
    test.to_csv(PROCESSED_DATA_DIR / 'test_metadata.csv', index=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    
    print(f"\nSkin tone distribution (0=dark, 1=light):")
    print(f"Mean: {df['skin_tone'].mean():.3f}")
    print(f"Std: {df['skin_tone'].std():.3f}")
    print(f"Min: {df['skin_tone'].min():.3f}")
    print(f"Max: {df['skin_tone'].max():.3f}")
    
    print("\nPreprocessing complete!")
    print(f"Data saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    organize_dataset()