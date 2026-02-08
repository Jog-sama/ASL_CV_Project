"""
Train classical ML model (MediaPipe + Random Forest)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, CLASSICAL_MODEL_PATH, ASL_CLASSES
from models.classical import ClassicalASLModel


def load_images_from_paths(image_paths, max_samples=None):
    """Load images from file paths"""
    if max_samples:
        image_paths = image_paths[:max_samples]
    
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    
    return images


def main():
    """Train and evaluate classical model"""
    print("=" * 60)
    print("Training Classical ML Model (MediaPipe + Random Forest)")
    print("=" * 60)
    
    # Load metadata
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_metadata.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_metadata.csv')
    
    # For faster training, sample subset (remove this for full training)
    SAMPLE_SIZE = 5000  # Use all data if you have time
    
    print(f"\nUsing {SAMPLE_SIZE} training samples for faster training...")
    train_sample = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
    
    # Load images
    print("\nLoading training images...")
    X_train = load_images_from_paths(train_sample['path'].tolist())
    y_train = train_sample['class'].values
    
    print("\nLoading test images...")
    X_test = load_images_from_paths(test_df['path'].tolist()[:1000])  # Sample 1000 for testing
    y_test = test_df['class'].values[:1000]
    
    # Train model
    print("\nInitializing model...")
    model = ClassicalASLModel(n_estimators=100)
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report (Top 10 classes):")
    report = classification_report(y_test, y_pred, target_names=ASL_CLASSES, zero_division=0)
    print('\n'.join(report.split('\n')[:15]))  # Print first 15 lines
    
    # Save model
    print(f"\nSaving model to {CLASSICAL_MODEL_PATH}")
    model.save(CLASSICAL_MODEL_PATH)
    
    print("\nClassical model training complete!")


if __name__ == "__main__":
    main()