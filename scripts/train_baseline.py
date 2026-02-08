"""
Train naive baseline model
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, BASELINE_MODEL_PATH, ASL_CLASSES
from models.baseline import NaiveBaseline


def main():
    """Train and evaluate baseline model"""
    print("=" * 60)
    print("Training Naive Baseline Model")
    print("=" * 60)
    
    # Load metadata
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_metadata.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_metadata.csv')
    
    # Extract labels
    y_train = train_df['class'].values
    y_test = test_df['class'].values
    
    # Train model
    print("\nTraining model (most frequent class strategy)...")
    model = NaiveBaseline(strategy='most_frequent')
    model.fit(y_train)
    
    print(f"Most frequent class: {model.most_frequent_class}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(y_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=ASL_CLASSES, zero_division=0))
    
    # Save model
    print(f"\nSaving model to {BASELINE_MODEL_PATH}")
    model.save(BASELINE_MODEL_PATH)
    
    print("\nBaseline training complete!")


if __name__ == "__main__":
    main()