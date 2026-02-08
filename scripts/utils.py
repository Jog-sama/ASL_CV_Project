"""
Utility functions for ASL Fairness Benchmark
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import albumentations as A
from pathlib import Path
import json


def get_skin_tone_estimate(image):
    """
    Estimate skin tone using HSV color space analysis.
    Returns a value between 0-1 (0=dark, 1=light)
    
    Args:
        image: numpy array (H, W, 3) in RGB
    
    Returns:
        float: skin tone estimate
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin pixels
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Get average brightness of skin pixels
    if mask.sum() > 0:
        skin_pixels = image[mask > 0]
        brightness = np.mean(skin_pixels)
        return brightness / 255.0
    else:
        return 0.5  # default


def add_gaussian_noise(image, noise_level=0.1):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_occlusion(image, size=40):
    """Add random square occlusion to image"""
    h, w = image.shape[:2]
    x = np.random.randint(0, max(1, w - size))
    y = np.random.randint(0, max(1, h - size))
    
    occluded = image.copy()
    occluded[y:y+size, x:x+size] = np.random.randint(0, 255, (size, size, 3))
    return occluded


def adjust_brightness(image, factor=1.0):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, test_loader, class_names, device='cpu'):
    """
    Evaluate model and return metrics
    
    Args:
        model: trained model
        test_loader: DataLoader for test data
        class_names: list of class names
        device: device to run on
    
    Returns:
        dict: evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'report': report
    }


def save_experiment_results(results, filepath):
    """Save experiment results to JSON"""
    # Convert numpy types to Python types
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            serializable_results[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count