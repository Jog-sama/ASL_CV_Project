"""
Run adversarial robustness experiments
Tests model performance under various perturbations INCLUDING SKIN TONE FAIRNESS
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, DEEP_MODEL_PATH, OUTPUT_DIR,
    ASL_CLASSES, DEVICE, BATCH_SIZE,
    NOISE_LEVELS, OCCLUSION_SIZES, BRIGHTNESS_FACTORS
)
from models.deep_learning import ASLDataset, ASLEfficientNet, get_transforms
from scripts.utils import (
    add_gaussian_noise, add_occlusion, adjust_brightness,
    plot_confusion_matrix, save_experiment_results
)


def evaluate_with_perturbation(model, dataset, device, perturbation_fn, param):
    """
    Evaluate model with specific perturbation
    
    Args:
        model: trained model
        dataset: test dataset
        device: device to run on
        perturbation_fn: function to apply perturbation
        param: perturbation parameter
    
    Returns:
        accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    transform = get_transforms(augment=False)
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Param {param}", leave=False):
            img_path, label = dataset.image_paths[i], dataset.labels[i]
            
            # Load and apply perturbation
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if perturbation_fn:
                img_rgb = perturbation_fn(img_rgb, param)
            
            # Transform and predict
            from PIL import Image
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            
            total += 1
            correct += (predicted.item() == label)
    
    return 100 * correct / total


def run_noise_experiment(model, test_dataset, device):
    """Test robustness to Gaussian noise"""
    print("\n" + "="*60)
    print("Experiment 1: Gaussian Noise Robustness")
    print("="*60)
    
    results = {'noise_levels': [], 'accuracies': []}
    
    for noise_level in NOISE_LEVELS:
        print(f"\nTesting with noise level: {noise_level}")
        acc = evaluate_with_perturbation(
            model, test_dataset, device,
            add_gaussian_noise if noise_level > 0 else None, noise_level
        )
        
        results['noise_levels'].append(noise_level)
        results['accuracies'].append(acc)
        print(f"Accuracy: {acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['noise_levels'], results['accuracies'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (σ)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Model Robustness to Gaussian Noise', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(results['accuracies'])-5, 100])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'noise_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def run_occlusion_experiment(model, test_dataset, device):
    """Test robustness to occlusions"""
    print("\n" + "="*60)
    print("Experiment 2: Occlusion Robustness")
    print("="*60)
    
    results = {'occlusion_sizes': [], 'accuracies': []}
    
    for size in OCCLUSION_SIZES:
        print(f"\nTesting with occlusion size: {size}px")
        acc = evaluate_with_perturbation(
            model, test_dataset, device,
            add_occlusion if size > 0 else None, size
        )
        
        results['occlusion_sizes'].append(size)
        results['accuracies'].append(acc)
        print(f"Accuracy: {acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['occlusion_sizes'], results['accuracies'], marker='s', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Occlusion Size (pixels)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Model Robustness to Hand Occlusion', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(results['accuracies'])-5, 100])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'occlusion_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def run_brightness_experiment(model, test_dataset, device):
    """Test robustness to brightness changes"""
    print("\n" + "="*60)
    print("Experiment 3: Lighting Condition Robustness")
    print("="*60)
    
    results = {'brightness_factors': [], 'accuracies': []}
    
    for factor in BRIGHTNESS_FACTORS:
        print(f"\nTesting with brightness factor: {factor}")
        acc = evaluate_with_perturbation(
            model, test_dataset, device,
            adjust_brightness, factor
        )
        
        results['brightness_factors'].append(factor)
        results['accuracies'].append(acc)
        print(f"Accuracy: {acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['brightness_factors'], results['accuracies'], marker='^', linewidth=2, markersize=8, color='green')
    plt.xlabel('Brightness Factor', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Model Robustness to Lighting Conditions', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Normal brightness')
    plt.ylim([min(results['accuracies'])-5, 100])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'brightness_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def run_skin_tone_experiment(model, test_dataset, test_df, device):
    """Test fairness across skin tones - THE KEY EXPERIMENT"""
    print("\n" + "="*60)
    print("🔍 Experiment 4: SKIN TONE FAIRNESS ANALYSIS")
    print("="*60)
    
    model.eval()
    
    # Group by skin tone quartiles
    skin_tones = test_df['skin_tone'].values
    quartiles = np.percentile(skin_tones, [25, 50, 75])
    
    groups = {
        'Dark (Q1)': skin_tones <= quartiles[0],
        'Medium-Dark (Q2)': (skin_tones > quartiles[0]) & (skin_tones <= quartiles[1]),
        'Medium-Light (Q3)': (skin_tones > quartiles[1]) & (skin_tones <= quartiles[2]),
        'Light (Q4)': skin_tones > quartiles[2]
    }
    
    results = {'groups': [], 'accuracies': [], 'counts': []}
    transform = get_transforms(augment=False)
    
    for group_name, mask in groups.items():
        indices = np.where(mask)[0]
        correct = 0
        total = 0
        
        print(f"\n{group_name}: {len(indices)} samples")
        
        with torch.no_grad():
            for idx in tqdm(indices, desc=group_name, leave=False):
                img_path = test_dataset.image_paths[idx]
                label = test_dataset.labels[idx]
                
                # Load and predict
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                from PIL import Image
                img_pil = Image.fromarray(img_rgb)
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                
                total += 1
                correct += (predicted.item() == label)
        
        accuracy = 100 * correct / total
        results['groups'].append(group_name)
        results['accuracies'].append(accuracy)
        results['counts'].append(total)
        
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Calculate disparity
    max_acc = max(results['accuracies'])
    min_acc = min(results['accuracies'])
    disparity = max_acc - min_acc
    
    print(f"\n📊 FAIRNESS METRICS:")
    print(f"├─ Max Accuracy: {max_acc:.2f}%")
    print(f"├─ Min Accuracy: {min_acc:.2f}%")
    print(f"└─ Disparity: {disparity:.2f}%")
    
    if disparity < 2.0:
        print("✅ EXCELLENT: Model shows minimal bias across skin tones")
    elif disparity < 5.0:
        print("⚠️  CAUTION: Model shows moderate disparity")
    else:
        print("❌ CONCERN: Model shows significant bias")
    
    # Plot results
    plt.figure(figsize=(12, 7))
    colors = ['#8B4513', '#D2691E', '#F4A460', '#FFE4C4']
    bars = plt.bar(results['groups'], results['accuracies'], color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Skin Tone Group', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('🎯 Model Fairness Across Skin Tones\n(Lower disparity = More fair)', 
              fontsize=16, fontweight='bold')
    plt.ylim([min(results['accuracies'])-5, 100])
    
    # Add value labels on bars
    for bar, acc, count in zip(bars, results['accuracies'], results['counts']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Add disparity annotation
    plt.text(0.98, 0.98, f'Disparity: {disparity:.2f}%', 
             transform=plt.gca().transAxes,
             fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             ha='right', va='top')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'skin_tone_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add disparity to results
    results['disparity'] = disparity
    
    return results


def main():
    """Run all experiments including FAIRNESS analysis"""
    print("=" * 60)
    print("🔬 Running Complete Fairness Benchmark Experiments")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_metadata.csv')
    
    # Sample subset for faster experiments
    SAMPLE_SIZE = 1000
    test_sample = test_df.sample(n=min(SAMPLE_SIZE, len(test_df)), random_state=42)
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(ASL_CLASSES)}
    test_labels = [label_to_idx[label] for label in test_sample['class']]
    
    # Create dataset
    test_dataset = ASLDataset(
        test_sample['path'].tolist(),
        test_labels,
        transform=None
    )
    
    # Load trained model
    print(f"\nLoading model from {DEEP_MODEL_PATH}")
    checkpoint = torch.load(DEEP_MODEL_PATH, map_location=DEVICE)
    
    model = ASLEfficientNet(num_classes=len(ASL_CLASSES), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    try:
        print(f"Model performance: Val Acc {checkpoint['val_acc']:.2f}%, Test Acc {checkpoint['test_acc']:.2f}%")
    except:
        print(f"Model loaded from checkpoint epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Run ALL experiments
    all_results = {}
    
    all_results['noise'] = run_noise_experiment(model, test_dataset, DEVICE)
    all_results['occlusion'] = run_occlusion_experiment(model, test_dataset, DEVICE)
    all_results['brightness'] = run_brightness_experiment(model, test_dataset, DEVICE)
    all_results['skin_tone'] = run_skin_tone_experiment(model, test_dataset, test_sample, DEVICE)
    
    # Save all results
    save_experiment_results(all_results, OUTPUT_DIR / 'experiment_results.json')
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"\n📂 Results saved to: {OUTPUT_DIR}\n")
    print("Generated files:")
    print("  📊 noise_experiment.png")
    print("  📊 occlusion_experiment.png")
    print("  📊 brightness_experiment.png")
    print("  🎯 skin_tone_experiment.png  ← KEY FAIRNESS RESULT!")
    print("  📄 experiment_results.json")
    print("\n" + "="*60)
    
    # Summary
    print("\n📈 QUICK SUMMARY:")
    print(f"  Skin Tone Disparity: {all_results['skin_tone']['disparity']:.2f}%")
    print(f"  Noise Robustness: {all_results['noise']['accuracies'][0]:.1f}% → {all_results['noise']['accuracies'][-1]:.1f}%")
    print(f"  Occlusion Robustness: {all_results['occlusion']['accuracies'][0]:.1f}% → {all_results['occlusion']['accuracies'][-1]:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()