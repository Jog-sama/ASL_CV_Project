"""
Train deep learning model (MobileNetV2)
"""
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, DEEP_MODEL_PATH, ASL_CLASSES, 
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NUM_CLASSES
)
from models.deep_learning import (
    ASLDataset, ASLMobileNet, get_transforms, 
    train_epoch, validate
)


def main():
    """Train and evaluate deep learning model"""
    print("=" * 60)
    print("Training Deep Learning Model (MobileNetV2)")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load metadata
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_metadata.csv')
    val_df = pd.read_csv(PROCESSED_DATA_DIR / 'val_metadata.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_metadata.csv')
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(ASL_CLASSES)}
    
    # Convert labels to indices
    train_labels = [label_to_idx[label] for label in train_df['class']]
    val_labels = [label_to_idx[label] for label in val_df['class']]
    test_labels = [label_to_idx[label] for label in test_df['class']]
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ASLDataset(
        train_df['path'].tolist(),
        train_labels,
        transform=get_transforms(augment=True)
    )
    
    val_dataset = ASLDataset(
        val_df['path'].tolist(),
        val_labels,
        transform=get_transforms(augment=False)
    )
    
    test_dataset = ASLDataset(
        test_df['path'].tolist(),
        test_labels,
        transform=get_transforms(augment=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nInitializing MobileNetV2...")
    model = ASLMobileNet(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': label_to_idx
            }, DEEP_MODEL_PATH)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("Evaluating best model on test set...")
    print("="*60)
    
    checkpoint = torch.load(DEEP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\nDeep learning model training complete!")


if __name__ == "__main__":
    main()