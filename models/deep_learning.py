"""
Deep Learning Model for ASL Recognition
Uses MobileNetV2 and EfficientNet-B3 with transfer learning
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=False, img_size=224):
    """Get image transforms"""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class ASLMobileNet(nn.Module):
    """MobileNetV2-based ASL classifier"""
    
    def __init__(self, num_classes=29, pretrained=True):
        super(ASLMobileNet, self).__init__()
        
        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace final classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ASLEfficientNet(nn.Module):
    """EfficientNet-B3 for SOTA performance"""
    
    def __init__(self, num_classes=29, pretrained=True):
        super(ASLEfficientNet, self).__init__()
        
        # Use weights parameter for newer PyTorch
        if pretrained:
            try:
                weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            except:
                weights = 'IMAGENET1K_V1'
        else:
            weights = None
            
        self.model = models.efficientnet_b3(weights=weights)
        
        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100*correct/total})
    
    return running_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total