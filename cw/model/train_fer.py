import os
import re
import json
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def get_filtered_paths(archive_dir):
    archive_path = Path(archive_dir)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    # 1. Train set: strictly Training_*.jpg
    for c in CLASSES:
        cdir = archive_path / "train" / c
        if not cdir.exists():
            continue
        for f in cdir.iterdir():
            if re.match(r"^Training_\d+\.jpg$", f.name):
                train_paths.append(str(f))
                train_labels.append(CLASS_TO_IDX[c])
                
    # 2. Test directory: PublicTest_*.jpg (val) and PrivateTest_*.jpg (test)
    for c in CLASSES:
        cdir = archive_path / "test" / c
        if not cdir.exists():
            continue
        for f in cdir.iterdir():
            if re.match(r"^PublicTest_\d+\.jpg$", f.name):
                val_paths.append(str(f))
                val_labels.append(CLASS_TO_IDX[c])
            elif re.match(r"^PrivateTest_\d+\.jpg$", f.name):
                test_paths.append(str(f))
                test_labels.append(CLASS_TO_IDX[c])
                
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def compute_train_mean_std(train_paths):
    print("Computing mean and std strictly on training set...")
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = len(train_paths) * 48 * 48
    
    for p in train_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        norm_img = img.astype(np.float32) / 255.0
        total_sum += np.sum(norm_img)
        total_sq_sum += np.sum(norm_img ** 2)
        
    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(max(var, 1e-6))
    return float(mean), float(std)

class FERDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("L") # 48x48 Grayscale PIL image
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 24x24
            nn.Dropout2d(0.2)
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 12x12
            nn.Dropout2d(0.25)
        )
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 6x6
            nn.Dropout2d(0.3)
        )
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 3x3
            nn.Dropout2d(0.4)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return avg_loss, acc, np.array(all_preds), np.array(all_targets)

if __name__ == "__main__":
    print("Starting FER2013 Pipeline Verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using compute device:", device)
    
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = get_filtered_paths("archive")
    print(f"Data counts -> Train: {len(train_p)}, Val (PublicTest): {len(val_p)}, Test (PrivateTest): {len(test_p)}")
    
    mean, std = compute_train_mean_std(train_p)
    print(f"Calculated Train Mean: {mean:.4f}, Std: {std:.4f}")
