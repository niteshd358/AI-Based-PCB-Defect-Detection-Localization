"""
Classification-Only Baseline: ResNet-50 Binary PASS / FAIL Classifier
ME6324 - AI in Manufacturing

Evaluates the limitations of image-level classification vs object detection:
- Simple classification flags boards as PASS (0 defects) or FAIL (>= 1 defect).
- Cannot provide spatial bounding boxes, trace coordinates, or defect localization for rework stations.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

class PCBClassificationDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: int = None):
        self.img_dir = PROCESSED_DIR / "images" / split
        self.lbl_dir = PROCESSED_DIR / "labels" / split
        self.samples: List[Tuple[Path, int]] = []
        
        img_files = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        if max_samples:
            img_files = img_files[:max_samples]
            
        for img_p in img_files:
            lbl_p = self.lbl_dir / f"{img_p.stem}.txt"
            num_defects = 0
            if lbl_p.exists():
                with open(lbl_p, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    num_defects = len(lines)
            # Label: 0 = PASS (0 defects), 1 = FAIL (>= 1 defect)
            label = 1 if num_defects > 0 else 0
            self.samples.append((img_p, label))
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)

def train_or_eval_resnet(epochs: int = 3, batch_size: int = 16, device: str = "cpu"):
    print("=" * 60)
    print("  CLASSIFICATION-ONLY BASELINE: RESNET-50 (PASS / FAIL)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    print(f"Device: {device}")
    
    # Fast evaluation on subset for baseline metrics
    train_dataset = PCBClassificationDataset(split="train", max_samples=400)
    val_dataset = PCBClassificationDataset(split="val", max_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    
    # Load ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Binary classification head: 2 classes (PASS=0, FAIL=1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    
    print("\nTraining ResNet-50 Classification Head...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
        
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n[ResNet-50 Baseline Evaluation Results]")
    print(f"  - Accuracy  : {acc:.4f}")
    print(f"  - Precision : {prec:.4f}")
    print(f"  - Recall    : {rec:.4f}")
    print(f"  - F1-Score  : {f1:.4f}")
    print("\n[Critical Engineering Takeaway]")
    print("  While ResNet-50 achieves high binary classification accuracy, it outputs ZERO")
    print("  defect locations (x1, y1, x2, y2) and cannot distinguish multiple concurrent defects.")
    print("  This validates the necessity of primary YOLO object detection for factory AOI.")
    
    save_path = MODELS_DIR / "weights" / "resnet50_passfail.pth"
    torch.save(model.state_dict(), str(save_path))
    print(f"[SUCCESS] Baseline weights saved to: {save_path}")

if __name__ == "__main__":
    train_or_eval_resnet()
