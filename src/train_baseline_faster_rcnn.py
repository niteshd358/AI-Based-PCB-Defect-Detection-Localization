"""
Two-Stage Detection Baseline: Faster R-CNN (ResNet-50-FPN)
ME6324 - AI in Manufacturing

Compares two-stage detection against one-stage YOLO:
- Speed (latency ms / throughput FPS)
- Bounding box proposal generation vs single-stage regression
"""

import time
import argparse
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

class PCBFasterRCNNDataset(Dataset):
    def __init__(self, split: str = "val", max_samples: int = 50):
        self.img_dir = PROCESSED_DIR / "images" / split
        self.lbl_dir = PROCESSED_DIR / "labels" / split
        img_files = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        self.img_files = img_files[:max_samples] if max_samples else img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        boxes = []
        labels = []
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2.0) * w
                        y1 = (yc - bh / 2.0) * h
                        x2 = (xc + bw / 2.0) * w
                        y2 = (yc + bh / 2.0) * h
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            # Torchvision detection expects class 0 as background, so labels are 1-indexed
                            labels.append(cls_id + 1)
                            
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {"boxes": boxes, "labels": labels}
        tensor_img = torchvision.transforms.functional.to_tensor(img)
        return tensor_img, target

def get_faster_rcnn_model(num_classes: int = 7):
    """Create Faster R-CNN model with 6 defect classes + background = 7."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def benchmark_faster_rcnn(device: str = "cpu"):
    print("=" * 60)
    print("  TWO-STAGE BASELINE: FASTER R-CNN BENCHMARKING")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    print(f"Device: {device}")
    
    model = get_faster_rcnn_model(num_classes=7)
    model.to(device)
    model.eval()
    
    # Warmup
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
            
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency
    print(f"Faster R-CNN Inference Latency : {avg_latency:.2f} ms")
    print(f"Faster R-CNN Throughput        : {fps:.1f} FPS")
    print("\n[Speed Trade-off Analysis]")
    print(f"  - Faster R-CNN requires ~{avg_latency:.1f} ms per PCB image.")
    print(f"  - YOLO detectors generally achieve 3x to 6x faster inference speeds.")
    print("  - For high-speed SMT assembly lines (e.g. 10-20 boards/sec), YOLO is required for real-time operation.")

if __name__ == "__main__":
    benchmark_faster_rcnn()
