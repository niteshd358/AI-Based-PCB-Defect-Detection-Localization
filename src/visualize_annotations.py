"""
Bounding Box Alignment & Dataset Verification Tool
Visualizes ground-truth bounding boxes on PCB images to verify:
1. Coordinate normalization / un-normalization accuracy
2. Alignment with actual physical PCB defect anomalies
3. Class balance and label distributions across splits
"""

import math
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

# Color palette (BGR for OpenCV)
CLASS_COLORS = [
    (0, 165, 255),   # missing_hole: Orange
    (255, 0, 0),     # mouse_bite: Blue
    (0, 0, 255),     # open_circuit: Bright Red
    (255, 0, 255),   # short: Magenta
    (0, 255, 255),   # spur: Yellow
    (0, 255, 0)      # spurious_copper: Green
]

def load_yolo_labels(label_file: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """Parse YOLO txt file and return pixel-coordinate bounding boxes (cls_id, x1, y1, x2, y2)."""
    boxes = []
    if not label_file.exists():
        return boxes
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((xc - w / 2.0) * img_w)
                y1 = int((yc - h / 2.0) * img_h)
                x2 = int((xc + w / 2.0) * img_w)
                y2 = int((yc + h / 2.0) * img_h)
                boxes.append((cls_id, max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)))
    return boxes

def draw_annotations(img: np.ndarray, boxes: List[Tuple[int, int, int, int, int]]) -> np.ndarray:
    """Draw bounding boxes and class tags on PCB image."""
    canvas = img.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cls_name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"class_{cls_id}"
        
        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Label badge
        label = f"{cls_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        badge_y1 = max(0, y1 - lh - 6)
        badge_y2 = y1
        badge_x2 = min(canvas.shape[1], x1 + lw + 6)
        
        cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return canvas

def generate_verification_grid(split: str = "train", num_images: int = 9, output_filename: str = "annotation_verification_grid.jpg"):
    """Render a grid of images with overlaid ground truth annotations."""
    img_dir = PROCESSED_DIR / "images" / split
    lbl_dir = PROCESSED_DIR / "labels" / split
    
    if not img_dir.exists():
        print(f"[ERROR] Image directory {img_dir} does not exist. Run convert_to_yolo.py first.")
        return
        
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not img_files:
        print(f"[ERROR] No images found in {img_dir}")
        return
        
    selected = img_files[:min(num_images, len(img_files))]
    annotated_images = []
    
    for img_p in selected:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_yolo_labels(lbl_p, w, h)
        annotated = draw_annotations(img, boxes)
        annotated_images.append((img_p.name, annotated, len(boxes)))
        
    # Build grid
    cols = min(3, len(annotated_images))
    rows = math.ceil(len(annotated_images) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for idx, (name, img_bgr, count) in enumerate(annotated_images):
        r, c = idx // cols, idx % cols
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[r, c].imshow(img_rgb)
        axes[r, c].set_title(f"{name} ({count} defects)", fontsize=10)
        axes[r, c].axis("off")
        
    for idx in range(len(annotated_images), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
        
    plt.tight_layout()
    out_path = BASE_DIR / "data" / output_filename
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[SUCCESS] Verification grid saved to: {out_path}")

def print_dataset_statistics():
    """Print per-class defect counts across train, val, and test splits."""
    stats = {split: {cls_name: 0 for cls_name in TARGET_CLASSES} for split in ["train", "val", "test"]}
    image_counts = {}
    
    for split in ["train", "val", "test"]:
        lbl_dir = PROCESSED_DIR / "labels" / split
        img_dir = PROCESSED_DIR / "images" / split
        image_counts[split] = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        if lbl_dir.exists():
            for lbl_p in lbl_dir.glob("*.txt"):
                with open(lbl_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id < len(TARGET_CLASSES):
                                stats[split][TARGET_CLASSES[cls_id]] += 1
                                
    print("\n" + "=" * 50)
    print("       PCB DATASET SPLIT & CLASS STATISTICS")
    print("=" * 50)
    for split in ["train", "val", "test"]:
        print(f"\n[{split.upper()}] Images: {image_counts.get(split, 0)}")
        for cls_name in TARGET_CLASSES:
            print(f"  - {cls_name:<18}: {stats[split][cls_name]} instances")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    print_dataset_statistics()
    generate_verification_grid()
