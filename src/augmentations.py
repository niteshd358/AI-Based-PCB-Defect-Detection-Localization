"""
Domain Adaptation & Real-World Augmentation Pipeline
Uses Albumentations to transform flatbed/dataset PCB scans into realistic
webcam/factory floor conditions:
1. Lighting & glare shifts (Brightness, Contrast, Hue/Saturation)
2. Lens distortions and angle misalignments (ShiftScaleRotate, Affine, Perspective)
3. Sensor artifacts & defocus (Gaussian noise, Motion blur)
4. Symmetrical variations (Rotations, Flips)
"""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

def get_train_transforms(img_size: int = 640):
    """
    Albumentations pipeline for real-world domain adaptation.
    Bboxes are formatted in YOLO format: [x_center, y_center, width, height] normalized.
    """
    return A.Compose([
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            translate_percent=(-0.06, 0.06),
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        
        # Color & lighting variations (crucial for webcam/desk lighting adaptation)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        
        # Sensor noise and camera focus artifacts
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def apply_augmentations(img: np.ndarray, bboxes: List[List[float]], class_labels: List[int], transforms=None):
    """Apply albumentations pipeline to an image and its YOLO bboxes."""
    if transforms is None:
        transforms = get_train_transforms(img.shape[0])
        
    augmented = transforms(image=img, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']

def visualize_augmentations(image_path: Path, label_path: Path, num_variants: int = 6, output_path: Path = None):
    """Generate a multi-panel visual demonstration of the domain adaptation pipeline."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not load image {image_path}")
        return
        
    h, w = img.shape[:2]
    bboxes = []
    class_labels = []
    
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bboxes.append([xc, yc, bw, bh])
                    class_labels.append(cls_id)
                    
    transforms = get_train_transforms(h)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_variants):
        if i == 0:
            aug_img, aug_boxes, aug_labels = img.copy(), bboxes, class_labels
            title = "Original Image"
        else:
            aug_img, aug_boxes, aug_labels = apply_augmentations(img, bboxes, class_labels, transforms)
            title = f"Augmented Variant #{i} (Lighting/Noise/Warp)"
            
        # Draw bboxes
        disp_img = aug_img.copy()
        dh, dw = disp_img.shape[:2]
        for box, lbl in zip(aug_boxes, aug_labels):
            xc, yc, bw, bh = box
            x1 = int((xc - bw / 2.0) * dw)
            y1 = int((yc - bh / 2.0) * dh)
            x2 = int((xc + bw / 2.0) * dw)
            y2 = int((yc + bh / 2.0) * dh)
            cls_name = TARGET_CLASSES[int(lbl)] if int(lbl) < len(TARGET_CLASSES) else f"cls_{lbl}"
            cv2.putText(disp_img, cls_name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
        disp_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(disp_rgb)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")
        
    plt.tight_layout()
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "augmentation_demo.jpg"
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"[SUCCESS] Augmentation demo saved to {output_path}")

if __name__ == "__main__":
    test_img = Path(__file__).resolve().parent.parent / "data" / "processed" / "images" / "train" / "pcb_train_00000.jpg"
    test_lbl = Path(__file__).resolve().parent.parent / "data" / "processed" / "labels" / "train" / "pcb_train_00000.txt"
    if test_img.exists():
        visualize_augmentations(test_img, test_lbl)
