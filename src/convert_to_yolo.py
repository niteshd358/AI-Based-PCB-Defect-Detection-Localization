"""
Unified Dataset Converter to YOLO Format
Parses and standardizes:
1. PKU-Market-PCB (COCO format JSONs: train.json, val.json, test.json)
2. DeepPCB (pair scans + defect annotation text files)
3. Synthetic PCB samples (if present)

Outputs partitioned data directly to:
  data/processed/images/{train, val, test}
  data/processed/labels/{train, val, test}
And generates data/pcb_data.yaml.
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import yaml

TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

# DeepPCB type mapping: 1: open, 2: short, 3: mousebite, 4: spur, 5: copper, 6: pin-hole
DEEPPCB_TYPE_MAP = {
    1: 2,  # open -> open_circuit
    2: 3,  # short -> short
    3: 1,  # mousebite -> mouse_bite
    4: 4,  # spur -> spur
    5: 5,  # copper -> spurious_copper
    6: 0   # pin-hole -> missing_hole
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PKU_DIR = RAW_DIR / "PKU-Market-PCB"
DEEPPCB_DIR = RAW_DIR / "DeepPCB"

def convert_coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [xmin, ymin, width, height] to YOLO [xc, yc, w, h] normalized."""
    xmin, ymin, w, h = bbox
    xc = (xmin + w / 2.0) / img_w
    yc = (ymin + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

def convert_box_coords_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to YOLO (xc, yc, w, h) normalized."""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

def process_pku_market_pcb(max_per_split: int = None):
    """Process PKU-Market-PCB dataset containing COCO JSON annotations."""
    if not PKU_DIR.exists():
        print("[INFO] PKU-Market-PCB folder not found.")
        return
        
    print("[INFO] Processing PKU-Market-PCB dataset (COCO JSON)...")
    split_map = {
        "train": PKU_DIR / "train.json",
        "val": PKU_DIR / "val.json",
        "test": PKU_DIR / "test.json"
    }
    
    for split, json_path in split_map.items():
        if not json_path.exists():
            continue
            
        with open(json_path, "r") as f:
            coco_data = json.load(f)
            
        images_dir = PKU_DIR / "images" / split
        if not images_dir.exists():
            images_dir = PKU_DIR / "images"
            
        # Map category IDs to target class IDs
        cat_id_to_target = {}
        for cat in coco_data.get("categories", []):
            name = cat["name"].strip().lower()
            if name in CLASS_NAME_TO_ID:
                cat_id_to_target[cat["id"]] = CLASS_NAME_TO_ID[name]
                
        # Group annotations by image_id
        img_annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
            
        out_img_dir = PROCESSED_DIR / "images" / split
        out_lbl_dir = PROCESSED_DIR / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        images_list = coco_data.get("images", [])
        if max_per_split:
            images_list = images_list[:max_per_split]
            
        count = 0
        for img_info in images_list:
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]
            
            src_img = images_dir / file_name
            if not src_img.exists():
                continue
                
            dest_name = f"pku_{split}_{img_id:05d}"
            dest_img = out_img_dir / f"{dest_name}.jpg"
            dest_lbl = out_lbl_dir / f"{dest_name}.txt"
            
            lines = []
            for ann in img_annotations.get(img_id, []):
                cat_id = ann["category_id"]
                target_cls = cat_id_to_target.get(cat_id)
                if target_cls is not None:
                    xc, yc, w, h = convert_coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                    lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    
            shutil.copy(src_img, dest_img)
            with open(dest_lbl, "w") as f:
                f.write("\n".join(lines))
            count += 1
            
        print(f"  -> Converted {count} PKU images for {split} split.")

def process_deeppcb(max_samples: int = None):
    """Process DeepPCB dataset."""
    if not DEEPPCB_DIR.exists():
        print("[INFO] DeepPCB folder not found.")
        return
        
    print("[INFO] Processing DeepPCB dataset...")
    # Find all _test.jpg images
    test_images = list(DEEPPCB_DIR.rglob("*_test.jpg"))
    if not test_images:
        return
        
    if max_samples:
        test_images = test_images[:max_samples]
        
    random.seed(42)
    random.shuffle(test_images)
    
    total = len(test_images)
    n_train = int(total * 0.70)
    n_val = int(total * 0.20)
    
    splits = {
        "train": test_images[:n_train],
        "val": test_images[n_train:n_train + n_val],
        "test": test_images[n_train + n_val:]
    }
    
    for split, img_list in splits.items():
        out_img_dir = PROCESSED_DIR / "images" / split
        out_lbl_dir = PROCESSED_DIR / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for idx, img_path in enumerate(img_list):
            # Annotation text file location:
            # e.g., group00041/00041/00041000_test.jpg -> group00041/00041_not/00041000.txt
            stem_num = img_path.stem.replace("_test", "")
            parent_name = img_path.parent.name
            ann_folder = img_path.parent.parent / f"{parent_name}_not"
            ann_file = ann_folder / f"{stem_num}.txt"
            
            if not ann_file.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            lines = []
            with open(ann_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2, defect_type = map(float, parts[:5])
                        target_cls = DEEPPCB_TYPE_MAP.get(int(defect_type))
                        if target_cls is not None:
                            yolo_box = convert_box_coords_to_yolo(x1, y1, x2, y2, w, h)
                            if yolo_box:
                                xc, yc, bw, bh = yolo_box
                                lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                                
            dest_name = f"dpcb_{split}_{idx:05d}"
            shutil.copy(img_path, out_img_dir / f"{dest_name}.jpg")
            with open(out_lbl_dir / f"{dest_name}.txt", "w") as f:
                f.write("\n".join(lines))
            count += 1
            
        print(f"  -> Converted {count} DeepPCB images for {split} split.")

def main():
    print("=" * 60)
    print(" UNIFIED DATASET CONVERTER TO YOLO (PKU + DEEPPCB)")
    print("=" * 60)
    
    # Process both datasets
    process_pku_market_pcb()
    process_deeppcb()
    
    # Update pcb_data.yaml
    data_yaml_path = DATA_DIR / "pcb_data.yaml"
    processed_posix = str(PROCESSED_DIR.resolve()).replace("\\", "/")
    yaml_content = {
        "path": processed_posix,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)}
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\n[SUCCESS] Dataset converted. Updated YAML at: {data_yaml_path}")

if __name__ == "__main__":
    main()
