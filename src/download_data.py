"""
Dataset Downloader & Preparer for PCB Defect Detection
Supports:
1. Automated Kaggle API download (DeepPCB and PKU-Market-PCB)
2. Manual archive decompression from data/raw/
3. Synthetic PCB sample dataset generator for immediate offline testing & validation
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
import cv2
import numpy as np

# Class Definitions
DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
DEEPPCB_RAW = RAW_DIR / "DeepPCB"
PKU_RAW = RAW_DIR / "PKU-Market-PCB"

def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist in standard locations or environment."""
    home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
    local_kaggle = Path.cwd() / "kaggle.json"
    
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if home_kaggle.exists():
        return True
    if local_kaggle.exists():
        # Copy to home directory if found locally
        home_kaggle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_kaggle, home_kaggle)
        print(f"[INFO] Copied local kaggle.json to {home_kaggle}")
        return True
    return False

def download_from_kaggle(dataset_slug: str, target_dir: Path):
    """Download and extract a dataset using the Kaggle API."""
    if not check_kaggle_credentials():
        print("\n[WARNING] Kaggle credentials not found!")
        print("Please place your `kaggle.json` API token in either:")
        print(f"  - {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("  - The project root directory (`./kaggle.json`)")
        print("Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
        print("\nAlternatively, run with `--generate-sample` to build a local validation dataset immediately.\n")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[INFO] Authenticated with Kaggle. Downloading {dataset_slug}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
        print(f"[SUCCESS] Downloaded and extracted {dataset_slug} into {target_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_slug}: {e}")
        return False

def extract_manual_archives():
    """Extract any zip files found in data/raw/."""
    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"[INFO] Found manual zip: {zip_path.name}. Extracting...")
        extract_target = RAW_DIR
        if "deep" in zip_path.name.lower():
            extract_target = DEEPPCB_RAW
        elif "pku" in zip_path.name.lower() or "market" in zip_path.name.lower():
            extract_target = PKU_RAW
        
        extract_target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_target)
        print(f"[SUCCESS] Extracted {zip_path.name} to {extract_target}")

def generate_synthetic_pcb_dataset(num_samples: int = 150):
    """
    Generate realistic PCB sample images with ground truth annotations
    across all 6 defect classes for immediate verification, test-runs, and CI.
    """
    print(f"\n[INFO] Generating {num_samples} realistic synthetic PCB inspection samples...")
    raw_synthetic_dir = RAW_DIR / "Synthetic_PCB"
    img_dir = raw_synthetic_dir / "images"
    lbl_dir = raw_synthetic_dir / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    img_size = 640
    
    for i in range(num_samples):
        # Base substrate (FR-4 green PCB)
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        green_val = np.random.randint(40, 75)
        img[:, :] = (np.random.randint(15, 30), green_val, np.random.randint(15, 30))
        
        # Substrate texture / woven glass pattern
        noise = np.random.normal(0, 5, (img_size, img_size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Draw copper traces (gold / bright copper color: BGR ~ [40, 180, 220])
        copper_color = (45 + np.random.randint(-5, 5), 185 + np.random.randint(-10, 10), 225 + np.random.randint(-10, 10))
        via_color = (180, 200, 210)  # Silver/tinned pads
        
        # Generate grid of IC pads & traces
        traces = []
        for y in range(80, img_size - 80, 70):
            pt1 = (40, y + np.random.randint(-5, 5))
            pt2 = (img_size - 40, y + np.random.randint(-5, 5))
            thickness = np.random.choice([8, 12, 16])
            cv2.line(img, pt1, pt2, copper_color, thickness)
            traces.append((pt1, pt2, thickness))
            
        for x in range(100, img_size - 100, 90):
            pt1 = (x, 40)
            pt2 = (x, img_size - 40)
            thickness = np.random.choice([8, 12])
            cv2.line(img, pt1, pt2, copper_color, thickness)
            traces.append((pt1, pt2, thickness))
            
        # Draw component solder pads & vias
        pads = []
        for px in range(120, img_size - 100, 110):
            for py in range(120, img_size - 100, 100):
                radius = 16
                cv2.circle(img, (px, py), radius, via_color, -1)
                # Drill hole (black center)
                cv2.circle(img, (px, py), 7, (20, 20, 20), -1)
                pads.append((px, py, radius))
                
        # Inject realistic defects
        annotations = []
        num_defects = np.random.randint(1, 5) if i % 10 != 0 else 0  # 10% golden/pass boards
        
        for _ in range(num_defects):
            defect_id = np.random.randint(0, 6)
            
            if defect_id == 0:  # Missing hole: Pad has copper/solder but no drill hole
                pad_idx = np.random.randint(0, len(pads))
                px, py, r = pads[pad_idx]
                # Cover hole with solder/copper
                cv2.circle(img, (px, py), 8, via_color, -1)
                x1, y1 = max(0, px - r), max(0, py - r)
                x2, y2 = min(img_size - 1, px + r), min(img_size - 1, py + r)
                
            elif defect_id == 1:  # Mouse bite: Bite taken out of trace edge
                t_idx = np.random.randint(0, len(traces))
                (p1x, p1y), (p2x, p2y), th = traces[t_idx]
                mx = int(p1x + (p2x - p1x) * np.random.uniform(0.2, 0.8))
                my = int(p1y + (p2y - p1y) * np.random.uniform(0.2, 0.8))
                bite_r = th // 2 + 3
                cv2.circle(img, (mx, my + th // 2), bite_r, (25, green_val, 25), -1)
                x1, y1 = max(0, mx - bite_r - 4), max(0, my - bite_r - 4)
                x2, y2 = min(img_size - 1, mx + bite_r + 4), min(img_size - 1, my + bite_r + 4)
                
            elif defect_id == 2:  # Open circuit: Gap interrupting copper trace
                t_idx = np.random.randint(0, len(traces))
                (p1x, p1y), (p2x, p2y), th = traces[t_idx]
                mx = int(p1x + (p2x - p1x) * np.random.uniform(0.2, 0.8))
                my = int(p1y + (p2y - p1y) * np.random.uniform(0.2, 0.8))
                gap_size = th + 6
                cv2.circle(img, (mx, my), gap_size, (25, green_val, 25), -1)
                x1, y1 = max(0, mx - gap_size - 2), max(0, my - gap_size - 2)
                x2, y2 = min(img_size - 1, mx + gap_size + 2), min(img_size - 1, my + gap_size + 2)
                
            elif defect_id == 3:  # Short circuit: Bridge connecting two traces
                sx = np.random.randint(100, img_size - 120)
                sy = np.random.randint(100, img_size - 120)
                cv2.rectangle(img, (sx, sy), (sx + 24, sy + 18), copper_color, -1)
                x1, y1 = sx - 2, sy - 2
                x2, y2 = sx + 26, sy + 20
                
            elif defect_id == 4:  # Spur: Sharp protrusion from trace
                t_idx = np.random.randint(0, len(traces))
                (p1x, p1y), (p2x, p2y), th = traces[t_idx]
                mx = int(p1x + (p2x - p1x) * np.random.uniform(0.2, 0.8))
                my = int(p1y + (p2y - p1y) * np.random.uniform(0.2, 0.8))
                pts = np.array([[mx, my], [mx + 18, my + 14], [mx - 4, my + 6]], np.int32)
                cv2.fillPoly(img, [pts], copper_color)
                x1, y1 = min(pts[:, 0]) - 2, min(pts[:, 1]) - 2
                x2, y2 = max(pts[:, 0]) + 2, max(pts[:, 1]) + 2
                
            else:  # Spurious copper: Stray fleck of copper on substrate
                cx = np.random.randint(60, img_size - 60)
                cy = np.random.randint(60, img_size - 60)
                cr = np.random.randint(6, 14)
                cv2.circle(img, (cx, cy), cr, copper_color, -1)
                x1, y1 = cx - cr - 2, cy - cr - 2
                x2, y2 = cx + cr + 2, cy + cr + 2
                
            # Convert [x1, y1, x2, y2] to YOLO format: [class_id, x_center, y_center, width, height]
            bw = (x2 - x1)
            bh = (y2 - y1)
            if bw > 2 and bh > 2:
                xc = (x1 + x2) / 2.0 / img_size
                yc = (y1 + y2) / 2.0 / img_size
                w = bw / float(img_size)
                h = bh / float(img_size)
                annotations.append(f"{defect_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        # Save image and label
        img_filename = f"pcb_sample_{i:04d}.jpg"
        lbl_filename = f"pcb_sample_{i:04d}.txt"
        
        cv2.imwrite(str(img_dir / img_filename), img)
        with open(lbl_dir / lbl_filename, "w") as f:
            f.write("\n".join(annotations))
            
    print(f"[SUCCESS] Synthetic dataset generated: {num_samples} images with annotations in {raw_synthetic_dir}")

def main():
    parser = argparse.ArgumentParser(description="PCB Dataset Management Tool")
    parser.add_argument("--download-all", action="store_true", help="Attempt to download both Kaggle datasets")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic PCB dataset for immediate pipeline validation")
    parser.add_argument("--samples", type=int, default=150, help="Number of synthetic samples to generate")
    args = parser.parse_args()
    
    extract_manual_archives()
    
    if args.download_all:
        print("[INFO] Attempting to download DeepPCB from Kaggle...")
        download_from_kaggle("norbertkorovics/deeppcb", DEEPPCB_RAW)
        print("[INFO] Attempting to download PKU-Market-PCB from Kaggle...")
        download_from_kaggle("akhatova/pcb-defects", PKU_RAW)
        
    if args.generate_sample or (not any(RAW_DIR.iterdir()) and not args.download_all):
        generate_synthetic_pcb_dataset(args.samples)
        
if __name__ == "__main__":
    main()
