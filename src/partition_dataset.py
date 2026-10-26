"""
Custom Dataset Partitioner & User-Testing Isolation Tool
Allows user to:
1. Specify exact percentage splits: e.g. Train %, Val %, Test %, User-Testing %
2. Creates a completely separate, clean folder for 'user_testing' (for manual blind testing in Streamlit)
3. Generates a new dedicated yaml file: data/custom_data.yaml
"""

import os
import shutil
import random
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"

def partition_dataset(
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.10,
    user_test_pct: float = 0.05,
    output_dir_name: str = "custom_split",
    seed: int = 42
):
    random.seed(seed)
    
    total_pct = train_pct + val_pct + test_pct + user_test_pct
    assert abs(total_pct - 1.0) < 1e-4, f"Percentages must sum to 1.0! Current sum: {total_pct}"
    
    # Target directories
    target_base = BASE_DIR / "data" / output_dir_name
    user_testing_dir = BASE_DIR / "data" / "user_testing"
    
    # Collect all existing images across train, val, test
    all_image_paths = []
    for split in ["train", "val", "test"]:
        split_dir = IMAGES_DIR / split
        if split_dir.exists():
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                all_image_paths.extend(list(split_dir.glob(f"*{ext}")))
                
    total_images = len(all_image_paths)
    if total_images == 0:
        print("[ERROR] No images found in data/processed/images/! Run convert_to_yolo.py first.")
        return
        
    print("=" * 65)
    print("       PCB DATASET RE-PARTITIONING & ISOLATION TOOL")
    print("=" * 65)
    print(f"Total Source Images Found : {total_images}")
    print(f"Train Percentage          : {train_pct * 100:.1f}%")
    print(f"Validation Percentage     : {val_pct * 100:.1f}%")
    print(f"Test Percentage           : {test_pct * 100:.1f}%")
    print(f"User-Testing (Isolated)   : {user_test_pct * 100:.1f}%")
    print("=" * 65)
    
    random.shuffle(all_image_paths)
    
    n_train = int(total_images * train_pct)
    n_val = int(total_images * val_pct)
    n_test = int(total_images * test_pct)
    
    train_imgs = all_image_paths[:n_train]
    val_imgs = all_image_paths[n_train:n_train + n_val]
    test_imgs = all_image_paths[n_train + n_val:n_train + n_val + n_test]
    user_imgs = all_image_paths[n_train + n_val + n_test:]
    
    splits = {
        "train": (train_imgs, target_base / "images" / "train", target_base / "labels" / "train"),
        "val": (val_imgs, target_base / "images" / "val", target_base / "labels" / "val"),
        "test": (test_imgs, target_base / "images" / "test", target_base / "labels" / "test"),
        "user_testing": (user_imgs, user_testing_dir / "images", user_testing_dir / "ground_truth_labels")
    }
    
    for split_name, (img_list, img_dest, lbl_dest) in splits.items():
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for img_path in img_list:
            stem = img_path.stem
            # Copy image
            shutil.copy2(img_path, img_dest / img_path.name)
            
            # Find and copy corresponding label file
            found_lbl = False
            for s in ["train", "val", "test"]:
                lbl_src = LABELS_DIR / s / f"{stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dest / f"{stem}.txt")
                    found_lbl = True
                    break
                    
            copied_count += 1
            
        print(f"  -> [{split_name:<12}] Copied {copied_count:>4} images to {img_dest}")
        
    # Write custom yaml
    yaml_path = BASE_DIR / "data" / "custom_data.yaml"
    yaml_content = f"""path: {target_base.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print("\n[SUCCESS] Partitioning complete!")
    print(f"New YOLO Config File : {yaml_path}")
    print(f"Isolated User Folder : {user_testing_dir}")
    print(f"Total Isolated Test Boards for You/Examiners: {len(user_imgs)} boards")

def main():
    parser = argparse.ArgumentParser(description="Partition PCB dataset into custom splits")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (e.g. 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (e.g. 0.15)")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio (e.g. 0.10)")
    parser.add_argument("--user", type=float, default=0.05, help="Isolated user-testing ratio (e.g. 0.05)")
    parser.add_argument("--out", type=str, default="custom_split", help="Target subfolder under data/")
    args = parser.parse_args()
    
    partition_dataset(
        train_pct=args.train,
        val_pct=args.val,
        test_pct=args.test,
        user_test_pct=args.user,
        output_dir_name=args.out
    )

if __name__ == "__main__":
    main()
