"""
Exploratory Data Analysis (EDA) Script for PCB Defect Detection
Analyzes:
- Defect frequency distributions
- Bounding box aspect ratios and area distributions
- Image dimensions and quality checks
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_eda():
    records = []
    for split in ["train", "val", "test"]:
        lbl_dir = PROCESSED_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_p in lbl_dir.glob("*.txt"):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        records.append({
                            "split": split,
                            "class_id": cls_id,
                            "class_name": TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"cls_{cls_id}",
                            "box_area": w * h,
                            "aspect_ratio": w / (h + 1e-6)
                        })
                        
    if not records:
        print("[ERROR] No annotations found for EDA.")
        return
        
    df = pd.DataFrame(records)
    print("\n[EDA Summary Statistics]")
    print(df.groupby(["split", "class_name"]).size().unstack(fill_value=0))
    
    # Generate visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x="class_name", hue="split", palette="Set2")
    plt.title("Defect Class Distribution across Splits")
    plt.xticks(rotation=30)
    plt.xlabel("")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="class_name", y="box_area", palette="Set2")
    plt.title("Normalized Defect Box Area Distribution")
    plt.xticks(rotation=30)
    plt.xlabel("")
    plt.yscale("log")
    
    plt.tight_layout()
    out_fig = BASE_DIR / "data" / "eda_defect_distributions.png"
    plt.savefig(str(out_fig), dpi=150)
    plt.close()
    print(f"[SUCCESS] EDA plot saved to {out_fig}")

if __name__ == "__main__":
    run_eda()
