"""
YOLOv8 / YOLO11 Training Pipeline for PCB Defect Detection
ME6324 - AI in Manufacturing

Trains the primary one-stage detector on the unified PCB dataset.
Supports CPU and CUDA hardware acceleration with automatic batch scaling.
"""

import os
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "pcb_data.yaml"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

def train_yolo(
    data_yaml: str = None,
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 10,
    workers: int = 2,
    device: str = None,
    fraction: float = 1.0
):
    """
    Train Ultralytics YOLO model on PCB defect dataset.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(data_yaml) if data_yaml else DATA_YAML
    if not data_path.exists():
        raise FileNotFoundError(f"Configuration file {data_path} not found!")
        
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
        
    print("=" * 60)
    print("      PCB DEFECT DETECTION - YOLO TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Architecture : {model_name}")
    print(f"Dataset Config    : {DATA_YAML}")
    print(f"Hardware Device   : {device} ({'GPU Acceleration' if device != 'cpu' else 'CPU Mode'})")
    print(f"Epochs            : {epochs}")
    print(f"Image Resolution  : {imgsz}x{imgsz}")
    print(f"Batch Size        : {batch}")
    print(f"Dataset Fraction  : {fraction * 100:.1f}%")
    print("=" * 60)
    
    # Initialize YOLO model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
        fraction=fraction,
        project=str(MODELS_DIR / "yolo_runs"),
        name="pcb_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        # Hyperparameters adjusted for PCB domain:
        degrees=10.0,
        translate=0.1,
        scale=0.15,
        shear=2.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    # Save the trained weights to standard destination
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest_best = WEIGHTS_DIR / "best_yolo.pt"
        shutil.copy(best_pt, dest_best)
        print(f"\n[SUCCESS] Optimal weights saved to: {dest_best}")
    else:
        print(f"\n[INFO] Run completed. Check results in: {results.save_dir}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on PCB Defect Dataset")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to data yaml (e.g. data/pcb_data.yaml or data/custom_data.yaml)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to train on (e.g. 0.05 for fast test)")
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction
    )

if __name__ == "__main__":
    main()
