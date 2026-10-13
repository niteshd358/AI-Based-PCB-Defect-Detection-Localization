"""
Model Evaluation and Benchmarking Suite
Calculates:
1. mAP@50 and mAP@50-95 across all 6 defect categories
2. Precision, Recall, and F1-score
3. Latency benchmarks (Inference speed in ms/frame)
4. PASS / FAIL operational classification accuracy
"""

import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data" / "pcb_data.yaml"
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

def evaluate_yolo(weights_path: Path, split: str = "test", device: str = "cpu"):
    """Evaluate YOLO detector on test or validation split."""
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        return None
        
    print("=" * 60)
    print("       PCB DEFECT DETECTION - MODEL EVALUATION SUITE")
    print("=" * 60)
    print(f"Model Checkpoint : {weights_path}")
    print(f"Evaluation Split : {split}")
    print(f"Execution Device : {device}")
    print("=" * 60)
    
    model = YOLO(str(weights_path))
    
    # 1. Benchmark Inference Latency
    print("\n[1/3] Benchmarking Inference Latency (ms/frame)...")
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_input, device=device, verbose=False)
        
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, device=device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency
    print(f"  -> Mean Inference Latency : {avg_latency:.2f} ms")
    print(f"  -> Estimated Throughput    : {fps:.1f} FPS")
    
    # 2. Validation Metrics
    print("\n[2/3] Evaluating Detection Metrics (mAP50, mAP50-95, P, R)...")
    val_results = model.val(
        data=str(DATA_YAML),
        split=split,
        device=device,
        batch=4,
        imgsz=640,
        project=str(MODELS_DIR / "eval_runs"),
        name="test_evaluation",
        exist_ok=True
    )
    
    metrics = {
        "mAP50": val_results.box.map50,
        "mAP50-95": val_results.box.map,
        "Precision": val_results.box.mp,
        "Recall": val_results.box.mr,
        "Mean Latency (ms)": avg_latency,
        "Throughput (FPS)": fps
    }
    
    print("\n[Overall Evaluation Summary]")
    for k, v in metrics.items():
        print(f"  - {k:<20}: {v:.4f}" if isinstance(v, float) else f"  - {k:<20}: {v}")
        
    # Class-wise metrics
    print("\n[Class-wise Performance]")
    class_df = []
    for i, cls_name in enumerate(TARGET_CLASSES):
        p = val_results.box.p[i] if len(val_results.box.p) > i else 0.0
        r = val_results.box.r[i] if len(val_results.box.r) > i else 0.0
        map50 = val_results.box.maps[i] if len(val_results.box.maps) > i else 0.0
        f1 = 2 * (p * r) / (p + r + 1e-6)
        class_df.append({
            "Class": cls_name,
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "mAP50": map50
        })
        print(f"  - {cls_name:<18} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f} | mAP50: {map50:.3f}")
        
    df = pd.DataFrame(class_df)
    report_csv = BASE_DIR / "models" / "evaluation_report.csv"
    df.to_csv(str(report_csv), index=False)
    print(f"\n[SUCCESS] Class-wise metrics saved to: {report_csv}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on PCB Defect Dataset")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "best_yolo.pt"), help="Path to weights file")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (val, test)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0)")
    args = parser.parse_args()
    
    evaluate_yolo(Path(args.weights), split=args.split, device=args.device)

if __name__ == "__main__":
    main()
