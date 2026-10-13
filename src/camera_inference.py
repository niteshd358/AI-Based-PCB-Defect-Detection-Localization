"""
Real-Time Webcam & USB Camera AOI Inference Stream
Captures live camera frames, applies preprocessing & aspect ratio letterboxing,
runs YOLO inference, and draws real-time defect bounding boxes with PASS/FAIL banners.
"""

import time
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from src.image_enhancement import enhance_pcb_image

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "weights" / "best_yolo.pt"

TARGET_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

CLASS_COLORS = [
    (0, 165, 255),   # missing_hole: Orange
    (255, 0, 0),     # mouse_bite: Blue
    (0, 0, 255),     # open_circuit: Bright Red
    (255, 0, 255),   # short: Magenta
    (0, 255, 255),   # spur: Yellow
    (0, 255, 0)      # spurious_copper: Green
]

def preprocess_frame(frame: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    top = (target_size - nh) // 2
    bottom = target_size - nh - top
    left = (target_size - nw) // 2
    right = target_size - nw - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (left, top)

def run_camera_stream(
    camera_index: int = 0,
    model_path: str = None,
    conf_thres: float = 0.35,
    iou_thres: float = 0.45,
    enable_enhancement: bool = True
):
    if model_path is None or not Path(model_path).exists():
        model_path = str(WEIGHTS_PATH)
        if not Path(model_path).exists():
            print(f"[WARN] Custom weights not found at {model_path}. Loading standard yolov8n.pt...")
            model_path = "yolov8n.pt"
            
    print(f"[INFO] Initializing YOLO from {model_path}...")
    model = YOLO(model_path)
    
    print(f"[INFO] Opening camera device index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}. Check USB connection or privacy permissions.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    
    print("\n[ACTIVE] Live Camera Inspection Started.")
    print("Controls:")
    print("  'e': Toggle Real-World Lighting Enhancement (CLAHE + White Balance)")
    print("  'q' or ESC: Exit Inspection Stream\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break
            
        # Optional industrial camera preprocessing
        if enable_enhancement:
            input_frame = enhance_pcb_image(frame, enable_wb=True, enable_clahe=True, enable_glare_suppression=True)
        else:
            input_frame = frame
            
        # Run inference
        results = model.predict(input_frame, conf=conf_thres, iou=iou_thres, verbose=False)
        result = results[0]
        
        annotated_frame = input_frame.copy()
        defects_found = []
        
        # Parse detections
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                cls_name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"cls_{cls_id}"
                defects_found.append((cls_name, conf, (x1, y1, x2, y2)))
                
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{cls_name} {conf*100:.0f}%"
                cv2.putText(annotated_frame, label_text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Draw Quality Verdict Badge
        verdict = "PASS" if len(defects_found) == 0 else f"FAIL ({len(defects_found)} Defects)"
        verdict_color = (0, 200, 0) if len(defects_found) == 0 else (0, 0, 230)
        
        cv2.rectangle(annotated_frame, (10, 10), (320, 65), (30, 30, 30), -1)
        cv2.putText(annotated_frame, f"STATUS: {verdict}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)
        
        # FPS Counter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("PCB Automated Optical Inspection (AOI)", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e') or key == ord('E'):
            enable_enhancement = not enable_enhancement
            print(f"[TOGGLE] Lighting Preprocessing: {'ENABLED' if enable_enhancement else 'DISABLED'}")
        elif key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stream stopped.")

def main():
    parser = argparse.ArgumentParser(description="Real-Time PCB Camera Inspection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    args = parser.parse_args()
    
    run_camera_stream(
        camera_index=args.camera,
        model_path=args.weights,
        conf_thres=args.conf,
        iou_thres=args.iou
    )

if __name__ == "__main__":
    main()
