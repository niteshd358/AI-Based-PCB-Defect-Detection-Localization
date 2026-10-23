# AI-Based PCB Defect Detection & Localization (AOI)
**Course:** ME6324 – AI in Manufacturing  
**Domain:** Automated Optical Inspection (AOI) for Quality Assurance in PCB Fabrication & Assembly

---

## 1. Project Overview
This project replaces manual, fatigue-prone visual inspection with a state-of-the-art deep learning-based Automated Optical Inspection (AOI) system for printed circuit board (PCB) manufacturing.

📖 **Comprehensive Documentation:** For the full 20-page equivalent engineering technical documentation covering mathematical formulations, multi-task loss derivation, baseline model benchmarks, Grad-CAM explainability mechanics, and manufacturing FMEA analysis, refer to [DETAILED_PROJECT_REPORT.md](file:///d:/Ai_In_manf/ai%20in%20manufactruing/DETAILED_PROJECT_REPORT.md).

### Target Defect Categories (6 Classes)
0. **`missing_hole`**: Unpunched or missing drill via / component hole.
1. **`mouse_bite`**: Small chunk eroded along the edge of a copper trace.
2. **`open_circuit`**: Unintended gap/break interrupting a copper track.
3. **`short`**: Unintended copper bridge connecting isolated traces.
4. **`spur`**: Sharp protrusion jutting out from a copper trace.
5. **`spurious_copper`**: Stray flecks/patches of copper on the substrate.

---

## 2. Directory Structure
```
ai-in-manufacturing/
├── data/
│   ├── raw/
│   │   ├── DeepPCB/
│   │   ├── PKU-Market-PCB/
│   │   └── Synthetic_PCB/
│   ├── processed/
│   │   ├── images/ (train, val, test)
│   │   └── labels/ (train, val, test)
│   ├── pcb_data.yaml
│   ├── annotation_verification_grid.jpg
│   └── augmentation_demo.jpg
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── convert_to_yolo.py
│   ├── visualize_annotations.py
│   ├── augmentations.py
│   ├── train_yolo.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── camera_inference.py
├── models/
│   └── weights/
├── app/
│   └── main.py
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 3. Kaggle API Configuration
To download the full DeepPCB and PKU-Market-PCB datasets:
1. Log in to [Kaggle](https://www.kaggle.com/) -> **Settings** -> **Create New Token**.
2. Save the downloaded `kaggle.json` into:
   - `C:\Users\<Username>\.kaggle\kaggle.json` OR
   - The project root directory: `./kaggle.json`
3. Execute automated download:
   ```bash
   python src/download_data.py --download-all
   ```
*Note: If you have already downloaded the zip files manually, simply drop them into `data/raw/` and run `python src/download_data.py` to auto-extract them.*

---

## 4. Pipeline Execution Commands

### Step 1: Format Conversion & Split Partitioning
Convert raw annotations (VOC XML / DeepPCB txt) into normalized YOLO bounding boxes and generate `data.yaml`:
```bash
python src/convert_to_yolo.py
```

### Step 2: Annotation Alignment Verification
Verify bounding box alignment and print class distribution statistics:
```bash
python src/visualize_annotations.py
```

### Step 3: Domain Adaptation Augmentation Demo
Preview real-world lighting, perspective warp, and camera noise augmentations:
```bash
python src/augmentations.py
```

### Step 4: YOLO Detector Training
Train YOLOv8 / YOLO11:
```bash
python src/train_yolo.py --model yolov8n.pt --epochs 30 --batch 8 --imgsz 640
```

### Step 5: Real-Time Streamlit Inspection Dashboard
Launch the interactive inspection portal:
```bash
streamlit run app/main.py
```
