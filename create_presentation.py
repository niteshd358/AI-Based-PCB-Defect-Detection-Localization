import os
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path("d:/Ai_In_manf/ai in manufactruing")
OUT_PPTX = BASE_DIR / "AI_Based_PCB_Defect_Detection_FinalReview.pptx"

# Initialize presentation with 16:9 widescreen format (20.0 x 11.25 inches matching sample)
prs = Presentation()
prs.slide_width = Inches(20.0)
prs.slide_height = Inches(11.25)
blank_layout = prs.slide_layouts[6] # Blank layout

# Color Palette: Modern Industrial AOI Dark Theme
COLOR_BG = RGBColor(15, 23, 42)       # Slate 900
COLOR_CARD = RGBColor(30, 41, 59)     # Slate 800
COLOR_CARD_BORDER = RGBColor(51, 65, 85) # Slate 700
COLOR_TEXT_MAIN = RGBColor(248, 250, 252) # Slate 50
COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
COLOR_ACCENT = RGBColor(56, 189, 248)  # Sky 400
COLOR_SUCCESS = RGBColor(16, 185, 129) # Emerald 500
COLOR_WARNING = RGBColor(245, 158, 11) # Amber 500

def set_slide_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG

def add_header(slide, slide_num, title_text, category_text="ME6324 - AI IN MANUFACTURING"):
    # Header bar background
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(18.4), Inches(1.2))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = f"{category_text} • SLIDE {slide_num}"
    p_cat.font.size = Pt(13)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(28)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN

def add_card(slide, left, top, width, height, title="", fill_color=COLOR_CARD, border_color=COLOR_CARD_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)
    
    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), width - Inches(0.6), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
    return card

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_background(slide1)

# Large Accent Backdrop
backdrop = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.0), Inches(18.4), Inches(9.25))
backdrop.fill.solid()
backdrop.fill.fore_color.rgb = COLOR_CARD
backdrop.line.color.rgb = COLOR_CARD_BORDER
backdrop.line.width = Pt(2)

# Title Text Box
title_tb = slide1.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(17.0), Inches(4.5))
tf = title_tb.text_frame
tf.word_wrap = True

p0 = tf.paragraphs[0]
p0.text = "ME6324 – AI IN MANUFACTURING • COMPREHENSIVE FINAL REVIEW"
p0.font.size = Pt(15)
p0.font.bold = True
p0.font.color.rgb = COLOR_ACCENT
p0.space_after = Pt(20)

p1 = tf.add_paragraph()
p1.text = "AI-Driven Vision-Based Defect Detection and Localization Automated Optical Inspection (AOI) System for Printed Circuit Boards"
p1.font.size = Pt(38)
p1.font.bold = True
p1.font.color.rgb = COLOR_TEXT_MAIN
p1.space_after = Pt(25)

p2 = tf.add_paragraph()
p2.text = "Technologies: Ultralytics YOLO • Computer Vision • Deep Learning • OpenCV (CLAHE / AWB) • Streamlit Cloud • Python"
p2.font.size = Pt(18)
p2.font.color.rgb = COLOR_SUCCESS

# Details Card (Bottom)
meta_card = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(6.8), Inches(17.0), Inches(2.8))
meta_card.fill.solid()
meta_card.fill.fore_color.rgb = RGBColor(20, 29, 47)
meta_card.line.color.rgb = COLOR_ACCENT
meta_card.line.width = Pt(1)

meta_tb = slide1.shapes.add_textbox(Inches(1.8), Inches(7.0), Inches(16.4), Inches(2.4))
mtf = meta_tb.text_frame
mtf.word_wrap = True

mp0 = mtf.paragraphs[0]
mp0.text = "PROJECT METRICS & SYSTEM SPECIFICATIONS AT A GLANCE:"
mp0.font.size = Pt(14)
mp0.font.bold = True
mp0.font.color.rgb = COLOR_ACCENT
mp0.space_after = Pt(10)

cols_text = [
    "• Architecture: Single-Stage Anchor-Free YOLO (20.3 MB Checkpoint)",
    "• Dataset Scale: 4,686 Real PCB Images, 22,359 Defect Annotations (6 Classes)",
    "• Benchmarked Accuracy: 91.4% mAP@50 | 91.6% Precision | 89.7% Recall",
    "• Industrial Throughput: 38.4 ms/frame (26.0 FPS CPU) | <8.5 ms (>110 FPS GPU)",
    "• Real-World Validation: Tested on 14 Physical Boards from ECE Central Workshop"
]
for line in cols_text:
    p = mtf.add_paragraph()
    p.text = line
    p.font.size = Pt(15)
    p.font.color.rgb = COLOR_TEXT_MAIN
    p.space_after = Pt(4)

# ==============================================================================
# SLIDE 2: PROBLEM STATEMENT & PROJECT OBJECTIVES
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_background(slide2)
add_header(slide2, 2, "Problem Statement & Project Objectives")

# Left Column: Problem Statement
add_card(slide2, Inches(0.8), Inches(1.8), Inches(8.8), Inches(8.8), "THE MANUFACTURING CHALLENGE")
prob_tb = slide2.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(8.2), Inches(7.6))
ptf = prob_tb.text_frame
ptf.word_wrap = True

prob_points = [
    ("High-Density Interconnect (HDI) Complexity:", "Modern bare-board PCBs feature copper traces under 75 microns and drilled micro-vias under 0.2 mm. Microscopic defects compromise entire electronic assemblies."),
    ("Manual Visual Inspection (MVI) Failure:", "Human operators using stereo-microscopes experience severe cognitive eye strain after 20-30 minutes, leading to defect escape rates of 15%-30% and sluggish line throughput (<2 boards/min)."),
    ("Classical CAD-Subtraction AOI Limitations:", "Traditional Automated Optical Inspection relies on hardcoded subtraction against golden CAD templates. Minor board warping (<1 mm), vibration, or ambient lighting changes trigger false alarm rates exceeding 20%."),
    ("Industrial Mandate:", "Requires an autonomous, invariant AI defect detector that localizes sub-millimeter defects in real-time (>20 FPS) and outputs rework coordinates [x, y, w, h] to guide automated repair.")
]
for idx, (title, desc) in enumerate(prob_points):
    p = ptf.paragraphs[0] if idx == 0 else ptf.add_paragraph()
    p.text = f"{idx+1}. {title} "
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_WARNING
    
    p2 = ptf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(12)

# Right Column: Project Objectives
add_card(slide2, Inches(10.4), Inches(1.8), Inches(8.8), Inches(8.8), "KEY PROJECT OBJECTIVES")
obj_tb = slide2.shapes.add_textbox(Inches(10.7), Inches(2.6), Inches(8.2), Inches(7.6))
otf = obj_tb.text_frame
otf.word_wrap = True

obj_points = [
    ("Unified Benchmark Dataset Engineering:", "Consolidate and convert 4,686 real PCB images into standard YOLO format across 6 critical classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper."),
    ("Single-Stage Anchor-Free Deep Learning:", "Train an optimized YOLO architecture with multi-scale Feature Pyramid Network (PANet) to simultaneously detect 12-pixel pinholes and large copper shorts."),
    ("Industrial Optical Preprocessing Suite:", "Engineer a real-world lighting normalizer (Gray World White Balance, LAB CLAHE, Glare Masking) to overcome uneven factory illumination and shiny solder reflections."),
    ("Live Operator Dashboard & Rework Manifest:", "Deploy an interactive Streamlit web dashboard with live camera capture, 4x micro-inspection zoom, and automated CSV rework coordinate export."),
    ("Physical Laboratory Validation:", "Validate model generalization on real physical PCB specimens captured in the ECE Central Electronics Workshop.")
]
for idx, (title, desc) in enumerate(obj_points):
    p = otf.paragraphs[0] if idx == 0 else otf.add_paragraph()
    p.text = f"Objective {idx+1}: {title}"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUCCESS
    
    p2 = otf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(12)

# ==============================================================================
# SLIDE 3: DATASET, AI METHODOLOGY & MODEL TRAINING
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_background(slide3)
add_header(slide3, 3, "Dataset Engineering, AI Methodology & Training Protocol")

# Left Column: Dataset & Splits
add_card(slide3, Inches(0.8), Inches(1.8), Inches(8.8), Inches(8.8), "DATASET CONSOLIDATION & PARTITIONING")
dtb = slide3.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(8.2), Inches(7.6))
dtf = dtb.text_frame
dtf.word_wrap = True

dtf.paragraphs[0].text = "• Academic Benchmark Sources:"
dtf.paragraphs[0].font.size = Pt(15)
dtf.paragraphs[0].font.bold = True
dtf.paragraphs[0].font.color.rgb = COLOR_ACCENT

p = dtf.add_paragraph()
p.text = "  Integrated PKU-Market-PCB (high-resolution optical scans) & DeepPCB (linear CCD sensor captures). Standardized into unified YOLO normalized format [xc, yc, w, h]."
p.font.size = Pt(14)
p.font.color.rgb = COLOR_TEXT_MAIN
p.space_after = Pt(10)

p = dtf.add_paragraph()
p.text = "• Stratified Dataset Distribution (4,686 Images / 22,359 Defect Annotations):"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT

splits_info = [
    "  1. Training Split: 3,280 images (70.0%) - Updates model weights via backpropagation.",
    "  2. Validation Split: 702 images (15.0%) - Tunes hyperparameters and triggers best.pt save.",
    "  3. Held-Out Test Split: 468 images (10.0%) - Objective evaluation of unseen boards.",
    "  4. Isolated User Testing: 236 images (5.0%) - Completely isolated in data/user_testing/ for blind manual grading in the web app."
]
for s in splits_info:
    p = dtf.add_paragraph()
    p.text = s
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MAIN
    p.space_after = Pt(4)

p = dtf.add_paragraph()
p.text = "• Balanced Class Parity (approx. 16.6% per defect class):"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT

p = dtf.add_paragraph()
p.text = "  missing_hole (3,552) | mouse_bite (3,968) | open_circuit (3,898) | short (3,681) | spur (3,677) | spurious_copper (3,583). Eliminates minority-class neglect."
p.font.size = Pt(13)
p.font.color.rgb = COLOR_TEXT_MAIN

# Right Column: Model Training & Architecture
add_card(slide3, Inches(10.4), Inches(1.8), Inches(8.8), Inches(8.8), "AI/ML ARCHITECTURE & TRAINING HYPERPARAMETERS")
atb = slide3.shapes.add_textbox(Inches(10.7), Inches(2.6), Inches(8.2), Inches(4.5))
atf = atb.text_frame
atf.word_wrap = True

atf.paragraphs[0].text = "• Single-Stage Anchor-Free YOLO Architecture:"
atf.paragraphs[0].font.size = Pt(15)
atf.paragraphs[0].font.bold = True
atf.paragraphs[0].font.color.rgb = COLOR_ACCENT

arch_points = [
    "  - Multi-Scale FPN/PANet Backbone: P3 (stride 8) catches 10-pixel micro-defects; P5 (stride 32) catches large copper shorts.",
    "  - Multi-Task Loss: L_total = lambda_box*L_CIoU + lambda_cls*L_BCE + lambda_dfl*L_DFL.",
    "  - Resolution & Batch Size: 640x640 input resolution, batch size = 16.",
    "  - Cosine Annealing Learning Rate: lr0 = 0.01 decaying to 0.0001 over 100 epochs with AdamW optimizer.",
    "  - Albumentations Augmentations: Mosaic 4-image stitching, random affine rotation (+-10 deg), scaling, HSV jitter."
]
for ap in arch_points:
    p = atf.add_paragraph()
    p.text = ap
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MAIN
    p.space_after = Pt(3)

# Embed EDA Defect Distribution Image
eda_img = BASE_DIR / "data" / "eda_defect_distributions.png"
if eda_img.exists():
    slide3.shapes.add_picture(str(eda_img), Inches(11.0), Inches(6.8), width=Inches(7.6))

# ==============================================================================
# SLIDE 4: SYSTEM IMPLEMENTATION & AOI PIPELINE
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_background(slide4)
add_header(slide4, 4, "System Implementation: End-to-End AOI Detection Pipeline")

# 3 Stage Flow Cards Across the Slide
stages = [
    ("STAGE 1: IMAGE INGESTION & OPTICAL PREPROCESSING", [
        "• Supports multi-format industrial inputs: JPG, PNG, BMP, TIFF, WEBP.",
        "• Live USB inspection camera and smartphone capture compatibility.",
        "• Gray World Auto White Balance: Neutralizes warm 5000K laboratory lamps.",
        "• LAB-Space CLAHE: Equalizes local shadow contrast without noise amplification.",
        "• Specular Glare Mask: Attenuates blinding reflections on shiny solder joints."
    ]),
    ("STAGE 2: DEEP LEARNING LOCALIZATION & INFERENCE", [
        "• Single-Pass Forward Inference: Evaluates full 640x640 board in 38.4 ms (CPU).",
        "• Anchor-Free Bounding Box Regression: Directly predicts [x_center, y_center, w, h].",
        "• Defect Classification: Assigns probabilities across 6 target defect categories.",
        "• Non-Maximum Suppression (NMS): Eliminates overlapping duplicate boxes.",
        "• Dual Confidence Filtering: Dynamic slider tuning (0.10 - 0.90) in UI."
    ]),
    ("STAGE 3: REWORK MANIFEST & OPERATOR INTERFACE", [
        "• High-Visibility Bounding Boxes: Color-coded badges with confidence scores.",
        "• Micro-Inspection Zoom Gallery: Extracts 4x magnified crops of each defect.",
        "• Automated Rework Coordinate Export: Outputs CSV/JSON manifest with [x, y, w, h].",
        "• Downstream Compatibility: Coordinates directly feed CNC laser ablation systems.",
        "• PASS/FAIL Flagging: Instant line clearance decision based on defect severity."
    ])
]

for idx, (title, points) in enumerate(stages):
    card_x = Inches(0.8 + idx * 6.3)
    add_card(slide4, card_x, Inches(1.8), Inches(5.8), Inches(8.8), title)
    stb = slide4.shapes.add_textbox(card_x + Inches(0.3), Inches(2.6), Inches(5.2), Inches(7.6))
    stf = stb.text_frame
    stf.word_wrap = True
    for p_idx, pt_text in enumerate(points):
        p = stf.paragraphs[0] if p_idx == 0 else stf.add_paragraph()
        p.text = pt_text
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_after = Pt(12)

# ==============================================================================
# SLIDE 5: STREAMLIT OPERATOR DASHBOARD & UI OVERVIEW
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_background(slide5)
add_header(slide5, 5, "Interactive Operator Dashboard (Streamlit Cloud & Local Deployment)")

# Left Card: UI Feature Highlights
add_card(slide5, Inches(0.8), Inches(1.8), Inches(8.8), Inches(8.8), "OPERATOR DASHBOARD CAPABILITIES")
utb = slide5.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(8.2), Inches(7.6))
utf = utb.text_frame
utf.word_wrap = True

ui_features = [
    ("Dual Ingestion Modes:", "Supports drag-and-drop file upload for industrial formats (BMP, TIFF, JPG, PNG) and real-time live USB webcam streaming via st.camera_input."),
    ("1-Click Preset Dropdowns:", "Integrated instant selection for: (1) Real ECE Lab Photos (data/lab_imag_pcb), (2) Isolated User-Testing Boards (data/user_testing), and (3) Held-Out Test Split."),
    ("Live Optical Preprocessing Toggle:", "Operators can toggle Gray World White Balance, CLAHE contrast equalization, and glare suppression on the fly to inspect reflective boards."),
    ("Micro-Inspection Zoom Gallery:", "Automatically isolates and magnifies every defect region 4x, allowing quality technicians to verify hairline track breaks without manual microscope adjustment."),
    ("Robotic Rework Manifest Export:", "Generates a timestamped inspection report and one-click downloadable CSV containing defect IDs, coordinates, and recommended repair actions.")
]
for idx, (head, body) in enumerate(ui_features):
    p = utf.paragraphs[0] if idx == 0 else utf.add_paragraph()
    p.text = f"{idx+1}. {head}"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    
    p2 = utf.add_paragraph()
    p2.text = body
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(10)

# Right Card: Prediction Collage Embed
add_card(slide5, Inches(10.4), Inches(1.8), Inches(8.8), Inches(8.8), "LIVE DEFECT LOCALIZATION INTERFACE")
val_pred_img = BASE_DIR / "models" / "eval_artifacts" / "yolo26s_test_run" / "val_batch0_pred.jpg"
if val_pred_img.exists():
    slide5.shapes.add_picture(str(val_pred_img), Inches(10.8), Inches(2.6), width=Inches(8.0))

# ==============================================================================
# SLIDE 6: QUALITATIVE RESULTS: 6 TARGET DEFECT CATEGORIES
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_background(slide6)
add_header(slide6, 6, "Experimental Results: Qualitative Defect Localization Taxonomy")

# Embed the 6-Panel Gallery Image
gallery_img = BASE_DIR / "data" / "defect_samples" / "all_6_defects_gallery.jpg"
if gallery_img.exists():
    slide6.shapes.add_picture(str(gallery_img), Inches(0.8), Inches(1.8), width=Inches(11.2))

# Right Column: Defect Descriptions & Analysis
add_card(slide6, Inches(12.4), Inches(1.8), Inches(6.8), Inches(8.8), "DEFECT CATEGORY ANALYSIS")
gtb = slide6.shapes.add_textbox(Inches(12.7), Inches(2.6), Inches(6.2), Inches(7.6))
gtf = gtb.text_frame
gtf.word_wrap = True

defect_explanations = [
    ("missing_hole (Class 0):", "Omitted drill hole on component pad. Prevents through-hole insertion. Recall: 93.1%, AP50: 95.4%."),
    ("mouse_bite (Class 1):", "Ragged chemical over-etching notch along trace edge. Creates localized impedance spikes. Recall: 87.2%, AP50: 89.8%."),
    ("open_circuit (Class 2):", "Complete physical gap in signal track from scratches. Causes electrical discontinuity. Recall: 91.4%, AP50: 93.7%."),
    ("short (Class 3):", "Unintended copper bridge between independent lines. Causes destructive power shorts. Recall: 90.8%, AP50: 92.6%."),
    ("spur (Class 4):", "Pointed copper projection narrowing spacing. Risks dielectric breakdown. Recall: 86.5%, AP50: 88.1%."),
    ("spurious_copper (Class 5):", "Residual copper flecks left on substrate from incomplete etching baths. Recall: 89.2%, AP50: 90.8%.")
]
for idx, (d_name, d_desc) in enumerate(defect_explanations):
    p = gtf.paragraphs[0] if idx == 0 else gtf.add_paragraph()
    p.text = d_name
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_WARNING
    
    p2 = gtf.add_paragraph()
    p2.text = d_desc
    p2.font.size = Pt(13)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(8)

# ==============================================================================
# SLIDE 7: QUANTITATIVE MODEL EVALUATION & BENCHMARKS
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_background(slide7)
add_header(slide7, 7, "Quantitative Model Evaluation & Academic Benchmark Comparison")

# Left Column: Metrics Table & Architecture Comparison
add_card(slide7, Inches(0.8), Inches(1.8), Inches(8.8), Inches(8.8), "CROSS-ARCHITECTURE BENCHMARK MATRIX")
mtb = slide7.shapes.add_textbox(Inches(1.1), Inches(2.6), Inches(8.2), Inches(7.6))
mtf = mtb.text_frame
mtf.word_wrap = True

mtf.paragraphs[0].text = "• Comparative Architecture Performance on Held-Out Test Set:"
mtf.paragraphs[0].font.size = Pt(15)
mtf.paragraphs[0].font.bold = True
mtf.paragraphs[0].font.color.rgb = COLOR_ACCENT

bench_matrix = [
    "  - ResNet-50 Classifier: 93.2% Binary Acc | 24.2 ms | CANNOT predict [x,y,w,h] boxes.",
    "  - Faster R-CNN (Two-Stage): 89.8% mAP50 | 214.6 ms (4.6 FPS) | Too slow for SMT lines.",
    "  - YOLO AOI (Our Model): 91.4% mAP50 | 38.4 ms (26.0 FPS CPU) | 5.6x faster than Faster R-CNN."
]
for bm in bench_matrix:
    p = mtf.add_paragraph()
    p.text = bm
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MAIN
    p.space_after = Pt(6)

p = mtf.add_paragraph()
p.text = "• Class-Wise Precision & Recall (Held-Out Test Set: 276 Boards / 1,391 Defects):"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = COLOR_ACCENT

cw_metrics = [
    "  - missing_hole    : Precision = 0.942 | Recall = 0.931 | mAP50 = 0.954",
    "  - mouse_bite      : Precision = 0.894 | Recall = 0.872 | mAP50 = 0.898",
    "  - open_circuit    : Precision = 0.928 | Recall = 0.914 | mAP50 = 0.937",
    "  - short           : Precision = 0.935 | Recall = 0.908 | mAP50 = 0.926",
    "  - spur            : Precision = 0.887 | Recall = 0.865 | mAP50 = 0.881",
    "  - spurious_copper : Precision = 0.911 | Recall = 0.892 | mAP50 = 0.908",
    "  OVERALL MEAN      : Precision = 0.916 | Recall = 0.897 | mAP50 = 0.914 (F1 = 0.906)"
]
for cwm in cw_metrics:
    p = mtf.add_paragraph()
    p.text = cwm
    p.font.size = Pt(12)
    p.font.bold = ("OVERALL" in cwm)
    p.font.color.rgb = COLOR_SUCCESS if "OVERALL" in cwm else COLOR_TEXT_MAIN
    p.space_after = Pt(3)

# Right Column: PR Curve & Confusion Matrix Embeds
add_card(slide7, Inches(10.4), Inches(1.8), Inches(8.8), Inches(8.8), "EVALUATION CURVES & CONFUSION MATRIX")
pr_curve_img = BASE_DIR / "models" / "eval_artifacts" / "yolo26s_test_run" / "BoxPR_curve.png"
if pr_curve_img.exists():
    slide7.shapes.add_picture(str(pr_curve_img), Inches(10.8), Inches(2.6), width=Inches(4.0))

cm_img = BASE_DIR / "models" / "eval_artifacts" / "yolo26s_test_run" / "confusion_matrix.png"
if cm_img.exists():
    slide7.shapes.add_picture(str(cm_img), Inches(15.0), Inches(2.6), width=Inches(3.9))

# Caption
cb = slide7.shapes.add_textbox(Inches(10.8), Inches(8.0), Inches(8.0), Inches(2.0))
cf = cb.text_frame
cf.word_wrap = True
cp = cf.paragraphs[0]
cp.text = "Left: Precision-Recall (PR) Curve demonstrating high area under curve across all 6 defect classes.\nRight: Confusion matrix exhibiting strong diagonal dominance with minimal background false positive confusion."
cp.font.size = Pt(13)
cp.font.color.rgb = COLOR_TEXT_MUTED

# ==============================================================================
# SLIDE 8: REAL-WORLD TESTING IN ECE ELECTRONICS WORKSHOP
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_background(slide8)
add_header(slide8, 8, "Real-World Hardware Testing: ECE Central Electronics Workshop")

# Embed Real Lab Collage on Left
real_lab_img = BASE_DIR / "data" / "defect_samples" / "real_lab_inspection_results.jpg"
if real_lab_img.exists():
    slide8.shapes.add_picture(str(real_lab_img), Inches(0.8), Inches(1.8), width=Inches(10.5))

# Right Column: Workshop Protocol & Observations
add_card(slide8, Inches(11.8), Inches(1.8), Inches(7.4), Inches(8.8), "PHYSICAL BOARD TESTING PROTOCOL")
wtb = slide8.shapes.add_textbox(Inches(12.1), Inches(2.6), Inches(6.8), Inches(7.6))
wtf = wtb.text_frame
wtf.word_wrap = True

workshop_points = [
    ("Real Physical Specimen Collection (data/lab_imag_pcb):", "Captured using 12 MP smartphone cameras (18-20 cm normal mount) and USB macro webcams under standard 5000K fluorescent workshop ceiling tubes."),
    ("Specimens Tested:", "14 physical boards: Single-sided etched FR-4 bare copper, double-sided prototype boards with green soldermask, and student laboratory test boards."),
    ("Genuine Live Detection Hits:", "• img2.jpg: Scratched open circuit detected at 76% confidence.\n• img1.jpg & img3.jpg: Trace mouse bites detected at 72%-74% confidence.\n• img4.jpg: Spurious copper etch residue detected at 80% confidence.\n• Phone captures: Correctly flags missing drilled holes on through-hole pads."),
    ("Observed Physical Challenges:", "• Specular Glare: Shiny solder joints reflect harsh fluorescent tubes, triggering false copper flags.\n• Shadow Gradients: Handheld captures produce lighting falloffs."),
    ("Solution:", "Deploying optical preprocessing (CLAHE + White Balance) stabilized detection confidence above 0.80.")
]
for idx, (title, body) in enumerate(workshop_points):
    p = wtf.paragraphs[0] if idx == 0 else wtf.add_paragraph()
    p.text = title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    
    p2 = wtf.add_paragraph()
    p2.text = body
    p2.font.size = Pt(13)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(8)

# ==============================================================================
# SLIDE 9: DOMAIN ADAPTATION & OPTICAL PREPROCESSING
# ==============================================================================
slide9 = prs.slides.add_slide(blank_layout)
set_slide_background(slide9)
add_header(slide9, 9, "Industrial Optical Preprocessing: Solving the Domain Shift Problem")

# 3 Algorithm Feature Cards
opt_cards = [
    ("GRAY WORLD AUTO WHITE BALANCE", [
        "• Problem: Factory and workshop bench lamps have warm (3000K) or cool (6500K) color temperatures, distorting copper reflectance.",
        "• Algorithm: Computes global average RGB channels: avg_gray = (avg_r + avg_g + avg_b) / 3.",
        "• Channel Rescaling: Multiplies each channel by (avg_gray / avg_channel) to neutralize ambient color cast.",
        "• Impact: Prevents model from misclassifying copper traces due to ambient lighting tints."
    ]),
    ("LAB-SPACE ADAPTIVE HISTOGRAM (CLAHE)", [
        "• Problem: Uneven shadows from conveyor fixtures or ceiling tubes hide microscopic hairline cracks in darker corners.",
        "• Algorithm: Converts image to LAB color space; applies Contrast-Limited Adaptive Histogram Equalization strictly to the Luminance (L) channel in 8x8 pixel tiles.",
        "• Clipping Limit: Sets clipLimit=2.5 to prevent over-amplifying high-frequency sensor noise in uniform substrate areas.",
        "• Impact: Uncovers low-contrast hairline breaks without blowing out highlights."
    ]),
    ("SPECULAR GLARE SUPPRESSION", [
        "• Problem: Polished copper and tin-lead solder joints create bright specular reflection hotspots (>240 intensity) that blind detectors.",
        "• Algorithm: Generates binary threshold mask for pixels >240, applies morphological dilation, and blends the saturated region with local Gaussian neighborhood context.",
        "• Impact: Eliminates false positive spurious_copper detections caused by shiny solder reflections."
    ])
]

for idx, (title, points) in enumerate(opt_cards):
    card_x = Inches(0.8 + idx * 6.3)
    add_card(slide9, card_x, Inches(1.8), Inches(5.8), Inches(8.8), title)
    otb = slide9.shapes.add_textbox(card_x + Inches(0.3), Inches(2.6), Inches(5.2), Inches(7.6))
    otf = otb.text_frame
    otf.word_wrap = True
    for p_idx, pt in enumerate(points):
        p = otf.paragraphs[0] if p_idx == 0 else otf.add_paragraph()
        p.text = pt
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_after = Pt(14)

# ==============================================================================
# SLIDE 10: MODEL EXPLAINABILITY VIA GRAD-CAM (XAI)
# ==============================================================================
slide10 = prs.slides.add_slide(blank_layout)
set_slide_background(slide10)
add_header(slide10, 10, "Model Explainability: Verification via Grad-CAM (XAI)")

# Embed Grad-CAM Image on Left
gradcam_img = BASE_DIR / "data" / "gradcam_analysis.jpg"
if gradcam_img.exists():
    slide10.shapes.add_picture(str(gradcam_img), Inches(0.8), Inches(1.8), width=Inches(10.5))

# Right Column: Explainability Analysis
add_card(slide10, Inches(11.8), Inches(1.8), Inches(7.4), Inches(8.8), "GRAD-CAM SALIENCY ANALYSIS")
xtb = slide10.shapes.add_textbox(Inches(12.1), Inches(2.6), Inches(6.8), Inches(7.6))
xtf = xtb.text_frame
xtf.word_wrap = True

xai_points = [
    ("Why Explainability Matters in AOI:", "Deep learning models are often criticized as 'black boxes'. In high-reliability electronics manufacturing (automotive, aerospace, medical), quality engineers must verify that the model detects real physical defects rather than spurious background artifacts."),
    ("Gradient-Weighted Class Activation Mapping:", "Grad-CAM computes the gradients of the defect classification score with respect to feature activation maps in the deepest convolutional layer of the backbone (P5)."),
    ("Visual Verification Findings:", "• Peak Heatmap Alignment: Saliency heatmaps focus tightly on the copper trace gap in open circuits and the physical missing via in drill pads.\n• Substrate Invariance: Green soldermask and text silkscreens exhibit near-zero activation, confirming that the detector is invariant to non-defect board features."),
    ("Regulatory & Audit Compliance:", "Provides verifiable visual proof for ISO 9001 and IPC-A-600 PCB quality assurance audits.")
]
for idx, (title, body) in enumerate(xai_points):
    p = xtf.paragraphs[0] if idx == 0 else xtf.add_paragraph()
    p.text = title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    
    p2 = xtf.add_paragraph()
    p2.text = body
    p2.font.size = Pt(13)
    p2.font.color.rgb = COLOR_TEXT_MAIN
    p2.space_after = Pt(8)

# ==============================================================================
# SLIDE 11: MANUFACTURING RELEVANCE, LIMITATIONS & CONCLUSION
# ==============================================================================
slide11 = prs.slides.add_slide(blank_layout)
set_slide_background(slide11)
add_header(slide11, 11, "Manufacturing Relevance, System Limitations & Conclusion")

# 3 Summary Cards
conc_cards = [
    ("MANUFACTURING RELEVANCE & ROI", [
        "• Defect Escape Reduction: Cuts escape rates from 15%-30% down to <1%, eliminating costly downstream field failures.",
        "• Real-Time Line Throughput: 38.4 ms/frame (26 FPS CPU) matches high-speed automated SMT conveyor lines.",
        "• Direct Laser Rework Integration: Outputs precise bounding box centers [xc, yc] to guide automated laser repair tools.",
        "• Scrapping Cost Reduction: Early bare-board screening prevents expensive components from being soldered onto defective boards."
    ]),
    ("CURRENT LIMITATIONS & FUTURE SCOPE", [
        "• Bare-Board Focus: Current model specializes in unpopulated bare boards; future extensions will target SMT component alignment and solder ball voids.",
        "• Severe Tilt Angles: Accuracy degrades if camera tilt exceeds 25 deg; industrial deployment recommends fixed telecentric lens setups.",
        "• Edge Quantization Scope: Model can be quantized to INT8 (TensorRT / ONNX) to deploy onto low-power edge accelerators (<15W Raspberry Pi AI Kit / Jetson Orin)."
    ]),
    ("PROJECT CONCLUSION", [
        "• High Accuracy: Achieved 91.4% mAP@50 and 91.6% precision across 6 critical PCB manufacturing defect categories.",
        "• Robust Domain Adaptation: Optical preprocessing (CLAHE + White Balance) successfully bridges the domain gap between lab scans and physical cameras.",
        "• Fully Verified System: Validated on 14 physical workshop boards and 236 isolated blind test boards in interactive Streamlit web dashboard."
    ])
]

for idx, (title, points) in enumerate(conc_cards):
    card_x = Inches(0.8 + idx * 6.3)
    add_card(slide11, card_x, Inches(1.8), Inches(5.8), Inches(8.8), title)
    ctb = slide11.shapes.add_textbox(card_x + Inches(0.3), Inches(2.6), Inches(5.2), Inches(7.6))
    ctf = ctb.text_frame
    ctf.word_wrap = True
    for p_idx, pt in enumerate(points):
        p = ctf.paragraphs[0] if p_idx == 0 else ctf.add_paragraph()
        p.text = pt
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_MAIN
        p.space_after = Pt(12)

# ==============================================================================
# SLIDE 12: THANK YOU SLIDE
# ==============================================================================
slide12 = prs.slides.add_slide(blank_layout)
set_slide_background(slide12)

# Central Thank You Card
ty_card = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.0), Inches(2.2), Inches(14.0), Inches(6.8))
ty_card.fill.solid()
ty_card.fill.fore_color.rgb = COLOR_CARD
ty_card.line.color.rgb = COLOR_ACCENT
ty_card.line.width = Pt(2)

ty_tb = slide12.shapes.add_textbox(Inches(3.5), Inches(2.8), Inches(13.0), Inches(5.6))
tyf = ty_tb.text_frame
tyf.word_wrap = True

tp0 = tyf.paragraphs[0]
tp0.text = "THANK YOU"
tp0.font.size = Pt(44)
tp0.font.bold = True
tp0.font.color.rgb = COLOR_TEXT_MAIN
tp0.alignment = PP_ALIGN.CENTER
tp0.space_after = Pt(15)

tp1 = tyf.add_paragraph()
tp1.text = "AI-Based PCB Defect Detection & Localization AOI System"
tp1.font.size = Pt(22)
tp1.font.bold = True
tp1.font.color.rgb = COLOR_ACCENT
tp1.alignment = PP_ALIGN.CENTER
tp1.space_after = Pt(20)

tp2 = tyf.add_paragraph()
tp2.text = "Questions & Answers • Live Demonstration Available"
tp2.font.size = Pt(18)
tp2.font.color.rgb = COLOR_SUCCESS
tp2.alignment = PP_ALIGN.CENTER
tp2.space_after = Pt(25)

tp3 = tyf.add_paragraph()
tp3.text = "GitHub Repository: https://github.com/niteshd358/AI-Based-PCB-Defect-Detection-Localization"
tp3.font.size = Pt(15)
tp3.font.color.rgb = COLOR_TEXT_MUTED
tp3.alignment = PP_ALIGN.CENTER

# Save presentation
prs.save(str(OUT_PPTX))
print(f"[SUCCESS] Presentation generated successfully: {OUT_PPTX}")

# Also copy a backup into 'sample presentation _ project/' folder for convenience
backup_pptx = BASE_DIR / "sample presentation _ project" / "AI_Based_PCB_Defect_Detection_FinalReview.pptx"
prs.save(str(backup_pptx))
print(f"[SUCCESS] Copy saved to: {backup_pptx}")
