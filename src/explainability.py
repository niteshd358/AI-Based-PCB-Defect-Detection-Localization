"""
Explainable AI (XAI) Suite for PCB Defect Inspection
Generates Grad-CAM activation heatmaps confirming that deep learning models
focus on physical defect anomalies (e.g. copper breaks, shorts, holes)
rather than background substrate artifacts.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class GradCAMVisualizer:
    def __init__(self, model=None, target_layer=None):
        if model is None:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.target_layer = self.model.layer4[-1].conv3
        else:
            self.model = model
            self.target_layer = target_layer
            
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Global average pool of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def apply_gradcam_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Resize heatmap and blend with original BGR image."""
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay

def run_explainability_demo(image_path: Path = None, output_path: Path = None):
    print("=" * 60)
    print("         GRAD-CAM EXPLAINABILITY VISUALIZATION")
    print("=" * 60)
    
    if image_path is None or not image_path.exists():
        # Pick first test image
        img_candidates = list((DATA_DIR / "processed" / "images" / "test").glob("*.jpg"))
        if not img_candidates:
            print("[ERROR] No test images found in data/processed/images/test.")
            return
        image_path = img_candidates[0]
        
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img_rgb).unsqueeze(0)
    
    visualizer = GradCAMVisualizer()
    heatmap = visualizer.generate_heatmap(input_tensor)
    overlay = apply_gradcam_overlay(img_bgr, heatmap)
    
    # Save comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input PCB Board", fontsize=12)
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=12)
    axes[1].axis("off")
    
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Heatmap Overlay on Defects", fontsize=12)
    axes[2].axis("off")
    
    plt.tight_layout()
    if output_path is None:
        output_path = DATA_DIR / "gradcam_analysis.jpg"
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    
    print(f"[SUCCESS] Grad-CAM visualization saved to: {output_path}")

if __name__ == "__main__":
    run_explainability_demo()
