"""
Real-World Industrial Camera Preprocessing & Lighting Normalization Suite
Adapts live desk camera, smartphone, and USB microscope captures to match
the training distribution of flatbed optical scans:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for uneven shadows
2. Automatic White Balance (Gray World) for color temperature variations
3. Glare & Specular Reflection attenuation on shiny copper/solder
4. Dynamic Brightness, Contrast & Unsharp Mask sharpening
"""

import cv2
import numpy as np

def apply_auto_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    """Gray World White Balance algorithm to remove warm/cool indoor lighting tints."""
    result = image_bgr.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # Scale channels
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-6)), 0, 255)
    
    return result.astype(np.uint8)

def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """Adaptive histogram equalization in LAB color space to normalize shadows & glare."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def suppress_glare(image_bgr: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Attenuate blown-out specular highlights on shiny copper traces and solder pads."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Inpaint or blend highlight regions with slight local blur
    if np.sum(mask) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
        mask_3ch = cv2.merge([mask_dilated, mask_dilated, mask_dilated]) / 255.0
        result = (image_bgr * (1.0 - 0.4 * mask_3ch) + blurred * (0.4 * mask_3ch)).astype(np.uint8)
        return result
    return image_bgr

def adjust_brightness_contrast(image_bgr: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """Adjust exposure parameters."""
    return cv2.convertScaleAbs(image_bgr, alpha=contrast, beta=brightness)

def unsharp_mask(image_bgr: np.ndarray, sigma: float = 1.0, strength: float = 1.2) -> np.ndarray:
    """Sharpen slightly out-of-focus camera captures to accentuate micro trace defects."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def enhance_pcb_image(
    image_bgr: np.ndarray,
    enable_wb: bool = True,
    enable_clahe: bool = True,
    clahe_clip: float = 2.5,
    enable_glare_suppression: bool = True,
    enable_sharpening: bool = False,
    brightness: int = 0,
    contrast: float = 1.0
) -> np.ndarray:
    """Full preprocessing pipeline for raw camera captures."""
    res = image_bgr.copy()
    
    # 1. White balance (neutralizes warm desk lamps)
    if enable_wb:
        res = apply_auto_white_balance(res)
        
    # 2. Glare suppression
    if enable_glare_suppression:
        res = suppress_glare(res)
        
    # 3. CLAHE local contrast normalization
    if enable_clahe:
        res = apply_clahe(res, clip_limit=clahe_clip)
        
    # 4. Exposure tuning
    if brightness != 0 or contrast != 1.0:
        res = adjust_brightness_contrast(res, brightness=brightness, contrast=contrast)
        
    # 5. Sharpening
    if enable_sharpening:
        res = unsharp_mask(res)
        
    return res
