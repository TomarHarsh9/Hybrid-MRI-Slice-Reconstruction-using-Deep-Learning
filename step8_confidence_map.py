import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from step2_missing_detection import simulate_missing_slices
from step3_reconstruction import linear_reconstruct, bspline_reconstruct
from step4_weight import compute_weight_map
from step5_ml_model import load_model, generate_ml_volume
from step6_fusion import adaptive_fusion

# FIX ML VOLUME (SAFE, NO ARTIFACTS)
def _fix_ml_volume(ml_volume, target_shape):
    D, H, W, C = target_shape
    fixed = []
    for i in range(D):
        s = None
        if isinstance(ml_volume, (list, tuple)) and i < len(ml_volume):
            s = ml_volume[i]
        elif isinstance(ml_volume, np.ndarray) and i < ml_volume.shape[0]:
            s = ml_volume[i]

        if s is None:
            fixed.append(np.zeros((H, W, C), dtype=np.float32))
            continue

        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 2: s = s[..., None]

        if s.shape != (H, W, C):
            fixed.append(np.zeros((H, W, C), dtype=np.float32))
        else:
            fixed.append(s)
    return np.stack(fixed, axis=0).astype(np.float32)

# STEP 8: CONFIDENCE MAP (REFINED)
def compute_confidence(linear, bspline, ml_volume, mask):
    """
    Refined confidence:
    - norm_var: uses 99th percentile to be robust against noise.
    - norm_grad: uses Laplacian to find uncertainty at fine tissue edges.
    - formula: uses exponential decay for a smoother medical heatmap.
    """
    D, H, W, C = linear.shape
    ml = _fix_ml_volume(ml_volume, (D, H, W, C))

    # Use channel 0
    lin = linear[..., 0].astype(np.float32)
    bsp = bspline[..., 0].astype(np.float32)
    mlc = ml[..., 0].astype(np.float32)

    # 1. VARIANCE (Disagreement)
    stack = np.stack([lin, bsp, mlc], axis=0)
    var = np.var(stack, axis=0)
    
    # Use percentile to avoid outliers darkening the whole map
    var_limit = np.percentile(var, 99) + 1e-8
    norm_var = np.clip(var / var_limit, 0, 1)

    # 2. LAPLACIAN (Structural Uncertainty)
    # Detects high-frequency details where reconstruction is hardest
    grad_list = []
    for i in range(D):
        g = cv2.Laplacian(lin[i], cv2.CV_32F)
        grad_list.append(np.abs(g))
    
    grad = np.stack(grad_list, axis=0)
    grad_limit = np.percentile(grad, 99) + 1e-8
    norm_grad = np.clip(grad / grad_limit, 0, 1)

    # 3. FINAL FORMULA
    # Confidence is high where methods agree and structures are smooth
    confidence_maps = np.exp(-(0.5 * norm_var + 0.1 * norm_grad))

    # MASK HANDLING: Original data is 100% certain
    if mask is not None:
        confidence_maps[mask] = 1.0

    return np.clip(confidence_maps, 0.0, 1.0).astype(np.float32)

def _normalize(img):
    img = img.astype(np.float32)
    mn, mx = np.min(img), np.max(img)
    if mx - mn < 1e-8: return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def visualize(original, reconstructed, confidence):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(_normalize(original), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(_normalize(reconstructed), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(confidence, cmap='hot')
    plt.title("Confidence Map")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = "processed_data"
    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))
    
    corrupted, mask = simulate_missing_slices(volume)
    linear = linear_reconstruct(corrupted, mask)
    bspline = bspline_reconstruct(corrupted, mask)
    weights = compute_weight_map(corrupted, mask)
    model = load_model()
    ml_volume = generate_ml_volume(model, corrupted, mask)
    final = adaptive_fusion(linear, bspline, ml_volume, weights, mask)
    
    confidence_maps = compute_confidence(linear, bspline, ml_volume, mask)

    # SELECT BEST SLICE BY ENERGY
    missing_indices = np.where(mask == False)[0]
    slice_energy = np.sum(volume[..., 0], axis=(1, 2))
    idx = missing_indices[np.argmax(slice_energy[missing_indices])]
    
    visualize(volume[idx, :, :, 0], final[idx, :, :, 0], confidence_maps[idx])