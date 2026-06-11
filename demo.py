import numpy as np
import matplotlib.pyplot as plt

from step2_missing_detection import simulate_missing_slices
from step3_reconstruction import linear_reconstruct, bspline_reconstruct
from step4_weight import compute_weight_map
from step5_ml_model import load_model, generate_ml_volume
from step6_fusion import adaptive_fusion
from step8_confidence_map import compute_confidence

# =========================
# LOAD DATA
# =========================
path = "processed_data/BraTS-GLI-00005-100.npy"
volume = np.load(path)

print("Loaded shape:", volume.shape)

# =========================
# STEP 2 — Missing slices
# =========================
corrupted, mask = simulate_missing_slices(volume)

# =========================
# STEP 3 — Reconstruction
# =========================
linear = linear_reconstruct(corrupted, mask)
bspline = bspline_reconstruct(corrupted, mask)

# =========================
# STEP 4 — Weights
# =========================
weights = compute_weight_map(corrupted, mask)

# FIX weights shape
D, H, W, C = linear.shape
fixed_weights = []

for i in range(len(weights)):
    w = weights[i]

    if w is None:
        w = np.ones((H, W, C))
    else:
        w = np.array(w)

    if w.ndim == 0:
        w = np.ones((H, W, C))

    elif w.ndim == 2:
        w = np.expand_dims(w, -1)

    if w.ndim == 3 and w.shape[-1] == 1:
        w = np.repeat(w, C, axis=-1)

    if w.shape != (H, W, C):
        w = np.ones((H, W, C))

    fixed_weights.append(w)

fixed_weights = np.stack(fixed_weights, axis=0)

# =========================
# STEP 5 — ML
# =========================
model = load_model()
ml_volume = generate_ml_volume(model, corrupted, mask)

# FIX ML shape
ml_fixed = []

for i in range(D):
    s = ml_volume[i]

    if s is None:
        s = np.zeros((H, W, C))

    s = np.array(s)

    if s.ndim == 2:
        s = np.expand_dims(s, -1)

    if s.ndim == 3 and s.shape[-1] == 1:
        s = np.repeat(s, C, axis=-1)

    if s.shape != (H, W, C):
        s = np.zeros((H, W, C))

    ml_fixed.append(s)

ml_fixed = np.stack(ml_fixed, axis=0)

# =========================
# STEP 6 — Fusion
# =========================
final = adaptive_fusion(linear, bspline, ml_fixed, fixed_weights, mask)

# =========================
# STEP 8 — Confidence
# =========================
confidence = compute_confidence(linear, bspline, ml_fixed, mask)

# Normalize
confidence = confidence - confidence.min()
confidence = confidence / (confidence.max() + 1e-8)

# =========================
# SHOW 5 SLICES
# =========================
indices = [40, 80, 100, 120, 150]

for idx in indices:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(volume[idx, :, :, 0], cmap='gray')
    ax[0].set_title("Original")

    ax[1].imshow(final[idx, :, :, 0], cmap='gray')
    ax[1].set_title("Reconstructed")

    ax[2].imshow(confidence[idx, :, :, 0], cmap='hot')
    ax[2].set_title("Confidence")

    for a in ax:
        a.axis("off")

    plt.show()