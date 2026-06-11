import os
import numpy as np

from step2_missing_detection import simulate_missing_slices
from step3_reconstruction import linear_reconstruct, bspline_reconstruct
from step4_weight import compute_weight_map
from step5_ml_model import load_model, generate_ml_volume


# =========================
# FUSION FUNCTION
# =========================
def adaptive_fusion(linear, bspline, ml_volume, weights, mask):
    D, H, W, C = linear.shape

    final = linear.copy()

    for i in range(D):

        if mask[i]:
            continue

        w = weights[i]
        ml_slice = ml_volume[i]

        if w is None or ml_slice is None:
            continue

        # expand dims
        w = w[:, :, None]                 # (H,W,1)
        ml_slice = ml_slice[:, :, None]   # (H,W,1)

        # expand ML to 4 channels
        ml_slice = np.repeat(ml_slice, C, axis=2)

        # classical interpolation
        classical = (linear[i] + bspline[i]) / 2

        # fusion equation
        final[i] = w * ml_slice + (1 - w) * classical

    return final


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    DATA_PATH = "processed_data"

    # load one patient
    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))

    print("Loaded:", file)
    print("Shape:", volume.shape)

    # Step 2
    corrupted, mask = simulate_missing_slices(volume)

    # Step 3
    linear = linear_reconstruct(corrupted, mask)
    bspline = bspline_reconstruct(corrupted, mask)

    # Step 4
    weights = compute_weight_map(corrupted, mask)

    # Step 5
    model = load_model()
    ml_volume = generate_ml_volume(model, corrupted, mask)

    # Step 6
    final = adaptive_fusion(linear, bspline, ml_volume, weights, mask)

    print("Fusion complete:", final.shape)