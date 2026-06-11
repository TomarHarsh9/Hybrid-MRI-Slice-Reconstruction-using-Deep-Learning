import os
import numpy as np
import cv2

from step2_missing_detection import simulate_missing_slices


# =========================
# GRADIENT FUNCTION
# =========================
def compute_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)
    return grad


# =========================
# WEIGHT MAP
# =========================
def compute_weight_map(corrupted, mask):
    D, H, W, C = corrupted.shape

    weights = [None] * D

    for i in range(D):

        if mask[i]:
            continue   # skip known slices

        # find previous available slice
        prev_idx = i - 1
        while prev_idx >= 0 and not mask[prev_idx]:
            prev_idx -= 1

        # find next available slice
        next_idx = i + 1
        while next_idx < D and not mask[next_idx]:
            next_idx += 1

        if prev_idx < 0 or next_idx >= D:
            continue

        # use T1c (channel 0)
        prev_slice = corrupted[prev_idx, :, :, 0]
        next_slice = corrupted[next_idx, :, :, 0]

        grad_prev = compute_gradient(prev_slice)
        grad_next = compute_gradient(next_slice)

        grad = (grad_prev + grad_next) / 2

        # normalize safely
        max_val = np.max(grad) + 1e-8
        w = grad / max_val

        weights[i] = w

    return weights


# =========================
# TEST
# =========================
if __name__ == "__main__":
    DATA_PATH = "processed_data"

    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))

    corrupted, mask = simulate_missing_slices(volume)

    weights = compute_weight_map(corrupted, mask)

    print("Step 4 completed: Weight maps generated")