import numpy as np
from scipy.interpolate import interp1d


# =========================
# LINEAR INTERPOLATION
# =========================
def linear_reconstruct(corrupted, mask):
    reconstructed = corrupted.copy()
    D = corrupted.shape[0]

    for i in range(D):
        if not mask[i]:

            prev_idx = i - 1
            next_idx = i + 1

            # find nearest available previous slice
            while prev_idx >= 0 and not mask[prev_idx]:
                prev_idx -= 1

            # find nearest available next slice
            while next_idx < D and not mask[next_idx]:
                next_idx += 1

            if prev_idx >= 0 and next_idx < D:
                reconstructed[i] = (corrupted[prev_idx] + corrupted[next_idx]) / 2

    return reconstructed


# =========================
# B-SPLINE INTERPOLATION
# =========================
def bspline_reconstruct(corrupted, mask):
    D, H, W, C = corrupted.shape
    reconstructed = corrupted.copy()

    x = np.arange(D)
    known_x = x[mask]

    for c in range(C):
        for i in range(H):
            for j in range(W):

                y = corrupted[:, i, j, c]
                known_y = y[mask]

                if len(known_y) < 4:
                    continue

                try:
                    f = interp1d(
                        known_x,
                        known_y,
                        kind='cubic',
                        fill_value="extrapolate"
                    )
                    reconstructed[:, i, j, c] = f(x)

                except:
                    continue

    return reconstructed


# =========================
# TEST (RUN THIS FILE ONLY)
# =========================
if __name__ == "__main__":
    import os
    from step2_missing_detection import simulate_missing_slices

    DATA_PATH = "processed_data"

    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))

    corrupted, mask = simulate_missing_slices(volume)

    linear = linear_reconstruct(corrupted, mask)
    bspline = bspline_reconstruct(corrupted, mask)

    print("Original:", volume.shape)
    print("Linear:", linear.shape)
    print("B-spline:", bspline.shape)