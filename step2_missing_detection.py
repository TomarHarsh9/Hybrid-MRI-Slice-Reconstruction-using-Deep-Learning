import numpy as np


# =========================
# STEP 2: MISSING SLICE DETECTION
# =========================

def simulate_missing_slices(volume, step=3):
    """
    Simulate missing slices

    Args:
        volume: (D, H, W, C)
        step: remove every 'step' slice

    Returns:
        corrupted_volume
        mask (True = available, False = missing)
    """

    corrupted = volume.copy()

    # True = available, False = missing
    mask = np.ones(volume.shape[0], dtype=bool)

    for i in range(0, volume.shape[0], step):
        corrupted[i] = 0
        mask[i] = False

    return corrupted, mask


# =========================
# TEST (RUN THIS FILE ALONE)
# =========================
if __name__ == "__main__":
    import os

    DATA_PATH = "processed_data"

    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))

    corrupted, mask = simulate_missing_slices(volume)

    print("Original shape:", volume.shape)
    print("Corrupted shape:", corrupted.shape)

    print("Total slices:", len(mask))
    print("Missing slices:", np.sum(mask == False))