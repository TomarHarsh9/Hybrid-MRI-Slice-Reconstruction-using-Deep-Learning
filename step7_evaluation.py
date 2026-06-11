import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from step2_missing_detection import simulate_missing_slices
from step3_reconstruction import linear_reconstruct, bspline_reconstruct
from step4_weight import compute_weight_map
from step5_ml_model import load_model, generate_ml_volume
from step6_fusion import adaptive_fusion


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(volume, linear, bspline, ml_volume, final, mask):

    psnr_scores = {
        "linear": [],
        "bspline": [],
        "ml": [],
        "fusion": []
    }

    ssim_scores = {
        "linear": [],
        "bspline": [],
        "ml": [],
        "fusion": []
    }

    D = volume.shape[0]

    for i in range(D):

        if mask[i]:  # skip known slices
            continue

        gt = volume[i, :, :, 0]   # use T1c channel

        lin = linear[i, :, :, 0]
        bs = bspline[i, :, :, 0]
        ml = ml_volume[i]
        fu = final[i, :, :, 0]

        # skip empty slices
        if np.std(gt) < 1e-5:
            continue

        # ---- PSNR ----
        psnr_scores["linear"].append(psnr(gt, lin, data_range=1))
        psnr_scores["bspline"].append(psnr(gt, bs, data_range=1))
        psnr_scores["ml"].append(psnr(gt, ml, data_range=1))
        psnr_scores["fusion"].append(psnr(gt, fu, data_range=1))

        # ---- SSIM ----
        ssim_scores["linear"].append(ssim(gt, lin, data_range=1))
        ssim_scores["bspline"].append(ssim(gt, bs, data_range=1))
        ssim_scores["ml"].append(ssim(gt, ml, data_range=1))
        ssim_scores["fusion"].append(ssim(gt, fu, data_range=1))

    return psnr_scores, ssim_scores


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    DATA_PATH = "processed_data"

    file = os.listdir(DATA_PATH)[0]
    volume = np.load(os.path.join(DATA_PATH, file))

    print("Loaded:", file)

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

    # Step 7
    psnr_scores, ssim_scores = evaluate(
        volume, linear, bspline, ml_volume, final, mask
    )

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n===== FINAL RESULTS =====")

    for method in psnr_scores:
        print(f"\n{method.upper()}:")

        print("PSNR:", np.mean(psnr_scores[method]))
        print("SSIM:", np.mean(ssim_scores[method]))