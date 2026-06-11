from fastapi import FastAPI, UploadFile, File
import numpy as np
import base64
import cv2
import io
from fastapi.middleware.cors import CORSMiddleware

# ===== IMPORT PIPELINE =====
from step2_missing_detection import simulate_missing_slices
from step3_reconstruction import linear_reconstruct, bspline_reconstruct
from step4_weight import compute_weight_map
from step5_ml_model import load_model, generate_ml_volume
from step6_fusion import adaptive_fusion
from step8_confidence_map import compute_confidence

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

def to_base64(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        volume = np.load(io.BytesIO(contents))

        # 1. RUN PIPELINE
        corrupted, mask = simulate_missing_slices(volume)
        linear = linear_reconstruct(corrupted, mask)
        bspline = bspline_reconstruct(corrupted, mask)
        weights = compute_weight_map(corrupted, mask)
        ml_volume = generate_ml_volume(model, corrupted, mask)
        final = adaptive_fusion(linear, bspline, ml_volume, weights, mask)
        confidence_full = compute_confidence(linear, bspline, ml_volume, mask)

        # 2. AUTOMATED SLICE SELECTION
        # Find missing slices and sort them by pixel energy (where the brain is)
        missing_indices = np.where(mask == False)[0]
        if len(missing_indices) == 0:
            return {"error": "No missing slices found"}
        
        slice_energies = np.sum(volume[..., 0], axis=(1, 2))
        # Get top 5 missing slices with most content
        top_indices = missing_indices[np.argsort(slice_energies[missing_indices])[::-1][:5]]
        # Sort them numerically so they appear in order
        top_indices = sorted(top_indices)

        results = []
        for idx in top_indices:
            orig_slice = volume[idx, ..., 0]
            recon_slice = final[idx, ..., 0]
            
            # 3. METRICS (NORMALIZED)
            o = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)
            r = (recon_slice - recon_slice.min()) / (recon_slice.max() - recon_slice.min() + 1e-8)
            mse = np.mean((o - r) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            
            ssim_val = np.corrcoef(o.flatten(), r.flatten())[0, 1]
            if np.isnan(ssim_val): ssim_val = 0.0

            # 4. COLOR HEATMAP
            conf_raw = confidence_full[idx]
            c_min, c_max = conf_raw.min(), conf_raw.max()
            conf_norm = (conf_raw - c_min) / (c_max - c_min + 1e-8)       
            conf_img = (conf_raw * 255).astype(np.uint8)
            conf_color = cv2.applyColorMap(conf_img, cv2.COLORMAP_HOT)
            
            _, buf = cv2.imencode('.png', conf_color)
            confidence_base64 = base64.b64encode(buf).decode('utf-8')
            results.append({
                "slice_index": int(idx),
                "psnr": float(psnr),
                "ssim": float(ssim_val),
                "original": to_base64(orig_slice),
                "reconstructed": to_base64(recon_slice),
                "confidence": base64.b64encode(buf).decode('utf-8')
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}