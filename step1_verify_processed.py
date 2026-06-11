import os
import numpy as np
import nibabel as nib

# =========================
# CONFIG
# =========================

BASE_PATH = r"E:\DIP_Project\processed_data"
NUM_PATIENTS = 10   # change as needed (5–20 recommended)

OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# HELPER FUNCTIONS
# =========================

def normalize(x):
    # simple min-max normalization
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def load_modality(patient_path, patient_id, mod):
    file_path = os.path.join(patient_path, f"{patient_id}-{mod}.nii.gz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    img = nib.load(file_path)
    data = img.get_fdata()

    return data


# =========================
# MAIN LOOP
# =========================

patients = sorted(os.listdir(BASE_PATH))[:NUM_PATIENTS]

print(f"Total patients to process: {len(patients)}\n")

for idx, pid in enumerate(patients):

    print(f"[{idx+1}/{len(patients)}] Processing: {pid}")

    patient_path = os.path.join(BASE_PATH, pid)

    try:
        # load all 4 modalities
        t1c = load_modality(patient_path, pid, "t1c")
        t1n = load_modality(patient_path, pid, "t1n")
        t2f = load_modality(patient_path, pid, "t2f")
        t2w = load_modality(patient_path, pid, "t2w")

        # normalize
        t1c = normalize(t1c)
        t1n = normalize(t1n)
        t2f = normalize(t2f)
        t2w = normalize(t2w)

        # stack → (4, H, W, D)
        volume = np.stack([t1c, t1n, t2f, t2w], axis=0)

        # reorder → (D, H, W, C)
        volume = np.transpose(volume, (3, 1, 2, 0))

        print("Shape:", volume.shape)

        # save each patient separately
        save_path = os.path.join(OUTPUT_DIR, f"{pid}.npy")
        np.save(save_path, volume)

        print("Saved:", save_path, "\n")

    except Exception as e:
        print(f" Error with {pid}: {e}\n")


print("STEP 1 COMPLETED")