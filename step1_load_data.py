import os
import numpy as np

BASE_PATH = "processed_data"

files = sorted([f for f in os.listdir(BASE_PATH) if f.endswith(".npy")])

print("Total files:", len(files))

for i, file in enumerate(files[:5]):   # test first 5
    path = os.path.join(BASE_PATH, file)

    data = np.load(path)

    print(f"[{i+1}] {file}")
    print("Shape:", data.shape)
    print("-" * 40)