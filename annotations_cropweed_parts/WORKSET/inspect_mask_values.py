from PIL import Image
import numpy as np
import os

mask_dir = r"DATASET_READY\masks"

files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
all_values = set()

for f in files[:30]:
    path = os.path.join(mask_dir, f)
    arr = np.array(Image.open(path))
    vals = np.unique(arr)
    print(f"{f}: {vals}")
    all_values.update(vals.tolist())

print("\nAll unique values seen:", sorted(all_values))
