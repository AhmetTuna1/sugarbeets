import numpy as np
from PIL import Image
import os

mask_dir = r"annotations_cropweed_parts\WORKSET\masks"
files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
all_values = set()

for f in files[1000:1050]:
    path = os.path.join(mask_dir, f)
    arr = np.array(Image.open(path))
    vals = np.unique(arr)
    all_values.update(vals.tolist())

print("Unique values in next batch:", sorted(all_values))
