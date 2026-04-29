from PIL import Image
import numpy as np
import os

input_dir = r"DATASET_READY\masks"
output_dir = r"DATASET_READY\binary_masks"
os.makedirs(output_dir, exist_ok=True)

WEED_VALUE = 10000  # Based on inspect output: 0 background, 10000 weed

count = 0
for f in os.listdir(input_dir):
    if not f.lower().endswith(".png"):
        continue

    path = os.path.join(input_dir, f)
    arr = np.array(Image.open(path))

    # set pixels to 255 where it matches WEED_VALUE, else 0
    binary = np.where(arr == WEED_VALUE, 255, 0).astype(np.uint8)
    Image.fromarray(binary).save(os.path.join(output_dir, f))
    count += 1

print(f"Done processing {count} binary masks.")
