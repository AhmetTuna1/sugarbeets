import numpy as np
from PIL import Image
import os

path = "annotations_cropweed_parts/ijrr_sugarbeets_2016_annotations/CKA_160427/annotations/dlp/iMapCleaned/1461671136_16770475.png"
if os.path.exists(path):
    arr = np.array(Image.open(path))
    print("iMapCleaned unique values:", np.unique(arr))
else:
    print("iMapCleaned file not found")

path_color = "annotations_cropweed_parts/ijrr_sugarbeets_2016_annotations/CKA_160427/annotations/dlp/colorCleaned/1461671136_16770475.png"
if os.path.exists(path_color):
    arr_color = np.array(Image.open(path_color))
    print("colorCleaned unique colors:")
    print(np.unique(arr_color.reshape(-1, arr_color.shape[2]), axis=0))
else:
    print("colorCleaned file not found")
