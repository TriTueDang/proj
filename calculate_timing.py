import pandas as pd

# Data from thesis.typ and analysis
utkface_samples = 23705
facemask_samples = 853

detectors = [
    {"name": "BlazeFace", "utk_time": 124, "mask_time": 9},
    {"name": "Haar Cascade", "utk_time": 318, "mask_time": 57},
    {"name": "FaceRecognition (HOG)", "utk_time": 1096, "mask_time": 81},
    {"name": "MTCNN", "utk_time": 3246, "mask_time": 170},
    {"name": "SCRFD", "utk_time": 8774, "mask_time": 1535},
]

print("| Detektor | UTKFace (ms/snim.) | Face Mask (ms/snim.) |")
print("| :--- | :---: | :---: |")

for d in detectors:
    utk_ms = (d["utk_time"] / utkface_samples) * 1000
    mask_ms = (d["mask_time"] / facemask_samples) * 1000
    print(f"| {d['name']} | {utk_ms:.2f} | {mask_ms:.2f} |")
