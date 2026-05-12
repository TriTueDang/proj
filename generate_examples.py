import cv2
import os
import kagglehub
from detectors.scrfd_detector import SCRFDDetector
from detectors.blazeface_detector import BlazeFaceDetector
from detectors.haar_detector import HaarCascadeDetector
from utils.face_mask_loader import load_face_mask_data

# Create output directory in thesis project
output_dir = r"c:\Users\programming\PROJ\tul-thesis-typst\tul-thesis\images_results"
os.makedirs(output_dir, exist_ok=True)

# # 1. UTKFace Examples
# utk_path = r"c:\Users\programming\proj25\proj\images\UTKFace"
# utk_samples = [
#     "59_1_0_20170110160643688.jpg.chip.jpg",
#     "10_0_0_20161220222308131.jpg.chip.jpg"
# ]

detector_scrfd = SCRFDDetector()

# for i, filename in enumerate(utk_samples):
#     img_path = os.path.join(utk_path, filename)
#     image = cv2.imread(img_path)
#     if image is None: continue

#     # Detection
#     bboxes = detector_scrfd.detect(image)

#     # Draw
#     for (x, y, w, h) in bboxes:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imwrite(os.path.join(output_dir, f"utk_example_{i}.jpg"), image)
#     print(f"Saved UTK example {i}: {filename}")

# 2. Face Mask Examples
mask_dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
# Load a few images
dataset = load_face_mask_data(mask_dataset_path, max_images=1)

if dataset:
    data = dataset[0] # Take the first image

    # 2a. Image with detections
    image_det = data[1].copy()
    bboxes = detector_scrfd.detect(image_det)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(image_det, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "mask_example_det.jpg"), image_det)
    print("Saved Mask example with detections")

    # 2b. Image with ground truth
    image_gt = data[1].copy()
    gt_boxes = data[2] # Ground truth boxes
    labels = data[3]
    for (x, y, w, h), label in zip(gt_boxes, labels):
        color = (255, 0, 0) if label == "with_mask" else (0, 0, 255)
        cv2.rectangle(image_gt, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_gt, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(os.path.join(output_dir, "mask_example_gt.jpg"), image_gt)
    print("Saved Mask example with ground truth")
