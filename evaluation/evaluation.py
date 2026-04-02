import numpy as np

def iou(box1, box2):
    # Compute Intersection over Union
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate(detector, dataset, iou_threshold=0.5):
    from collections import defaultdict
    stats = defaultdict(list)
    def get_age_group(age):
        if age <= 2: return "0-2"
        if age <= 5: return "3-5"
        if age <= 12: return "6-12"
        if age <= 18: return "13-18"
        if age <= 25: return "19-25"
        if age <= 35: return "26-35"
        if age <= 50: return "36-50"
        if age <= 70: return "51-70"
        if age <= 90: return "71-90"
        return "90+"

    for image, (age, gender, race) in dataset:
        gt_box = [0, 0, image.shape[1], image.shape[0]]  # approximace
        pred_boxes = detector.detect(image)

        matched = any(iou(gt_box, pb) >= iou_threshold for pb in pred_boxes)
        stats[(gender, race, get_age_group(age))].append(int(matched))

    return stats
