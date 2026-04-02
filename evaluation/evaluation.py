from evaluation.iou_func import iou

def evaluate(detector, dataset, iou_threshold=0.5):
    """
    Evaluates the face detector performance across different demographic groups.
    This function iterates through a dataset, performs detection, and calculates
    the detection success rate based on Intersection over Union (IoU). Results are
    categorized by gender, race, and age groups to identify potential bias or
    performance variations.
    """

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
        # Create a ground truth box covering the whole image (UTK face dataset is cropped) for face detection evaluation
        gt_box = [0, 0, image.shape[1], image.shape[0]]
        pred_boxes = detector.detect(image)

        matched = any(iou(gt_box, pb) >= iou_threshold for pb in pred_boxes)
        stats[(gender, race, get_age_group(age))].append(int(matched))

    return stats
