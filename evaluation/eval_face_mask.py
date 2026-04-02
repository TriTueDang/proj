from evaluation.iou_func import iou

MASK_STATUS_MAP = {
    'with_mask': 'with_mask',
    'without_mask': 'without_mask',
    'incorrect_mask': 'incorrect_mask',
    'mask_weared_incorrect': 'incorrect_mask',
}

def normalize_mask_label(label):
    if label is None:
        return 'unknown'
    normalized = str(label).strip().lower().replace(' ', '_')
    return MASK_STATUS_MAP.get(normalized, normalized)

def evaluate_detector(detector, dataset, iou_threshold=0.5):
    """
    Evaluate detector performance across each face mask wearing category.
    Returns a dictionary with TP, FP, and FN metrics for each mask-status category.
    """
    categories = ['with_mask', 'incorrect_mask', 'without_mask']
    stats = {cat: {'TP': 0, 'FP': 0, 'FN': 0, 'TotalGT': 0} for cat in categories}
    stats['background'] = {'TP': 0, 'FP': 0, 'FN': 0, 'TotalGT': 0}

    for _, image, gt_boxes, gt_labels in dataset:
        normalized_labels = [normalize_mask_label(lbl) for lbl in gt_labels]
        pred_boxes = detector.detect(image)
        matched_gt = set()

        for label in normalized_labels:
            if label in stats:
                stats[label]['TotalGT'] += 1
            else:
                stats['background']['TotalGT'] += 1

        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, g_box in enumerate(gt_boxes):
                current_iou = iou(p_box, g_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_label = normalized_labels[best_gt_idx]
                if gt_label in stats:
                    stats[gt_label]['TP'] += 1
                else:
                    stats['background']['TP'] += 1
                matched_gt.add(best_gt_idx)
            else:
                if best_gt_idx >= 0:
                    fp_label = normalized_labels[best_gt_idx]
                    if fp_label in stats:
                        stats[fp_label]['FP'] += 1
                    else:
                        stats['background']['FP'] += 1
                else:
                    stats['background']['FP'] += 1

        for gt_idx, gt_label in enumerate(normalized_labels):
            if gt_idx not in matched_gt:
                if gt_label in stats:
                    stats[gt_label]['FN'] += 1
                else:
                    stats['background']['FN'] += 1

    return stats
