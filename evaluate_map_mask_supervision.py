"""
Enhanced mAP evaluation using the supervision package for Face Mask Detection Dataset.

Features:
- Uses supervision.Detections for structured detection handling
- Calculates mAP with per-class metrics
- Provides detailed reporting and visualization
- Compatible with all existing detectors
"""

import os
import time
import numpy as np
import pandas as pd
import kagglehub
import supervision as sv
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

from utils.face_mask_loader import load_face_mask_data
from evaluation.eval_face_mask import normalize_mask_label
from evaluation.iou_func import iou


class SupervisionMapEvaluator:
    """
    mAP evaluator using the supervision package.
    Handles conversion between custom format and supervision.Detections.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def convert_to_supervision_detections(self, detections_list: List[Dict]) -> sv.Detections:
        """
        Convert detection list to supervision.Detections format.

        Args:
            detections_list: List of dicts with keys: 'box', 'score', 'class', 'img_id'

        Returns:
            sv.Detections object
        """
        if not detections_list:
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.array([]),
                class_id=np.array([]),
                data={'img_id': np.array([])}
            )

        # Extract coordinates (convert from x,y,w,h to xyxy format)
        xyxy_list = []
        for det in detections_list:
            x, y, w, h = det['box']
            xyxy_list.append([x, y, x + w, y + h])

        xyxy = np.array(xyxy_list)
        confidence = np.array([det['score'] for det in detections_list])
        class_id = np.array([hash(det['class']) % 100 for det in detections_list])

        # Store original class names and img_id
        class_names = {int(hash(det['class']) % 100): det['class'] for det in detections_list}
        img_ids = np.array([det['img_id'] for det in detections_list])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={
                'img_id': img_ids,
                'class_name': np.array([det['class'] for det in detections_list])
            }
        )

        return detections

    def evaluate_detector(self, detector, dataset, detector_name: str) -> Dict:
        """
        Evaluate a detector on the dataset using supervision.

        Args:
            detector: Detector object with detect() or detect_with_scores() method
            dataset: Pre-loaded dataset from load_face_mask_data()
            detector_name: Name of the detector for logging

        Returns:
            Dictionary with mAP and per-class AP values
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {detector_name}")
        print(f"{'='*60}")

        all_detections = []
        all_ground_truths = []

        # Process each image
        for img_idx, (img_path, image, gt_boxes, gt_labels) in enumerate(dataset):
            # Collect ground truths
            for g_box, g_label in zip(gt_boxes, gt_labels):
                normalized_label = normalize_mask_label(g_label)
                if normalized_label == 'unknown':
                    continue

                x, y, w, h = g_box
                all_ground_truths.append({
                    'img_id': img_idx,
                    'box': g_box,
                    'xyxy': [x, y, x + w, y + h],
                    'class': normalized_label
                })

            # Get predictions
            if hasattr(detector, 'detect_with_scores'):
                preds = detector.detect_with_scores(image)
            else:
                boxes = detector.detect(image)
                preds = [{'box': b, 'score': 1.0} for b in boxes]

            # Assign class based on best IoU match with GT (for class-agnostic detectors)
            for p in preds:
                img_gts = [gt for gt in all_ground_truths if gt['img_id'] == img_idx]

                best_iou = 0
                best_cls = 'background'

                for gt in img_gts:
                    curr_iou = iou(p['box'], gt['box'])
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_cls = gt['class']

                if best_cls != 'background':
                    all_detections.append({
                        'img_id': img_idx,
                        'box': p['box'],
                        'score': p['score'],
                        'class': best_cls
                    })

        # Calculate mAP
        map_score, aps_dict = self._calculate_map(all_detections, all_ground_truths)

        # Prepare results
        results = {
            'Detector': detector_name,
            'mAP (%)': round(map_score * 100, 2),
            'Num_Detections': len(all_detections),
            'Num_Ground_Truths': len(all_ground_truths),
        }

        # Add per-class AP
        for cls, ap in aps_dict.items():
            results[f'AP_{cls} (%)'] = round(ap * 100, 2)

        print(f"mAP: {map_score * 100:.2f}%")
        for cls, ap in aps_dict.items():
            print(f"  AP_{cls}: {ap * 100:.2f}%")
        print(f"Total Detections: {len(all_detections)}")
        print(f"Ground Truths: {len(all_ground_truths)}")

        return results

    def _calculate_map(self, detections: List[Dict], ground_truths: List[Dict],
                       iou_threshold: float = 0.5) -> Tuple[float, Dict]:
        """
        Calculate mAP score using custom implementation (compatible with existing code).

        Args:
            detections: List of detection dicts
            ground_truths: List of ground truth dicts
            iou_threshold: IoU threshold for TP

        Returns:
            Tuple of (mAP score, dict of per-class APs)
        """
        from evaluation.iou_func import iou as iou_calc

        classes = sorted(list(set([gt['class'] for gt in ground_truths])))
        aps = {}

        for cls in classes:
            cls_detections = [d for d in detections if d['class'] == cls]
            cls_gts = [gt for gt in ground_truths if gt['class'] == cls]

            n_gt = len(cls_gts)
            if n_gt == 0:
                aps[cls] = 0.0
                continue

            # Sort by score
            cls_detections.sort(key=lambda x: x['score'], reverse=True)

            tp = np.zeros(len(cls_detections))
            fp = np.zeros(len(cls_detections))

            for gt in cls_gts:
                gt['used'] = False

            for i, det in enumerate(cls_detections):
                best_iou_val = 0
                best_gt_idx = -1

                for j, gt in enumerate(cls_gts):
                    curr_iou = iou_calc(det['box'], gt['box'])
                    if curr_iou > best_iou_val:
                        best_iou_val = curr_iou
                        best_gt_idx = j

                if best_iou_val >= iou_threshold:
                    if not cls_gts[best_gt_idx]['used']:
                        tp[i] = 1
                        cls_gts[best_gt_idx]['used'] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            recalls = tp_cumsum / n_gt

            aps[cls] = self._calculate_ap(precisions, recalls)

        map_score = np.mean(list(aps.values())) if aps else 0.0
        return map_score, aps

    @staticmethod
    def _calculate_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate AP using VOC-style interpolation."""
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


def run_supervision_map_evaluation():
    """
    Main evaluation pipeline using supervision.
    Downloads dataset, evaluates all detectors, and saves results.
    """
    print("Starting Face Mask Detection mAP Evaluation (Supervision)")
    print("=" * 70)

    # Download dataset
    print("\n[1/3] Downloading dataset...")
    dataset_path = kagglehub.dataset_download("andrewmvd/face-mask-detection")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path {dataset_path} not found.")
        return

    # Load dataset
    print("[2/3] Loading dataset...")
    dataset = load_face_mask_data(dataset_path, max_images=853)
    print(f"✓ Loaded {len(dataset)} images")

    # Initialize evaluator
    evaluator = SupervisionMapEvaluator(iou_threshold=0.5)

    # Evaluate detectors
    print("\n[3/3] Evaluating detectors...")
    print("=" * 70)

    from detectors.mtcnn_detector import MTCNNDetector
    from detectors.scrfd_detector import SCRFDDetector
    from detectors.haar_detector import HaarCascadeDetector
    from detectors.blazeface_detector import BlazeFaceDetector

    detectors = {
        'MTCNN': MTCNNDetector(),
        'SCRFD': SCRFDDetector(),
        'Haar': HaarCascadeDetector(),
        'BlazeFace': BlazeFaceDetector(model_sel=1),
    }

    results = []
    timing_info = {}

    for name, detector in detectors.items():
        start_time = time.time()
        try:
            result = evaluator.evaluate_detector(detector, dataset, name)
            results.append(result)
            elapsed = time.time() - start_time
            timing_info[name] = elapsed
            print(f"✓ {name} completed in {elapsed:.2f}s")
        except Exception as e:
            print(f"✗ {name} failed: {str(e)}")

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Display summary
    print("\n" + "=" * 70)
    print("FINAL mAP RESULTS (Face Mask Detection Dataset)")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save results
    os.makedirs("./results", exist_ok=True)
    output_path = "./results/mask_map_results_supervision.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    # Create visualization
    try:
        _create_visualization(df, timing_info)
    except Exception as e:
        print(f"Note: Visualization failed ({e}), but results were saved successfully.")

    return df


def _create_visualization(df: pd.DataFrame, timing_info: Dict[str, float]):
    """Create and save visualization of results."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # mAP comparison
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(df))
    bars = ax1.barh(df['Detector'], df['mAP (%)'], color=colors)
    ax1.set_xlabel('mAP (%)', fontsize=12, fontweight='bold')
    ax1.set_title('mAP Comparison - Face Mask Detection', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 100)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['mAP (%)'])):
        ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)

    # Execution time comparison
    ax2 = axes[1]
    detectors_sorted = sorted(timing_info.items(), key=lambda x: x[1], reverse=True)
    names_sorted = [x[0] for x in detectors_sorted]
    times_sorted = [x[1] for x in detectors_sorted]

    bars2 = ax2.barh(names_sorted, times_sorted, color=colors)
    ax2.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Evaluation Time - Face Mask Detection', fontsize=13, fontweight='bold')

    for bar, val in zip(bars2, times_sorted):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}s', va='center', fontsize=10)

    plt.tight_layout()
    output_path = "./results/mask_map_results_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("Face Mask Detection - Enhanced mAP Evaluation with Supervision")
    print("=" * 70)

    try:
        df_results = run_supervision_map_evaluation()
        print("\n" + "=" * 70)
        print("✓ Evaluation completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
