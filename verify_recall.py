import time
import kagglehub
import os
from evaluation.eval_face_mask import evaluate_detector
from utils.face_mask_loader import load_face_mask_data
from detectors.mtcnn_detector import MTCNNDetector
from detectors.haar_detector import HaarCascadeDetector
from detectors.scrfd_detector import SCRFDDetector
from detectors.blazeface_detector import BlazeFaceDetector

def summarize_results(stats, name):
    print(f"\nFinal Results for {name}:")
    tp = sum(s['TP'] for s in stats.values())
    fp = sum(s['FP'] for s in stats.values())
    fn = sum(s['FN'] for s in stats.values())
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    
    print(f"  Recall:    {recall:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    
    print("Loading a subset of 50 images for quick verification...")
    # Loading 50 images for quick verification
    dataset = load_face_mask_data(path, max_images=50)
    
    detectors = {
        "MTCNN": MTCNNDetector(),
        "HAAR": HaarCascadeDetector(),
        "SCRFD": SCRFDDetector(),
        "BlazeFace": BlazeFaceDetector()
    }
    
    for name, detector in detectors.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()
        stats = evaluate_detector(detector, dataset)
        end_time = time.time()
        summarize_results(stats, name)
        print(f"  Took {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
