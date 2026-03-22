import cv2
import sys
import os

sys.path.append(os.getcwd())

from detectors.dlib_detector import DlibHOGDetector
from detectors.face_recognition_detector import FaceRecognitionDetector

def test_detectors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    print(f"Testing image: {image_path}")

    # Test DlibHOGDetector
    print("\n--- DlibHOGDetector ---")
    dlib_detector = DlibHOGDetector()
    dlib_boxes = dlib_detector.detect(image)
    print(f"Found {len(dlib_boxes)} faces.")
    for i, (x, y, w, h) in enumerate(dlib_boxes):
        print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")

    # Test FaceRecognitionDetector
    print("\n--- FaceRecognitionDetector ---")
    fr_detector = FaceRecognitionDetector(model="hog")
    fr_boxes = fr_detector.detect(image)
    print(f"Found {len(fr_boxes)} faces.")
    for i, (x, y, w, h) in enumerate(fr_boxes):
        print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")

if __name__ == "__main__":
    sample_image = 'images/UTKFace/1_0_0_20161219140627985.jpg.chip.jpg'
    test_detectors(sample_image)
