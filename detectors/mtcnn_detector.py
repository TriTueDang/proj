from mtcnn import MTCNN
import cv2
class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.conf_threshold = 0.3

    def detect(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_image)
        boxes = [r['box'] for r in results if r['confidence'] >= self.conf_threshold]
        return boxes