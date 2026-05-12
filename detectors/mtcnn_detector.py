from mtcnn import MTCNN
import cv2
class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.conf_threshold = 0.3

    def detect(self, image):
        detections = self.detect_with_scores(image)
        return [det['box'] for det in detections]

    def detect_with_scores(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_image)
        detections = []
        for r in results:
            score = float(r.get('confidence', 0.0))
            if score >= self.conf_threshold:
                detections.append({'box': tuple(r['box']), 'score': score})
        return detections