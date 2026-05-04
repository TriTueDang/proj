from mtcnn import MTCNN

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.conf_threshold = 0.3

    def detect(self, image):
        results = self.detector.detect_faces(image)
        boxes = [r['box'] for r in results if r['confidence'] >= self.conf_threshold]
        return boxes