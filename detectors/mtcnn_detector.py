from mtcnn import MTCNN

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, image):
        results = self.detector.detect_faces(image)
        boxes = [r['box'] for r in results]
        return boxes