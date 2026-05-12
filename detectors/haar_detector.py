import cv2

class HaarCascadeDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, image):
        detections = self.detect_with_scores(image)
        return [det['box'] for det in detections]

    def detect_with_scores(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), maxSize=(300, 300))
        return [{'box': tuple(box), 'score': 1.0} for box in boxes]
