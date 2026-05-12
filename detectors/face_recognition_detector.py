import face_recognition
import cv2

class FaceRecognitionDetector:
    def __init__(self, model="hog"):
        # face_recognition uses dlib's HOG or CNN models under the hood.
        # model="hog" is faster, model="cnn" is more accurate but requires GPU
        self.model = model

    def detect(self, image):
        detections = self.detect_with_scores(image)
        return [det['box'] for det in detections]

    def detect_with_scores(self, image):
        # face_recognition expects RGB images, while OpenCV uses BGR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations (top, right, bottom, left)
        face_locations = face_recognition.face_locations(rgb_image, model=self.model)
        
        detections = []
        for (top, right, bottom, left) in face_locations:
            x = left
            y = top
            w = right - left
            h = bottom - top
            detections.append({'box': (x, y, w, h), 'score': 1.0})
        return detections
