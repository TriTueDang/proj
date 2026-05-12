import dlib
import cv2

class DlibHOGDetector:
    def __init__(self):
        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        detections = self.detect_with_scores(image)
        return [det['box'] for det in detections]

    def detect_with_scores(self, image):
        # Convert to grayscale as dlib works better with it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces. The second parameter '1' is for upsampling (helps find smaller faces)
        faces = self.detector(gray, 1)
        
        detections = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            detections.append({'box': (x, y, w, h), 'score': 1.0})
        return detections
