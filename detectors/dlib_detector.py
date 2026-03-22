import dlib
import cv2

class DlibHOGDetector:
    def __init__(self):
        # Initialize dlib's face detector (HOG-based)
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        # Convert to grayscale as dlib works better with it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces. The second parameter '1' is for upsampling (helps find smaller faces)
        faces = self.detector(gray, 1)
        
        boxes = []
        for face in faces:
            # Convert dlib's rectangle to [x, y, w, h]
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            boxes.append((x, y, w, h))
            
        return boxes
