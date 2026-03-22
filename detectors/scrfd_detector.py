from insightface.app import FaceAnalysis

class SCRFDDetector:
    def __init__(self):
        # Initialize the SCRFD face detector
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(160, 160))
        self.confidence_threshold = 0.5

    def detect(self, image):
        faces = self.app.get(image)
        boxes = []
        for face in faces:
            if face.det_score >= self.confidence_threshold:
                # Extract the bounding box coordinates
                box = face.bbox  # [x1, y1, x2, y2]
                x, y, x2, y2 = [int(b) for b in box]
                boxes.append((x, y, x2 - x, y2 - y))
        return boxes

