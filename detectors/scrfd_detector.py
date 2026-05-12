from insightface.app import FaceAnalysis

class SCRFDDetector:
    def __init__(self):
        # Initialize the SCRFD face detector
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.confidence_threshold = 0.3

    def detect(self, image):
        detections = self.detect_with_scores(image)
        return [det['box'] for det in detections]

    def detect_with_scores(self, image):
        faces = self.app.get(image)
        detections = []
        for face in faces:
            score = float(face.det_score)
            if score >= self.confidence_threshold:
                box = face.bbox  # [x1, y1, x2, y2]
                x, y, x2, y2 = [int(b) for b in box]
                detections.append({'box': (x, y, x2 - x, y2 - y), 'score': score})
        return detections

