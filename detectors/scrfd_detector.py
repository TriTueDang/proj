from insightface.app import FaceAnalysis

class SCRFDDetector:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])

        self.app.prepare(ctx_id=0, det_size=(160, 160))

    def detect(self, image):
        faces = self.app.get(image)
        boxes = []
        for face in faces:
            box = face.bbox  # [x1, y1, x2, y2]
            x, y, x2, y2 = [int(b) for b in box]
            boxes.append((x, y, x2 - x, y2 - y))
        return boxes
