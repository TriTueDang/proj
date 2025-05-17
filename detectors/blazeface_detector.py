import cv2
import mediapipe as mp

class BlazeFaceDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        # Convert to RGB as required by mediapipe, assume fixed 200x200 size for UTKFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        boxes = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] >= 0.5:  # Confidence threshold
                    # Extract bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * 200)
                    y = int(bboxC.ymin * 200)
                    w = int(bboxC.width * 200)
                    h = int(bboxC.height * 200)
                    boxes.append((x, y, w, h))
        return boxes


