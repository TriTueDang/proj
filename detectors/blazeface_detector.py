import cv2
import mediapipe as mp

class BlazeFaceDetector:
    def __init__(self, model_sel=0):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection = model_sel,
            min_detection_confidence=0.3
        )

    def detect(self, image):
        # Convert to RGB as required by mediapipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        boxes = []
        if results.detections:
            h_img, w_img, _ = image.shape
            for detection in results.detections:
                if detection.score[0] >= 0.3:  # Confidence threshold
                    # Extract bounding box
                    bboxC = detection.location_data.relative_bounding_box

                    x1 = int(bboxC.xmin * w_img)
                    y1 = int(bboxC.ymin * h_img)
                    x2 = int((bboxC.xmin + bboxC.width) * w_img)
                    y2 = int((bboxC.ymin + bboxC.height) * h_img)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_img, x2)
                    y2 = min(h_img, y2)

                    w = x2 - x1
                    h = y2 - y1

                    if w > 0 and h > 0:
                        boxes.append((x1, y1, w, h))
        return boxes


