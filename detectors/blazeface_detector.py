import cv2
import mediapipe as mp

class BlazeFaceDetector:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        # Use model_selection=0 for close-range, small face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image):
        # Convert to RGB as required by mediapipe, assume fixed 200x200 size for UTKFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        boxes = []
        if results.detections:
            for detection in results.detections:
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * 200)
                y = int(bboxC.ymin * 200)
                w = int(bboxC.width * 200)
                h = int(bboxC.height * 200)
                boxes.append((x, y, w, h))
        return boxes

# Example usage
if __name__ == '__main__':
    detector = BlazeFaceDetector(model_selection=0, min_detection_confidence=0.5)
    image = cv2.imread('./face recognition/images/UTKFace/1_0_0_20161219140623097.jpg.chip.jpg')
    if image.shape != (200, 200, 3):
        image = cv2.resize(image, (200, 200))
    boxes = detector.detect(image)
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
