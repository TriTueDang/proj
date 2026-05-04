import cv2
import numpy as np
import supervision as sv
from detectors.haar_detector import HaarCascadeDetector
import matplotlib.pyplot as plt
# 1. Load image
image = cv2.imread('lena.png')
detector = HaarCascadeDetector()

# 2. Get raw results [[x, y, w, h], ...]
raw_detections = detector.detect(image)

# 3. Convert [x, y, w, h] to [x1, y1, x2, y2]
if len(raw_detections) > 0:
    # Convert to numpy array if not already
    raw_detections = np.array(raw_detections)

    # Calculate x2 and y2
    xyxy = raw_detections.copy()
    xyxy[:, 2] = raw_detections[:, 0] + raw_detections[:, 2] # x + w
    xyxy[:, 3] = raw_detections[:, 1] + raw_detections[:, 3] # y + h

    class_id = np.zeros(len(xyxy), dtype=int)

    # Initialize Detections with the class_id
    detections = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        class_id=class_id
    )

    # Initialize Detections object
    # detections = sv.Detections(xyxy=xyxy.astype(np.float32))
else:
    # Handle case with no detections
    detections = sv.Detections.empty()

# 4. Visualization
# Note: Newer versions of supervision separate the box and label annotators
box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# 5. Display

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()