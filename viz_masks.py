import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import kagglehub
from utils.face_mask_loader import load_face_mask_data
from detectors.scrfd_detector import SCRFDDetector
from detectors.blazeface_detector import BlazeFaceDetector
from detectors.mtcnn_detector import MTCNNDetector

def visualize_detections(detector, data, num_samples=3):
    # Download/get path to the dataset


    if not data:
        print("No data found.")
        return

    plt.figure(figsize=(15, 5 * num_samples))

    for i, (img_path, image, gt_boxes, labels) in enumerate(data[:num_samples]):
        # Create a copy for drawing
        viz_img = image.copy()

        # 1. Draw Ground Truth boxes (Green)
        for box, label in zip(gt_boxes, labels):
            x, y, w, h = box
            cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 2. Run detector
        pred_boxes = detector.detect(image)

        # 3. Draw Predicted boxes (Blue)
        for box in pred_boxes:
            x, y, w, h = box
            cv2.rectangle(viz_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display result
        ax = plt.subplot(num_samples, 1, i + 1)
        plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Snímek: {os.path.basename(img_path)}")
        plt.axis('off')

        # Add legend
        green_patch = mpatches.Patch(color='green', label='Ground Truth')
        blue_patch = mpatches.Patch(color='blue', label='Detekce')
        ax.legend(handles=[green_patch, blue_patch], loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Inicializace nejlepšího detektoru pro masky (SCRFD)
    print("Inicializace detektoru...")

    detector = BlazeFaceDetector(model_sel=1)
    # Spuštění vizualizace na prvních 3 snímcích
    path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    dataset = load_face_mask_data(path, max_images=853)
    visualize_detections(detector, dataset, num_samples=3)
