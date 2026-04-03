import os
import cv2
import xml.etree.ElementTree as ET

def load_face_mask_data(dataset_path, max_images=None):
    """
    Load data from the face-mask dataset (annotations and images).
    Returns a list of tuples: (image_path, numpy_image, bounding_boxes, labels).
    Bounding boxes are in (x, y, width, height) format.
    """
    annotations_dir = os.path.join(dataset_path, "annotations")
    images_dir = os.path.join(dataset_path, "images")

    data = []

    if not os.path.exists(annotations_dir) or not os.path.exists(images_dir):
        print("Error: The annotations or images directory is missing.")
        return data

    # Iterate over all XML files in annotations
    for filename in sorted(os.listdir(annotations_dir)):
        if not filename.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get the corresponding image filename from XML
        img_filename = root.find("filename").text
        img_path = os.path.join(images_dir, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            continue

        boxes = []
        labels = []

        # Load bounding box and label data
        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Convert to (x, y, width, height)
            w = xmax - xmin
            h = ymax - ymin
            boxes.append((xmin, ymin, w, h))
            labels.append(label)

        data.append((img_path, image, boxes, labels))

        if max_images and len(data) >= max_images:
            break

    return data