import os
import cv2

def parse_filename(filename):
    """
    Parses the filename to extract age, gender and race
    """
    parts = filename.split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
        return age, gender, race
    except (ValueError, IndexError):
        return None

def load_utkface_images(path, max_images=None):
    """
    Loads images from the UTKFace dataset and returns a list of tuples (image, label).
    """
    data = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            label = parse_filename(filename)
            if label is None:
                continue
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            if image is not None:
                data.append((image, label))

            if max_images and len(data) >= max_images:
                break
    return data
