import os
import cv2

def parse_filename(filename):
    """
    Parses the filename to extract age, gender and race
    """
    parts = filename.split("_")
    if len(parts) < 4:
        return None
    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    return age, gender, race

def load_utkface_images(path, max_images=None):
    """
    Loads images from the UTKFace dataset and returns a list of tuples (image, label).
    """
    data = []
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith(".jpg"):
            label = parse_filename(filename)
            if label is None:
                continue
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            if image is not None:
                data.append((image, label))
        if max_images and i >= max_images:
            break
    return data
