import cv2
import numpy as np
from bounding_box import BoundingBox

MIN_AREA = 20  # Minimum pixel area to consider a component a real symbol


# Returns a list of bounding boxes for each character in the image
def segment(image: np.ndarray) -> list[BoundingBox]:
    # Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize: Otsu's thresholding ensures we get clean black/white
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    bounding_boxes = []

    # Label 0 is the background, so we skip it and start from 1
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        bounding_boxes.append(BoundingBox(x=x, y=y, width=w, height=h))

    # Sort left to right, top to bottom (reading order)
    bounding_boxes.sort(key=lambda b: (b.y, b.x))

    return bounding_boxes
