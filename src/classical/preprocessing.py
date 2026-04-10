import cv2
import numpy as np

from segmentation import BoundingBox


def load_image(image_path: str) -> np.ndarray:
    """Load BGR image from disk (OpenCV convention)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale + light blur for segmentation; 2D array for `segment`."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return cv2.GaussianBlur(gray, (3, 3), 0)


def crop_character(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop a single region; shape matches input (grayscale 2D or BGR 3D)."""
    h, w = image.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(w, box.x + box.width)
    y1 = min(h, box.y + box.height)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=np.uint8)
    return image[y0:y1, x0:x1].copy()
