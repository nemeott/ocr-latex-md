import cv2
import numpy as np


def extract_features(symbol_image: np.ndarray) -> np.ndarray:
    """Resize crop to 8×8 grayscale grid, flatten, normalize to [0, 1]."""
    if symbol_image.size == 0:
        return np.zeros(8 * 8, dtype=np.float64)
    if symbol_image.ndim == 3:
        gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = symbol_image
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float64) / 255.0).ravel()
