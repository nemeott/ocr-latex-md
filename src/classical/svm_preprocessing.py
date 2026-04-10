import cv2
import numpy as np
from bounding_box import BoundingBox
from PIL import Image as PILImage
from preprocessing import preprocess


def svm_load_image(path_or_image, label: str, size: int = 28) -> list:

    #this function was wrriten with the help of LLM's
    #I asked the llm what functions from the opncv library we needed to use to load the image

    """
    This function help us to load the image and convert it to the format we need for the SVM training

    Args:
        path_or_image: The path to the image or the image itself
        label: The label of the image
        size: The size of the image

    Returns:
        [label, binary.flatten()]: The label and the binary image(numpy array)
    """

    if isinstance(path_or_image, str):
        image = cv2.imread(path_or_image, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not load image at '{path_or_image}'")


    elif isinstance(path_or_image, PILImage.Image):  # PIL Image

        image = np.array(path_or_image.convert("L"))

    elif isinstance(path_or_image, np.ndarray):  # Already a numpy array

        image = path_or_image.copy()


        if image.ndim == 3:


            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError(f"Expected a file path, PIL Image, or np.ndarray")

    # Pad to square to preserve aspect ratio before resizing
    h, w = image.shape[:2]
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    # Binarize before resizing to preserve fine stroke detail
    binary = preprocess(image)

    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_AREA)

    # Re-binarize to clean up gray interpolation artifacts from resize
    ignore, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)

    # Convert from 0/255 to 0/1
    binary = (binary > 0).astype(np.uint8)

    return [label, binary.flatten()]


def svm_reshape_image(flat_pixels: np.ndarray, new_n: int, new_m: int, orig_side: int = 28) -> np.ndarray:

    """
    This function help us to reshape the image to the format we need for the SVM training

    Args:
        flat_pixels: The flattened image
        new_n: The new height of the image
        new_m: The new width of the image
        orig_side: The side length of the original square image (default 28)
    Returns:
        image: The reshaped image
    """

    image = flat_pixels.reshape(orig_side, orig_side).astype(np.float64)

    old_n, old_m = image.shape

    # Determine resize target: shrink axes that need downsizing
    resize_n = min(old_n, new_n)
    resize_m = min(old_m, new_m)

    # Downsample with cv2 if either axis needs to shrink
    if resize_n < old_n or resize_m < old_m:

        image = cv2.resize(image, (resize_m, resize_n), interpolation=cv2.INTER_AREA)

    # Pad with blank pixels if either axis needs to grow
    pad_n = new_n - image.shape[0]

    pad_m = new_m - image.shape[1]

    if pad_n > 0 or pad_m > 0:

        image = np.pad(

            image,
            ((pad_n // 2, pad_n - pad_n // 2), (pad_m // 2, pad_m - pad_m // 2)),
            mode="constant",
            constant_values=0,
        )

    return image.astype(flat_pixels.dtype)
