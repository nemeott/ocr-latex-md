import cv2
import numpy as np
from bounding_box import BoundingBox
from data_loading import load_math_writing, load_iam_lines
from svm_preprocessing import svm_load_image


def load_image(image_path: str) -> np.ndarray:

    """
    This function help us to load the image

    Args:
        image_path: The path to the image
    Returns:
        image: The loaded image
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:

    """
    This function help us to preprocess the image

    Args:
        image: The image to preprocess
    Returns:
        binary: The binary image
    """

    #This codebelow help us to convert the image to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    #This cope below help us to remove noise from the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #This function help us to binarize the image, keeping only the black and white pixels
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)

    return binary


def crop_character(image: np.ndarray, box: BoundingBox) -> np.ndarray:

    """
    This function help us to crop the character from the image

    Args:
        image: The image to crop the character from
        box: The bounding box of the character
    Returns:
        cropped: The cropped character
    """

    #This function is still a work in progress
    #It is a good starting point for further improvements...

    h_img, w_img = image.shape[:2]

    # Clamp to image boundaries
    x1 = max(0, box.x)

    y1 = max(0, box.y)
    x2 = min(w_img, box.x + box.width)
    y2 = min(h_img, box.y + box.height)

    cropped = image[y1:y2, x1:x2]

    #pad to square (preserve aspect ratio)
    ch, cw = cropped.shape[:2]
    side = max(ch, cw)
    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left
    squared = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    #Resize to 28x28 to match SVM training format
    resized = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)

    #we rebinarize the image again because the resize introduces gray interpolation artifacts
    ignored, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)

    return resized


def _load_dataset(math_split, text_split, n_math=None, n_text=None):

    """
    This function help us to load the dataset and convert it to the format we need for the SVM training

    Right now this is only for SVM training, but I plan on extending it to for general purpose training in the future. 

    Args:
        math_split: The split of the math dataset
        text_split: The split of the text dataset
        n_math: The number of math samples to load
        n_text: The number of text samples to load
    Returns:
        data_list: A list of the data
    """

    data_list = []

    for ds, img_key, txt_key, type_label, n_cap in [
        (load_math_writing(math_split), "png", "txt", 1, n_math),
        (load_iam_lines(text_split), "image", "text", 0, n_text),
    ]:

        if n_cap is None:
            count = len(ds)
        else:
            count = min(n_cap, len(ds))

        for i in range(count):
            pixels = svm_load_image(ds[i][img_key], ds[i][txt_key])[1]
            data_list.append([[type_label, ds[i][txt_key]], pixels])

    return data_list


def prepare_training_data(n_math=None, n_text=None):
    return _load_dataset("train", "train", n_math, n_text)


def prepare_test_data(n_math=None, n_text=None):
    return _load_dataset("validation", "test", n_math, n_text)
