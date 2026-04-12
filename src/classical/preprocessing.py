import cv2
import numpy as np
from joblib import Parallel, delayed

from bounding_box import BoundingBox
from data_loading import load_iam_lines, load_math_writing


def load_image(image_path: str) -> np.ndarray:
    # this function was wrriten with the help of LLM's
    # I asked the llm what functions from the opncv library we needed to use to load the image

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
    # this function was wrriten with the help of LLM
    # I utilized LLM to help me understand what methods I needed to use to implement the image processing methods I needed. So functions like gaussianblur, threshold...

    """
    This function help us to preprocess the image

    Args:
        image: The image to preprocess
    Returns:
        binary: The binary image
    """

    # This codebelow help us to convert the image to grayscale
    if image.ndim == 3:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE to normalize contrast across different datasets (IAM vs MathWriting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray = clahe.apply(gray)

    # This cope below help us to remove noise from the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize with Otsu's method
    ignore, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to fix broken strokes before segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def crop_character(image: np.ndarray, box: BoundingBox, size: int = 28) -> np.ndarray:
    # this function was wrriten with the help of LLM's
    # I needed help with the logic of the function, so I asked the llm to help me with this logic to implement what I needed this funciton to do

    """
    This function help us to crop the character from the image.

    Matches the preprocessing used in svm_load_image so that inference
    features are consistent with training features.

    Args:
        image: The original image (color or grayscale) to crop from
        box: The bounding box of the character
        size: The output size (default 28 to match SVM training format)
    Returns:
        binary: The preprocessed, binarized character image (size x size), values 0/1
    """

    h_img, w_img = image.shape[:2]

    # Clamp to image boundaries
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(w_img, box.x + box.width)
    y2 = min(h_img, box.y + box.height)

    cropped = image[y1:y2, x1:x2]

    # Convert to grayscale if needed
    if cropped.ndim == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Pad to square (preserve aspect ratio)
    ch, cw = cropped.shape[:2]
    side = max(ch, cw)
    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left
    squared = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    # Binarize using the shared preprocessing pipeline (CLAHE, blur, Otsu, morph close)
    binary = preprocess(squared)

    # Resize to match SVM training format
    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_AREA)

    # Re-binarize because resize introduces gray interpolation artifacts
    ignored, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)

    # Convert from 0/255 to 0/1 to match svm_load_image output
    binary = (binary > 0).astype(np.uint8)

    return binary


def _load_dataset(math_split, text_split, n_math=None, n_text=None, n_jobs=-2):
    """
    This function help us to load the dataset and convert it to the format we need for the SVM training

    Right now this is only for SVM training, but I plan on extending it to for general purpose training in the future.

    Args:
        math_split: The split of the math dataset
        text_split: The split of the text dataset
        n_math: The number of math samples to load
        n_text: The number of text samples to load
        n_jobs: Number of parallel jobs for preprocessing (-2 = all cores but one, -1 = all cores, 1 = sequential)
    Returns:
        data_list: A list of the data
    """

    from svm_preprocessing import svm_load_image

    data_list = []

    for ds, img_key, txt_key, type_label, n_cap in [
        (load_math_writing(math_split), "png", "txt", 1, n_math),
        (load_iam_lines(text_split), "image", "text", 0, n_text),
    ]:

        # Used LLM here to help me implmenet parallel processing for the dataset loading.

        count = len(ds) if n_cap is None else min(n_cap, len(ds))

        results = Parallel(n_jobs=n_jobs)(
            delayed(svm_load_image)(ds[i][img_key], ds[i][txt_key]) for i in range(count)
        )

        for label, pixels in results:
            data_list.append([[type_label, label], pixels])

    return data_list


def prepare_training_data(n_math=None, n_text=None, n_jobs=-2):
    return _load_dataset("train", "train", n_math, n_text, n_jobs)


def prepare_test_data(n_math=None, n_text=None, n_jobs=-2):
    return _load_dataset("validation", "test", n_math, n_text, n_jobs)


def remove_characters(data_list: list[list[list[int]]]):
    for data in data_list:
        label = data[0][1].strip()
        while "  " in label:
            label = label.replace("  ", " ")
        data[0][1] = label

    return data_list


def remove_characters_from_decoder_output(text: str) -> str:
    # strips leading/trailing whitespace and collapses runs of spaces in a single string.
    # intended for cleaning CRNN decoder outputs that often have extra spaces.

    for char in text:
        if char == "\t":
            text = text.replace("\t", "")
        elif char == "\n":
            text = text.replace("\n", "")
        elif char == "\r":
            text = text.replace("\r", "")

    while text and text[0] == " ":
        text = text[1:]

    while text and text[-1] == " ":
        text = text[:-1]

    while "  " in text:
        text = text.replace("  ", " ")

    return text


def remove_spaces_before_characters(text: str) -> str:

    punctuation = ".,!?;:)]}'\"-"
    for char in punctuation:
        while " " + char in text:
            text = text.replace(" " + char, char)

    return text

