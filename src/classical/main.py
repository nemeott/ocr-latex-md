"""Simple run script for the classical pipeline.

This version uses the ensemble SVM model.
It:
 - Loads the ensemble SVM (expects `src/classical/models/final_ens_svm.pkl`).
 - Segments the image into bounding boxes.
 - Optionally filters out boxes smaller than `min_area` (width*height) if specified.
 - Crops each character, preprocesses to the PCA's expected input size.
 - Predicts with the ensemble and renders the AST as LaTeX-marked Markdown.

Usage:
    python src/classical/main.py path/to/image.jpg [--min-area N]
"""

import argparse
import os
import sys
from pathlib import Path

# Allow `python main.py` here or `python src/classical/main.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load mappings once at module load (adjust paths as needed)
import os

from classifier import SymbolClassifier
from label_maps import load_emnist_mapping, load_hasy_mapping, map_symbol_value
from preprocessing import crop_character, load_image
from segmentation import segment
from structure import AST

EMNIST_MAP = load_emnist_mapping(os.path.join(os.path.dirname(__file__), "emnist-byclass-mapping.txt"))
HASY_MAP = load_hasy_mapping(os.path.join(os.path.dirname(__file__), "hasy-symbols.csv"))


def main(image_path: str = "example.png", min_area: int = 0) -> str:
    """Classical OCR demo: segment on full image, crop chars from original BGR, classify.

    Supports PNG, JPG, and JPEG images.

    Returns the rendered LaTeX/Markdown string.
    """
    classifier = SymbolClassifier()

    # Load image and segment to get bounding boxes
    image = load_image(image_path)
    boxes = segment(image)  # Runs `preprocess` internally for connected components
    if not boxes:
        raise RuntimeError("No symbols detected in the image.")

    # Optionally filter out tiny boxes before proceeding
    if min_area > 0:
        boxes = [b for b in boxes if (b.width * b.height) >= min_area]
    if not boxes:
        raise RuntimeError(f"No symbols remaining after filtering with min_area={min_area}.")

    # Crop characters from the original image
    cropped_chars = [crop_character(image, box) for box in boxes]

    # # Save all cropped characters to a folder for debugging
    # save_dir = "cropped_chars"
    # os.makedirs(save_dir, exist_ok=True)
    # for idx, (crop, box) in enumerate(zip(cropped_chars, boxes)):
    #     arr = np.array(crop)
    #     arr = (arr * 255).astype(np.uint8)
    #     img = Image.fromarray(arr)
    #     img.save(os.path.join(save_dir, f"{idx}.png"))
    #     print(f"Crop {idx}: dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    # print(f"Saved {len(cropped_chars)} cropped chars to {save_dir}/")

    # Extract features and predict symbols for each cropped character
    symbols = classifier.predict_batch(cropped_chars, boxes)

    # Map symbol values to human-readable output
    for symbol in symbols:
        print(f"Predicted: {symbol.value} (type: {symbol.type}), box: {symbol.box}")
        symbol.value = map_symbol_value(symbol, EMNIST_MAP, HASY_MAP)

    # Build AST and render LaTeX/Markdown using the predicted symbols and their bounding boxes
    ast = AST(symbols)
    return ast.render_latex_markdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run ensemble SVM on an image and print LaTeX/Markdown.",
    )
    parser.add_argument("image", help="Path to input image (e.g. test.jpg)")
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="Minimum bounding-box area (width*height) to keep for prediction. Boxes smaller than this are ignored.",
    )
    args = parser.parse_args()

    # Basic path check
    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    print(main(str(img_path), min_area=args.min_area))
