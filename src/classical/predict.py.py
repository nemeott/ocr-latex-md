"""Simple run script for the classical pipeline.

This version uses the ensemble SVM model.
It:
 - Loads the ensemble SVM (expects `src/classical/models/final_ens_svm.pkl`).
 - Segments the image into bounding boxes.
 - Crops each character, preprocesses to the PCA's expected input size.
 - Predicts with the ensemble and renders the AST as LaTeX-marked Markdown.

Usage:
    python src/classical/run.py path/to/image.jpg [--min-area N]
"""

import argparse
from pathlib import Path

from classifier import (
    LoadEnsembleSVM,
    PredictEnsembleSVM,
    PreprocessInputs,
)
from preprocessing import crop_character, load_image
from segmentation import segment
from structure import AST
from symbol import Symbol, SymbolType

MODELS_DIR = Path(__file__).resolve().parent / "models"
ENSEMBLE_NAME = "final_ens_svm"


def _get_pca_side(pca_obj, fallback_side_length: int = 28) -> int:
    """Given a fitted PCA object, return the expected square side length.

    (i.e., sqrt(n_features_in_)). Falls back to `fallback` if unknown.
    """
    if pca_obj is None:
        return fallback_side_length

    if hasattr(pca_obj, "n_features_in_"):
        try:
            n = int(pca_obj.n_features_in_)
            side_length = round(n**0.5)
            if side_length * side_length == n:
                return side_length

            # Not a perfect square: warn and return
            print(
                f"Warning: PCA expects {n} features; using side={side_length} (side*side={side_length * side_length}).",
            )
            return side_length
        except Exception:
            pass

    # Fallback
    print(f"Warning: could not determine PCA feature count; using fallback side={fallback_side_length}.")
    return fallback_side_length


def predict_image(image_path: str, min_area: int = 0) -> str:
    """Predict the LaTeX/Markdown output for the given image using the ensemble model.

    Filters out bounding boxes smaller than `min_area` (area = width * height) if min_area > 0.

    Raises RuntimeError if the ensemble model cannot be loaded or predictions are malformed.
    """
    # Load ensemble model (raise helpful error if missing)
    try:
        models, kmeans, pca = LoadEnsembleSVM(str(MODELS_DIR / ENSEMBLE_NAME))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ensemble model from '{MODELS_DIR / ENSEMBLE_NAME}.pkl'. "
            "Make sure the file exists and is a valid ensemble SVM pickle.",
        ) from e

    # Determine side expected by PCA
    side = _get_pca_side(pca, fallback_side_length=28)

    # Load image and segment characters
    image = load_image(image_path)
    bounding_boxes = segment(image)

    # Optionally filter out tiny boxes before proceeding
    if min_area > 0:
        bounding_boxes = [b for b in bounding_boxes if (b.width * b.height) >= min_area]

    if not bounding_boxes:
        return ""

    # Create list of samples for PreprocessInputs: each sample is [label, binary.flatten()]
    samples: list[list] = []
    for box in bounding_boxes:
        cropped = crop_character(image, box)  # Usually returns 28x28 binary array
        samples.append([[0, "?"], cropped.flatten()])  # Dummy label for PreprocessInputs

    X, _ = PreprocessInputs(samples, n=side, m=side)  # noqa: N806

    # Predict using ensemble
    predictions = PredictEnsembleSVM(models, kmeans, pca, X)

    # Build Symbol objects from predictions
    symbols: list[Symbol] = []
    for box, prediction in zip(bounding_boxes, predictions, strict=True):
        # Expect prediction to be sequence-like [type, label]
        try:
            symbol_type_value = int(prediction[0])
            symbol_value = str(prediction[1])
        except Exception as e:
            raise RuntimeError(f"Unexpected prediction shape/value: {prediction!r}") from e

        symbol_type = SymbolType.MATH if symbol_type_value == 1 else SymbolType.TEXT
        symbols.append(Symbol(symbol_value, symbol_type, box))

    # print(symbols)
    ast = AST(symbols)
    return ast.render_latex_markdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
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

    # Predict and print LaTeX/Markdown output
    latex_markdown_prediction = predict_image(str(img_path), min_area=args.min_area)
    print(latex_markdown_prediction)
