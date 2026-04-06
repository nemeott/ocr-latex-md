# from features import extract_features
import sys
from pathlib import Path

from classifier import LoadEnsembleSVM, LoadGeneralSVM, PredictEnsembleSVM, PredictGeneralSVM
from preprocessing import crop_character, load_image, preprocess
from segmentation import segment
from structure import AST
from symbol import Symbol

# Basic outline for classical OCR pipeline for LaTeX to Markdown conversion

# Load bundled pretrained models (saved under src/classical/models/*.pkl).
# Note: LoadGeneralSVM/LoadEnsembleSVM append ".pkl", so pass the path WITHOUT extension.
MODELS_DIR = Path(__file__).resolve().parent / "models"

try:
    general_model, general_pca = LoadGeneralSVM(str(MODELS_DIR / "final_gen_svm"))
    ensemble_models, ensemble_k_means, ensemble_pca = LoadEnsembleSVM(str(MODELS_DIR / "final_ens_svm"))
except Exception as e:
    raise ValueError(
        f"Can't load pretrained models from '{MODELS_DIR}'. "
        f"Expected 'final_gen_svm.pkl' and 'final_ens_svm.pkl'. Original error: {e}"
    ) from None

# TODO: define `image` before running the pipeline (e.g., `image = preprocess(load_image(path))`)
# Segment each character
boxes = segment(image)

symbols: list[Symbol] = []
for box in boxes:
    # Get the cropped image of the character
    cropped = crop_character(image, box)

    # Extract the features from the cropped image
    features = extract_features(cropped)

    # Predict the symbol using a trained classifier
    symbol: Symbol = PredictEnsembleSVM(ensemble_models, ensemble_k_means, ensemble_pca, features)
    symbol2: Symbol = PredictGeneralSVM(general_model, general_pca, features)

    # Append the predicted symbol and its bounding box to the list
    symbols.append(symbol)

ast = AST(symbols)
markdown = ast.render_latex_markdown()

print(markdown)
