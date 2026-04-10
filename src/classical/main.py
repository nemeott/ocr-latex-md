import os
import sys

# Allow `python main.py` here or `python src/classical/main.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import Symbol, SymbolClassifier
from features import extract_features
from preprocessing import crop_character, load_image, preprocess
from segmentation import segment
from structure import AST


def main(image_path: str = "example.png") -> str:
    classifier = SymbolClassifier()

    image = load_image(image_path)
    image = preprocess(image)

    boxes = segment(image)

    symbols: list[Symbol] = []
    for box in boxes:
        cropped = crop_character(image, box)
        features = extract_features(cropped)
        symbol: Symbol = classifier.predict(features, box)
        symbols.append(symbol)

    ast = AST(symbols)
    return ast.render_latex_markdown()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "example.png"
    print(main(path))
