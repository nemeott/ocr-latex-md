# from features import extract_features
from preprocessing import crop_character, load_image, preprocess, prepare_training_data, prepare_test_data
from segmentation import segment
from structure import AST
from symbol import Symbol
from classifier import LoadGeneralSVM, LoadEnsembleSVM, PredictGeneralSVM, PredictEnsembleSVM

# Basic outline for classical OCR pipeline for LaTeX to Markdown conversion

train_data = prepare_training_data()
test_data = prepare_test_data()

classifier = SymbolClassifier()
# classifier.train() # TODO: Train classifier or load pre-trained model

try:
    # Load the models
    generalModel, generalPCA = LoadGeneralSVM("general_svm")
    ensembleModels, ensembleKMeans, ensemblePCA = LoadEnsembleSVM("ensemble_svm")
except Exception:
    # Loading failed
    raise ValueError("Can't load model. Ensure filenames are correct and models are trained.")

# Segment each character
boxes = segment(image)

symbols: list[Symbol] = []
for box in boxes:
    # Get the cropped image of the character
    cropped = crop_character(image, box)

    # Extract the features from the cropped image
    features = extract_features(cropped)

    # Predict the symbol using a trained classifier
    symbol: Symbol = PredictEnsembleSVM(ensembleModels, ensembleKMeans, ensemblePCA, features)
    symbol2: Symbol = PredictGeneralSVM(generalModel, generalPCA, features)

    # Append the predicted symbol and its bounding box to the list
    symbols.append(symbol)

ast = AST(symbols)
markdown = ast.render_latex_markdown()

print(markdown)
