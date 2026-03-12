# from classifier import
# from features import extract_features
from preprocessing import crop_character, load_image, preprocess
from segmentation import segment
from structure import AST
from symbol import Symbol

# Basic outline for classical OCR pipeline for LaTeX to Markdown conversion

train_data = prepare_training_data(n_math=115, n_text=115) #Both of these load in 115 samples. If you want entire dataset, leave parameters defaul. 
test_data = prepare_test_data(n_math=115, n_text=115)

classifier = SymbolClassifier()
# classifier.train() # TODO: Train classifier or load pre-trained model

# Segment each character
boxes = segment(image)

symbols: list[Symbol] = []
for box in boxes:
    # Get the cropped image of the character
    cropped = crop_character(image, box)

    # Extract the features from the cropped image
    features = extract_features(cropped)

    # Predict the symbol using a trained classifier
    symbol: Symbol = classifier.predict(features)

    # Append the predicted symbol and its bounding box to the list
    symbols.append(symbol)

ast = AST(symbols)
markdown = ast.render_latex_markdown()

print(markdown)
