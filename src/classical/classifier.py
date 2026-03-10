from segmentation import BoundingBox
from symbol import Symbol, SymbolType


# Classify symbols in images using classical ML techniques (SVM)
# Can use a MultiOutputClassifier (scikit-learn) to predict both symbol and type (TEXT or MATH)
class SymbolClassifier:
    def train(self, X, y):
        pass

    # Need to return a string label for the predicted symbol somehow (could split into another function)
    def predict(self, X) -> Symbol:
        return Symbol("a", SymbolType.TEXT, BoundingBox(0, 0, 10, 10))
