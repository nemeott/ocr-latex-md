from enum import Enum


class SymbolType(Enum):
    TEXT = 0
    MATH = 1


class Symbol:
    """Represents a symbol in the image, which can be either text or math."""

    def __init__(self, symbol_value: str, symbol_type: SymbolType) -> None:
        self.value = symbol_value
        self.type = symbol_type


# Classify symbols in images using classical ML techniques (SVM)
class SymbolClassifier:
    def train(self, X, y):
        pass

    # Need to return a string label for the predicted symbol somehow (could split into another function)
    def predict(self, X) -> Symbol:
        return Symbol("a", SymbolType.TEXT)
