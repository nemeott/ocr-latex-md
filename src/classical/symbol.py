from enum import Enum

from bounding_box import BoundingBox


class SymbolType(Enum):
    """Represents the type of a symbol."""

    TEXT = 0
    MATH = 1


class Symbol:
    """Represents a symbol in the image, which can be either text or math."""

    def __init__(self, symbol_value: str, symbol_type: SymbolType, bounding_box: BoundingBox) -> None:
        self.value = symbol_value
        self.type = symbol_type
        self.box: BoundingBox = bounding_box

    def __str__(self) -> str:
        return f"Symbol('{self.value}', {self.type}, {self.box})"

    def __repr__(self) -> str:
        return self.__str__()
