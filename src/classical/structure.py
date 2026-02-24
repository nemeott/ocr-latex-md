from classifier import Symbol
from segmentation import BoundingBox

# Detects the structure of the document, such superscripts, subscripts, fractions, etc.
# Uses the spatial relationships between the symbols to build an abstract syntax tree (AST)

# Example AST
# [
#     Text("Hello world, here is a formula: ")
#     MathNode
#     ├── Superscript
#     │    ├── base: Text("x")
#     │    └── exponent: Text("2")
#     ├── Text("+")
#     └── Text("y")
# ]


class Node:
    pass


class Text(Node):
    """Represents a piece of text. Can be a single symbol or a sequence of symbols."""

    def __init__(self, text: str) -> None:
        self.text = text


class MathNode(Node):
    """Represents a mathematical expression, which can contain multiple child nodes, including Text nodes."""

    def __init__(self) -> None:
        self.children: list[Node] = []

    def add(self, node: Node) -> None:
        """Adds a child node to the MathNode."""
        self.children.append(node)

    def clear(self) -> None:
        """Clears all child nodes from the MathNode."""
        self.children.clear()


class Superscript(Node):
    def __init__(self, base: Node, superscript: Node) -> None:
        self.base = base
        self.superscript = superscript


class Subscript(Node):
    def __init__(self, base: Node, subscript: Node) -> None:
        self.base = base
        self.subscript = subscript


class Fraction(Node):
    def __init__(self, numerator: Node, denominator: Node) -> None:
        self.numerator = numerator
        self.denominator = denominator


#
# Abstract Syntax Tree (AST)
#


class AST:
    def __init__(self, symbols: list[tuple[Symbol, BoundingBox]]) -> None:
        self.root: list[Node] = self._build_structure(symbols)

    # Build an AST from the list of symbols and their bounding boxes
    @staticmethod
    def _build_structure(symbols: list[tuple[Symbol, BoundingBox]]) -> list[Node]:
        nodes: list[Node] = []
        math_node = MathNode()

        for label, box in symbols:
            # TODO: Use bounding box spatial relationships
            # TODO: Clustering based on box positions?
            nodes.append(Text(label.value))

        return nodes

    # TODO: Render methods
    # Render the AST to LaTeX & Markdown
    def render_latex_markdown(self) -> str:
        return "a"
