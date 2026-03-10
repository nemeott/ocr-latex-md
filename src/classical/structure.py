from segmentation import BoundingBox
from symbol import Symbol, SymbolType

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

    def __str__(self) -> str:
        return self.text


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

    def __str__(self) -> str:
        return f"${''.join(str(child) for child in self.children)}$"


# TODO: Multiline math node


class Superscript(Node):
    def __init__(self, base: Node, superscript: Node) -> None:
        self.base = base
        self.superscript = superscript

    def __str__(self) -> str:
        return f"{self.base}^{self.superscript}"


class Subscript(Node):
    def __init__(self, base: Node, subscript: Node) -> None:
        self.base = base
        self.subscript = subscript

    def __str__(self) -> str:
        return f"{self.base}_{self.subscript}"


class Fraction(Node):
    def __init__(self, numerator: Node, denominator: Node) -> None:
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self) -> str:
        return f"\\frac{{{self.numerator}}}{{{self.denominator}}}"


#
# Abstract Syntax Tree (AST)
#


class AST:
    def __init__(self, symbols: list[Symbol]) -> None:
        self.root: list[Node] = self._build_structure(symbols)

    # Build an AST from the list of symbols and their bounding boxes
    @staticmethod
    def _build_structure(symbols: list[Symbol]) -> list[Node]:
        nodes: list[Node] = []
        math_node = MathNode()

        for symbol in symbols:
            # TODO: Use bounding box spatial relationships
            # TODO: Clustering based on box positions?

            label_text = Text(symbol.value)
            if symbol.type == SymbolType.MATH:
                math_node.add(label_text)
            else:
                # Add and clear the math node if we encounter a text symbol after math symbols
                if math_node.children:
                    nodes.append(math_node)
                    math_node.clear()

                nodes.append(label_text)

        if math_node.children:
            nodes.append(math_node)

        return nodes

    # TODO: Render methods
    # Render the AST to LaTeX & Markdown
    def render_latex_markdown(self) -> str:
        return "".join(str(node) for node in self.root)


# Testing
if __name__ == "__main__":
    """
    Hello world!
    $x+y^2$
    """
    symbols: list[Symbol] = [
        Symbol("Hello world! ", SymbolType.TEXT, BoundingBox(0, 20, 100, 10)),
        Symbol("x", SymbolType.MATH, BoundingBox(0, 0, 10, 10)),
        Symbol("+", SymbolType.MATH, BoundingBox(10, 0, 10, 10)),
        Symbol("y", SymbolType.MATH, BoundingBox(20, 0, 10, 10)),
        Symbol("2", SymbolType.MATH, BoundingBox(30, 8, 5, 5)),  # Slightly above the y
    ]

    ast = AST(symbols)

    print(ast.render_latex_markdown())
