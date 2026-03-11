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


SCRIPT_MARGIN = 0.7  # Has to be 70% to the right
FRACTION_MARGIN = 0.5  # Has to be 50% above/below

FRACTION_SYMBOL = "_"


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
# Math Spatial Relation Tree
#
class MathTree:
    """A helper class to build a math expression tree based on the spatial relationships of the symbols.

    Assumes increasing y is upwards to make understanding equations easier.
    """

    def __init__(self) -> None:
        self.children: list[Symbol] = []

    def add(self, symbol: Symbol) -> None:
        """Adds a child node to the MathTree."""
        self.children.append(symbol)

    def clear(self) -> None:
        """Clears all child nodes from the MathTree."""
        self.children.clear()

    @staticmethod
    def is_superscript_of(a: Symbol, b: Symbol) -> bool:
        """Returns True if B is a superscript of A."""
        return (
            b.box.center_x() > (a.box.center_x() + a.box.width * SCRIPT_MARGIN)  # B right A within margin
            and b.box.center_y() > (a.box.center_y() + a.box.height * SCRIPT_MARGIN)  # B above A within margin
        )

    @staticmethod
    def is_subscript_of(a: Symbol, b: Symbol) -> bool:
        """Returns True if B is a subscript of A."""
        return (
            b.box.center_x() > (a.box.center_x() + a.box.width * SCRIPT_MARGIN)  # B right of A within margin
            and b.box.center_y() < (a.box.center_y() - a.box.height * SCRIPT_MARGIN)  # B below A within margin
        )

    @staticmethod
    def is_above(a: Symbol, b: Symbol) -> bool:
        """Returns True if B is above A."""
        return b.box.center_y() > a.box.center_y() + a.box.height * FRACTION_MARGIN  # B above A within margin

    @staticmethod
    def is_below(a: Symbol, b: Symbol) -> bool:
        """Returns True if B is below A."""
        return b.box.center_y() < a.box.center_y() - a.box.height * FRACTION_MARGIN  # B below A within margin

    @staticmethod
    def symbol_priority(symbol: Symbol) -> int:
        """Returns the priority of a symbol based on its type and value.

        Fraction symbols have the highest priority, followed by math symbols, and anything else.
        """
        if symbol.value == FRACTION_SYMBOL:
            return 2
        if symbol.type == SymbolType.MATH:
            return 1
        return 0

    @staticmethod
    def dominates(a: Symbol, b: Symbol) -> bool:  # noqa: PLR0911
        """Determines if A dominates B based on their bounding boxes.

        Uses the following rules to determine the dominant symbol:
            1. B is in region of A, and A is not in region of B
            2. Symbol class (e.g. fraction > superscript > subscript > normal)
            2. Area is used as a tiebreaker if both A and B are in each other's region
        """
        # Spatial relation checks
        a_has_b = (
            MathTree.is_superscript_of(a, b)
            or MathTree.is_subscript_of(a, b)
            or MathTree.is_above(a, b)
            or MathTree.is_below(a, b)
        )
        b_has_a = (
            MathTree.is_superscript_of(b, a)
            or MathTree.is_subscript_of(b, a)
            or MathTree.is_above(b, a)
            or MathTree.is_below(b, a)
        )

        # A has B within region and B does not have A: then A dominates B
        if a_has_b and not b_has_a:
            return True
        # B has A within region and A does not have B: then B dominates A
        if b_has_a and not a_has_b:
            return False

        # Symbol class priority (operators, fraction bar, etc.)
        if MathTree.symbol_priority(a) > MathTree.symbol_priority(b):
            return True
        if MathTree.symbol_priority(b) > MathTree.symbol_priority(a):
            return False

        # Size comparison (area, then height) since priorities are equal
        if a.box.area() > b.box.area():
            return True
        return a.box.height > b.box.height

    @staticmethod
    def get_dominant_symbol(symbols: list[Symbol]) -> Symbol:
        """Determines the dominant symbol in a list of symbols based on their bounding boxes."""
        if len(symbols) == 1:
            return symbols[0]

        # Find the symbol that dominates the most other symbols
        best_symbol = symbols[0]
        best_score = 0

        for candidate in symbols:
            score = sum(1 for other in symbols if candidate != other and MathTree.dominates(candidate, other))
            if score > best_score:
                best_score = score
                best_symbol = candidate

        return best_symbol

    def to_math_node(self) -> MathNode:
        """Converts the MathTree to a MathNode."""
        math_node = MathNode()

        # Sort symbols by their position (left to right)
        symbols = self.children
        symbols.sort(key=lambda s: s.box.x)

        dominant_symbol = MathTree.get_dominant_symbol(symbols)

        # TODO: Build math node

        return math_node


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
        math_tree = MathTree()

        for symbol in symbols:
            # TODO: Use bounding box spatial relationships
            # TODO: Clustering based on box positions?

            label_text = Text(symbol.value)
            if symbol.type == SymbolType.MATH:
                math_tree.add(symbol)
            else:
                # Add and clear the math node if we encounter a text symbol after math symbols
                if math_tree.children:
                    nodes.append(math_tree.to_math_node())
                    math_tree.clear()

                nodes.append(label_text)

        if math_tree.children:
            nodes.append(math_tree.to_math_node())

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
        Symbol("1", SymbolType.MATH, BoundingBox(8, -7, 6, 6)),  # Slightly below the x (subscript)
        Symbol("+", SymbolType.MATH, BoundingBox(10, 0, 10, 10)),
        Symbol("y", SymbolType.MATH, BoundingBox(20, 0, 10, 10)),
        Symbol("2", SymbolType.MATH, BoundingBox(30, 8, 5, 5)),  # Slightly above the y (superscript)
        Symbol("_", SymbolType.MATH, BoundingBox(23, -15, 30, 5)),  # Fraction below
        Symbol("13", SymbolType.MATH, BoundingBox(20, -27, 10, 10)),  # Denominator
    ]

    ast = AST(symbols)

    # Fraction line should be dominant symbol here
    print(MathTree.get_dominant_symbol(symbols[1:]))
    print("Dominance scores:")
    for candidate in symbols[1:]:
        score = sum(1 for other in symbols[1:] if candidate != other and MathTree.dominates(candidate, other))
        print(f"  {candidate.value} (area={candidate.box.area()}, height={candidate.box.height}): score={score}")

    print()

    # FIXME: Use structure detection
    print(ast.render_latex_markdown())
