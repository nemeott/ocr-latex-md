from bounding_box import BoundingBox
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


SCRIPT_MARGIN = 0.3  # Has to be 30% above/below and right of the base symbol
FRACTION_MARGIN = 0.5  # Has to be 50% above/below the base symbol to be considered a fraction component

BASELINE_TOLERANCE = 0.20  # Expands dominant symbol vertical band by this fraction of its height

FRACTION_SYMBOL = "_"


class Node:
    pass


class Text(Node):
    """Represents a piece of text. Can be a single symbol or a sequence of symbols."""

    def __init__(self, text: str) -> None:
        self.text: str = text

    def __str__(self) -> str:
        return self.text


class MathNode(Node):
    """Represents a mathematical expression, which can contain multiple child nodes, including Text nodes."""

    def __init__(self, children: list[Node] | None = None) -> None:
        self.children: list[Node] = [] if children is None else children

    def add(self, node: Node) -> None:
        """Adds a child node to the MathNode."""
        self.children.append(node)

    def clear(self) -> None:
        """Clears all child nodes from the MathNode."""
        self.children.clear()

    def render(self) -> str:
        """Renders children without adding math mode delimiters."""
        return "".join(str(child) for child in self.children)

    def __str__(self) -> str:
        return f"${self.render()}$"


# TODO: Multiline math node


class Superscript(Node):
    def __init__(self, base: Node, superscript: Node) -> None:
        self.base: Node = base
        self.superscript: Node = superscript

    def __str__(self) -> str:
        # Use braces for multi-character superscripts
        sup = str(self.superscript)
        if len(sup) == 1:
            return f"{self.base}^{sup}"
        return f"{self.base}^{{{sup}}}"


class Subscript(Node):
    def __init__(self, base: Node, subscript: Node) -> None:
        self.base: Node = base
        self.subscript: Node = subscript

    def __str__(self) -> str:
        # Use braces for multi-character subscripts
        sub = str(self.subscript)
        if len(sub) == 1:
            return f"{self.base}_{sub}"
        return f"{self.base}_{{{sub}}}"


class Fraction(Node):
    def __init__(self, numerator: Node, denominator: Node) -> None:
        self.numerator: Node = numerator
        self.denominator: Node = denominator

    @staticmethod
    def _render_child(node: Node) -> str:
        r"""Render a child node suitable for embedding inside \frac{...}{...}.

        If the child is a MathNode, don't include the surrounding $...$.
        """
        if isinstance(node, MathNode):
            return node.render()
        return str(node)

    def __str__(self) -> str:
        numer = Fraction._render_child(self.numerator)
        denom = Fraction._render_child(self.denominator)
        return f"\\frac{{{numer}}}{{{denom}}}"


#
# Math Spatial Relation Tree
#
class MathTree:
    """A helper class to build a math expression tree based on the spatial relationships of the symbols.

    Assumes increasing y is upwards to make understanding spatial relations and equations easier.
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
    def is_superscript_of(b: Symbol, a: Symbol) -> bool:
        """Returns True if B is a superscript of A."""
        return (
            b.box.center_x() > (a.box.center_x() + a.box.width * SCRIPT_MARGIN)  # B right A within margin
            and b.box.center_y() > (a.box.center_y() + a.box.height * SCRIPT_MARGIN)  # B above A within margin
        )

    @staticmethod
    def is_subscript_of(b: Symbol, a: Symbol) -> bool:
        """Returns True if B is a subscript of A."""
        return (
            b.box.center_x() > (a.box.center_x() + a.box.width * SCRIPT_MARGIN)  # B right of A within margin
            and b.box.center_y() < (a.box.center_y() - a.box.height * SCRIPT_MARGIN)  # B below A within margin
        )

    # TODO: Refactor
    @staticmethod
    def is_above(b: Symbol, a: Symbol) -> bool:
        """Returns True if B is above A."""
        return b.box.center_y() > a.box.center_y() + a.box.height * FRACTION_MARGIN  # B above A within margin

    @staticmethod
    def is_below(b: Symbol, a: Symbol) -> bool:
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
    def dominates(a: Symbol, b: Symbol) -> bool:
        """Determines if A dominates B based on their bounding boxes.

        Uses the following rules to determine the dominant symbol:
            1. B is in region of A, and A is not in region of B
            2. Symbol class (e.g. fraction > math > normal)
            2. Area is used as a tiebreaker if both A and B are in each other's region
        """
        # Spatial relation checks
        a_has_b = (
            MathTree.is_superscript_of(b, a)
            or MathTree.is_subscript_of(b, a)
            or MathTree.is_above(b, a)
            or MathTree.is_below(b, a)
        )
        b_has_a = (
            MathTree.is_superscript_of(a, b)
            or MathTree.is_subscript_of(a, b)
            or MathTree.is_above(a, b)
            or MathTree.is_below(a, b)
        )

        # A has B within region and B does not have A: then A dominates B
        if a_has_b and not b_has_a:
            return True
        # B has A within region and A does not have B: then B dominates A
        if b_has_a and not a_has_b:
            return False

        # Symbol class priority (fraction, math, other)
        a_priority = MathTree.symbol_priority(a)
        b_priority = MathTree.symbol_priority(b)
        if a_priority > b_priority:
            return True
        if b_priority > a_priority:
            return False

        # Size comparison (area, then height) since priorities are equal
        if a.box.area() > b.box.area():
            return True
        return a.box.height > b.box.height

    @staticmethod
    def get_dominant_symbol_index(symbols: list[Symbol]) -> int:
        """Determines the dominant symbol in a list of symbols based on their bounding boxes."""
        if len(symbols) == 1:
            return 0

        # Find the symbol that dominates the most other symbols
        best_score = 0
        best_index = 0
        for i, candidate in enumerate(symbols):
            score = sum(1 for other in symbols if candidate != other and MathTree.dominates(candidate, other))
            if score > best_score:
                best_score = score
                best_index = i

        return best_index

    @staticmethod
    def _get_anchor_index(sym: Symbol, anchors: list[Symbol]) -> int:
        """Returns index of best baseline anchor for a symbol based on x-overlap, else nearest by x distance."""
        sx0 = sym.box.x
        sx1 = sym.box.x + sym.box.width
        scx = sym.box.center_x()

        best_overlap = -1
        best_dist = float("inf")
        best_i = 0

        for i, anc in enumerate(anchors):
            ax0 = anc.box.x
            ax1 = anc.box.x + anc.box.width
            acx = anc.box.center_x()

            overlap = max(0, min(sx1, ax1) - max(sx0, ax0))
            dist = abs(scx - acx)

            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best_overlap = overlap
                best_dist = dist
                best_i = i

        return best_i

    @staticmethod
    def _split_fraction_groups(
        symbols: list[Symbol],
        bar: Symbol,
    ) -> tuple[list[Symbol], list[Symbol], list[Symbol], list[Symbol]]:
        """Split fraction candidates plus leftover symbols not participating in the fraction.

        Heuristic:
        - Only consider symbols that horizontally overlap the bar (common for printed fractions)
        - Use sign of center_y - bar_center_y to decide numerator vs denominator

        Returns: (left_syms, numerator_syms, denominator_syms, right_syms)
        """
        bar_x0 = bar.box.x
        bar_x1 = bar.box.x + bar.box.width
        bar_cy = bar.box.center_y()

        left_syms: list[Symbol] = []
        numerators: list[Symbol] = []
        denominators: list[Symbol] = []
        right_syms: list[Symbol] = []

        for sym in symbols:
            if sym is bar:
                continue

            # Only consider symbols that horizontally overlap the bar
            sx0 = sym.box.x
            sx1 = sym.box.x + sym.box.width
            overlaps_bar = max(0, min(sx1, bar_x1) - max(sx0, bar_x0)) > 0

            # Use vertical position to determine numerator or denominator if overlapping
            if overlaps_bar:
                if sym.box.center_y() > bar_cy:
                    numerators.append(sym)
                    continue
                if sym.box.center_y() < bar_cy:
                    denominators.append(sym)
                    continue

            # Non-overlapping symbols: route left/right based on horizontal position relative to bar
            scx = sym.box.center_x()
            bcx = bar.box.center_x()
            if scx <= bcx:
                left_syms.append(sym)
            elif bcx < scx:
                right_syms.append(sym)

        return left_syms, numerators, denominators, right_syms

    def to_math_node_no_fraction(self) -> MathNode:
        """Converts the MathTree to a MathNode, but does NOT apply the fraction-special-case.

        This is used when recursively rendering numerator/denominator groups inside a fraction to
        prevent the fraction bar heuristic from re-triggering and causing infinite recursion.
        """
        math_node = MathNode()

        # Sort symbols by their position (left to right)
        symbols = list(self.children)
        symbols.sort(key=lambda s: s.box.x)

        if not symbols:
            return math_node

        # Find dominant symbol (but ignore fraction-bar special-casing here)
        dominant_symbol_index = MathTree.get_dominant_symbol_index(symbols)
        dominant_symbol = symbols[dominant_symbol_index]

        # Baseline band for expressions
        dom_top = dominant_symbol.box.top()
        dom_bottom = dominant_symbol.box.bottom()
        dom_h = dom_top - dom_bottom
        tol = BASELINE_TOLERANCE * dom_h

        baseline_top = dom_top - tol
        baseline_bottom = dom_bottom + tol

        # Partition into baseline, above, below
        baseline_syms: list[Symbol] = []
        above_syms: list[Symbol] = []
        below_syms: list[Symbol] = []

        for sym in symbols:
            if baseline_bottom <= sym.box.center_y() <= baseline_top:  # On band
                baseline_syms.append(sym)
            elif sym.box.center_y() > baseline_top:  # Above band
                above_syms.append(sym)
            elif sym.box.center_y() < baseline_bottom:  # Below band
                below_syms.append(sym)
            else:
                # Fallback
                baseline_syms.append(sym)

        if not baseline_syms:
            # If baseline detection failed, return all symbols as they came in x order
            for sym in symbols:
                math_node.add(Text(sym.value))
            return math_node

        # Group superscripts/subscripts by baseline anchor
        supers_by_anchor: dict[int, list[Symbol]] = {i: [] for i in range(len(baseline_syms))}
        subs_by_anchor: dict[int, list[Symbol]] = {i: [] for i in range(len(baseline_syms))}

        for sym in above_syms:
            anchor_index = MathTree._get_anchor_index(sym, baseline_syms)
            supers_by_anchor[anchor_index].append(sym)

        for sym in below_syms:
            anchor_index = MathTree._get_anchor_index(sym, baseline_syms)
            subs_by_anchor[anchor_index].append(sym)

        # Emit baseline left-to-right, attaching scripts where present
        for i, base_sym in enumerate(baseline_syms):
            base_node: Node = Text(base_sym.value)

            # Only attach scripts that are reasonably to the right (reuse existing heuristics)
            # Prevents some cases where a symbol above earlier baseline gets incorrectly attached
            super_syms = [s for s in supers_by_anchor[i] if MathTree.is_superscript_of(s, base_sym)]
            sub_syms = [s for s in subs_by_anchor[i] if MathTree.is_subscript_of(s, base_sym)]

            super_syms.sort(key=lambda s: s.box.x)
            sub_syms.sort(key=lambda s: s.box.x)

            if super_syms:
                sup_text = Text("".join(s.value for s in super_syms))
                base_node = Superscript(base_node, sup_text)

            if sub_syms:
                sub_text = Text("".join(s.value for s in sub_syms))
                base_node = Subscript(base_node, sub_text)

            math_node.add(base_node)

        return math_node

    def to_math_node(self) -> MathNode:
        """Converts the MathTree to a MathNode.

        Current implementation:
        1) If the dominant symbol is a fraction bar AND it has both numerator and denominator groups, build a Fraction.
        2) Otherwise, build a baseline band from a non-bar dominant symbol (with tolerance).
        3) Pick baseline symbols whose vertical center lies in that band.
        4) Attach non-baseline symbols as simple superscripts/subscripts to the nearest baseline anchor by x.
        """
        math_node = MathNode()

        # Sort symbols by their position (left to right)
        symbols = list(self.children)
        symbols.sort(key=lambda s: s.box.x)

        if not symbols:
            return math_node

        # Find dominant symbol
        dominant_symbol_index = MathTree.get_dominant_symbol_index(symbols)
        dominant_symbol = symbols[dominant_symbol_index]

        # Fraction handling: only treat as a fraction if we can form BOTH numerator and denominator.
        # If we can't, we should NOT let the bar influence baseline parsing
        if dominant_symbol.value == FRACTION_SYMBOL:
            left_syms, numerator_syms, denominator_syms, right_syms = MathTree._split_fraction_groups(
                symbols, dominant_symbol
            )

            if numerator_syms and denominator_syms:
                # IMPORTANT:
                # `to_math_node_no_fraction()` returns a *MathNode*, whose __str__ wraps with `$...$`.
                # When we embed that MathNode inside another MathNode (this `math_node`), we get
                # nested dollars like: `$$x+$\frac{...}{...}$`.
                #
                # To avoid that, inline left/right content by appending their children directly.
                if left_syms:
                    left_tree = MathTree()
                    for s in left_syms:
                        left_tree.add(s)
                    left_node = left_tree.to_math_node_no_fraction()
                    math_node.children.extend(left_node.children)

                # Render the fraction itself
                numerator_tree = MathTree()
                for s in numerator_syms:
                    numerator_tree.add(s)

                denominator_tree = MathTree()
                for s in denominator_syms:
                    denominator_tree.add(s)

                numerator_node: Node = numerator_tree.to_math_node_no_fraction()
                denominator_node: Node = denominator_tree.to_math_node_no_fraction()

                math_node.add(Fraction(numerator_node, denominator_node))

                # Render right side (inline children to avoid nested `$...$`)
                if right_syms:
                    right_tree = MathTree()
                    for s in right_syms:
                        right_tree.add(s)
                    right_node = right_tree.to_math_node_no_fraction()
                    math_node.children.extend(right_node.children)

                return math_node

        # Baseline band for non-fraction expressions
        dom_top = dominant_symbol.box.top()
        dom_bottom = dominant_symbol.box.bottom()
        dom_h = dom_top - dom_bottom
        tol = BASELINE_TOLERANCE * dom_h

        baseline_top = dom_top - tol
        baseline_bottom = dom_bottom + tol

        # Partition into baseline, above, below
        baseline_syms: list[Symbol] = []
        above_syms: list[Symbol] = []
        below_syms: list[Symbol] = []

        for sym in symbols:
            if baseline_bottom <= sym.box.center_y() <= baseline_top:  # On band
                baseline_syms.append(sym)
            elif sym.box.center_y() > baseline_top:  # Above band
                above_syms.append(sym)
            elif sym.box.center_y() < baseline_bottom:  # Below band
                below_syms.append(sym)
            else:
                # Fallback
                baseline_syms.append(sym)

        if not baseline_syms:
            # If baseline detection failed, return all symbols as they came in x order
            for sym in symbols:
                math_node.add(Text(sym.value))
            return math_node

        # Group superscripts/subscripts by baseline anchor
        supers_by_anchor: dict[int, list[Symbol]] = {i: [] for i in range(len(baseline_syms))}
        subs_by_anchor: dict[int, list[Symbol]] = {i: [] for i in range(len(baseline_syms))}

        for sym in above_syms:
            anchor_index = MathTree._get_anchor_index(sym, baseline_syms)
            supers_by_anchor[anchor_index].append(sym)

        for sym in below_syms:
            anchor_index = MathTree._get_anchor_index(sym, baseline_syms)
            subs_by_anchor[anchor_index].append(sym)

        # Emit baseline left-to-right, attaching scripts where present
        for i, base_sym in enumerate(baseline_syms):
            base_node: Node = Text(base_sym.value)

            # Only attach scripts that are reasonably to the right (reuse existing heuristics)
            # Prevents some cases where a symbol above earlier baseline gets incorrectly attached
            super_syms = [s for s in supers_by_anchor[i] if MathTree.is_superscript_of(base_sym, s)]
            sub_syms = [s for s in subs_by_anchor[i] if MathTree.is_subscript_of(base_sym, s)]

            super_syms.sort(key=lambda s: s.box.x)
            sub_syms.sort(key=lambda s: s.box.x)

            if super_syms:
                sup_text = Text("".join(s.value for s in super_syms))
                base_node = Superscript(base_node, sup_text)

            if sub_syms:
                sub_text = Text("".join(s.value for s in sub_syms))
                base_node = Subscript(base_node, sub_text)

            math_node.add(base_node)

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
    r"""
    Hello world!
    $x_1+y^2$

    Hello world!
    $x_1+\frac{y^2}{13}$

    Hello world!
    $\frac{x_1+y^2}{13}$
    """
    symbols: list[Symbol] = [
        Symbol("Hello world! ", SymbolType.TEXT, BoundingBox(0, 20, 100, 10)),
        Symbol("x", SymbolType.MATH, BoundingBox(0, 0, 10, 10)),
        Symbol("1", SymbolType.MATH, BoundingBox(8, -7, 6, 6)),  # Slightly below the x (subscript)
        Symbol("+", SymbolType.MATH, BoundingBox(10, 0, 10, 10)),
        Symbol("y", SymbolType.MATH, BoundingBox(20, 0, 10, 10)),
        Symbol("2", SymbolType.MATH, BoundingBox(30, 8, 5, 5)),  # Slightly above the y (superscript)
        Symbol("_", SymbolType.MATH, BoundingBox(23, -15, 35, 5)),  # Fraction below
        Symbol("65", SymbolType.MATH, BoundingBox(20, -27, 10, 10)),  # Denominator
    ]

    ast = AST(symbols)

    # Fraction line should be dominant symbol here
    print("Dominance scores:")
    for candidate in symbols[1:]:
        score = sum(1 for other in symbols[1:] if candidate != other and MathTree.dominates(candidate, other))
        print(f"  {candidate.value} (area={candidate.box.area()}, height={candidate.box.height}): score={score}")

    print()

    print(ast.render_latex_markdown())
