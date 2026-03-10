class BoundingBox:
    """Represents a bounding box for a character in the image."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def center(self) -> tuple[float, float]:
        """Returns the center coordinates of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def __str__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
