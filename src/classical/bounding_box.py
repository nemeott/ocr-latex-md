class BoundingBox:
    """Represents a bounding box for a character in the image."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def top(self) -> int:
        """Returns the top y coordinate of the bounding box."""
        return self.y + self.height

    def bottom(self) -> int:
        """Returns the bottom y coordinate of the bounding box."""
        return self.y

    def center_x(self) -> float:
        """Returns the x-coordinate of the center of the bounding box."""
        return self.x + self.width / 2.0

    def center_y(self) -> float:
        """Returns the y-coordinate of the center of the bounding box."""
        return self.y + self.height / 2.0

    def center(self) -> tuple[float, float]:
        """Returns the center coordinates of the bounding box."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def area(self) -> int:
        """Returns the area of the bounding box."""
        return self.width * self.height

    def __str__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
