"""Simple segmentation visualizer.

Usage:
    python src/classical/visualize_segmentation.py path/to/image.jpg
    python src/classical/visualize_segmentation.py path/to/image.jpg --label --output out.png

Draws bounding boxes from `segmentation.segment` over the original image and saves the result.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from bounding_box import BoundingBox
from preprocessing import load_image
from segmentation import segment

PRESET_COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "white": (255, 255, 255)}


def parse_color(s: str) -> tuple[int, int, int]:
    """Parse a color string into a BGR tuple. Accepts preset names or 'B,G,R' format."""
    s = s.strip().lower()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError("Color triplet must have 3 integers (B,G,R).")
        return tuple(int(x) for x in parts)  # ty:ignore[invalid-return-type]

    return PRESET_COLORS.get(s, (0, 0, 255))  # Default to red


def draw_boxes(
    img: np.ndarray,
    boxes: Iterable[BoundingBox],
    color: tuple[int, int, int] = (0, 0, 255),  # Red
    thickness: int = 2,
    label: bool = False,
) -> np.ndarray:
    """Draw bounding boxes on the image using OpenCV."""
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        out = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        out = img.copy()

    # Determine font scale based on image size
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.0, max(h, w) / 1000.0))
    text_thick = max(1, thickness // 2)

    for i, box in enumerate(boxes):
        x1 = max(0, round(box.x))
        y1 = max(0, round(box.y))
        x2 = min(w - 1, round(box.x + box.width))
        y2 = min(h - 1, round(box.y + box.height))

        # Draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # Optionally draw label
        if label:
            text = str(i)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thick)
            ty = y1 - 4
            if ty - text_height < 0:
                ty = y1 + text_height + 4

            # Background
            cv2.rectangle(out, (x1 - 1, ty - text_height - 1), (x1 + text_width + 1, ty + 2), (0, 0, 0), -1)
            cv2.putText(out, text, (x1, ty), font, font_scale, (255, 255, 255), text_thick, lineType=cv2.LINE_AA)

    return out


def default_out_path(inp: Path) -> Path:
    """Given an input image path, return a default output path with '_seg' suffix."""
    return inp.with_name(inp.stem + "_seg" + inp.suffix)


def main():
    """Parse arguments, load image, segment, draw boxes, and save visualization."""
    parser = argparse.ArgumentParser(description="Visualize segmentation bounding boxes.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path (default: input_seg.ext)")
    parser.add_argument("--label", action="store_true", help="Draw numeric labels on boxes")
    parser.add_argument("--color", default="red", help="Color name (red/green/blue) or B,G,R triplet")
    parser.add_argument("--thickness", type=int, default=2, help="Rectangle thickness")
    parser.add_argument("--min-area", type=int, default=0, help="Filter boxes smaller than this area (display only)")
    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.image)
    if not input_path.exists():
        raise SystemExit(f"Image not found: {input_path}")

    color = parse_color(args.color)
    image = load_image(str(input_path))
    bounding_boxes = segment(image)

    # Optionally filter out small boxes before drawing
    if args.min_area > 0:
        bounding_boxes = [b for b in bounding_boxes if b.width * b.height >= args.min_area]

    # Draw boxes and save visualization
    visualization = draw_boxes(image, bounding_boxes, color=color, thickness=args.thickness, label=args.label)
    out_path = Path(args.output) if args.output else default_out_path(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), visualization)
    if not ok:
        print(f"Failed to write {out_path}")
    else:
        print(f"Wrote visualization to {out_path}")


if __name__ == "__main__":
    main()
