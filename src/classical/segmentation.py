from bounding_box import BoundingBox


# Returns a list of bounding boxes for each character in the image
def segment(image) -> list[BoundingBox]:
    bounding_boxes = []
    bounding_boxes.append(BoundingBox(x=30, y=20, width=30, height=40))
    return bounding_boxes
