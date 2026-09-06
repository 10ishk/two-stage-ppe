"""Dependency-light bounding-box geometry in XYXY pixel coordinates."""

from typing import Iterable, Tuple

BBox = Tuple[float, float, float, float]


def as_box(values: Iterable[float]) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in values)
    return x1, y1, x2, y2


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def clip_box(box: BBox, width: int, height: int) -> BBox:
    """Clip a box to image boundaries represented by width and height."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def expand_box(box: BBox, padding: float, width: int, height: int) -> BBox:
    """Expand every side by a fraction of box width/height, then clip."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    x1, y1, x2, y2 = box
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    return clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def crop_to_global(box: BBox, crop_origin: tuple[float, float]) -> BBox:
    """Translate a crop-relative box into full-image coordinates."""
    ox, oy = crop_origin
    x1, y1, x2, y2 = box
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy


def iou(a: BBox, b: BBox) -> float:
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def ioa(subject: BBox, container: BBox) -> float:
    """Return intersection divided by the subject box area."""
    area = box_area(subject)
    return intersection_area(subject, container) / area if area > 0 else 0.0


def valid_box(box: BBox) -> bool:
    return box[2] > box[0] and box[3] > box[1]

