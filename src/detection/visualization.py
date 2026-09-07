"""Visual QA: draw YOLO bounding boxes on sampled images for human inspection.

Read-only against the dataset; all outputs are written under a caller-supplied
output directory (never back into DATA_SETS).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .annotation_validation import YoloBox


def draw_boxes(image, boxes: list[YoloBox], *, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Draw normalized YOLO boxes onto a copy of a BGR image.

    Args:
        image: BGR image array (as returned by ``cv2.imread``).
        boxes: Validated YOLO boxes (normalized coordinates) to draw.
        color: BGR box color.
        thickness: Line thickness in pixels.

    Returns:
        A new BGR image array with boxes drawn; the input is not mutated.
    """

    annotated = image.copy()
    height, width = image.shape[:2]

    for box in boxes:
        x_min = int((box.x_center - box.width / 2) * width)
        y_min = int((box.y_center - box.height / 2) * height)
        x_max = int((box.x_center + box.width / 2) * width)
        y_max = int((box.y_center + box.height / 2) * height)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)

    return annotated


def save_annotated_image(image_path: Path, boxes: list[YoloBox], output_path: Path) -> None:
    """Load an image, draw its boxes, and save the annotated copy.

    Args:
        image_path: Source image to load.
        boxes: Boxes to draw on it.
        output_path: Destination path for the annotated copy.

    Raises:
        ValueError: If the source image cannot be decoded.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to decode image for visualization: {image_path}")

    annotated = draw_boxes(image, boxes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
