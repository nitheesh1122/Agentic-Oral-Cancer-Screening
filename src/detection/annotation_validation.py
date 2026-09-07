"""YOLO detection-label parsing and validation.

A YOLO label file has one row per object:
    class_id x_center y_center width height
with all five values whitespace-separated and the four geometry values
normalized to [0, 1] relative to image width/height. This project uses a
single unified class (``0 = lesion``), so any other class id is invalid.

Functions here are pure and side-effect-free so they can be unit-tested with
synthetic label text without touching the real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

VALID_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class YoloBox:
    """A single validated YOLO detection annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(slots=True)
class LabelValidationResult:
    """Outcome of validating one label file's content.

    Attributes:
        boxes: Successfully parsed and validated boxes.
        errors: Human-readable descriptions of every rejected row.
        is_empty: True if the label file had zero rows (no annotations).
    """

    boxes: list[YoloBox] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_empty: bool = False

    @property
    def is_valid(self) -> bool:
        """True if every row parsed was valid (empty files are considered valid)."""
        return not self.errors


def parse_yolo_label_text(text: str) -> LabelValidationResult:
    """Parse and validate the full text content of one YOLO label file.

    Args:
        text: Raw text content of a ``.txt`` label file.

    Returns:
        A ``LabelValidationResult`` with parsed boxes and/or per-row errors.
    """

    result = LabelValidationResult()
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        result.is_empty = True
        return result

    for line_no, line in enumerate(lines, start=1):
        error = _validate_row(line, line_no, result.boxes)
        if error:
            result.errors.append(error)

    return result


def _validate_row(line: str, line_no: int, boxes: list[YoloBox]) -> str | None:
    """Validate one label row; on success appends to ``boxes`` and returns None."""

    parts = line.split()
    if len(parts) != 5:
        return f"line {line_no}: expected 5 fields (class x y w h), got {len(parts)}: {line!r}"

    raw_class_id, raw_x, raw_y, raw_w, raw_h = parts

    try:
        class_id = int(raw_class_id)
    except ValueError:
        return f"line {line_no}: class id is not an integer: {raw_class_id!r}"

    try:
        x_center, y_center, width, height = (float(v) for v in (raw_x, raw_y, raw_w, raw_h))
    except ValueError:
        return f"line {line_no}: non-numeric coordinate value in {line!r}"

    for name, value in (("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)):
        if math.isnan(value) or math.isinf(value):
            return f"line {line_no}: {name} is NaN/infinite: {value}"

    if class_id != VALID_CLASS_ID:
        return f"line {line_no}: invalid class id {class_id} (only {VALID_CLASS_ID} = lesion is valid)"

    if not (0.0 <= x_center <= 1.0):
        return f"line {line_no}: x_center {x_center} outside [0, 1]"
    if not (0.0 <= y_center <= 1.0):
        return f"line {line_no}: y_center {y_center} outside [0, 1]"
    if not (0.0 < width <= 1.0):
        return f"line {line_no}: width {width} outside (0, 1]"
    if not (0.0 < height <= 1.0):
        return f"line {line_no}: height {height} outside (0, 1]"

    x_min, x_max = x_center - width / 2, x_center + width / 2
    y_min, y_max = y_center - height / 2, y_center + height / 2
    # Small epsilon tolerates float rounding at the exact image boundary.
    eps = 1e-4
    if x_min < -eps or x_max > 1.0 + eps or y_min < -eps or y_max > 1.0 + eps:
        return (
            f"line {line_no}: bounding box [{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}] "
            "extends outside the image boundary"
        )

    boxes.append(YoloBox(class_id=class_id, x_center=x_center, y_center=y_center, width=width, height=height))
    return None
