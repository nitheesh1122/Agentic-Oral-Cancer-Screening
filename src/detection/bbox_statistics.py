"""Aggregate statistics over parsed YOLO bounding boxes.

Pure functions over plain numbers/lists so they're trivially unit-testable;
no dependency on the dataset filesystem layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotation_validation import YoloBox


@dataclass(slots=True)
class Percentiles:
    p0: float
    p25: float
    p50: float
    p75: float
    p100: float
    mean: float


def _percentiles(values: list[float]) -> Percentiles:
    if not values:
        return Percentiles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = p / 100.0 * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return Percentiles(
        p0=ordered[0], p25=_pct(25), p50=_pct(50), p75=_pct(75), p100=ordered[-1],
        mean=sum(ordered) / n,
    )


@dataclass(slots=True)
class BoundingBoxStatistics:
    count: int
    width: Percentiles
    height: Percentiles
    area_fraction: Percentiles
    aspect_ratio: Percentiles
    tiny_boxes: int
    near_full_image_boxes: int
    extreme_aspect_ratio_boxes: int


@dataclass(slots=True)
class PerImageBoxCounts:
    zero: int = 0
    one: int = 0
    two: int = 0
    three_or_more: int = 0
    counts: list[int] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.counts)

    @property
    def min_annotations(self) -> int:
        return min(self.counts) if self.counts else 0

    @property
    def max_annotations(self) -> int:
        return max(self.counts) if self.counts else 0

    @property
    def mean_annotations(self) -> float:
        return sum(self.counts) / len(self.counts) if self.counts else 0.0

    @property
    def median_annotations(self) -> float:
        if not self.counts:
            return 0.0
        ordered = sorted(self.counts)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def compute_per_image_box_counts(box_counts_per_image: list[int]) -> PerImageBoxCounts:
    """Bucket per-image annotation counts into 0 / 1 / 2 / 3+ lesion buckets.

    Args:
        box_counts_per_image: One entry per image = number of valid boxes in it.

    Returns:
        A ``PerImageBoxCounts`` with bucket totals and raw counts.
    """

    result = PerImageBoxCounts(counts=list(box_counts_per_image))
    for n in box_counts_per_image:
        if n == 0:
            result.zero += 1
        elif n == 1:
            result.one += 1
        elif n == 2:
            result.two += 1
        else:
            result.three_or_more += 1
    return result


def compute_bbox_statistics(
    boxes: list[YoloBox],
    *,
    tiny_area_threshold: float = 0.001,
    near_full_area_threshold: float = 0.95,
    extreme_aspect_ratio: float = 5.0,
) -> BoundingBoxStatistics:
    """Compute width/height/area/aspect-ratio distributions and flag outliers.

    Args:
        boxes: Parsed, already-validated YOLO boxes (normalized coordinates).
        tiny_area_threshold: Boxes with area fraction below this are flagged as suspiciously tiny.
        near_full_area_threshold: Boxes with area fraction above this are flagged as near-full-image.
        extreme_aspect_ratio: Boxes with width/height or height/width above this are flagged.

    Returns:
        A ``BoundingBoxStatistics`` summary. All-zero/empty if ``boxes`` is empty.
    """

    widths = [b.width for b in boxes]
    heights = [b.height for b in boxes]
    areas = [b.width * b.height for b in boxes]
    aspect_ratios = [b.width / b.height for b in boxes if b.height > 0]

    tiny = sum(1 for a in areas if a < tiny_area_threshold)
    near_full = sum(1 for a in areas if a > near_full_area_threshold)
    extreme_ar = sum(1 for ar in aspect_ratios if ar > extreme_aspect_ratio or ar < 1 / extreme_aspect_ratio)

    return BoundingBoxStatistics(
        count=len(boxes),
        width=_percentiles(widths),
        height=_percentiles(heights),
        area_fraction=_percentiles(areas),
        aspect_ratio=_percentiles(aspect_ratios),
        tiny_boxes=tiny,
        near_full_image_boxes=near_full,
        extreme_aspect_ratio_boxes=extreme_ar,
    )
