"""Unit tests for YOLO label parsing/validation (src/detection/annotation_validation.py)."""

from __future__ import annotations

from src.detection.annotation_validation import parse_yolo_label_text


class TestValidLabels:
    def test_single_valid_box(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.4 0.4\n")
        assert result.is_valid
        assert not result.is_empty
        assert len(result.boxes) == 1
        box = result.boxes[0]
        assert box.class_id == 0
        assert box.x_center == 0.5

    def test_multiple_valid_boxes(self):
        result = parse_yolo_label_text("0 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.2 0.2\n")
        assert result.is_valid
        assert len(result.boxes) == 2

    def test_box_touching_image_boundary_is_valid(self):
        result = parse_yolo_label_text("0 0.05 0.5 0.1 0.5\n")
        assert result.is_valid

    def test_empty_file_is_valid_and_flagged_empty(self):
        result = parse_yolo_label_text("")
        assert result.is_valid
        assert result.is_empty
        assert result.boxes == []

    def test_whitespace_only_file_is_treated_as_empty(self):
        result = parse_yolo_label_text("   \n\n  \n")
        assert result.is_empty


class TestInvalidLabels:
    def test_wrong_field_count(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.4\n")
        assert not result.is_valid
        assert "expected 5 fields" in result.errors[0]

    def test_extra_fields(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.4 0.4 0.1\n")
        assert not result.is_valid

    def test_non_integer_class_id(self):
        result = parse_yolo_label_text("lesion 0.5 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "class id" in result.errors[0]

    def test_invalid_class_id(self):
        result = parse_yolo_label_text("1 0.5 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "invalid class id" in result.errors[0]

    def test_non_numeric_coordinate(self):
        result = parse_yolo_label_text("0 abc 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "non-numeric" in result.errors[0]

    def test_nan_coordinate(self):
        result = parse_yolo_label_text("0 nan 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "NaN" in result.errors[0]

    def test_infinite_coordinate(self):
        result = parse_yolo_label_text("0 inf 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "NaN/infinite" in result.errors[0]

    def test_x_center_out_of_range(self):
        result = parse_yolo_label_text("0 1.5 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert "x_center" in result.errors[0]

    def test_negative_width(self):
        result = parse_yolo_label_text("0 0.5 0.5 -0.1 0.4\n")
        assert not result.is_valid
        assert "width" in result.errors[0]

    def test_zero_width(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.0 0.4\n")
        assert not result.is_valid

    def test_zero_height(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.4 0.0\n")
        assert not result.is_valid

    def test_box_extends_outside_image_boundary(self):
        result = parse_yolo_label_text("0 0.9 0.9 0.5 0.5\n")
        assert not result.is_valid
        assert "outside the image boundary" in result.errors[0]

    def test_one_bad_row_does_not_block_other_valid_rows(self):
        result = parse_yolo_label_text("0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.4 0.4\n")
        assert not result.is_valid
        assert len(result.boxes) == 1
        assert len(result.errors) == 1
