"""Phase 2 entry point: validate the processed YOLOv11 detection dataset.

Read-only against DATA_SETS/PROCESSED/detection_yolo and its manifest.
Writes a validation report, CSVs of any issues found, and a visual-QA sample
under outputs/module1/detection/dataset_validation/. Never modifies the
dataset, never trains a model.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.annotation_validation import YoloBox
from src.detection.bbox_statistics import compute_bbox_statistics, compute_per_image_box_counts
from src.detection.dataset_validation import validate_detection_dataset
from src.detection.leakage_validation import find_cross_split_duplicates, find_patient_split_conflicts
from src.detection.visualization import save_annotated_image

DATASET_ROOT = Path(r"D:\FYP\DATA_SETS\PROCESSED\detection_yolo")
MANIFEST_PATH = Path(r"D:\FYP\DATA_SETS\PROCESSED\reports\split_manifest.csv")
CROSS_DATASET_DUP_REPORT = Path(r"D:\FYP\DATA_SETS\PROCESSED\reports\cross_dataset_duplicate_report.csv")
OUTPUT_DIR = REPO_ROOT / "outputs" / "module1" / "detection" / "dataset_validation"
SPLITS = ("train", "val", "test")
SAMPLES_PER_SOURCE = 8
RANDOM_SEED = 42


def load_manifest_rows() -> list[dict[str, str]]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_source_lookup(rows: list[dict[str, str]]) -> dict[str, str]:
    """Map detection-image filename -> source_dataset, from the manifest."""
    lookup = {}
    for row in rows:
        if row["task"] == "detection" and row["status"] == "INCLUDED" and row["derived_path"]:
            filename = Path(row["derived_path"]).name
            lookup[filename] = row["source_dataset"]
    return lookup


def check_cross_dataset_duplicates_excluded(rows: list[dict[str, str]]) -> dict:
    """Verify the known CODE<->SMART-OM cross-dataset duplicates are absent
    from the final INCLUDED detection set, by checking their original_path
    against every INCLUDED detection row's original_path."""

    included_original_paths = {
        row["original_path"] for row in rows if row["task"] == "detection" and row["status"] == "INCLUDED"
    }

    with open(CROSS_DATASET_DUP_REPORT, newline="", encoding="utf-8") as f:
        dup_rows = list(csv.DictReader(f))

    still_present = []
    for dup_row in dup_rows:
        for key in ("example_path_a", "example_path_b"):
            path = dup_row.get(key, "")
            if path in included_original_paths:
                still_present.append((dup_row, key, path))

    return {"total_cross_dataset_dup_rows": len(dup_rows), "still_present_in_detection": still_present}


def main() -> None:
    print("=" * 60)
    print("Phase 2 — YOLOv11 Detection Dataset Validation")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Structural validation (images, labels, pairing)
    print("\n[1/6] Structural validation (images/labels/pairing)...")
    t0 = time.perf_counter()
    split_reports = validate_detection_dataset(DATASET_ROOT, SPLITS)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    total_images = sum(len(r.pairs) for r in split_reports.values())
    total_missing_labels = sum(len(r.missing_labels) for r in split_reports.values())
    total_orphan_labels = sum(len(r.orphan_labels) for r in split_reports.values())
    total_unreadable = sum(len(r.unreadable_images) for r in split_reports.values())
    total_invalid_labels = sum(len(r.invalid_labels) for r in split_reports.values())
    total_empty_labels = sum(len(r.empty_labels) for r in split_reports.values())
    total_valid_pairs = sum(len(r.valid_pairs) for r in split_reports.values())

    # 2. Manifest / source lookup
    print("[2/6] Loading manifest and building source lookup...")
    manifest_rows = load_manifest_rows()
    source_lookup = build_source_lookup(manifest_rows)

    source_counts: dict[str, int] = defaultdict(int)
    unmatched_source = 0
    for report in split_reports.values():
        for pair in report.pairs:
            if pair.image_path is not None:
                source = source_lookup.get(pair.image_path.name)
                if source:
                    source_counts[source] += 1
                else:
                    unmatched_source += 1

    # 3. Cross-dataset duplicate re-verification
    print("[3/6] Verifying known cross-dataset (CODE<->SMART-OM) duplicates are excluded...")
    cross_dup_check = check_cross_dataset_duplicates_excluded(manifest_rows)

    # 4. Leakage checks (hash-based cross-split, patient-level)
    print("[4/6] Leakage checks (hash-based cross-split + patient-level)...")
    t0 = time.perf_counter()
    dup_report = find_cross_split_duplicates({s: DATASET_ROOT / "images" / s for s in SPLITS})
    print(f"  hashed {dup_report.total_images_hashed} images in {time.perf_counter() - t0:.1f}s")
    patient_report = find_patient_split_conflicts(manifest_rows, task="detection")

    # 5. Bounding-box statistics
    print("[5/6] Bounding-box statistics...")
    all_boxes: list[YoloBox] = []
    per_image_counts: list[int] = []
    for report in split_reports.values():
        for pair in report.pairs:
            if pair.label is not None and pair.label.is_valid:
                all_boxes.extend(pair.label.boxes)
                per_image_counts.append(len(pair.label.boxes))
    bbox_stats = compute_bbox_statistics(all_boxes)
    box_count_buckets = compute_per_image_box_counts(per_image_counts)

    # 6. Visual QA sampling
    print("[6/6] Visual QA sampling...")
    rng = random.Random(RANDOM_SEED)
    visual_qa_dir = OUTPUT_DIR / "visual_qa"
    by_source_valid_pairs: dict[str, list] = defaultdict(list)
    for report in split_reports.values():
        for pair in report.valid_pairs:
            if pair.image_path is not None:
                source = source_lookup.get(pair.image_path.name, "UNKNOWN")
                by_source_valid_pairs[source].append(pair)

    visual_qa_manifest = []
    for source, pairs in by_source_valid_pairs.items():
        sample = rng.sample(pairs, min(SAMPLES_PER_SOURCE, len(pairs)))
        for pair in sample:
            boxes = pair.label.boxes if pair.label else []
            out_path = visual_qa_dir / source / f"{pair.split}_{pair.stem}.jpg"
            save_annotated_image(pair.image_path, boxes, out_path)
            visual_qa_manifest.append(
                {"source": source, "split": pair.split, "stem": pair.stem, "num_boxes": len(boxes), "output": str(out_path)}
            )

    # Also visualize the worst bbox-statistic outliers regardless of source
    outlier_dir = visual_qa_dir / "_outliers"
    outlier_pairs = []
    for report in split_reports.values():
        for pair in report.valid_pairs:
            if pair.label is None:
                continue
            for box in pair.label.boxes:
                area = box.width * box.height
                ar = box.width / box.height if box.height else 0
                if area < 0.001 or area > 0.95 or ar > 5.0 or (ar and ar < 0.2):
                    outlier_pairs.append(pair)
                    break
    outlier_sample = rng.sample(outlier_pairs, min(10, len(outlier_pairs)))
    for pair in outlier_sample:
        boxes = pair.label.boxes if pair.label else []
        out_path = outlier_dir / f"{pair.split}_{pair.stem}.jpg"
        save_annotated_image(pair.image_path, boxes, out_path)

    # ---- Write CSV issue reports ----
    def _write_pairs_csv(path: Path, pairs: list, note: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "stem", "image_path", "label_path", "detail"])
            for p in pairs:
                detail = ""
                if p.label is not None and p.label.errors:
                    detail = "; ".join(p.label.errors)
                elif p.image is not None and p.image.error:
                    detail = p.image.error
                writer.writerow([p.split, p.stem, p.image_path, p.label_path, detail])

    all_missing = [p for r in split_reports.values() for p in r.missing_labels]
    all_orphan = [p for r in split_reports.values() for p in r.orphan_labels]
    all_invalid = [p for r in split_reports.values() for p in r.invalid_labels]
    all_empty = [p for r in split_reports.values() for p in r.empty_labels]
    all_unreadable = [p for r in split_reports.values() for p in r.unreadable_images]

    _write_pairs_csv(OUTPUT_DIR / "missing_labels.csv", all_missing, "missing label")
    _write_pairs_csv(OUTPUT_DIR / "orphan_labels.csv", all_orphan, "orphan label")
    _write_pairs_csv(OUTPUT_DIR / "invalid_labels.csv", all_invalid, "invalid label")
    _write_pairs_csv(OUTPUT_DIR / "empty_labels.csv", all_empty, "empty label")
    _write_pairs_csv(OUTPUT_DIR / "unreadable_images.csv", all_unreadable, "unreadable image")

    with open(OUTPUT_DIR / "visual_qa_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "split", "stem", "num_boxes", "output"])
        writer.writeheader()
        writer.writerows(visual_qa_manifest)

    # ---- Print summary to console (full report assembled by caller/report writer) ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images discovered: {total_images}")
    for split, report in split_reports.items():
        print(f"  {split}: {len(report.pairs)} pairs "
              f"({len(report.valid_pairs)} valid, {len(report.missing_labels)} missing-label, "
              f"{len(report.orphan_labels)} orphan-label, {len(report.invalid_labels)} invalid-label, "
              f"{len(report.empty_labels)} empty-label, {len(report.unreadable_images)} unreadable-image)")
    print(f"Total valid trainable pairs: {total_valid_pairs}")
    print(f"Source distribution: {dict(source_counts)} (unmatched: {unmatched_source})")
    print(f"Cross-split hash duplicates: {len(dup_report.duplicate_groups)} "
          f"(hashed {dup_report.total_images_hashed} images)")
    print(f"Patient-level split conflicts: {len(patient_report.conflicting_patients)} "
          f"(of {patient_report.total_patients_checked} patients checked)")
    print(f"Cross-dataset dup rows still present in INCLUDED detection set: "
          f"{len(cross_dup_check['still_present_in_detection'])} / {cross_dup_check['total_cross_dataset_dup_rows']}")
    print(f"Bbox count: {bbox_stats.count}, tiny: {bbox_stats.tiny_boxes}, "
          f"near-full: {bbox_stats.near_full_image_boxes}, extreme-AR: {bbox_stats.extreme_aspect_ratio_boxes}")
    print(f"Per-image lesion counts: 0={box_count_buckets.zero} 1={box_count_buckets.one} "
          f"2={box_count_buckets.two} 3+={box_count_buckets.three_or_more}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")

    # Persist a machine-readable summary for the report writer to consume.
    import json

    summary = {
        "total_images": total_images,
        "splits": {
            s: {
                "pairs": len(r.pairs),
                "valid_pairs": len(r.valid_pairs),
                "missing_labels": len(r.missing_labels),
                "orphan_labels": len(r.orphan_labels),
                "invalid_labels": len(r.invalid_labels),
                "empty_labels": len(r.empty_labels),
                "unreadable_images": len(r.unreadable_images),
            }
            for s, r in split_reports.items()
        },
        "total_valid_pairs": total_valid_pairs,
        "total_missing_labels": total_missing_labels,
        "total_orphan_labels": total_orphan_labels,
        "total_invalid_labels": total_invalid_labels,
        "total_empty_labels": total_empty_labels,
        "total_unreadable_images": total_unreadable,
        "source_distribution": dict(source_counts),
        "unmatched_source": unmatched_source,
        "cross_split_hash_duplicates": len(dup_report.duplicate_groups),
        "total_images_hashed": dup_report.total_images_hashed,
        "patient_conflicts": len(patient_report.conflicting_patients),
        "total_patients_checked": patient_report.total_patients_checked,
        "cross_dataset_dup_still_present": len(cross_dup_check["still_present_in_detection"]),
        "cross_dataset_dup_total_rows": cross_dup_check["total_cross_dataset_dup_rows"],
        "bbox_stats": {
            "count": bbox_stats.count,
            "width_mean": bbox_stats.width.mean, "width_p50": bbox_stats.width.p50,
            "height_mean": bbox_stats.height.mean, "height_p50": bbox_stats.height.p50,
            "area_fraction_mean": bbox_stats.area_fraction.mean, "area_fraction_p50": bbox_stats.area_fraction.p50,
            "aspect_ratio_mean": bbox_stats.aspect_ratio.mean, "aspect_ratio_p50": bbox_stats.aspect_ratio.p50,
            "tiny_boxes": bbox_stats.tiny_boxes,
            "near_full_image_boxes": bbox_stats.near_full_image_boxes,
            "extreme_aspect_ratio_boxes": bbox_stats.extreme_aspect_ratio_boxes,
        },
        "per_image_box_counts": {
            "zero": box_count_buckets.zero, "one": box_count_buckets.one,
            "two": box_count_buckets.two, "three_or_more": box_count_buckets.three_or_more,
            "min": box_count_buckets.min_annotations, "max": box_count_buckets.max_annotations,
            "mean": box_count_buckets.mean_annotations, "median": box_count_buckets.median_annotations,
        },
    }
    with open(OUTPUT_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
