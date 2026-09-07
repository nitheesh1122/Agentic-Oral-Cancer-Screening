"""Data-leakage checks for the detection dataset: exact-duplicate images
across splits, and patient-level grouping violations across splits.

These are independent, from-scratch checks -- they do not trust the
pipeline's own duplicate/manifest reports, they re-derive the answer from
the actual files (image content hashes) and the manifest's raw rows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CrossSplitDuplicate:
    """One content hash that appears in more than one split."""

    file_hash: str
    splits: set[str]
    paths: list[Path]


@dataclass(slots=True)
class DuplicateLeakageReport:
    total_images_hashed: int
    duplicate_groups: list[CrossSplitDuplicate] = field(default_factory=list)

    @property
    def has_leakage(self) -> bool:
        return len(self.duplicate_groups) > 0


def hash_file(path: Path) -> str:
    """Return the MD5 hex digest of a file's raw bytes."""

    return hashlib.md5(path.read_bytes()).hexdigest()


def find_cross_split_duplicates(split_to_images_dir: dict[str, Path]) -> DuplicateLeakageReport:
    """Hash every image in every split and flag any hash shared across splits.

    Args:
        split_to_images_dir: Mapping of split name -> directory of image files
            for that split (e.g. ``{"train": .../images/train, ...}``).

    Returns:
        A ``DuplicateLeakageReport`` listing any hash found in more than one split.
    """

    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    total = 0

    for split, images_dir in split_to_images_dir.items():
        if not images_dir.is_dir():
            continue
        for path in images_dir.iterdir():
            if not path.is_file():
                continue
            file_hash = hash_file(path)
            hash_to_splits[file_hash].add(split)
            hash_to_paths[file_hash].append(path)
            total += 1

    groups = [
        CrossSplitDuplicate(file_hash=h, splits=splits, paths=hash_to_paths[h])
        for h, splits in hash_to_splits.items()
        if len(splits) > 1
    ]
    return DuplicateLeakageReport(total_images_hashed=total, duplicate_groups=groups)


@dataclass(slots=True)
class PatientLeakageReport:
    total_patients_checked: int
    conflicting_patients: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    @property
    def has_leakage(self) -> bool:
        return len(self.conflicting_patients) > 0


def find_patient_split_conflicts(
    rows: list[dict[str, str]],
    *,
    task: str,
    source_field: str = "source_dataset",
    patient_field: str = "patient_id",
    task_field: str = "task",
    status_field: str = "status",
    included_value: str = "INCLUDED",
    split_field: str = "split",
) -> PatientLeakageReport:
    """Check that no (source_dataset, patient_id) pair spans more than one split.

    Rows with a blank patient id are skipped (image-level grouping is the
    documented, intentional policy for sources without patient identifiers,
    e.g. ``merged``).

    Args:
        rows: Manifest rows as dicts (e.g. from ``csv.DictReader``).
        task: Only rows whose ``task_field`` equals this value are checked.

    Returns:
        A ``PatientLeakageReport`` listing any patient found across multiple splits.
    """

    patient_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        if row.get(task_field) != task or row.get(status_field) != included_value:
            continue
        patient_id = (row.get(patient_field) or "").strip()
        if not patient_id:
            continue
        key = (row.get(source_field, ""), patient_id)
        patient_splits[key].add(row.get(split_field, ""))

    conflicts = {k: v for k, v in patient_splits.items() if len(v) > 1}
    return PatientLeakageReport(total_patients_checked=len(patient_splits), conflicting_patients=conflicts)
