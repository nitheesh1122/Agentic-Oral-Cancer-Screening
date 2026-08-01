"""Image Quality Assessment Agent.

First stage of the agentic oral cancer screening pipeline. Produces a
structured, per-image result (see ``process_single_image``) that the
downstream YOLOv11 Lesion Detection Agent can consume directly via the
``processed_image`` file path.

Pipeline per image: validate -> BRISQUE -> blur -> brightness -> contrast ->
quality decision -> (if accepted) resize -> denoise -> CLAHE -> normalize ->
save -> comparison image.
"""

from __future__ import annotations

import time
from typing import Any

from .brisque_analyzer import BrisqueAnalyzer
from .config import DEFAULT_CONFIG, QualityConfig
from .metrics import compute_blur_score, compute_brightness, compute_contrast
from .preprocessing import build_comparison_image, run_preprocessing_pipeline
from .utils import (
    DiscoveredImage,
    discover_images,
    ensure_output_directories,
    get_logger,
    load_image,
    save_image,
    write_quality_report,
)

REQUIRED_MODULES: tuple[str, ...] = ("cv2", "numpy", "pandas", "torch", "pyiqa")


class ImageQualityAssessmentAgent:
    """Assesses raw oral image quality and preprocesses accepted images.

    Attributes:
        config: Paths and thresholds controlling agent behavior.
        logger: Module-scoped logger.
        brisque_analyzer: Reusable BRISQUE scorer.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG
        self.logger = get_logger(__name__, self.config.log_level)
        self.brisque_analyzer = BrisqueAnalyzer(device=self.config.brisque_device)

    def check_dependencies(self) -> None:
        """Verify that all required third-party modules can be imported.

        Raises:
            ImportError: If any required module is missing, naming the
                missing module and how to install it.
        """

        import importlib

        for module_name in REQUIRED_MODULES:
            try:
                importlib.import_module(module_name)
            except ImportError as exc:
                raise ImportError(
                    f"Missing required dependency '{module_name}' for the Image Quality "
                    "Assessment Agent. Install requirements with: pip install -r requirements.txt"
                ) from exc

    def discover(self) -> list[DiscoveredImage]:
        """Discover all raw images across auto-detected class subfolders."""

        return discover_images(self.config.raw_dir, self.config.supported_extensions)

    def assess_quality(
        self, brisque_score: float, blur_score: float, brightness: float, contrast: float
    ) -> tuple[str, bool, list[str]]:
        """Apply threshold rules to decide whether an image is usable.

        Args:
            brisque_score: BRISQUE score (lower is better).
            blur_score: Variance-of-Laplacian blur score (higher is sharper).
            brightness: Mean grayscale brightness (0-255 scale).
            contrast: Grayscale standard deviation.

        Returns:
            A tuple of (decision, proceed, reasons) where ``decision`` is
            "Accepted" or "Rejected", ``proceed`` is the boolean equivalent,
            and ``reasons`` lists every failed threshold check.
        """

        reasons: list[str] = []
        config = self.config

        if brisque_score > config.brisque_max:
            reasons.append(f"BRISQUE {brisque_score:.2f} exceeds max {config.brisque_max}")
        if blur_score < config.blur_min:
            reasons.append(f"Blur score {blur_score:.2f} below min {config.blur_min}")
        if not (config.brightness_min <= brightness <= config.brightness_max):
            reasons.append(
                f"Brightness {brightness:.2f} outside [{config.brightness_min}, {config.brightness_max}]"
            )
        if contrast < config.contrast_min:
            reasons.append(f"Contrast {contrast:.2f} below min {config.contrast_min}")

        decision = "Accepted" if not reasons else "Rejected"
        return decision, not reasons, reasons

    def process_single_image(self, item: DiscoveredImage) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the full quality-assessment and preprocessing pipeline on one image.

        Args:
            item: A discovered raw image and its class label.

        Returns:
            A tuple of (csv_record, agent_output) where ``csv_record`` matches
            the quality report schema and ``agent_output`` matches the
            structured handoff contract for the next pipeline stage.
        """

        start_time = time.perf_counter()

        try:
            image = load_image(item.path)
        except (FileNotFoundError, ValueError) as exc:
            self.logger.warning("Skipping invalid image %s: %s", item.path, exc)
            elapsed = time.perf_counter() - start_time
            return self._build_invalid_record(item, elapsed), self._build_invalid_output(item)

        original_height, original_width = image.shape[:2]

        brisque_score = self.brisque_analyzer.score_path(item.path)
        blur_score = compute_blur_score(image)
        brightness = compute_brightness(image)
        contrast = compute_contrast(image)

        decision, proceed, reasons = self.assess_quality(brisque_score, blur_score, brightness, contrast)

        processed_path = ""
        processed_width: Any = ""
        processed_height: Any = ""

        if proceed:
            _, persistable = run_preprocessing_pipeline(image, self.config)
            output_path = self.config.processed_dir / item.class_name / item.path.name
            save_image(persistable, output_path)
            processed_height, processed_width = persistable.shape[:2]
            processed_path = str(output_path)

            comparison = build_comparison_image(image, persistable)
            comparison_path = (
                self.config.comparisons_dir / f"{item.class_name}_{item.path.stem}_comparison.jpg"
            )
            save_image(comparison, comparison_path)
        else:
            self.logger.info("Rejected %s: %s", item.path.name, "; ".join(reasons))

        elapsed = time.perf_counter() - start_time

        record = {
            "Filename": item.path.name,
            "Class": item.class_name,
            "BRISQUE Score": round(brisque_score, 4),
            "Blur Score": round(blur_score, 4),
            "Brightness": round(brightness, 4),
            "Contrast": round(contrast, 4),
            "Decision": decision,
            "Original Width": original_width,
            "Original Height": original_height,
            "Processed Width": processed_width,
            "Processed Height": processed_height,
            "Processing Time": round(elapsed, 4),
        }

        output = {
            "filename": item.path.name,
            "class": item.class_name,
            "processed_image": processed_path,
            "brisque_score": round(brisque_score, 4),
            "blur_score": round(blur_score, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "quality_decision": decision,
            "proceed": proceed,
        }

        return record, output

    @staticmethod
    def _build_invalid_record(item: DiscoveredImage, elapsed: float) -> dict[str, Any]:
        return {
            "Filename": item.path.name,
            "Class": item.class_name,
            "BRISQUE Score": "",
            "Blur Score": "",
            "Brightness": "",
            "Contrast": "",
            "Decision": "Rejected (Invalid Image)",
            "Original Width": "",
            "Original Height": "",
            "Processed Width": "",
            "Processed Height": "",
            "Processing Time": round(elapsed, 4),
        }

    @staticmethod
    def _build_invalid_output(item: DiscoveredImage) -> dict[str, Any]:
        return {
            "filename": item.path.name,
            "class": item.class_name,
            "processed_image": "",
            "brisque_score": None,
            "blur_score": None,
            "brightness": None,
            "contrast": None,
            "quality_decision": "Rejected (Invalid Image)",
            "proceed": False,
        }

    def run(self) -> list[dict[str, Any]]:
        """Run the full agent pipeline end to end and print progress banners.

        Returns:
            A list of structured per-image output dictionaries, one per
            discovered image, ready to hand off to the Lesion Detection Agent.

        Raises:
            ImportError: If a required dependency is missing.
            FileNotFoundError: If the raw dataset directory does not exist.
            ValueError: If no images are discovered.
        """

        print("=========================================")
        print("Image Quality Assessment Agent")
        print("=========================================")
        print()

        print("Checking Dependencies...")
        self.check_dependencies()
        print("Dependency Check Passed")
        print()

        print("Loading Images...")
        items = self.discover()
        print(f"Images Loaded : {len(items)}")
        print()

        class_names = (item.class_name for item in items)
        ensure_output_directories(self.config.processed_dir, self.config.comparisons_dir, class_names)

        print("Running BRISQUE Assessment...")
        print("Calculating Blur...")
        print("Calculating Brightness...")
        print("Calculating Contrast...")
        print("Running Preprocessing...")
        print("Saving Processed Images...")
        print()

        records: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for item in items:
            record, output = self.process_single_image(item)
            records.append(record)
            outputs.append(output)

        print("Generating Report...")
        write_quality_report(records, self.config.csv_report_path)
        print()

        print("Pipeline Completed Successfully")
        return outputs
