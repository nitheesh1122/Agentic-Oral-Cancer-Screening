"""One-off smoke test: run the real ImageQualityAssessmentAgent pipeline
against a handful of real images from DATA_SETS/PROCESSED/classification/train,
writing outputs to outputs/module1/quality/ instead of the default dataset/ dir.

Not a pytest test (uses real BRISQUE model + real dataset paths) -- run
directly with `python tests/smoke_quality.py` for manual verification.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.image_quality.config import QualityConfig
from src.image_quality.quality_agent import ImageQualityAssessmentAgent
from src.image_quality.utils import DiscoveredImage

TRAIN_DIR = Path(r"D:\FYP\DATA_SETS\PROCESSED\classification\train")
OUTPUT_DIR = REPO_ROOT / "outputs" / "module1" / "quality" / "smoke_test"


def main() -> None:
    config = QualityConfig(
        dataset_root=TRAIN_DIR,
        raw_dir=TRAIN_DIR,
        processed_dir=OUTPUT_DIR / "processed",
        reports_dir=OUTPUT_DIR / "reports",
        comparisons_dir=OUTPUT_DIR / "reports" / "comparisons",
        csv_report_path=OUTPUT_DIR / "reports" / "quality_report.csv",
    )
    agent = ImageQualityAssessmentAgent(config=config)

    print("Checking dependencies...")
    agent.check_dependencies()
    print("OK")

    items: list[DiscoveredImage] = []
    for class_dir in sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob("*.jpg"))[:3]
        items.extend(DiscoveredImage(path=f, class_name=class_dir.name) for f in files)

    print(f"Selected {len(items)} images across {len(set(i.class_name for i in items))} classes")

    from src.image_quality.utils import ensure_output_directories

    ensure_output_directories(config.processed_dir, config.comparisons_dir, (i.class_name for i in items))

    records = []
    outputs = []
    t0 = time.perf_counter()
    for item in items:
        record, output = agent.process_single_image(item)
        records.append(record)
        outputs.append(output)
        print(
            f"  {item.class_name:25s} {item.path.name:30s} "
            f"BRISQUE={record['BRISQUE Score']!s:>10} decision={record['Decision']}"
        )
    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(items)} images in {elapsed:.2f}s ({elapsed/len(items):.2f}s/image)")

    from src.image_quality.utils import write_quality_report

    write_quality_report(records, config.csv_report_path)
    print(f"Report written to {config.csv_report_path}")

    accepted = sum(1 for o in outputs if o["proceed"])
    print(f"\nAccepted: {accepted}/{len(outputs)}")

    for o in outputs:
        assert set(o.keys()) == {
            "filename", "class", "processed_image", "brisque_score",
            "blur_score", "brightness", "contrast", "quality_decision", "proceed",
        }
        if o["proceed"]:
            assert Path(o["processed_image"]).is_file(), f"missing processed file for {o['filename']}"
    print("Output schema + file-existence checks passed.")


if __name__ == "__main__":
    main()
