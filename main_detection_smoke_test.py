"""Phase 3 entry point: YOLOv11 (Ultralytics YOLO11) verification + CPU smoke test.

Does NOT train the real model. Measures: model load time, one forward pass on
a real image, and a tiny 1-epoch training pass on a small subset of the
*training* split only (never val/test), to measure actual CPU throughput on
this machine. All outputs go under outputs/module1/detection/smoke_test/.
Nothing under DATA_SETS is read except as plain image/label files, and
nothing there is modified.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.dataset_validation import validate_detection_dataset

DATASET_ROOT = Path(r"D:\FYP\DATA_SETS\PROCESSED\detection_yolo")
OUTPUT_DIR = REPO_ROOT / "outputs" / "module1" / "detection" / "smoke_test"

SMOKE_TRAIN_COUNT = 150
SMOKE_VAL_COUNT = 30
RANDOM_SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 1
WORKERS = 0
DEVICE = "cpu"
MODEL_CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
MODEL_NAME = "yolo11n.pt"

FULL_TRAIN_COUNT = 4313
FULL_VAL_COUNT = 538


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Phase 3 smoke test starting at {timestamp}")

    # ---- 1. Environment ----
    import torch
    import torchvision
    import ultralytics

    env = {
        "timestamp_utc": timestamp,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": DEVICE,
    }
    print("Environment:", json.dumps(env, indent=2))

    # ---- 2. Reuse Phase 2 validation to get a known-valid train pool ----
    print("\nValidating dataset (reusing Phase 2 code) to select a clean smoke subset...")
    reports = validate_detection_dataset(DATASET_ROOT, splits=("train",))
    train_valid_pairs = reports["train"].valid_pairs
    print(f"  {len(train_valid_pairs)} valid train pairs available")

    rng = random.Random(RANDOM_SEED)
    shuffled = train_valid_pairs[:]
    rng.shuffle(shuffled)
    smoke_train_pairs = shuffled[:SMOKE_TRAIN_COUNT]
    smoke_val_pairs = shuffled[SMOKE_TRAIN_COUNT:SMOKE_TRAIN_COUNT + SMOKE_VAL_COUNT]
    print(f"  smoke train: {len(smoke_train_pairs)} images (from train split only)")
    print(f"  smoke val:   {len(smoke_val_pairs)} images (disjoint, also from train split -- test/val untouched)")

    smoke_train_txt = OUTPUT_DIR / "smoke_train.txt"
    smoke_val_txt = OUTPUT_DIR / "smoke_val.txt"
    smoke_train_txt.write_text("\n".join(str(p.image_path) for p in smoke_train_pairs), encoding="utf-8")
    smoke_val_txt.write_text("\n".join(str(p.image_path) for p in smoke_val_pairs), encoding="utf-8")

    smoke_yaml = OUTPUT_DIR / "smoke_dataset.yaml"
    smoke_yaml.write_text(
        f"train: {smoke_train_txt.as_posix()}\nval: {smoke_val_txt.as_posix()}\nnc: 1\nnames: ['lesion']\n",
        encoding="utf-8",
    )
    print(f"  wrote {smoke_yaml}")

    # ---- 3. Model load ----
    print("\nLoading YOLO model...")
    from ultralytics import YOLO

    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_checkpoint = MODEL_CHECKPOINT_DIR / MODEL_NAME
    # Ultralytics downloads to the current working directory by default; if we
    # already have a cached copy under models/checkpoints/, load that instead
    # of re-downloading (and if it downloads fresh below via bare MODEL_NAME,
    # it lands in the repo root, which we do not want -- so always load by
    # the explicit local path once cached).
    t0 = time.perf_counter()
    model = YOLO(str(local_checkpoint) if local_checkpoint.is_file() else MODEL_NAME)
    load_time = time.perf_counter() - t0
    print(f"  model load time: {load_time:.2f}s")

    # If it downloaded to cwd just now (first run), relocate it into models/checkpoints/.
    cwd_checkpoint = REPO_ROOT / MODEL_NAME
    if cwd_checkpoint.is_file() and not local_checkpoint.is_file():
        cwd_checkpoint.rename(local_checkpoint)
        print(f"  relocated downloaded checkpoint to {local_checkpoint}")

    # ---- 4. Forward-pass sanity check on one real image ----
    sample_image = smoke_train_pairs[0].image_path
    print(f"\nRunning single forward pass on: {sample_image}")
    t0 = time.perf_counter()
    predict_results = model.predict(source=str(sample_image), device=DEVICE, imgsz=IMG_SIZE, verbose=False)
    forward_time = time.perf_counter() - t0
    forward_ok = len(predict_results) == 1
    print(f"  forward pass time: {forward_time:.2f}s, ok={forward_ok}")

    # ---- 5. Training smoke test (1 epoch, small subset, train split only) ----
    print(f"\nRunning smoke training: {EPOCHS} epoch, {len(smoke_train_pairs)} train images, batch={BATCH_SIZE}...")
    t0 = time.perf_counter()
    train_results = model.train(
        data=str(smoke_yaml),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        workers=WORKERS,
        device=DEVICE,
        seed=RANDOM_SEED,
        project=str(OUTPUT_DIR),
        name="smoke_run",
        exist_ok=True,
        plots=False,
        cache=False,
        verbose=True,
    )
    total_smoke_time = time.perf_counter() - t0
    print(f"  total smoke train+val time: {total_smoke_time:.2f}s")

    # Ultralytics writes results.csv with a cumulative "time" column (seconds
    # elapsed since training start, measured by the framework itself) --
    # prefer that over our own wall clock (which also includes YOLO's
    # one-time train-setup overhead: dataloader build, AMP check, etc.).
    results_csv = OUTPUT_DIR / "smoke_run" / "results.csv"
    train_time = total_smoke_time
    if results_csv.is_file():
        import csv as _csv

        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if rows:
            time_key = next((k for k in rows[-1] if k.strip() == "time"), None)
            if time_key is not None:
                try:
                    train_time = float(rows[-1][time_key])
                except ValueError:
                    pass

    # ---- 6. Throughput ----
    train_images_per_sec = len(smoke_train_pairs) / train_time if train_time else 0.0
    sec_per_image = train_time / len(smoke_train_pairs) if smoke_train_pairs else 0.0
    sec_per_batch = sec_per_image * BATCH_SIZE
    batches_per_sec = train_images_per_sec / BATCH_SIZE if BATCH_SIZE else 0.0

    # ---- 7. Extrapolation to full dataset (scaled by measured throughput) ----
    estimated_full_epoch_train_time = FULL_TRAIN_COUNT / train_images_per_sec if train_images_per_sec else float("inf")
    # Reuse the same measured train throughput as a conservative proxy for val throughput
    # (val batches run forward-only, so this likely over-estimates val time slightly).
    estimated_full_epoch_val_time = FULL_VAL_COUNT / train_images_per_sec if train_images_per_sec else float("inf")
    estimated_full_epoch_total = estimated_full_epoch_train_time + estimated_full_epoch_val_time

    estimates = {
        n: {
            "seconds": estimated_full_epoch_total * n,
            "hours": estimated_full_epoch_total * n / 3600,
        }
        for n in (10, 25, 50, 100)
    }

    # ---- 8. Persist results ----
    summary = {
        "environment": env,
        "cpu": {"model": "13th Gen Intel(R) Core(TM) i5-1340P", "cores": 12, "threads": 16, "ram_gb": 15.65},
        "yolo": {
            "framework": "Ultralytics",
            "package_version": ultralytics.__version__,
            "model_family": "YOLO11 (YOLOv11)",
            "exact_checkpoint": MODEL_NAME,
            "checkpoint_source": "Ultralytics official release assets (auto-downloaded by the `ultralytics` package)",
        },
        "dataset": {
            "root": str(DATASET_ROOT),
            "full_train": FULL_TRAIN_COUNT,
            "full_val": FULL_VAL_COUNT,
            "full_test": 539,
            "smoke_train": len(smoke_train_pairs),
            "smoke_val": len(smoke_val_pairs),
        },
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "imgsz": IMG_SIZE,
            "workers": WORKERS, "device": DEVICE, "seed": RANDOM_SEED,
        },
        "measured": {
            "model_load_time_s": load_time,
            "forward_pass_time_s": forward_time,
            "forward_pass_ok": forward_ok,
            "total_smoke_time_s": total_smoke_time,
            "smoke_train_time_s": train_time,
            "sec_per_image": sec_per_image,
            "sec_per_batch": sec_per_batch,
            "images_per_second": train_images_per_sec,
            "batches_per_second": batches_per_sec,
        },
        "estimated_full_dataset": {
            "estimated_full_epoch_train_time_s": estimated_full_epoch_train_time,
            "estimated_full_epoch_val_time_s": estimated_full_epoch_val_time,
            "estimated_full_epoch_total_s": estimated_full_epoch_total,
            "epoch_estimates": estimates,
        },
    }
    with open(OUTPUT_DIR / "smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
