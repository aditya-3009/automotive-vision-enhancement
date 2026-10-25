"""Batch image processing with per-image runtime and optional reference metrics."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import cv2

from vision_enhance.methods import METHODS, enhance_image
from vision_enhance.metrics import compare_images

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
REPORT_FIELDS = [
    "input_path", "output_path", "method", "runtime_ms", "reference_path", "psnr", "ssim", "status"
]


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    method: str,
    reference_dir: str | Path | None = None,
    gamma: float = 0.8,
) -> list[dict[str, Any]]:
    """Process all supported images directly in input_dir and write a CSV report."""
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {source_dir}")
    if method.lower() not in METHODS:
        raise ValueError(f"Unsupported method {method!r}; choose one of {METHODS}")

    target_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(reference_dir) if reference_dir is not None else None
    records: list[dict[str, Any]] = []

    image_paths = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        relative_name = image_path.name
        output_path = target_dir / relative_name
        record: dict[str, Any] = {
            "input_path": str(image_path),
            "output_path": str(output_path),
            "method": method.lower(),
            "runtime_ms": "",
            "reference_path": "",
            "psnr": "",
            "ssim": "",
            "status": "ok",
        }
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            record["status"] = "read_error"
            records.append(record)
            continue

        start = time.perf_counter()
        try:
            enhanced = enhance_image(image, method=method, gamma=gamma)
        except (ValueError, cv2.error) as exc:
            record["status"] = f"enhancement_error: {exc}"
            records.append(record)
            continue
        record["runtime_ms"] = round((time.perf_counter() - start) * 1000.0, 3)

        if not cv2.imwrite(str(output_path), enhanced):
            record["status"] = "write_error"
            records.append(record)
            continue

        if ref_dir is not None:
            reference_path = ref_dir / relative_name
            record["reference_path"] = str(reference_path)
            reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            if reference is None:
                record["status"] = "missing_or_unreadable_reference"
            elif reference.shape != enhanced.shape:
                record["status"] = "reference_shape_mismatch"
            else:
                try:
                    values = compare_images(reference, enhanced)
                    record["psnr"] = round(values["psnr"], 6)
                    record["ssim"] = round(values["ssim"], 6)
                except ValueError as exc:
                    record["status"] = f"metric_error: {exc}"
        records.append(record)

    report_path = target_dir / "metrics.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records
