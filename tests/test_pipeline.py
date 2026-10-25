import csv

import cv2
import numpy as np

from vision_enhance.pipeline import process_directory


def test_pipeline_writes_enhanced_image_and_report(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    image = np.full((24, 24, 3), 60, dtype=np.uint8)
    assert cv2.imwrite(str(input_dir / "sample.png"), image)

    records = process_directory(input_dir, output_dir, method="gamma", gamma=0.8)

    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert (output_dir / "sample.png").exists()
    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "gamma"
    assert rows[0]["psnr"] == ""
