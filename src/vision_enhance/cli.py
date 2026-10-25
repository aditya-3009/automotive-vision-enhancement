"""CLI for running batch enhancement experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_enhance.methods import METHODS
from vision_enhance.pipeline import process_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a classical image-enhancement baseline.")
    parser.add_argument("--input", required=True, help="Directory containing input images")
    parser.add_argument("--output", required=True, help="Output directory for images and metrics.csv")
    parser.add_argument("--method", choices=METHODS, default="clahe")
    parser.add_argument("--gamma", type=float, default=0.8, help="Gamma value for --method gamma")
    parser.add_argument("--reference-dir", help="Optional directory of aligned reference images with matching filenames")
    args = parser.parse_args()

    records = process_directory(
        args.input, args.output, method=args.method,
        reference_dir=args.reference_dir, gamma=args.gamma,
    )
    counts: dict[str, int] = {}
    for row in records:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({
        "method": args.method,
        "processed_images": len(records),
        "status_counts": counts,
        "report": str(Path(args.output) / "metrics.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
