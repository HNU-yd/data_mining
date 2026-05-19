#!/usr/bin/env python3
"""Download the Hugging Face parquet mirror of CASIA-WebFace."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "data" / "raw" / "casia_webface_parquet"
BASE_URL = "https://huggingface.co/datasets/SaffalPoosh/casia_web_face/resolve/main/data"


def download_one(index: int, output_dir: Path) -> Path:
    name = f"train-{index:05d}-of-00020.parquet"
    target = output_dir / name
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return target
    url = f"{BASE_URL}/{name}"
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(target), url],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for i in range(20):
        path = download_one(i, args.output_dir)
        rows = pq.ParquetFile(path).metadata.num_rows
        total_rows += rows
        print(f"{path.name}: {rows} rows")
    print(f"CASIA-WebFace parquet rows: {total_rows}")


if __name__ == "__main__":
    main()
