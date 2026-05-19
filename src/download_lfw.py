#!/usr/bin/env python3
"""Download and prepare an LFW image directory for the verification task."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SOURCES = [
    {
        "name": "official_original",
        "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "archive": RAW_DIR / "lfw.tgz",
        "kind": "tgz",
        "root": RAW_DIR / "lfw",
    },
    {
        "name": "sklearn_figshare_original",
        "url": "https://ndownloader.figshare.com/files/5976018",
        "archive": RAW_DIR / "lfw.tgz",
        "kind": "tgz",
        "root": RAW_DIR / "lfw",
    },
    {
        "name": "hf_deepfunneled",
        "url": "https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main/lfw-deepfunneled.zip",
        "archive": RAW_DIR / "lfw-deepfunneled.zip",
        "kind": "zip",
        "root": RAW_DIR / "lfw-deepfunneled",
    },
]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(target), url])


def extract(archive: Path, kind: str) -> None:
    if kind == "tgz":
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(RAW_DIR)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW_DIR)
    else:
        raise ValueError(kind)
    shutil.rmtree(RAW_DIR / "__MACOSX", ignore_errors=True)


def image_count(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg")) if root.exists() else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download and extract even if images already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        root = source["root"]
        if not args.force and image_count(root) == 13233:
            print(f"Using existing {source['name']} at {root}")
            return

    last_error: Exception | None = None
    for source in SOURCES:
        try:
            print(f"Trying {source['name']}: {source['url']}")
            download(source["url"], source["archive"])
            extract(source["archive"], source["kind"])
            count = image_count(source["root"])
            if count != 13233:
                raise RuntimeError(f"Expected 13233 images, found {count} in {source['root']}")
            print(f"Prepared LFW images at {source['root']}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Failed {source['name']}: {exc}")
    raise RuntimeError("Could not download LFW from any configured source") from last_error


if __name__ == "__main__":
    main()
