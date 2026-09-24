"""
Dataset Versioning and SHA-256 Hash Verification for OSTutorLLM.

Computes exact SHA-256 digests and line counts for instruction datasets
(train, validation, test) to ensure strict experimental reproducibility.
"""

import hashlib
import json
import os
from typing import Dict, Any


DATASET_PATHS = {
    "train": "data/instruction/train.jsonl",
    "validation": "data/instruction/validation.jsonl",
    "test": "data/instruction/test.jsonl",
    "formatted_train": "data/instruction/formatted_train.jsonl",
    "formatted_val": "data/instruction/formatted_validation.jsonl",
    "formatted_test": "data/instruction/formatted_test.jsonl",
}


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash digest of a given file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_lines(filepath: str) -> int:
    """Count non-empty lines in a dataset file."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_dataset_hashes() -> Dict[str, Dict[str, Any]]:
    """
    Gather hashes, file sizes, and record counts for all instruction datasets.

    Returns:
        Dict mapping dataset key to metadata dictionary.
    """
    summary = {}
    for key, path in DATASET_PATHS.items():
        if os.path.exists(path):
            summary[key] = {
                "path": path,
                "sha256": compute_file_hash(path),
                "line_count": count_lines(path),
                "size_bytes": os.path.getsize(path),
            }
        else:
            summary[key] = {
                "path": path,
                "sha256": "FILE_NOT_FOUND",
                "line_count": 0,
                "size_bytes": 0,
            }
    return summary


def print_dataset_hashes() -> None:
    """Print clean summary of dataset version digests."""
    hashes = get_dataset_hashes()
    print("==========================================")
    print(" OSTutorLLM Dataset SHA-256 Hashes ")
    print("==========================================")
    for key, meta in hashes.items():
        print(f"[{key.upper()}] {meta['path']}")
        print(f"  Lines : {meta['line_count']}")
        print(f"  Size  : {meta['size_bytes']} bytes")
        print(f"  SHA256: {meta['sha256']}")
        print("------------------------------------------")


if __name__ == "__main__":
    print_dataset_hashes()
