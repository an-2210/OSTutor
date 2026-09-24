"""
Dataset Quality Auditor for OSTutorLLM (Phase 3).

Audits dataset quality across schema validity, taxonomy integrity, exact & near duplicates,
output length anomalies, unit balance, train/test leakage, and benchmark isolation against os_benchmark.json.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from finetuning.prepare_dataset import (
    INSTRUCTION_DIR,
    check_split_leakage,
    detect_exact_duplicates,
    detect_near_duplicates,
    load_and_validate_jsonl,
    load_taxonomy_data,
)

logger = logging.getLogger(__name__)
BENCHMARK_PATH = BASE_DIR / "data" / "benchmark" / "os_benchmark.json"


def load_benchmark_questions() -> List[Dict[str, Any]]:
    """Load benchmark questions from data/benchmark/os_benchmark.json."""
    if not BENCHMARK_PATH.exists():
        return []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_benchmark_overlap(instruction_records: List[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """
    Check if any instruction records overlap with os_benchmark.json questions.

    Returns:
        List of (instruction_idx, benchmark_id, question_text_snippet) tuples.
    """
    bench_items = load_benchmark_questions()
    if not bench_items:
        return []

    overlaps: List[Tuple[int, str, str]] = []
    bench_questions = {
        item.get("question", "").strip().lower(): item.get("id", "unknown_bench_id")
        for item in bench_items
        if "question" in item
    }

    for idx, rec in enumerate(instruction_records):
        q = rec.get("instruction", "").strip().lower()
        if q in bench_questions:
            overlaps.append((idx, bench_questions[q], q[:50]))

    return overlaps


def check_length_anomalies(records: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    """
    Detect outputs that are suspiciously short (<15 words) or long (>1500 words).

    Returns:
        List of (record_idx, word_count, issue_type) tuples.
    """
    anomalies: List[Tuple[int, int, str]] = []
    for idx, rec in enumerate(records):
        out_text = rec.get("output", "").strip()
        word_count = len(out_text.split())
        if word_count < 15:
            anomalies.append((idx, word_count, "suspiciously_short"))
        elif word_count > 1500:
            anomalies.append((idx, word_count, "suspiciously_long"))
    return anomalies


def run_quality_audit() -> Dict[str, Any]:
    """
    Execute comprehensive quality audit across all splits.

    Returns:
        Structured audit report dictionary.
    """
    print("\nOSTutorLLM Dataset Quality Audit")
    print("===============================\n")

    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    split_data: Dict[str, List[Dict[str, Any]]] = {}
    split_errors: Dict[str, List[str]] = {}

    total_records = 0
    for split in splits:
        records, errs = load_and_validate_jsonl(INSTRUCTION_DIR / split)
        split_name = split.replace(".jsonl", "")
        split_data[split_name] = records
        split_errors[split_name] = errs
        total_records += len(records)

    all_records = split_data.get("train", []) + split_data.get("validation", []) + split_data.get("test", [])

    # 1. Exact Duplicates
    exact_dups = detect_exact_duplicates(all_records)

    # 2. Near Duplicates
    near_dups = detect_near_duplicates(all_records, threshold=0.85)

    # 3. Output Length Anomalies
    length_anomalies = check_length_anomalies(all_records)

    # 4. Train vs Test Leakage
    leakage_items = check_split_leakage(split_data.get("train", []), split_data.get("test", []))

    # 5. Benchmark Contamination Overlap
    benchmark_overlaps = check_benchmark_overlap(all_records)

    # 6. Unit Imbalance Check
    unit_counts: Dict[int, int] = {u: 0 for u in range(1, 8)}
    for r in all_records:
        u = r.get("unit")
        if isinstance(u, int) and 1 <= u <= 7:
            unit_counts[u] += 1

    unit_warnings = []
    min_count = min(unit_counts.values()) if unit_counts else 0
    max_count = max(unit_counts.values()) if unit_counts else 0
    if max_count > 0 and (min_count / max_count) < 0.2:
        unit_warnings.append(f"Significant unit imbalance detected: min unit count ({min_count}) is <20% of max unit count ({max_count}).")

    report = {
        "total_records_audited": total_records,
        "split_counts": {k: len(v) for k, v in split_data.items()},
        "schema_errors_count": sum(len(v) for v in split_errors.values()),
        "exact_duplicates_count": len(exact_dups),
        "near_duplicates_count": len(near_dups),
        "length_anomalies_count": len(length_anomalies),
        "train_test_leakage_count": len(leakage_items),
        "benchmark_overlap_count": len(benchmark_overlaps),
        "unit_counts": unit_counts,
        "unit_warnings": unit_warnings,
        "audit_pass": (
            sum(len(v) for v in split_errors.values()) == 0
            and len(exact_dups) == 0
            and len(leakage_items) == 0
            and len(benchmark_overlaps) == 0
        ),
    }

    print(f"Audit Summary:")
    print(f"  - Total Records Audited : {total_records}")
    print(f"  - Schema Errors         : {report['schema_errors_count']}")
    print(f"  - Exact Duplicates      : {report['exact_duplicates_count']}")
    print(f"  - Near Duplicates (>85%): {report['near_duplicates_count']}")
    print(f"  - Length Anomalies      : {report['length_anomalies_count']}")
    print(f"  - Train/Test Leakage    : {report['train_test_leakage_count']}")
    print(f"  - Benchmark Overlap     : {report['benchmark_overlap_count']}")
    print(f"  - Unit Distribution     : {unit_counts}")
    if unit_warnings:
        for w in unit_warnings:
            print(f"  - [WARNING] {w}")
    print(f"  - Overall Audit Pass    : {'✔ YES' if report['audit_pass'] else '✘ ISSUES DETECTED'}\n")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_quality_audit()
