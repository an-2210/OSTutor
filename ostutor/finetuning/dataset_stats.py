"""
Dataset Statistics Generator for OSTutorLLM (Phase 3).

Calculates comprehensive counts, distributions across units, Bloom levels, difficulties,
task types, provenance sources, and dataset splits.
Generates data/instruction/dataset_statistics.json and dataset_statistics.txt.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from finetuning.prepare_dataset import INSTRUCTION_DIR, load_and_validate_jsonl

logger = logging.getLogger(__name__)


def compute_dataset_statistics() -> Dict[str, Any]:
    """
    Compute dataset statistics across all instruction splits.

    Returns:
        Dictionary containing detailed dataset statistics.
    """
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    all_records: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for split in splits:
        path = INSTRUCTION_DIR / split
        records, errs = load_and_validate_jsonl(path)
        split_name = split.replace(".jsonl", "")
        split_counts[split_name] = len(records)
        all_records.extend(records)

    total = len(all_records)
    if total == 0:
        logger.warning("No records found to compute statistics.")
        return {}

    by_unit: Dict[str, int] = {f"unit_{u}": 0 for u in range(1, 8)}
    by_bloom: Dict[str, int] = {}
    by_difficulty: Dict[str, int] = {}
    by_task_type: Dict[str, int] = {}
    by_source_type: Dict[str, int] = {}

    for rec in all_records:
        u = f"unit_{rec.get('unit')}"
        b = str(rec.get("bloom_level"))
        d = str(rec.get("difficulty"))
        t = str(rec.get("task_type"))
        s = str(rec.get("source_type"))

        by_unit[u] = by_unit.get(u, 0) + 1
        by_bloom[b] = by_bloom.get(b, 0) + 1
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
        by_task_type[t] = by_task_type.get(t, 0) + 1
        by_source_type[s] = by_source_type.get(s, 0) + 1

    stats = {
        "total_records": total,
        "split_counts": split_counts,
        "by_unit": by_unit,
        "by_bloom_level": by_bloom,
        "by_difficulty": by_difficulty,
        "by_task_type": by_task_type,
        "by_source_type": by_source_type,
        "bloom_percentages": {
            k: round((v / total) * 100, 2) for k, v in by_bloom.items()
        },
        "unit_percentages": {
            k: round((v / total) * 100, 2) for k, v in by_unit.items()
        },
    }

    return stats


def export_statistics_reports(stats: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Export statistics reports to JSON and formatted TXT files.

    Returns:
        Tuple of (json_path, txt_path).
    """
    json_path = INSTRUCTION_DIR / "dataset_statistics.json"
    txt_path = INSTRUCTION_DIR / "dataset_statistics.txt"

    # Export JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Build formatted text report
    lines = []
    lines.append("==================================================")
    lines.append(" OSTutorLLM Instruction Dataset Statistics Report ")
    lines.append("==================================================\n")
    lines.append(f"Total Instruction Examples : {stats.get('total_records')}\n")

    lines.append("1. Split Breakdown:")
    for split, count in stats.get("split_counts", {}).items():
        pct = (count / stats.get("total_records", 1)) * 100
        lines.append(f"   - {split:<12}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n2. Unit Breakdown (Syllabus Units 1-7):")
    for unit, count in stats.get("by_unit", {}).items():
        pct = stats.get("unit_percentages", {}).get(unit, 0.0)
        lines.append(f"   - {unit:<10}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n3. Bloom's Taxonomy Breakdown:")
    for bloom, count in stats.get("by_bloom_level", {}).items():
        pct = stats.get("bloom_percentages", {}).get(bloom, 0.0)
        lines.append(f"   - {bloom:<12}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n4. Difficulty Level Breakdown:")
    for diff, count in stats.get("by_difficulty", {}).items():
        pct = (count / stats.get("total_records", 1)) * 100
        lines.append(f"   - {diff:<10}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n5. Task Type Breakdown:")
    for task, count in sorted(stats.get("by_task_type", {}).items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats.get("total_records", 1)) * 100
        lines.append(f"   - {task:<25}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n6. Provenance (Source Type) Breakdown:")
    for src, count in sorted(stats.get("by_source_type", {}).items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats.get("total_records", 1)) * 100
        lines.append(f"   - {src:<25}: {count:>5} records ({pct:>5.1f}%)")

    lines.append("\n==================================================")

    report_content = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")

    print(report_content)
    logger.info(f"Exported dataset statistics to {json_path} and {txt_path}")
    return json_path, txt_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = compute_dataset_statistics()
    if stats:
        export_statistics_reports(stats)
