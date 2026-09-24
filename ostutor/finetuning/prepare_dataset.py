"""
Dataset Preparation and Schema Validation Module for OSTutorLLM Fine-Tuning.

Validates instruction JSONL records against predefined OS taxonomy, Bloom's taxonomy levels,
and allowed task types. Performs deduplication, summary statistics generation, and split creation.
"""

import json
import os
from typing import Any, Dict, List, Set, Tuple

ALLOWED_BLOOM_LEVELS: Set[str] = {
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
}

ALLOWED_TASK_TYPES: Set[str] = {
    "concept_explanation",
    "beginner_explanation",
    "comparison",
    "mcq_generation",
    "viva_generation",
    "numerical_problem",
    "problem_solving",
    "programming",
    "debugging",
    "analysis",
    "evaluation",
    "step_by_step_solution",
    "summarization",
}

REQUIRED_KEYS: Set[str] = {
    "unit",
    "topic",
    "subtopic",
    "bloom_level",
    "difficulty",
    "task_type",
    "instruction",
    "input",
    "output",
    "source_type",
}


def validate_record(record: Dict[str, Any], record_idx: int) -> List[str]:
    """
    Validate a single instruction-tuning record dictionary.

    Args:
        record: Instruction item dictionary.
        record_idx: Line index of the record for error messaging.

    Returns:
        List of validation error message strings (empty if valid).
    """
    errors: List[str] = []

    # Check missing fields
    missing_keys = REQUIRED_KEYS - set(record.keys())
    if missing_keys:
        errors.append(f"Record #{record_idx}: Missing required keys {sorted(list(missing_keys))}")

    # Validate unit number (1-7)
    unit = record.get("unit")
    if not isinstance(unit, int) or unit < 1 or unit > 7:
        errors.append(f"Record #{record_idx}: Invalid unit '{unit}'. Must be integer 1-7.")

    # Validate Bloom level
    bloom = record.get("bloom_level")
    if bloom not in ALLOWED_BLOOM_LEVELS:
        errors.append(
            f"Record #{record_idx}: Invalid Bloom level '{bloom}'. Allowed: {sorted(list(ALLOWED_BLOOM_LEVELS))}"
        )

    # Validate task type
    task_type = record.get("task_type")
    if task_type not in ALLOWED_TASK_TYPES:
        errors.append(
            f"Record #{record_idx}: Invalid task_type '{task_type}'. Allowed: {sorted(list(ALLOWED_TASK_TYPES))}"
        )

    # Check non-empty instruction and output
    if not str(record.get("instruction", "")).strip():
        errors.append(f"Record #{record_idx}: 'instruction' field cannot be empty.")
    if not str(record.get("output", "")).strip():
        errors.append(f"Record #{record_idx}: 'output' field cannot be empty.")

    return errors


def load_and_validate_jsonl(file_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a JSONL file and validate all contained records.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Tuple of (list of valid records, list of error messages).
    """
    if not os.path.exists(file_path):
        return [], [f"File not found: {file_path}"]

    records: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                record_errors = validate_record(data, idx)
                if record_errors:
                    all_errors.extend(record_errors)
                else:
                    records.append(data)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line #{idx}: Invalid JSON syntax - {str(e)}")

    return records, all_errors


def remove_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate instruction records based on instruction + input combination.

    Args:
        records: List of instruction records.

    Returns:
        Deduplicated list of instruction records.
    """
    seen: Set[str] = set()
    unique_records: List[Dict[str, Any]] = []

    for rec in records:
        key = f"{rec.get('instruction', '')}|||{rec.get('input', '')}"
        if key not in seen:
            seen.add(key)
            unique_records.append(rec)

    return unique_records


def generate_dataset_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over the dataset.

    Args:
        records: List of instruction records.

    Returns:
        Dictionary containing counts per unit, bloom level, task type, and total records.
    """
    stats: Dict[str, Any] = {
        "total_records": len(records),
        "by_unit": {},
        "by_bloom_level": {},
        "by_task_type": {},
    }

    for rec in records:
        u = f"unit_{rec.get('unit')}"
        b = str(rec.get("bloom_level"))
        t = str(rec.get("task_type"))

        stats["by_unit"][u] = stats["by_unit"].get(u, 0) + 1
        stats["by_bloom_level"][b] = stats["by_bloom_level"].get(b, 0) + 1
        stats["by_task_type"][t] = stats["by_task_type"].get(t, 0) + 1

    return stats


if __name__ == "__main__":
    train_path = os.path.join("data", "instruction", "train.jsonl")
    records, errors = load_and_validate_jsonl(train_path)
    if errors:
        print(f"Validation errors found in {train_path}:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"Validation successful for {train_path}! Total records: {len(records)}")
        print(json.dumps(generate_dataset_statistics(records), indent=2))
