"""
Dataset Preparation and Validation Module for OSTutorLLM Fine-Tuning (Phase 3).

Provides schema validation, taxonomy mapping verification, exact & near-duplicate detection,
train/validation/test split leakage verification, and dataset loader functions.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "data" / "taxonomy.json"
INSTRUCTION_DIR = BASE_DIR / "data" / "instruction"

logger = logging.getLogger(__name__)

ALLOWED_BLOOM_LEVELS: Set[str] = {
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
}

ALLOWED_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}

ALLOWED_TASK_TYPES: Set[str] = {
    "concept_explanation",
    "beginner_explanation",
    "advanced_explanation",
    "comparison",
    "mcq_generation",
    "viva_generation",
    "numerical_problem",
    "problem_solving",
    "algorithm_explanation",
    "step_by_step_solution",
    "programming",
    "debugging",
    "analysis",
    "evaluation",
    "scenario_analysis",
    "misconception_correction",
    "summarization",
    "exam_question_answer",
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


def load_taxonomy_data(taxonomy_path: Path = TAXONOMY_PATH) -> Tuple[Set[int], Dict[int, Set[str]], Dict[str, Set[str]]]:
    """
    Load valid units, topics, and subtopics from data/taxonomy.json.

    Returns:
        Tuple of (valid_units, unit_topics_map, topic_subtopics_map).
    """
    valid_units: Set[int] = set()
    unit_topics_map: Dict[int, Set[str]] = {}
    topic_subtopics_map: Dict[str, Set[str]] = {}

    if not taxonomy_path.exists():
        return valid_units, unit_topics_map, topic_subtopics_map

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        units_data = json.load(f)

    for u in units_data:
        uid = int(u["unit_id"].split("_")[1])
        valid_units.add(uid)
        unit_topics_map[uid] = set()

        for t in u.get("topics", []):
            t_name = t["name"]
            unit_topics_map[uid].add(t_name.lower())
            if t_name.lower() not in topic_subtopics_map:
                topic_subtopics_map[t_name.lower()] = set()

            for st in t.get("subtopics", []):
                topic_subtopics_map[t_name.lower()].add(st.lower())

    return valid_units, unit_topics_map, topic_subtopics_map


def validate_record(record: Dict[str, Any], record_idx: int, taxonomy_info: Optional[Tuple] = None) -> List[str]:
    """
    Validate a single instruction-tuning record dictionary against schema and taxonomy.

    Args:
        record: Instruction record dictionary.
        record_idx: 1-indexed line index of record.
        taxonomy_info: Preloaded taxonomy tuple.

    Returns:
        List of validation error strings.
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

    # Validate difficulty level
    diff = record.get("difficulty")
    if diff not in ALLOWED_DIFFICULTIES:
        errors.append(
            f"Record #{record_idx}: Invalid difficulty '{diff}'. Allowed: {sorted(list(ALLOWED_DIFFICULTIES))}"
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

    # Taxonomy checks if provided
    if taxonomy_info:
        valid_units, unit_topics_map, topic_subtopics_map = taxonomy_info
        if unit in valid_units:
            topic = str(record.get("topic", "")).strip().lower()
            if topic and topic not in unit_topics_map.get(unit, set()):
                errors.append(f"Record #{record_idx}: Topic '{record.get('topic')}' not mapped to Unit {unit} in taxonomy.")
            elif topic in topic_subtopics_map:
                subtopic = str(record.get("subtopic", "")).strip().lower()
                if subtopic and subtopic not in topic_subtopics_map[topic]:
                    # Warning logging for subtopic drift
                    pass

    return errors


def load_and_validate_jsonl(file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load a JSONL dataset file and validate all records.

    Args:
        file_path: Path to JSONL file.

    Returns:
        Tuple of (valid records list, error strings list).
    """
    if not file_path.exists():
        return [], [f"File not found: {file_path}"]

    taxonomy_info = load_taxonomy_data()
    records: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                record_errors = validate_record(data, idx, taxonomy_info)
                if record_errors:
                    all_errors.extend(record_errors)
                else:
                    records.append(data)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line #{idx}: Invalid JSON syntax - {str(e)}")

    return records, all_errors


def detect_exact_duplicates(records: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    """
    Identify exact duplicate records based on instruction + input.

    Returns:
        List of (idx_orig, idx_dup, instruction_key) tuples.
    """
    seen: Dict[str, int] = {}
    duplicates: List[Tuple[int, int, str]] = []

    for idx, rec in enumerate(records):
        key = f"{rec.get('instruction', '').strip()}|||{rec.get('input', '').strip()}".lower()
        if key in seen:
            duplicates.append((seen[key], idx, key[:60]))
        else:
            seen[key] = idx

    return duplicates


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract word n-grams from text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def detect_near_duplicates(
    records: List[Dict[str, Any]], threshold: float = 0.85
) -> List[Tuple[int, int, float, str]]:
    """
    Detect near-duplicate records using 3-gram Jaccard similarity.

    Args:
        records: List of instruction records.
        threshold: Jaccard similarity threshold (0.0 to 1.0).

    Returns:
        List of (idx_a, idx_b, similarity_score, instruction_a_snippet) tuples.
    """
    ngrams_list = [
        get_ngrams(f"{r.get('instruction', '')} {r.get('input', '')}") for r in records
    ]
    near_dups: List[Tuple[int, int, float, str]] = []

    num_records = len(records)
    for i in range(num_records):
        for j in range(i + 1, min(i + 100, num_records)):  # Windowed comparison
            sim = jaccard_similarity(ngrams_list[i], ngrams_list[j])
            if sim >= threshold:
                snippet = records[i].get("instruction", "")[:50]
                near_dups.append((i, j, round(sim, 4), snippet))

    return near_dups


def check_split_leakage(
    train_records: List[Dict[str, Any]], test_records: List[Dict[str, Any]]
) -> List[Tuple[int, int, str]]:
    """
    Check for data leakage between training set and test set.

    Returns:
        List of (train_idx, test_idx, instruction_key) overlapping pairs.
    """
    train_keys = {
        f"{r.get('instruction', '').strip()}|||{r.get('input', '').strip()}".lower(): idx
        for idx, r in enumerate(train_records)
    }
    leakage: List[Tuple[int, int, str]] = []

    for t_idx, test_rec in enumerate(test_records):
        key = f"{test_rec.get('instruction', '').strip()}|||{test_rec.get('input', '').strip()}".lower()
        if key in train_keys:
            leakage.append((train_keys[key], t_idx, key[:60]))

    return leakage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("OSTutorLLM Dataset Preparation & Validation")
    print("===========================================")

    for split_name in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        path = INSTRUCTION_DIR / split_name
        recs, errs = load_and_validate_jsonl(path)
        if errs:
            print(f"[ERRORS] {split_name} validation failed with {len(errs)} errors:")
            for e in errs[:5]:
                print(f"  - {e}")
        else:
            print(f"[PASS] {split_name}: {len(recs)} records validated successfully.")
