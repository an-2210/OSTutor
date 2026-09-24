"""
Automated Validation Suite for OSTutorLLM (Phase 1).

Tests directory structure, JSON validity, JSONL schema validity,
CSV headers, and python module imports.
"""

import csv
import json
import os
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_directory_structure():
    """Verify that all required repository directories exist."""
    required_dirs = [
        "data",
        "data/rag",
        "data/rag/raw",
        "data/rag/processed",
        "data/instruction",
        "data/benchmark",
        "rag",
        "finetuning",
        "evaluation",
        "backend",
        "frontend",
        "docs",
    ]
    for d in required_dirs:
        path = os.path.join(BASE_DIR, d)
        assert os.path.isdir(path), f"Directory missing: {d}"


def test_taxonomy_json_validity():
    """Verify data/taxonomy.json syntax and contents."""
    taxonomy_path = os.path.join(BASE_DIR, "data", "taxonomy.json")
    assert os.path.exists(taxonomy_path), "data/taxonomy.json does not exist"

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    assert isinstance(units, list), "taxonomy.json must contain a list of units"
    assert len(units) == 7, f"Expected 7 OS units in taxonomy, found {len(units)}"

    for idx, unit in enumerate(units, start=1):
        assert unit.get("unit_id") == f"unit_{idx}"
        assert "unit_name" in unit
        assert "bloom_levels" in unit
        assert "topics" in unit
        assert len(unit["topics"]) > 0


def test_benchmark_json_validity():
    """Verify data/benchmark/os_benchmark.json syntax and schema."""
    benchmark_path = os.path.join(BASE_DIR, "data", "benchmark", "os_benchmark.json")
    assert os.path.exists(benchmark_path), "os_benchmark.json does not exist"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    assert isinstance(items, list), "os_benchmark.json must contain a list"
    assert len(items) >= 3, "Benchmark should contain at least 3 sample records"

    required_keys = {"id", "unit", "topic", "subtopic", "bloom_level", "difficulty", "task_type", "question", "reference_answer"}
    for item in items:
        missing = required_keys - set(item.keys())
        assert not missing, f"Benchmark item missing keys: {missing}"


def test_instruction_jsonl_validity():
    """Verify train, validation, and test JSONL files in data/instruction/."""
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    from finetuning.prepare_dataset import load_and_validate_jsonl

    for split in splits:
        file_path = os.path.join(BASE_DIR, "data", "instruction", split)
        assert os.path.exists(file_path), f"Instruction split missing: {split}"

        records, errors = load_and_validate_jsonl(file_path)
        assert not errors, f"Validation errors in {split}: {errors}"
        assert len(records) > 0, f"Split {split} should contain records"


def test_metadata_csv_headers():
    """Verify data/rag/metadata.csv columns."""
    csv_path = os.path.join(BASE_DIR, "data", "rag", "metadata.csv")
    assert os.path.exists(csv_path), "metadata.csv missing"

    expected_headers = [
        "document_id",
        "filename",
        "unit",
        "topic",
        "subtopic",
        "source_type",
        "source",
        "license_or_access",
        "description",
    ]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == expected_headers, f"CSV headers mismatch. Got: {headers}"


def test_python_module_imports():
    """Verify that all placeholder modules can be imported cleanly."""
    import rag.ingest
    import rag.clean
    import rag.chunk
    import rag.embed
    import rag.retrieve
    import finetuning.prepare_dataset
    import finetuning.train
    import finetuning.evaluate
    import evaluation.baseline
    import evaluation.evaluate_rag
    import evaluation.compare_models
    import backend.main

    assert True
