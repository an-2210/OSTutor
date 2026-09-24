"""
Unit Test Suite for OSTutorLLM Instruction Dataset & Fine-Tuning Tools (Phase 3).
"""

import json
import unittest
from pathlib import Path

from finetuning.format_dataset import convert_record_to_chat_format, format_instruction_splits
from finetuning.prepare_dataset import (
    ALLOWED_BLOOM_LEVELS,
    ALLOWED_DIFFICULTIES,
    ALLOWED_TASK_TYPES,
    INSTRUCTION_DIR,
    REQUIRED_KEYS,
    check_split_leakage,
    detect_exact_duplicates,
    load_and_validate_jsonl,
    load_taxonomy_data,
)

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "os_benchmark.json"


class TestInstructionDataset(unittest.TestCase):
    """Test instruction-tuning dataset integrity, schema validation, and formatting."""

    def setUp(self):
        self.splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
        self.records_by_split = {}
        for split in self.splits:
            path = INSTRUCTION_DIR / split
            recs, errs = load_and_validate_jsonl(path)
            self.assertEqual(len(errs), 0, f"Schema validation errors in {split}: {errs}")
            self.assertGreater(len(recs), 0, f"Split {split} is empty")
            self.records_by_split[split] = recs

    def test_schema_required_fields(self):
        """Ensure all records contain all required keys."""
        for split, recs in self.records_by_split.items():
            for idx, r in enumerate(recs, start=1):
                missing = REQUIRED_KEYS - set(r.keys())
                self.assertEqual(len(missing), 0, f"{split} record #{idx} missing keys: {missing}")

    def test_valid_bloom_and_difficulty(self):
        """Ensure all records have valid Bloom and difficulty values."""
        for split, recs in self.records_by_split.items():
            for idx, r in enumerate(recs, start=1):
                self.assertIn(r["bloom_level"], ALLOWED_BLOOM_LEVELS, f"{split} #{idx}: Invalid Bloom")
                self.assertIn(r["difficulty"], ALLOWED_DIFFICULTIES, f"{split} #{idx}: Invalid difficulty")
                self.assertIn(r["task_type"], ALLOWED_TASK_TYPES, f"{split} #{idx}: Invalid task type")

    def test_taxonomy_alignment(self):
        """Verify units (1-7), topics, and subtopics align with data/taxonomy.json."""
        valid_units, unit_topics_map, topic_subtopics_map = load_taxonomy_data()
        self.assertEqual(len(valid_units), 7)

        for split, recs in self.records_by_split.items():
            for idx, r in enumerate(recs, start=1):
                unit = r["unit"]
                self.assertIn(unit, valid_units, f"{split} #{idx}: Invalid unit {unit}")

                topic = r["topic"].lower()
                self.assertIn(topic, unit_topics_map[unit], f"{split} #{idx}: Topic '{r['topic']}' not in Unit {unit}")

    def test_exact_duplicates(self):
        """Ensure zero exact duplicates across the full dataset."""
        all_recs = []
        for recs in self.records_by_split.values():
            all_recs.extend(recs)

        dups = detect_exact_duplicates(all_recs)
        self.assertEqual(len(dups), 0, f"Found exact duplicates in dataset: {dups[:3]}")

    def test_train_test_split_leakage(self):
        """Ensure zero instruction leakage between train and test splits."""
        train_recs = self.records_by_split["train.jsonl"]
        test_recs = self.records_by_split["test.jsonl"]

        leakage = check_split_leakage(train_recs, test_recs)
        self.assertEqual(len(leakage), 0, f"Data leakage detected between train and test: {leakage}")

    def test_benchmark_isolation(self):
        """Ensure dataset records do not contaminate os_benchmark.json."""
        if not BENCHMARK_PATH.exists():
            return
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            bench_items = json.load(f)

        bench_questions = {b.get("question", "").strip().lower() for b in bench_items}
        for split, recs in self.records_by_split.items():
            for r in recs:
                prompt = r.get("instruction", "").strip().lower()
                self.assertNotIn(prompt, bench_questions, f"Contamination: '{prompt}' found in benchmark!")

    def test_chat_formatting(self):
        """Verify record conversion into generic chat template format."""
        sample_rec = {
            "unit": 3,
            "topic": "CPU Scheduling",
            "subtopic": "Round Robin",
            "bloom_level": "Apply",
            "difficulty": "medium",
            "task_type": "problem_solving",
            "instruction": "Solve Round Robin for P1=4ms.",
            "input": "P1 burst=4ms",
            "output": "Step 1: Execute P1...",
            "source_type": "faculty_material",
        }
        chat_item = convert_record_to_chat_format(sample_rec)

        self.assertIn("messages", chat_item)
        self.assertIn("metadata", chat_item)
        self.assertEqual(len(chat_item["messages"]), 2)
        self.assertEqual(chat_item["messages"][0]["role"], "user")
        self.assertEqual(chat_item["messages"][1]["role"], "assistant")
        self.assertIn("P1 burst=4ms", chat_item["messages"][0]["content"])
        self.assertEqual(chat_item["metadata"]["unit"], 3)


if __name__ == "__main__":
    unittest.main()
