"""
Dataset Chat Formatter for OSTutorLLM (Phase 3).

Converts structured OS instruction records into model-agnostic chat format
(`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}`)
while preserving full research metadata for evaluation.
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


def convert_record_to_chat_format(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single structured instruction record into a generic chat message payload.

    Args:
        record: Structured JSONL instruction record.

    Returns:
        Chat payload with 'messages' list and 'metadata' dict.
    """
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    assistant_output = str(record.get("output", "")).strip()

    if user_input:
        user_content = f"{instruction}\n\nContext / Input:\n{user_input}"
    else:
        user_content = instruction

    chat_payload = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_output},
        ],
        "metadata": {
            "unit": record.get("unit"),
            "topic": record.get("topic"),
            "subtopic": record.get("subtopic"),
            "bloom_level": record.get("bloom_level"),
            "difficulty": record.get("difficulty"),
            "task_type": record.get("task_type"),
            "source_type": record.get("source_type"),
        },
    }

    return chat_payload


def format_instruction_splits() -> List[Path]:
    """
    Format train.jsonl, validation.jsonl, test.jsonl into generic chat format JSONL files.

    Returns:
        List of generated chat dataset Path objects.
    """
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    formatted_paths: List[Path] = []

    print("\nOSTutorLLM Chat Dataset Formatter")
    print("================================")

    for split in splits:
        input_path = INSTRUCTION_DIR / split
        out_name = f"formatted_{split}"
        output_path = INSTRUCTION_DIR / out_name

        records, errs = load_and_validate_jsonl(input_path)
        if errs:
            logger.warning(f"Validation warnings in {split}: {len(errs)}")

        formatted_records = [convert_record_to_chat_format(r) for r in records]

        with open(output_path, "w", encoding="utf-8") as f:
            for item in formatted_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        formatted_paths.append(output_path)
        print(f"  - Formatted {len(formatted_records):>4} records from {split:<16} -> {out_name}")

    print("================================\n")
    return formatted_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    format_instruction_splits()
