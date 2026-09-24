"""
Fine-Tuned Model Evaluator for OSTutorLLM.

Provides interface functions for evaluating fine-tuned OS tutor models against held-out
instruction test sets (`data/instruction/test.jsonl`).
"""

import json
import os
from typing import Any, Dict, List, Optional


def load_test_dataset(test_jsonl_path: str = "data/instruction/test.jsonl") -> List[Dict[str, Any]]:
    """
    Load held-out instruction test set records.

    Args:
        test_jsonl_path: Path to test.jsonl file.

    Returns:
        List of test record dictionaries.
    """
    if not os.path.exists(test_jsonl_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_jsonl_path}")

    records: List[Dict[str, Any]] = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_fine_tuned_model(
    model_checkpoint_path: str,
    test_jsonl_path: str = "data/instruction/test.jsonl",
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a fine-tuned model checkpoint on the instruction test dataset.

    Args:
        model_checkpoint_path: Path to fine-tuned LoRA adapter or merged weights.
        test_jsonl_path: Path to held-out test set.
        output_report_path: Optional path to save evaluation report JSON.

    Returns:
        Dictionary of computed metrics (BLEU, ROUGE, Exact Match, Concept Accuracy).
    """
    test_records = load_test_dataset(test_jsonl_path)

    print(f"[Model Evaluator] Loaded {len(test_records)} test records from: {test_jsonl_path}")
    print(f"[Model Evaluator] Evaluating checkpoint: {model_checkpoint_path}")

    # TODO: In Phase 5/7, generate predictions and compute ROUGE-L, BLEU, and LLM-as-a-Judge scores
    metrics_summary = {
        "checkpoint": model_checkpoint_path,
        "test_samples": len(test_records),
        "metrics": {
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "bleu_4": 0.0,
            "os_concept_accuracy": 0.0,
        },
        "status": "Placeholder evaluated. Full evaluation deferred to Phase 7.",
    }

    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    print("Fine-tuning evaluator placeholder module ready.")
