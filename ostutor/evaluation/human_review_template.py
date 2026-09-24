"""
Human Review Template Generator for OSTutorLLM (Phase 4).

Extracts a representative stratified sample of baseline evaluation responses
(30-50 examples across units, Bloom levels, and task types) and formats a template
JSONL file for blind human expert evaluation.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_human_review_template(
    results_path: str = "data/evaluation/base_model_test_results.jsonl",
    output_path: str = "data/evaluation/human_review_template.jsonl",
    sample_size: int = 35,
    seed: int = 42,
) -> str:
    """
    Generate stratified human review JSONL file.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Baseline results file not found at: {results_path}")

    records: List[Dict[str, Any]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "_metadata" in item:
                continue
            records.append(item)

    if not records:
        raise ValueError(f"No evaluation records found in {results_path}")

    # Stratified sampling logic
    random.seed(seed)
    if len(records) <= sample_size:
        sampled = records
    else:
        # Group by unit & task_type
        strata: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            key = f"U{r.get('unit', 0)}_{r.get('task_type', 'general')}"
            strata.setdefault(key, []).append(r)

        sampled = []
        # Sample proportionally from each stratum
        stratum_keys = sorted(strata.keys())
        per_stratum = max(1, sample_size // len(stratum_keys))

        for k in stratum_keys:
            items = strata[k]
            random.shuffle(items)
            sampled.extend(items[:per_stratum])

        # If we still need more to reach sample_size
        remaining = [r for r in records if r not in sampled]
        random.shuffle(remaining)
        needed = sample_size - len(sampled)
        if needed > 0:
            sampled.extend(remaining[:needed])

    # Build human review structure
    review_items = []
    for item in sampled:
        review_record = {
            "question_id": item.get("question_id"),
            "unit": item.get("unit"),
            "topic": item.get("topic"),
            "bloom_level": item.get("bloom_level"),
            "task_type": item.get("task_type"),
            "difficulty": item.get("difficulty"),
            "instruction": item.get("instruction"),
            "input": item.get("input"),
            "model_output": item.get("model_output"),
            "reference_output": item.get("reference_output"),
            "human_evaluation": {
                "correctness": None,      # Scale 1-5
                "relevance": None,        # Scale 1-5
                "clarity": None,          # Scale 1-5
                "groundedness": None,     # Scale 1-5
                "reasoning_quality": None,# Scale 1-5
                "bloom_alignment": None,  # Scale 1-5
                "overall_quality": None,  # Scale 1-5
                "reviewer_id": "",
                "comments": "",
            },
        }
        review_items.append(review_record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rev in review_items:
            f.write(json.dumps(rev, ensure_ascii=False) + "\n")

    print(f"Generated human review template with {len(review_items)} stratified records.")
    print(f"Template saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Human Review Template JSONL")
    parser.add_argument("--results", type=str, default="data/evaluation/base_model_test_results.jsonl", help="Input baseline results JSONL")
    parser.add_argument("--output", type=str, default="data/evaluation/human_review_template.jsonl", help="Output template JSONL path")
    parser.add_argument("--sample-size", type=int, default=35, help="Number of records to sample (30-50)")
    args = parser.parse_args()

    generate_human_review_template(
        results_path=args.results,
        output_path=args.output,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
