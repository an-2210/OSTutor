"""
Baseline Model Evaluation Module for OSTutorLLM.

Evaluates unmodified zero-shot / few-shot Base LLMs against `data/benchmark/os_benchmark.json`.
"""

import json
import os
from typing import Any, Dict, List, Optional


def load_benchmark(benchmark_path: str = "data/benchmark/os_benchmark.json") -> List[Dict[str, Any]]:
    """
    Load OS benchmark evaluation questions and reference answers.

    Args:
        benchmark_path: Path to os_benchmark.json file.

    Returns:
        List of benchmark test item dictionaries.
    """
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark dataset not found at: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_base_model(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    benchmark_path: str = "data/benchmark/os_benchmark.json",
) -> Dict[str, Any]:
    """
    Run baseline zero-shot evaluation of an un-tuned Base LLM without RAG context.

    Args:
        model_name: Base LLM name or local model path.
        benchmark_path: Path to benchmark JSON dataset.

    Returns:
        Dictionary of baseline evaluation metrics.
    """
    benchmark_data = load_benchmark(benchmark_path)
    print(f"[Baseline Evaluation] Model: {model_name}")
    print(f"[Baseline Evaluation] Evaluating on {len(benchmark_data)} benchmark items...")

    # TODO: In Phase 4, generate baseline responses and calculate similarity / accuracy scores
    results = {
        "configuration": "Base LLM (No RAG, No Fine-tuning)",
        "model_name": model_name,
        "total_questions": len(benchmark_data),
        "metrics": {
            "overall_accuracy": 0.0,
            "concept_explanation_score": 0.0,
            "problem_solving_score": 0.0,
        },
        "status": "Placeholder initialized. Base model evaluation deferred to Phase 4.",
    }
    return results


if __name__ == "__main__":
    res = evaluate_base_model()
    print(json.dumps(res, indent=2))
