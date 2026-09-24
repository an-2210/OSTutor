"""
Comparative Model Evaluation Utility for OSTutorLLM (Phase 4).

Compares performance metrics between the Base LLM and the Instruction-Tuned Model
(LoRA/QLoRA) on validation data.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import evaluate_response


def load_results_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL results file excluding metadata header."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found at: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "_metadata" not in item:
                records.append(item)
    return records


def compare_models(
    base_results_path: str = "data/evaluation/base_model_test_results.jsonl",
    tuned_results_path: str = "data/evaluation/fine_tuned_val_results.jsonl",
    json_output: str = "data/evaluation/model_comparison_report.json",
    txt_output: str = "data/evaluation/model_comparison_report.txt",
) -> Dict[str, Any]:
    """
    Compute comparative metrics between Base LLM and Instruction-Tuned LLM.
    """
    base_records = load_results_file(base_results_path)
    tuned_records = load_results_file(tuned_results_path)

    def evaluate_set(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        rouge_l_list = []
        concept_cov_list = []
        completion_list = []
        num_acc_list = []
        len_list = []

        for r in records:
            out = r.get("model_output", "")
            eval_res = evaluate_response(r, out)
            rouge_l_list.append(eval_res["rouge_l"])
            concept_cov_list.append(eval_res["concept_coverage"])
            completion_list.append(eval_res["task_completion"])
            len_list.append(len(out.split()))

            if eval_res["numerical_eval"]["numerical_applicable"]:
                num_acc_list.append(1.0 if eval_res["numerical_eval"]["is_correct"] else 0.0)

        n = len(records)
        return {
            "count": n,
            "mean_rouge_l": round(sum(rouge_l_list) / n, 4) if n else 0.0,
            "mean_concept_coverage": round(sum(concept_cov_list) / n, 4) if n else 0.0,
            "task_completion_rate": round(sum(completion_list) / n, 4) if n else 0.0,
            "numerical_accuracy": round(sum(num_acc_list) / len(num_acc_list), 4) if num_acc_list else None,
            "avg_word_count": round(sum(len_list) / n, 1) if n else 0.0,
        }

    base_stats = evaluate_set(base_records)
    tuned_stats = evaluate_set(tuned_records)

    delta = {
        "rouge_l_diff": round(tuned_stats["mean_rouge_l"] - base_stats["mean_rouge_l"], 4),
        "concept_coverage_diff": round(tuned_stats["mean_concept_coverage"] - base_stats["mean_concept_coverage"], 4),
        "task_completion_diff": round(tuned_stats["task_completion_rate"] - base_stats["task_completion_rate"], 4),
    }

    comparison_report = {
        "title": "Base LLM vs. Instruction-Tuned LLM Comparative Benchmark",
        "base_model": base_stats,
        "instruction_tuned_model": tuned_stats,
        "delta": delta,
    }

    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    txt_lines = [
        "==================================================",
        " Base LLM vs. Instruction-Tuned LLM Comparison ",
        "==================================================",
        "1. BASE LLM METRICS",
        f"   Evaluated Count      : {base_stats['count']}",
        f"   Mean ROUGE-L         : {base_stats['mean_rouge_l']:.4f}",
        f"   Mean Concept Coverage: {base_stats['mean_concept_coverage']:.4f}",
        f"   Task Completion Rate : {base_stats['task_completion_rate']*100:.2f}%",
        f"   Avg Word Count       : {base_stats['avg_word_count']}",
        "--------------------------------------------------",
        "2. INSTRUCTION-TUNED LLM METRICS",
        f"   Evaluated Count      : {tuned_stats['count']}",
        f"   Mean ROUGE-L         : {tuned_stats['mean_rouge_l']:.4f}",
        f"   Mean Concept Coverage: {tuned_stats['mean_concept_coverage']:.4f}",
        f"   Task Completion Rate : {tuned_stats['task_completion_rate']*100:.2f}%",
        f"   Avg Word Count       : {tuned_stats['avg_word_count']}",
        "--------------------------------------------------",
        "3. PERFORMANCE DELTA (Tuned - Base)",
        f"   ROUGE-L Delta        : {delta['rouge_l_diff']:+.4f}",
        f"   Concept Coverage Delta: {delta['concept_coverage_diff']:+.4f}",
        f"   Completion Rate Delta: {delta['task_completion_diff']*100:+.2f}%",
        "==================================================",
        f"Comparison report saved to: {json_output} and {txt_output}",
        "==================================================",
    ]

    txt_content = "\n".join(txt_lines)
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(txt_content)
    return comparison_report


def main():
    parser = argparse.ArgumentParser(description="Compare Base LLM vs Instruction-Tuned LLM")
    parser.add_argument("--base-results", type=str, default="data/evaluation/base_model_test_results.jsonl", help="Base model evaluation JSONL")
    parser.add_argument("--tuned-results", type=str, default="data/evaluation/fine_tuned_val_results.jsonl", help="Tuned model evaluation JSONL")
    parser.add_argument("--json-output", type=str, default="data/evaluation/model_comparison_report.json", help="Output JSON report path")
    parser.add_argument("--txt-output", type=str, default="data/evaluation/model_comparison_report.txt", help="Output TXT report path")
    args = parser.parse_args()

    compare_models(
        base_results_path=args.base_results,
        tuned_results_path=args.tuned_results,
        json_output=args.json_output,
        txt_output=args.txt_output,
    )


if __name__ == "__main__":
    main()
