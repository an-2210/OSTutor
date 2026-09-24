"""
Baseline Report Generator for OSTutorLLM (Phase 4).

Reads baseline evaluation results (data/evaluation/base_model_test_results.jsonl),
computes automatic domain metrics, unit/Bloom/task breakdowns, latency stats,
and outputs structured JSON and text reports.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import evaluate_response


def generate_baseline_report(
    results_path: str = "data/evaluation/base_model_test_results.jsonl",
    json_output: str = "data/evaluation/base_model_report.json",
    txt_output: str = "data/evaluation/base_model_report.txt",
) -> Dict[str, Any]:
    """
    Generate baseline evaluation report JSON and text files.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Baseline results file not found: {results_path}")

    metadata = {}
    records: List[Dict[str, Any]] = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "_metadata" in item:
                metadata = item["_metadata"]
            else:
                records.append(item)

    if not records:
        raise ValueError("No test result records found in results file.")

    total = len(records)

    # Compute metrics for each example
    evaluated_records = []
    unit_metrics = defaultdict(list)
    task_metrics = defaultdict(list)
    bloom_metrics = defaultdict(list)

    rouge_scores = []
    concept_cov_scores = []
    task_completion_scores = []
    numerical_correctness = []
    latencies = []

    for r in records:
        model_out = r.get("model_output", "")
        metrics = evaluate_response(r, model_out)

        r_eval = {**r, "metrics": metrics}
        evaluated_records.append(r_eval)

        rouge_scores.append(metrics["rouge_l"])
        concept_cov_scores.append(metrics["concept_coverage"])
        task_completion_scores.append(metrics["task_completion"])
        latencies.append(r.get("latency_seconds", 0.0))

        unit = r.get("unit", 0)
        task_type = r.get("task_type", "general")
        bloom = r.get("bloom_level", "Understand")

        unit_metrics[unit].append(metrics)
        task_metrics[task_type].append(metrics)
        bloom_metrics[bloom].append(metrics)

        if metrics["numerical_eval"]["numerical_applicable"]:
            numerical_correctness.append(1.0 if metrics["numerical_eval"]["is_correct"] else 0.0)

    # Summary aggregations
    mean_rouge = round(sum(rouge_scores) / total, 4) if total else 0.0
    mean_concept = round(sum(concept_cov_scores) / total, 4) if total else 0.0
    mean_completion = round(sum(task_completion_scores) / total, 4) if total else 0.0
    mean_num_acc = round(sum(numerical_correctness) / len(numerical_correctness), 4) if numerical_correctness else None
    mean_latency = round(sum(latencies) / total, 4) if total else 0.0

    # Distributions
    unit_dist = dict(Counter(r.get("unit") for r in records))
    bloom_dist = dict(Counter(r.get("bloom_level") for r in records))
    task_dist = dict(Counter(r.get("task_type") for r in records))
    diff_dist = dict(Counter(r.get("difficulty") for r in records))

    unit_breakdown = {}
    for u, m_list in unit_metrics.items():
        unit_breakdown[f"unit_{u}"] = {
            "count": len(m_list),
            "mean_rouge_l": round(sum(m["rouge_l"] for m in m_list) / len(m_list), 4),
            "mean_concept_coverage": round(sum(m["concept_coverage"] for m in m_list) / len(m_list), 4),
        }

    task_breakdown = {}
    for t, m_list in task_metrics.items():
        task_breakdown[t] = {
            "count": len(m_list),
            "mean_rouge_l": round(sum(m["rouge_l"] for m in m_list) / len(m_list), 4),
            "mean_concept_coverage": round(sum(m["concept_coverage"] for m in m_list) / len(m_list), 4),
        }

    report_data = {
        "evaluation_title": "OSTutorLLM Baseline Model Evaluation Report",
        "metadata": metadata,
        "summary": {
            "total_evaluated": total,
            "mean_rouge_l": mean_rouge,
            "mean_concept_coverage": mean_concept,
            "task_completion_rate": mean_completion,
            "numerical_accuracy": mean_num_acc,
            "mean_latency_seconds": mean_latency,
        },
        "distributions": {
            "units": unit_dist,
            "bloom_levels": bloom_dist,
            "task_types": task_dist,
            "difficulty": diff_dist,
        },
        "breakdowns": {
            "by_unit": unit_breakdown,
            "by_task_type": task_breakdown,
        },
    }

    # Save JSON report
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate human-readable text report
    txt_lines = [
        "==================================================",
        " OSTutorLLM Baseline Model Evaluation Report ",
        "==================================================",
        f"Model Name           : {metadata.get('model', 'Unknown')}",
        f"Hardware Accelerator : {metadata.get('hardware', 'Unknown')}",
        f"Evaluation Timestamp : {metadata.get('timestamp', 'Unknown')}",
        f"Dataset Hash (Test)  : {metadata.get('dataset_hash', 'Unknown')}",
        "--------------------------------------------------",
        "1. AGGREGATE EVALUATION METRICS",
        "--------------------------------------------------",
        f"Total Examples Evaluated: {total}",
        f"Mean ROUGE-L Score      : {mean_rouge:.4f}",
        f"Mean Concept Coverage   : {mean_concept:.4f}",
        f"Task Completion Rate    : {mean_completion * 100:.2f}%",
        f"Numerical Accuracy      : {f'{mean_num_acc * 100:.2f}%' if mean_num_acc is not None else 'N/A'}",
        f"Mean Latency per Item   : {mean_latency:.3f} s",
        "--------------------------------------------------",
        "2. DATASET DISTRIBUTIONS",
        "--------------------------------------------------",
        f"Units Distribution      : {unit_dist}",
        f"Bloom Level Distribution: {bloom_dist}",
        f"Task Type Distribution  : {task_dist}",
        f"Difficulty Distribution : {diff_dist}",
        "--------------------------------------------------",
        "3. BREAKDOWN BY UNIT",
        "--------------------------------------------------",
    ]

    for u_key, u_stats in sorted(unit_breakdown.items()):
        txt_lines.append(f"  {u_key.upper()}: count={u_stats['count']}, ROUGE-L={u_stats['mean_rouge_l']}, ConceptCov={u_stats['mean_concept_coverage']}")

    txt_lines.extend([
        "--------------------------------------------------",
        "4. BREAKDOWN BY TASK TYPE",
        "--------------------------------------------------",
    ])
    for t_key, t_stats in sorted(task_breakdown.items()):
        txt_lines.append(f"  {t_key}: count={t_stats['count']}, ROUGE-L={t_stats['mean_rouge_l']}, ConceptCov={t_stats['mean_concept_coverage']}")

    txt_lines.extend([
        "==================================================",
        f"Report saved to: {json_output} and {txt_output}",
        "==================================================",
    ])

    txt_content = "\n".join(txt_lines)
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(txt_content)
    return report_data


def main():
    parser = argparse.ArgumentParser(description="Generate Baseline Evaluation Report")
    parser.add_argument("--results", type=str, default="data/evaluation/base_model_test_results.jsonl", help="Input results JSONL")
    parser.add_argument("--json-output", type=str, default="data/evaluation/base_model_report.json", help="Output JSON report path")
    parser.add_argument("--txt-output", type=str, default="data/evaluation/base_model_report.txt", help="Output TXT report path")
    args = parser.parse_args()

    generate_baseline_report(
        results_path=args.results,
        json_output=args.json_output,
        txt_output=args.txt_output,
    )


if __name__ == "__main__":
    main()
