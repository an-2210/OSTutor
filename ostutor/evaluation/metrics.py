"""
OS Domain Metrics and Automatic Evaluation Utilities for OSTutorLLM (Phase 4).

Computes textual metrics (ROUGE-L, concept coverage) and specialized OS domain
metrics (numerical problem verification for scheduling, page replacement, address
translation, and deadlock analysis).
"""

import re
from typing import Any, Dict, List, Optional


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Compute exact string match ratio (case-insensitive, normalized whitespace)."""
    ref_norm = " ".join(reference.strip().lower().split())
    hyp_norm = " ".join(hypothesis.strip().lower().split())
    return 1.0 if ref_norm == hyp_norm else 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L LCS (Longest Common Subsequence) F1 score.
    Falls back to a standard LCS calculation if rouge_score package is missing.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return round(scores['rougeL'].fmeasure, 4)
    except Exception:
        # Heuristic LCS fallback
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0

        # Dynamic programming LCS
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        prec = lcs / len(hyp_tokens)
        rec = lcs / len(ref_tokens)
        if prec + rec == 0:
            return 0.0
        return round(2 * (prec * rec) / (prec + rec), 4)


def compute_concept_coverage(reference: str, hypothesis: str, expected_concepts: Optional[List[str]] = None) -> float:
    """
    Compute key concept / keyword coverage proxy score.
    Extracts significant OS terms from reference or uses explicitly provided concepts.
    """
    hyp_lower = hypothesis.lower()
    if expected_concepts:
        keywords = [k.lower() for k in expected_concepts if k]
    else:
        # Extract OS-specific domain technical words from reference
        raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', reference.lower())
        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "with", "that", "this",
            "are", "an", "be", "as", "by", "on", "at", "from", "or", "it", "has"
        }
        keywords = list(set([w for w in raw_words if w not in stopwords]))

    if not keywords:
        return 1.0

    matches = sum(1 for kw in keywords if kw in hyp_lower)
    return round(matches / len(keywords), 4)


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text string."""
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def evaluate_numerical_os_problem(task_type: str, reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Specialized checker for numerical OS problems (scheduling, memory, deadlocks).

    Checks whether key numerical answers present in reference appear in hypothesis.
    """
    ref_nums = extract_numbers(reference)
    hyp_nums = extract_numbers(hypothesis)

    if not ref_nums:
        return {"numerical_applicable": False, "match_ratio": 1.0, "is_correct": True}

    matches = sum(1 for n in ref_nums if n in hyp_nums)
    ratio = round(matches / len(ref_nums), 4) if ref_nums else 1.0
    is_correct = ratio >= 0.8

    details = {
        "numerical_applicable": True,
        "reference_numbers": ref_nums,
        "hypothesis_numbers": hyp_nums,
        "matched_count": matches,
        "total_expected": len(ref_nums),
        "match_ratio": ratio,
        "is_correct": is_correct,
    }

    # Task specific extra heuristics
    hyp_lower = hypothesis.lower()
    ref_lower = reference.lower()

    if "deadlock" in task_type.lower() or "banker" in ref_lower:
        safe_in_ref = "safe" in ref_lower and "unsafe" not in ref_lower
        unsafe_in_ref = "unsafe" in ref_lower
        if safe_in_ref:
            details["classification_correct"] = "safe" in hyp_lower and "unsafe" not in hyp_lower
        elif unsafe_in_ref:
            details["classification_correct"] = "unsafe" in hyp_lower

    return details


def evaluate_response(example: Dict[str, Any], model_output: str) -> Dict[str, Any]:
    """
    Evaluate a single model output record against reference.

    Returns dict of metrics for this example.
    """
    reference = example.get("reference_output", example.get("output", ""))
    task_type = example.get("task_type", "general")

    rouge_l = compute_rouge_l(reference, model_output)
    exact_match = compute_exact_match(reference, model_output)
    concept_cov = compute_concept_coverage(reference, model_output)
    num_eval = evaluate_numerical_os_problem(task_type, reference, model_output)

    is_completed = len(model_output.strip()) >= 20

    return {
        "rouge_l": rouge_l,
        "exact_match": exact_match,
        "concept_coverage": concept_cov,
        "task_completion": 1.0 if is_completed else 0.0,
        "numerical_eval": num_eval,
        "overall_score": round(0.4 * rouge_l + 0.4 * concept_cov + 0.2 * (1.0 if is_completed else 0.0), 4),
    }
