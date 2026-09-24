# OSTutorLLM Instruction Dataset Documentation

**OSTutorLLM Phase 3: Operating Systems Instruction-Tuning Dataset Repository**

---

## 1. Overview

This directory contains the curriculum-aligned instruction-tuning dataset for **OSTutorLLM**. The dataset is formatted as structured JSONL records mapped directly to the 7 Operating Systems units defined in `data/taxonomy.json`.

---

## 2. Directory Structure

```text
data/instruction/
├── train.jsonl                 # Training split (~80%, 984 records)
├── validation.jsonl            # Validation split (~10%, 123 records)
├── test.jsonl                  # Protected test split (~10%, 123 records)
├── formatted_train.jsonl       # Generic chat-format training split
├── formatted_validation.jsonl  # Generic chat-format validation split
├── formatted_test.jsonl        # Generic chat-format test split
├── dataset_statistics.json     # Machine-readable distribution metrics
├── dataset_statistics.txt      # Formatted text statistics report
└── README.md                   # This documentation file
```

---

## 3. Schema Specification

Each record in `train.jsonl`, `validation.jsonl`, and `test.jsonl` follows this canonical schema:

```json
{
  "unit": 3,
  "topic": "CPU Scheduling",
  "subtopic": "Round Robin",
  "bloom_level": "Apply",
  "difficulty": "medium",
  "task_type": "problem_solving",
  "instruction": "Solve the CPU scheduling problem step by step.",
  "input": "Processes P1, P2 and P3 have burst times ...",
  "output": "First construct the execution order ...",
  "source_type": "faculty_material"
}
```

### Required Fields:
- **`unit`** *(Integer 1-7)*: OS Syllabus Unit number.
- **`topic`** *(String)*: Topic name matching `data/taxonomy.json`.
- **`subtopic`** *(String)*: Subtopic name matching `data/taxonomy.json`.
- **`bloom_level`** *(String)*: Cognitive level (`Remember`, `Understand`, `Apply`, `Analyze`, `Evaluate`).
- **`difficulty`** *(String)*: Task complexity (`easy`, `medium`, `hard`).
- **`task_type`** *(String)*: Educational task category (e.g., `concept_explanation`, `numerical_problem`, `debugging`, `mcq_generation`, `viva_generation`).
- **`instruction`** *(String)*: The prompt or problem statement given to the tutor model.
- **`input`** *(String)*: Optional context, memory layout, process table, or snippet (may be empty string `""`).
- **`output`** *(String)*: Verified pedagogical model response.
- **`source_type`** *(String)*: Provenance origin (`faculty_material`, `course_material`, `synthetic_validated`, `manually_authored`).

---

## 4. Protected Test Set Notice

> **IMPORTANT RESEARCH NOTICE:**
> `data/instruction/test.jsonl` and `data/benchmark/os_benchmark.json` serve as **strictly protected evaluation sets**. They must NEVER be included in training prompts, fine-tuning data, or dataset generation prompts to prevent data contamination and leakage.
