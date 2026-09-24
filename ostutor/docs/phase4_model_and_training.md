# OSTutorLLM — Phase 4: Base Model Selection, Baseline Evaluation & Fine-Tuning Preparation

## 1. Overview
Phase 4 establishes the experimental baseline and fine-tuning infrastructure for **OSTutorLLM**:
1. Hardware inspection and accelerator detection (Apple Silicon MPS / NVIDIA CUDA).
2. Selection of open-weight Base LLM (`Qwen/Qwen2.5-1.5B-Instruct`).
3. SHA-256 dataset versioning and hashing.
4. Reproducible zero-shot baseline evaluation on the protected test set.
5. Automated domain metrics (ROUGE-L, concept coverage, specialized numerical OS problem checkers).
6. Human evaluation template generation (`data/evaluation/human_review_template.jsonl`).
7. LoRA fine-tuning framework and small pilot experiment execution.
8. Baseline vs. Fine-Tuned comparative analysis pipeline.

---

## 2. Hardware Inspection
Telemetrics are inspected dynamically without requiring admin privileges:
- **Primary Platform**: macOS Apple Silicon (`arm64`)
- **Accelerator**: Apple Silicon (MPS)
- **RAM**: 16 GB Unified Memory
- **PyTorch**: 2.11.0 with native `torch.backends.mps.is_available()`

Command:
```bash
python3 finetuning/hardware_info.py
```

---

## 3. Base Model Selection
- **Selected Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Fallback Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **License**: Apache 2.0 (permissive, open research distribution)
- **Parameters**: 1.5 Billion
- **Context Window**: 32,768 tokens (Fine-tuning max sequence length set to 1,024)
- **Selection Rationale**: High technical reasoning capability, low memory footprint (~3.0 GB FP16), native MPS FP16 support on Apple Silicon.

See [docs/model_selection.md](file:///Users/anweshalaha/Desktop/cyberthreat/ostutor/docs/model_selection.md) for full trade-off evaluation.

---

## 4. Protected Test Set Isolation Rule
The test set (`data/instruction/test.jsonl`, SHA-256: `0af07ada...`) is strictly protected:
- NEVER used in training (`train.py`).
- NEVER used in hyperparameter tuning or prompt optimization.
- Reference answers are NEVER provided to the model during inference.

Dataset hashes can be checked at any time:
```bash
python3 evaluation/dataset_hash.py
```

---

## 5. Baseline Evaluation Methodology
Baseline responses are generated using deterministic sampling (`temperature = 0.0`):
```bash
python3 evaluation/baseline.py --limit 5 --mock
python3 evaluation/generate_baseline_report.py
```

Evaluation metrics:
- **ROUGE-L**: Longest common subsequence match against reference output.
- **Concept Coverage**: Proxy metric evaluating domain technical keyword presence.
- **Numerical Correctness**: Domain-specific logic checking for CPU scheduling, page replacement, address translation, and deadlock safety.
- **Task Completion**: Validates structural completeness of tutor responses.

---

## 6. LoRA Fine-Tuning Preparation & Pilot Training
- **Framework**: Hugging Face `transformers` + `peft` (LoRA).
- **LoRA Parameters**: Rank $r=16$, $\alpha=32$, Dropout $0.05$.
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- **Learning Rate**: $2 \times 10^{-4}$, Linear scheduler, Warmup ratio $0.03$.
- **Batch Size**: 2, Gradient accumulation steps: 4.

Command:
```bash
python3 finetuning/train.py --limit 100 --mock
```

---

## 7. Comparative Evaluation Pipeline
Evaluates fine-tuned adapter on validation set and compares against base LLM:
```bash
python3 finetuning/evaluate.py --limit 5 --mock
python3 evaluation/compare_base_and_tuned.py
```

Outputs:
- `data/evaluation/model_comparison_report.json`
- `data/evaluation/model_comparison_report.txt`

---

## 8. File Structure Summary

```text
finetuning/
├── hardware_info.py         # Hardware telemetry and accelerator detection
├── model_config.py          # Centralized hyperparameter & model configuration
├── train.py                 # LoRA fine-tuning training & pilot script
├── evaluate.py              # Fine-tuned model validation evaluator
├── format_dataset.py        # Dataset formatting (Phase 3)
└── dataset_stats.py         # Dataset statistics (Phase 3)

evaluation/
├── baseline.py              # Base LLM baseline evaluation script
├── metrics.py               # ROUGE-L, concept coverage, & numerical OS checkers
├── generate_baseline_report.py # Baseline report generator
├── human_review_template.py # Blind human evaluation template generator
├── compare_base_and_tuned.py# Comparative model analysis script
└── dataset_hash.py          # Dataset SHA-256 version digest calculator

experiments/
├── baseline_config.json     # Baseline experiment config
├── pilot_lora_config.json   # LoRA pilot training config
└── README.md                # Experiments documentation
```
