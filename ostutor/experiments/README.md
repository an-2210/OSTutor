# OSTutorLLM Experiment Configurations (Phase 4)

This directory contains machine-readable experiment configurations for baseline evaluation and LoRA fine-tuning preparation.

## Files

1. **`baseline_config.json`**
   - Defines parameters for zero-shot baseline evaluation of the unmodified base LLM (`Qwen/Qwen2.5-1.5B-Instruct`).
   - Uses low-temperature deterministic generation (`temperature = 0.0`).
   - Evaluates exclusively against the protected test set (`data/instruction/test.jsonl`).

2. **`pilot_lora_config.json`**
   - Defines hyperparameters for Parameter-Efficient Fine-Tuning (PEFT / LoRA).
   - Configures rank $r=16$, $\alpha=32$, dropout $0.05$, and target projection modules (`q_proj`, `v_proj`, etc.).
   - Consumes only `formatted_train.jsonl` and `formatted_validation.jsonl`.
   - Strictly excludes `test.jsonl`.

## Running Experiments

### Hardware Check
```bash
python3 finetuning/hardware_info.py
```

### Baseline Inference & Evaluation
```bash
python3 evaluation/baseline.py --limit 5 --mock
python3 evaluation/generate_baseline_report.py
```

### Pilot Training Execution
```bash
python3 finetuning/train.py --limit 100 --mock
```

### Fine-Tuned Model Evaluation & Model Comparison
```bash
python3 finetuning/evaluate.py --limit 5 --mock
python3 evaluation/compare_base_and_tuned.py
```
