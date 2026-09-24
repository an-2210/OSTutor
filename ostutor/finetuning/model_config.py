"""
Centralized Model and Training Configuration for OSTutorLLM (Phase 4).

Defines base model identifiers, sequence limits, LoRA/QLoRA hyperparameters,
and evaluation sampling configurations.
"""

import os
from typing import Any, Dict, List

# Selected Open-Weight Base Model
# Qwen/Qwen2.5-1.5B-Instruct is a highly-capable, permissive (Apache 2.0) open-weight model
# optimized for local execution on Apple Silicon (MPS) and CUDA devices.
BASE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_BASE_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"

# Context & Sequence Length Limits
MAX_SEQ_LENGTH: int = 1024

# LoRA Hyperparameters
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Parameters
LEARNING_RATE: float = 2e-4
BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 4
NUM_TRAIN_EPOCHS: int = 3
WARMUP_RATIO: float = 0.03
WEIGHT_DECAY: float = 0.01

# Inference Parameters (Deterministic Baseline Evaluation)
INFERENCE_TEMPERATURE: float = 0.0
INFERENCE_TOP_P: float = 1.0
MAX_NEW_TOKENS: int = 512
RANDOM_SEED: int = 42

# Paths
TRAIN_DATA_PATH: str = "data/instruction/formatted_train.jsonl"
VAL_DATA_PATH: str = "data/instruction/formatted_validation.jsonl"
TEST_DATA_PATH: str = "data/instruction/test.jsonl"
OUTPUT_DIR: str = "checkpoints/ostutor_lora"
EVALUATION_DIR: str = "data/evaluation"


def get_model_config() -> Dict[str, Any]:
    """Return dictionary of centralized configuration settings."""
    return {
        "base_model": BASE_MODEL,
        "fallback_base_model": FALLBACK_BASE_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "inference_temperature": INFERENCE_TEMPERATURE,
        "inference_top_p": INFERENCE_TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "random_seed": RANDOM_SEED,
        "train_data_path": TRAIN_DATA_PATH,
        "val_data_path": VAL_DATA_PATH,
        "test_data_path": TEST_DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "evaluation_dir": EVALUATION_DIR,
    }
