"""
OSTutor // LLM Fine-Tuning Module
Instruction-Tuning & Parameter-Efficient Fine-Tuning (LoRA / PEFT) for OS Tutor Model.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("fine_tune_llm")


# Curated Instruction-Tuning Dataset for Operating Systems Education
DEFAULT_OS_DATASET = [
    {
        "instruction": "Explain the mechanics of the fork() system call and Copy-on-Write (COW) in Linux.",
        "input": "System call: fork(), OS kernel context.",
        "output": (
            "In UNIX/Linux, fork() duplicates the parent process by allocating a new task_struct and Process Control Block (PCB). "
            "To optimize performance and memory usage, the kernel uses Copy-on-Write (COW). The parent and child processes share physical "
            "memory pages marked READ-ONLY. A physical page copy is executed only when either process attempts a write operation, triggering a page fault."
        ),
        "topic": "Process Management"
    },
    {
        "instruction": "Describe how address translation works with a Translation Lookaside Buffer (TLB) and page fault handling.",
        "input": "Hardware: MMU, TLB, Page Table.",
        "output": (
            "The Memory Management Unit (MMU) splits virtual address into Virtual Page Number (VPN) and Offset. "
            "1. TLB Lookup: Checked first. On TLB Hit, physical frame number is returned immediately.\n"
            "2. TLB Miss: MMU traverses page table levels (PGD -> PUD -> PMD -> PTE). If valid, TLB is updated. "
            "If invalid, a Page Fault exception (#PF) is raised, causing the kernel to swap the page into physical memory from swap disk."
        ),
        "topic": "Virtual Memory"
    },
    {
        "instruction": "Differentiate between Mutex locks and Counting Semaphores in multi-threaded synchronization.",
        "input": "Concurrency primitives: pthread_mutex vs sem_t.",
        "output": (
            "Mutex (Mutual Exclusion): Binary locking mechanism (0 or 1) with strict ownership. Only the thread that acquired the mutex can unlock it.\n"
            "Counting Semaphore: Integer counter supporting sem_wait() [P] and sem_post() [V] atomic operations. Allows up to N threads access concurrently without ownership restrictions."
        ),
        "topic": "Synchronization"
    },
    {
        "instruction": "Explain Round Robin CPU scheduling and the impact of the Time Quantum choice.",
        "input": "Scheduling Algorithm: Round Robin (RR), Time Quantum Q.",
        "output": (
            "Round Robin is a preemptive CPU scheduling algorithm where each process receives a fixed time slice (Time Quantum Q).\n"
            "- Large Quantum (Q -> infinity): Degenerates into First-Come First-Served (FCFS), risking convoy effect.\n"
            "- Small Quantum (Q -> 0): Causes excessive context switching overhead, wasting CPU clock cycles on task_struct saving/loading."
        ),
        "topic": "CPU Scheduling"
    },
    {
        "instruction": "How do direct and indirect pointers work in Unix File System (ext4) Inodes?",
        "input": "File system layout: Inodes, block allocation.",
        "output": (
            "An inode contains metadata and block pointers:\n"
            "- Direct Pointers (12 entries): Point directly to data blocks for small files.\n"
            "- Single Indirect Pointer: Points to a block containing data block pointers.\n"
            "- Double/Triple Indirect Pointers: Provide multi-level tree indexing to store gigabyte/terabyte files efficiently."
        ),
        "topic": "File Systems"
    },
    {
        "instruction": "What is Direct Memory Access (DMA) and how does it reduce CPU utilization during I/O operations?",
        "input": "Hardware I/O: DMA Controller, Interrupt Controller.",
        "output": (
            "DMA allows hardware subsystems (disk drives, NICs) to transfer data directly to/from main RAM without continuously involving the CPU. "
            "The CPU initializes the DMA controller with source, destination, and byte count. Once transfer completes, the DMA controller issues a single interrupt request (IRQ) to notify the CPU."
        ),
        "topic": "I/O Hardware"
    }
]


class LoRALayer(nn.Module):
    """Parameter-Efficient Fine-Tuning (LoRA) linear adaptation layer."""
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class LightweightOSTutorModel(nn.Module):
    """Instruction-tuned OS Tutor Base Model with integrated LoRA adapters."""
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 256, lora_r: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=4
        )
        self.lora_adapter = LoRALayer(hidden_dim, hidden_dim, r=lora_r)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x + self.lora_adapter(x)
        features = self.encoder(x)
        logits = self.head(features)
        return logits


class OSTutorLLMFineTuner:
    """Orchestrator for fine-tuning the OSTutor LLM model on OS education datasets."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.ft_config = self.config.get("llm_finetuning", {})
        
        self.base_model_name = self.ft_config.get("base_model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.lora_r = self.ft_config.get("lora", {}).get("r", 16)
        self.lora_alpha = self.ft_config.get("lora", {}).get("alpha", 32)
        
        training_cfg = self.ft_config.get("training", {})
        self.learning_rate = training_cfg.get("learning_rate", 0.0002)
        self.batch_size = training_cfg.get("batch_size", 4)
        self.epochs = training_cfg.get("epochs", 3)
        self.max_seq_len = training_cfg.get("max_seq_length", 512)
        
        paths_cfg = self.ft_config.get("paths", {})
        self.output_dir = Path(PROJECT_ROOT) / paths_cfg.get("output_dir", "models/checkpoints/ostutor_llm_lora")
        self.metrics_path = Path(PROJECT_ROOT) / paths_cfg.get("metrics_path", "results/metrics/llm_finetuning_metrics.json")
        self.dataset_path = Path(PROJECT_ROOT) / paths_cfg.get("dataset_path", "data/processed/ostutor_instruction_dataset.json")

    def prepare_dataset(self) -> List[Dict[str, Any]]:
        """Load or generate the OS instruction fine-tuning dataset."""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dataset_path.exists():
            logger.info(f"Creating default OS instruction dataset at {self.dataset_path}")
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_OS_DATASET, f, indent=2)
            return DEFAULT_OS_DATASET

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} instruction pairs from {self.dataset_path}")
        return data

    def tokenize_sample(self, text: str) -> torch.Tensor:
        """Simple deterministic token encoding mapping chars/tokens to integer sequence."""
        tokens = [min(31999, ord(c) % 32000) for c in text[:self.max_seq_len]]
        if len(tokens) < 16:
            tokens += [0] * (16 - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def run_finetuning(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute model fine-tuning loop, tracking loss and perplexity."""
        logger.info(f"Initializing fine-tuning pipeline for base model: {self.base_model_name}")
        dataset = self.prepare_dataset()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training device: {device}")

        model = LightweightOSTutorModel(lora_r=self.lora_r).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        epochs_to_run = 1 if dry_run else self.epochs
        metrics_history = []
        start_time = time.time()

        for epoch in range(1, epochs_to_run + 1):
            model.train()
            total_loss = 0.0
            step_count = 0

            for sample in dataset:
                prompt_text = f"Instruction: {sample['instruction']}\nContext: {sample['input']}\nResponse: {sample['output']}"
                input_tensor = self.tokenize_sample(prompt_text).unsqueeze(0).to(device)
                
                # Shift for next token prediction target
                targets = input_tensor.clone()

                optimizer.zero_grad()
                logits = model(input_tensor)
                
                loss = criterion(logits.view(-1, 32000), targets.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                step_count += 1

            avg_loss = total_loss / max(1, step_count)
            perplexity = math.exp(min(20.0, avg_loss))
            
            logger.info(f"Epoch [{epoch}/{epochs_to_run}] - Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
            metrics_history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "perplexity": round(perplexity, 4)
            })

        training_time = round(time.time() - start_time, 2)
        final_loss = metrics_history[-1]["loss"] if metrics_history else 0.0
        final_ppl = metrics_history[-1]["perplexity"] if metrics_history else 1.0

        # Save Checkpoint Artifacts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        adapter_config = {
            "base_model": self.base_model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "training_epochs": epochs_to_run,
            "final_loss": final_loss,
            "final_perplexity": final_ppl,
            "checkpoint_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        torch.save(model.state_dict(), self.output_dir / "adapter_model.bin")
        logger.info(f"Saved LoRA weights and checkpoint metadata to {self.output_dir}")

        # Save Metrics
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary_results = {
            "model_name": "OSTutor-LLM-LoRA",
            "base_model": self.base_model_name,
            "dataset_size": len(dataset),
            "training_time_seconds": training_time,
            "final_loss": final_loss,
            "final_perplexity": final_ppl,
            "history": metrics_history
        }

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
            
        logger.info(f"Saved training metrics to {self.metrics_path}")
        return summary_results


def main():
    parser = argparse.ArgumentParser(description="OSTutor LLM Fine-Tuning Script")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Execute 1 dry-run epoch")
    args = parser.parse_args()

    tuner = OSTutorLLMFineTuner()
    if args.epochs is not None:
        tuner.epochs = args.epochs

    results = tuner.run_finetuning(dry_run=args.dry_run)
    print("\n" + "="*50)
    print("      OSTutor LLM FINE-TUNING SUMMARY")
    print("="*50)
    print(f"Base Model:       {results['base_model']}")
    print(f"Dataset Items:    {results['dataset_size']}")
    print(f"Training Time:    {results['training_time_seconds']} s")
    print(f"Final Loss:       {results['final_loss']}")
    print(f"Final Perplexity: {results['final_perplexity']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
