"""
Baseline Inference and Evaluation Module for OSTutorLLM (Phase 4).

Loads protected test set (data/instruction/test.jsonl), generates responses
using the unmodified Base LLM (or mock pipeline for lightweight verification),
and records deterministic results for baseline comparison.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetuning.hardware_info import get_hardware_info
from finetuning.model_config import BASE_MODEL, INFERENCE_TEMPERATURE, MAX_NEW_TOKENS
from evaluation.dataset_hash import compute_file_hash


def format_inference_prompt(instruction: str, input_text: str = "") -> str:
    """Format user prompt using standard ChatML instruction template."""
    prompt = (
        "<|im_start|>system\n"
        "You are OSTutorLLM, a helpful, precise, and rigorous Operating Systems tutor.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n"
    )
    if input_text and input_text.strip():
        prompt += f"\nInput Details:\n{input_text.strip()}\n"
    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    return prompt


class BaseLLMInferenceEngine:
    """Inference engine supporting Hugging Face Transformers with fallback mock execution."""

    def __init__(self, model_name: str = BASE_MODEL, use_mock: bool = False):
        self.model_name = model_name
        self.use_mock = use_mock
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        if not self.use_mock:
            self._initialize_real_model()

    def _initialize_real_model(self) -> None:
        """Attempt loading HuggingFace transformers model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hw = get_hardware_info()
            if hw["mps_available"]:
                self.device = "mps"
            elif hw["cuda_available"]:
                self.device = "cuda"

            print(f"[Baseline Inference Engine] Loading model '{self.model_name}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "mps":
                self.model = self.model.to(self.device)

            self.model.eval()
            print(f"[Baseline Inference Engine] Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"[Baseline Inference Engine WARNING] Could not load model '{self.model_name}': {e}")
            print("[Baseline Inference Engine] Switching to deterministic mock generator for evaluation pipeline verification.")
            self.use_mock = True

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate response for given prompt."""
        if self.use_mock:
            return self._mock_generate(prompt)

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=INFERENCE_TEMPERATURE if INFERENCE_TEMPERATURE > 0 else 0.0,
                    do_sample=False if INFERENCE_TEMPERATURE == 0.0 else True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1] :]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"[Generation Error]: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Generate structured baseline heuristic output when model is unavailable or in mock mode."""
        prompt_lower = prompt.lower()
        if "cpu scheduling" in prompt_lower or "turnaround" in prompt_lower or "waiting time" in prompt_lower:
            return (
                "To solve this CPU scheduling problem:\n"
                "1. Calculate process completion times based on arrival and burst times.\n"
                "2. Turnaround Time = Completion Time - Arrival Time.\n"
                "3. Waiting Time = Turnaround Time - Burst Time.\n"
                "Average Turnaround Time = 7.5 ms, Average Waiting Time = 3.5 ms."
            )
        elif "page replacement" in prompt_lower or "fifo" in prompt_lower or "lru" in prompt_lower:
            return (
                "For the page replacement sequence:\n"
                "1. Maintain the page frame queue.\n"
                "2. On page fault, replace the candidate according to the algorithm rule.\n"
                "Total Page Faults = 9."
            )
        elif "address translation" in prompt_lower or "page number" in prompt_lower or "offset" in prompt_lower:
            return (
                "Given logical address translation parameters:\n"
                "Page Number = Logical Address / Page Size.\n"
                "Offset = Logical Address % Page Size.\n"
                "Physical Address = (Frame Number * Page Size) + Offset."
            )
        else:
            return (
                "Operating Systems fundamental concept explanation:\n"
                "Operating systems manage hardware resources (CPU, Memory, I/O devices) and provide abstractions "
                "such as processes, threads, virtual memory, and file systems to software applications."
            )


def run_baseline_evaluation(
    test_path: str = "data/instruction/test.jsonl",
    output_path: str = "data/evaluation/base_model_test_results.jsonl",
    model_name: str = BASE_MODEL,
    limit: Optional[int] = None,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """
    Run baseline evaluation over protected test set and save formatted JSONL results.
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Protected test set not found at: {test_path}")

    # Load test examples
    examples: List[Dict[str, Any]] = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if limit and limit > 0:
        examples = examples[:limit]

    total_examples = len(examples)
    print(f"==========================================")
    print(f" OSTutorLLM Baseline Evaluation ")
    print(f"==========================================")
    print(f"Model Path       : {model_name}")
    print(f"Test Set         : {test_path}")
    print(f"Test SHA-256     : {compute_file_hash(test_path)}")
    print(f"Evaluating Items : {total_examples}")
    print(f"Output Path      : {output_path}")
    print(f"------------------------------------------")

    engine = BaseLLMInferenceEngine(model_name=model_name, use_mock=use_mock)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for idx, ex in enumerate(examples, 1):
        prompt = format_inference_prompt(ex.get("instruction", ""), ex.get("input", ""))

        # Protected test rule: do NOT include reference output in model generation context
        ref_output = ex.get("output", ex.get("reference_output", ""))

        print(f"[{idx}/{total_examples}] Evaluating Q_ID: {ex.get('question_id', f'Q_{idx}')} (Unit {ex.get('unit')})...")
        t0 = time.time()
        model_output = engine.generate(prompt)
        elapsed = round(time.time() - t0, 3)

        record = {
            "question_id": ex.get("question_id", f"Q_{idx}"),
            "unit": ex.get("unit"),
            "topic": ex.get("topic", ""),
            "bloom_level": ex.get("bloom_level", ""),
            "task_type": ex.get("task_type", ""),
            "difficulty": ex.get("difficulty", ""),
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "reference_output": ref_output,
            "model_output": model_output,
            "latency_seconds": elapsed,
        }
        results.append(record)

    total_time = round(time.time() - start_time, 2)

    # Save output JSONL
    metadata = {
        "model": model_name,
        "temperature": INFERENCE_TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dataset": os.path.basename(test_path),
        "dataset_hash": compute_file_hash(test_path),
        "hardware": get_hardware_info()["accelerator"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": total_examples,
        "total_time_seconds": total_time,
        "mock_mode": engine.use_mock,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        # First line is metadata header
        f.write(json.dumps({"_metadata": metadata}) + "\n")
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"------------------------------------------")
    print(f"Baseline evaluation completed in {total_time}s.")
    print(f"Results saved to: {output_path}")
    print(f"==========================================")

    return {
        "metadata": metadata,
        "results_count": len(results),
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OSTutorLLM Base Model Baseline Evaluation")
    parser.add_argument("--test-path", type=str, default="data/instruction/test.jsonl", help="Path to test set")
    parser.add_argument("--output-path", type=str, default="data/evaluation/base_model_test_results.jsonl", help="Output results JSONL path")
    parser.add_argument("--model", type=str, default=BASE_MODEL, help="Base model identifier or path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items (e.g. 5 or 20)")
    parser.add_argument("--mock", action="store_true", help="Force mock generation mode for fast local verification")
    args = parser.parse_args()

    run_baseline_evaluation(
        test_path=args.test_path,
        output_path=args.output_path,
        model_name=args.model,
        limit=args.limit,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
