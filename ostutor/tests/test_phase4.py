"""
Unit Tests for Phase 4 Base Model Selection, Baseline Evaluation & Fine-Tuning Preparation.
"""

import json
import os
import tempfile
import unittest

from finetuning.hardware_info import get_hardware_info
from finetuning.model_config import BASE_MODEL, get_model_config
from finetuning.train import run_pilot_training
from evaluation.dataset_hash import compute_file_hash, get_dataset_hashes
from evaluation.baseline import format_inference_prompt, run_baseline_evaluation
from evaluation.metrics import compute_concept_coverage, compute_rouge_l, evaluate_numerical_os_problem
from evaluation.human_review_template import generate_human_review_template


class TestPhase4Implementation(unittest.TestCase):

    def test_hardware_info(self):
        """Test hardware detection utility structure."""
        hw = get_hardware_info()
        self.assertIn("platform", hw)
        self.assertIn("architecture", hw)
        self.assertIn("cpu_count", hw)
        self.assertIn("ram_gb", hw)
        self.assertIn("accelerator", hw)
        self.assertIn("mps_available", hw)
        self.assertIn("cuda_available", hw)
        self.assertGreater(hw["cpu_count"], 0)
        self.assertGreater(hw["ram_gb"], 0)

    def test_model_config(self):
        """Test centralized model configuration parameters."""
        config = get_model_config()
        self.assertEqual(config["base_model"], BASE_MODEL)
        self.assertEqual(config["max_seq_length"], 1024)
        self.assertEqual(config["lora_r"], 16)
        self.assertIn("q_proj", config["target_modules"])
        self.assertEqual(config["inference_temperature"], 0.0)

    def test_dataset_hashing(self):
        """Test dataset SHA-256 digest calculator."""
        hashes = get_dataset_hashes()
        self.assertIn("train", hashes)
        self.assertIn("validation", hashes)
        self.assertIn("test", hashes)
        self.assertEqual(len(hashes["test"]["sha256"]), 64)

    def test_inference_prompt_formatting(self):
        """Test ChatML instruction prompt formatting."""
        prompt = format_inference_prompt(
            instruction="Explain Round Robin scheduling.",
            input_text="Time quantum = 2ms"
        )
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("OSTutorLLM", prompt)
        self.assertIn("Explain Round Robin scheduling.", prompt)
        self.assertIn("Time quantum = 2ms", prompt)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_training_dataset_exclusion(self):
        """Test strict protection preventing test set from being passed to training."""
        with self.assertRaises(ValueError):
            run_pilot_training(train_path="data/instruction/test.jsonl")

    def test_metrics_calculation(self):
        """Test automatic metrics and numerical OS problem checker."""
        ref = "Average waiting time is 5.5 ms and total page faults is 9."
        hyp = "The waiting time average is 5.5 ms with total page faults equal to 9."

        rouge = compute_rouge_l(ref, hyp)
        self.assertGreater(rouge, 0.4)

        cov = compute_concept_coverage(ref, hyp)
        self.assertGreater(cov, 0.5)

        num_eval = evaluate_numerical_os_problem("cpu_scheduling", ref, hyp)
        self.assertTrue(num_eval["numerical_applicable"])
        self.assertTrue(num_eval["is_correct"])

    def test_human_review_template_schema(self):
        """Test human review template generator schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_results_path = os.path.join(tmpdir, "results.jsonl")
            output_template_path = os.path.join(tmpdir, "human_review.jsonl")

            with open(test_results_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"_metadata": {"model": "test"}}) + "\n")
                rec = {
                    "question_id": "TEST_Q1",
                    "unit": 3,
                    "topic": "Scheduling",
                    "bloom_level": "Apply",
                    "task_type": "numerical_problem",
                    "difficulty": "medium",
                    "instruction": "Calculate SJF",
                    "input": "",
                    "reference_output": "Answer 10",
                    "model_output": "Answer 10",
                }
                f.write(json.dumps(rec) + "\n")

            generate_human_review_template(
                results_path=test_results_path,
                output_path=output_template_path,
                sample_size=1,
            )

            self.assertTrue(os.path.exists(output_template_path))
            with open(output_template_path, "r", encoding="utf-8") as f:
                data = json.loads(f.readline())
                self.assertEqual(data["question_id"], "TEST_Q1")
                self.assertIn("human_evaluation", data)
                self.assertIsNone(data["human_evaluation"]["correctness"])

    def test_experiment_configs(self):
        """Test validity of machine-readable experiment JSON files."""
        baseline_cfg_path = "experiments/baseline_config.json"
        pilot_cfg_path = "experiments/pilot_lora_config.json"

        self.assertTrue(os.path.exists(baseline_cfg_path))
        self.assertTrue(os.path.exists(pilot_cfg_path))

        with open(baseline_cfg_path, "r", encoding="utf-8") as f:
            cfg1 = json.load(f)
            self.assertEqual(cfg1["experiment_id"], "EXP-001-BASELINE")

        with open(pilot_cfg_path, "r", encoding="utf-8") as f:
            cfg2 = json.load(f)
            self.assertEqual(cfg2["experiment_id"], "EXP-002-PILOT-LORA")
            self.assertTrue(cfg2["datasets"]["protected_test_excluded"])


if __name__ == "__main__":
    unittest.main()
