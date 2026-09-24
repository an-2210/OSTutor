# OSTutorLLM Model Selection & Hardware Compatibility

## 1. Overview
This document presents the criteria, candidates, trade-offs, and final decision for selecting the open-weight base Large Language Model (LLM) for **OSTutorLLM Phase 4**.

---

## 2. Selection Criteria
Candidates were evaluated across ten technical and operational criteria:

1. **Open-Weight Availability & Licensing**: Must be freely available for research and education (e.g., Apache 2.0 or permissive Llama/Gemma user licenses).
2. **Model Parameter Size**: 0.5B to 3B parameters to enable responsive local inference and fine-tuning on consumer hardware (16 GB–24 GB RAM / VRAM).
3. **Memory Footprint**: Fits cleanly in memory (VRAM or unified Mac RAM) when loaded in FP16/BF16 (~1B–3B params require ~3–6 GB RAM).
4. **Apple Silicon (MPS) Compatibility**: Full support for PyTorch `mps` backend without requiring unsupported `bitsandbytes` CUDA kernels on macOS.
5. **NVIDIA CUDA Compatibility**: Seamless integration with standard PyTorch CUDA execution and optional 4-bit QLoRA on Linux/Windows workstations.
6. **LoRA / PEFT Integration**: Architecture compatible with standard Hugging Face PEFT target module projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, etc.).
7. **Instruction-Following Capability**: High base performance on structured technical reasoning, educational explanations, and numerical problem solving.
8. **Context Window Length**: Supports context length of at least 4,096 to 32,768 tokens (Phase 4 uses `MAX_SEQ_LENGTH = 1024` for instruction tuning).
9. **Tokenizer Efficiency**: High-density tokenizer capable of tokenizing technical code snippets and OS formulas efficiently.
10. **Ease of Reproducible Local Setup**: Accessible without complex gated access barriers or custom proprietary runtimes.

---

## 3. Candidates Evaluated

| Model Candidate | Parameters | License | Architecture Features | MPS (Apple Silicon) | CUDA (NVIDIA) | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-1.5B-Instruct** | 1.5B | Apache 2.0 | RoPE, SwiGLU, 32k context | Excellent (FP16) | Excellent | **SELECTED PRIMARY** |
| **Qwen/Qwen2.5-0.5B-Instruct** | 0.5B | Apache 2.0 | RoPE, SwiGLU, 32k context | Excellent (FP16) | Excellent | **SELECTED FALLBACK** |
| **meta-llama/Llama-3.2-1B-Instruct** | 1.2B | Llama 3.2 | GQA, 128k context | Good (FP16) | Good | Alternative |
| **meta-llama/Llama-3.2-3B-Instruct** | 3.2B | Llama 3.2 | GQA, 128k context | Moderate (RAM limit) | Excellent | Alternative |
| **google/gemma-2-2b-it** | 2.6B | Gemma | Sliding window, 8k context | Good (FP16) | Good | Alternative |
| **mistralai/Mistral-7B-Instruct-v0.3** | 7.3B | Apache 2.0 | Sliding window, 32k context | Heavy (RAM strain) | Requires 16GB+ VRAM | Excluded (Size limit) |

---

## 4. Hardware Constraints & Decision Rationale

### Primary Development Hardware
- **System**: Apple Silicon Mac (Apple M-series)
- **RAM**: 16 GB Unified Memory
- **Acceleration**: PyTorch `mps` (Metal Performance Shaders)

### Selected Primary Base LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- **Memory Requirements**: ~3.0 GB in FP16 precision. Fits easily within 16 GB RAM while leaving ample headroom for gradient tracking, optimizer states, and baseline evaluation.
- **Architectural Strengths**: Demonstrates state-of-the-art benchmark results for sub-3B instruction models in technical domain reasoning, code execution, and mathematical problem-solving.
- **Licensing**: Permissive Apache 2.0 license enables seamless research distribution and reproducible open-science deployment.
- **Quantization & LoRA Strategy**:
  - **Apple Silicon (MPS)**: Native FP16 LoRA adapter training via PEFT (bypasses macOS `bitsandbytes` CUDA kernel incompatibilities).
  - **NVIDIA CUDA**: Supports both standard FP16 LoRA and 4-bit QLoRA (`bitsandbytes`).

---

## 5. Limitations
1. **Domain Generalization**: As a 1.5B parameter model, baseline out-of-the-box knowledge on complex OS niche concepts (e.g., specific Linux kernel 6.x data structures) may require fine-tuning or RAG grounding.
2. **Context Window Overhead**: While context window supports up to 32k, sequence length for Phase 4 fine-tuning is capped at 1,024 tokens to optimize local training throughput.

---

## 6. Model Download & Git Policy
- Model weights must **NEVER** be committed to Git repositories.
- Local cache paths (`~/.cache/huggingface`, `models/`, `checkpoints/`) are strictly isolated in `.gitignore`.
