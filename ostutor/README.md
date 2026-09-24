# OSTutorLLM

**OSTutorLLM: A Retrieval-Augmented and Instruction-Tuned Large Language Model for Operating Systems Education**

---

## 1. Project Overview
**OSTutorLLM** is a research-oriented domain-specific AI tutor designed to assist undergraduate students in learning Operating Systems (OS) concepts, solving numerical CPU/memory problems, practicing viva questions, and debugging concurrency code.

---

## 2. Problem Statement
General-purpose Large Language Models (LLMs) often suffer from:
- Factual hallucinations when explaining complex kernel operations or hardware interactions.
- Generic explanations that lack pedagogical structure aligned with course syllabi.
- Inaccurate numerical problem solving (e.g., Gantt charts, page fault calculations, disk seek calculations).

---

## 3. Aim
To construct and evaluate a specialized AI tutoring system tailored for Operating Systems education by combining:
1. A **Retrieval-Augmented Generation (RAG) pipeline** utilizing verified OS course materials.
2. An **Instruction-Tuned Open-Source LLM** trained on OS-specific instruction-response pairs categorized by Bloom's taxonomy.
3. A **4-Way Empirical Evaluation Framework** comparing Base LLM, RAG + Base LLM, Instruction-Tuned LLM, and RAG + Instruction-Tuned LLM.

---

## 4. Phase 2 — RAG Pipeline Implementation

Phase 2 establishes the complete end-to-end RAG preprocessing, embedding, FAISS vector indexing, and semantic retrieval engine.

### Quick Start Commands

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build FAISS Vector Index:**
   Discovers raw documents in `data/rag/raw/`, extracts page/slide text, cleans, chunks, generates embeddings, and saves the FAISS index to `data/rag/index/`:
   ```bash
   python3 rag/build_index.py
   ```

3. **Test Interactive Retrieval via CLI:**
   Query top-K relevant OS chunks:
   ```bash
   python3 rag/test_retrieval.py "What is the difference between preemptive and non-preemptive scheduling?"
   ```
   Query with specific OS Unit filtering:
   ```bash
   python3 rag/test_retrieval.py --unit 3 --top-k 3 "Round Robin scheduling"
   ```

4. **Run Batch Retrieval Evaluation:**
   Evaluates Top-1, Top-3, Top-5 unit matching and keyword coverage across a 21-question benchmark suite:
   ```bash
   python3 rag/evaluate_retrieval.py
   ```

5. **Run RAG Test Suite:**
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py"
   ```

---

## 5. RAG vs. Fine-Tuning Explanation
- **Retrieval-Augmented Generation (RAG):** Supplies factual **domain knowledge and context** from verified lecture notes, lab manuals, and faculty slides at query time. It eliminates hallucinations and provides precise page/slide source citations.
- **Instruction Fine-Tuning:** Teaches the model **OS-specific instructional behavior**, step-by-step problem-solving methods, pedagogical tone, and structured response formats.
- **Combined (RAG + Fine-Tuned LLM):** Yields a reliable, accurate tutor capable of providing structured, pedagogically sound, and context-grounded answers.

---

## 6. Dataset Strategy
- **RAG Knowledge Base:** Contains verified institution-provided lecture notes, faculty slide packs, lab manuals, open documentation, and past question papers in `.pdf`, `.pptx`, `.txt`, `.docx` formats.
  > *Governance Note:* Raw copyrighted files under `data/rag/raw/` are excluded from public source control via `.gitignore`.
- **Instruction Dataset:** Formatted as JSONL records containing `unit`, `topic`, `subtopic`, `bloom_level`, `difficulty`, `task_type`, `instruction`, `input`, `output`, and `source_type`.
- **Benchmark Suite (`data/benchmark/os_benchmark.json`):** Independent test suite strictly isolated from training data to evaluate model performance across all 7 OS units.

---

## 7. Seven Operating Systems Units
The system directly maps to a 7-unit OS curriculum:
1. **Unit 1 — Introduction to Operating Systems** *(Bloom: Understand)*
2. **Unit 2 — OS Principles** *(Bloom: Understand)*
3. **Unit 3 — Scheduling** *(Bloom: Apply / Analyze)*
4. **Unit 4 — Concurrency** *(Bloom: Analyze / Apply)*
5. **Unit 5 — Memory Management** *(Bloom: Analyze / Evaluate)*
6. **Unit 6 — Virtualization and File System Management** *(Bloom: Understand / Apply / Analyze)*
7. **Unit 7 — Storage Management, Protection and Security** *(Bloom: Analyze / Evaluate)*

---

## 8. Current Development Status
- **Phase 1 Complete:** Repository structure established, 7-unit taxonomy created, dataset schemas defined, sample records added, and documentation created.
- **Phase 2 Complete:** Complete RAG ingestion, cleaning, token chunking, SentenceTransformers embedding generation, FAISS vector indexing, retrieval CLI, and batch evaluator implemented and verified.
- **Next Step:** Phase 3 (Instruction Dataset Curation for LoRA Fine-Tuning).
