# OSTutorLLM Development Plan & Project Roadmap

**OSTutorLLM: A Retrieval-Augmented and Instruction-Tuned Large Language Model for Operating Systems Education**

---

## Roadmap Phases

### Phase 1: Dataset Infrastructure and Taxonomy (Current Phase)
- **Objective:** Establish workspace structure, JSON/JSONL dataset schemas, metadata catalogs, 7-unit taxonomy, and placeholder python modules.
- **Deliverables:**
  - `data/taxonomy.json` mapping Units 1–7 with Bloom levels.
  - `data/rag/metadata.csv` catalog schema.
  - `data/instruction/` sample JSONL split files (`train.jsonl`, `validation.jsonl`, `test.jsonl`).
  - `data/benchmark/os_benchmark.json` benchmark suite schema and initial samples.
  - Documentation (`README.md`, `architecture.md`, `dataset.md`, `development_plan.md`).
  - Modular python interface placeholders.

### Phase 2: RAG Preprocessing and Retrieval
- **Objective:** Build ingestion, text cleaning, chunking, embedding generation, and vector database retrieval.
- **Deliverables:**
  - PDF/PPT text extractors in `rag/ingest.py`.
  - Sentence-transformers embedding integration in `rag/embed.py`.
  - Vector similarity index setup (FAISS / Chroma) in `rag/retrieve.py`.
  - Retrieval precision & recall validation scripts.

### Phase 3: Instruction Dataset Curation
- **Objective:** Expand the OS instruction dataset to cover all 7 units, Bloom levels, and task types.
- **Deliverables:**
  - Curated, validated dataset of 1,000+ instruction-input-output records.
  - Deduplicated and verified train/val/test splits using `prepare_dataset.py`.

### Phase 4: Base Model Baseline Evaluation
- **Objective:** Evaluate zero-shot performance of un-tuned Base LLMs (Llama-3, Qwen-2.5, etc.) on `os_benchmark.json`.
- **Deliverables:**
  - Baseline execution script in `evaluation/baseline.py`.
  - Baseline metrics report (Accuracy, BLEU, ROUGE).

### Phase 5: LoRA/QLoRA Instruction Fine-Tuning
- **Objective:** Execute parameter-efficient fine-tuning (PEFT) on open-source base LLMs using instruction dataset.
- **Deliverables:**
  - QLoRA training script in `finetuning/train.py`.
  - Fine-tuned model checkpoints & adapter weights.
  - Evaluation on held-out test set via `finetuning/evaluate.py`.

### Phase 6: RAG + Fine-Tuned Model Integration
- **Objective:** Integrate retriever context injection with fine-tuned LLM inference pipeline.
- **Deliverables:**
  - Unified RAG + Fine-Tuned LLM execution interface in `backend/main.py`.

### Phase 7: Comprehensive Comparative Evaluation
- **Objective:** Conduct rigorous 4-way research evaluation comparing:
  1. Base LLM
  2. RAG + Base LLM
  3. Instruction-tuned LLM
  4. RAG + Instruction-tuned LLM
- **Deliverables:**
  - Comparative benchmark report generated via `evaluation/compare_models.py`.
  - Quantitative analysis tables for paper/thesis documentation.

### Phase 8: Minimal Web Frontend
- **Objective:** Develop a clean, interactive web user interface for student interaction.
- **Deliverables:**
  - Student query portal for conceptual Q&A, viva practice, and numerical problems.
  - Integration with backend REST endpoints.

### Phase 9: Personalization and Adaptive Learning
- **Objective:** Introduce student skill tracking and Bloom level adaptive problem recommendations.
- **Deliverables:**
  - Adaptive problem difficulty selector.
  - Student progress feedback module.

### Phase 10: Final Experiments, Thesis & Documentation
- **Objective:** Finalize experimental evaluation, error analysis, code cleanup, and project documentation.
- **Deliverables:**
  - Complete project thesis / research paper draft.
  - Finalized, reproducible codebase ready for open-source publication.

---

## Developer Responsibility Division (2 Student Developers)

| Developer | Core Responsibilities | Key Directory Focus |
| :--- | :--- | :--- |
| **Developer A (Data & RAG Lead)** | OS Taxonomy mapping, raw document ingestion, text cleaning, embedding model selection, vector DB indexing, retrieval tuning, and RAG evaluation. | `data/rag/`, `rag/`, `evaluation/evaluate_rag.py` |
| **Developer B (LLM & Fine-Tuning Lead)** | Instruction dataset curation, schema validation, QLoRA fine-tuning setup, model training execution, baseline evaluation, backend API, and 4-way comparison. | `data/instruction/`, `finetuning/`, `evaluation/baseline.py`, `backend/` |
