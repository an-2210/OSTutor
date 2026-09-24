# System Architecture: OSTutorLLM

**OSTutorLLM: A Retrieval-Augmented and Instruction-Tuned Large Language Model for Operating Systems Education**

---

## 1. Overview & High-Level System Architecture

OSTutorLLM combines **Retrieval-Augmented Generation (RAG)** with **Instruction Fine-Tuning** to deliver an accurate, non-hallucinating, and pedagogically sound AI tutor for Operating Systems education.

### Interactive Inference Architecture

```mermaid
flowchart TD
    Student[Student User] -->|Asks Question / Practice Problem| Frontend[Frontend Web UI]
    Frontend -->|HTTP API Request| Backend[Backend / API Service]
    Backend --> QueryProc[Query Processor]
    
    subgraph RAG Retrieval Pipeline (Phase 2 Implemented)
        QueryProc -->|Query String| Retriever[Vector Retriever]
        Retriever -->|Embed Query| Embedder[SentenceTransformers]
        Embedder -->|Vector Search| FAISS[(FAISS Vector Index)]
        FAISS -->|Metadata Lookup| ChunkStore[JSONL Chunk Metadata Store]
        ChunkStore -->|Ranked Chunks| Context[Relevant OS Context]
    end
    
    QueryProc -->|Prompt + OS Context| Model[Instruction-Tuned Open-Source LLM]
    Context -->|Augment Prompt| Model
    Model -->|Structured Tutor Response + Citations| Backend
    Backend -->|JSON Payload| Frontend
    Frontend -->|Rendered Answer & Guidance| Student
```

---

## 2. Off-Line Model Training & RAG Indexing Pipeline

```mermaid
flowchart TD
    subgraph RAG Preprocessing Pipeline (Phase 2)
        RawDocs[OS Documents: PDF, PPTX, TXT, DOCX] -->|ingest.py| ExtractedPages[Page/Slide Extractor]
        ExtractedPages -->|clean.py| CleanedText[Normalized Text]
        CleanedText -->|chunk.py| Chunks[Chunks JSONL Store]
        Chunks -->|embed.py| Embeddings[Dense Vector Matrix]
        Embeddings -->|retrieve.py| FAISSIndex[(FAISS IndexFlatIP)]
    end
    
    subgraph Fine-Tuning Pipeline (Phases 3-5)
        Taxonomy[Syllabus Taxonomy] --> InstructGen[Instruction Dataset]
        InstructGen -->|prepare_dataset.py| Validate[Data Validation]
        Validate --> Split[Train/Val/Test Split]
        Split --> QLoRA[LoRA / QLoRA Training]
        QLoRA --> InstructModel[Instruction-Tuned Model]
    end
```

---

## 3. Implementation Status & Subsystem Architecture

### Completed Subsystems:
- **Config & Core Settings (`config.py`):** Centralized parameters for chunking (~700 tokens, ~100 overlap), embedding model (`sentence-transformers/all-MiniLM-L6-v2`), and paths.
- **Document Ingestion (`rag/ingest.py`):** PDF (`pypdf`), PPTX (`python-pptx`), TXT, DOCX (`python-docx`) page/slide level text extraction and catalog validation.
- **Text Cleaning (`rag/clean.py`):** Rejoins hyphenated line breaks, removes headers/footers, normalizes whitespace, preserves section headings (`#`).
- **Chunking Engine (`rag/chunk.py`):** Generates deterministic chunk IDs (`doc_p001_chunk01`), preserves metadata, saves to `data/rag/processed/chunks.jsonl`.
- **Dense Embedding Generator (`rag/embed.py`):** SentenceTransformers embedding engine with L2 normalization, supporting Apple Silicon (MPS), CUDA, and CPU.
- **FAISS Vector Store & Retriever (`rag/retrieve.py`):** Cosine similarity search using `faiss.IndexFlatIP` with metadata persistence (`faiss.index`, `chunks.json`) and unit/topic metadata filtering.
- **Index Build Orchestration (`rag/build_index.py`):** Automated pipeline builder.
- **Interactive CLI Tester (`rag/test_retrieval.py`):** Command-line retrieval query runner.
- **Batch Evaluator (`rag/evaluate_retrieval.py`):** 21-question evaluation suite covering all 7 OS units.
