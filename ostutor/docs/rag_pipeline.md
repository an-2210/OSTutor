# OSTutorLLM RAG Pipeline Specification & Documentation

**OSTutorLLM Phase 2: Retrieval-Augmented Generation Architecture**

---

## 1. Overview

The **OSTutorLLM RAG Pipeline** transforms raw Operating Systems course materials (PDF, PPTX, TXT, DOCX) into a metadata-indexed vector search system. It provides accurate, ground-truth context to eliminate LLM hallucinations and support student problem-solving across all 7 OS syllabus units.

```
OS Raw Documents (.pdf, .pptx, .txt, .docx)
                    ↓
Document Discovery (`rag/ingest.py`)
                    ↓
Text Extraction & Metadata Attachment (`metadata.csv` validation)
                    ↓
Text Cleaning & Normalization (`rag/clean.py`)
                    ↓
Token Chunking & JSONL Persistence (`rag/chunk.py` -> `chunks.jsonl`)
                    ↓
SentenceTransformers Embedding Generation (`rag/embed.py`)
                    ↓
FAISS IndexFlatIP Vector Construction (`rag/retrieve.py` -> `faiss.index`)
                    ↓
Semantic Vector Retrieval API & CLI (`rag/test_retrieval.py`)
```

---

## 2. Supported Formats & Extraction Strategy

- **PDF Documents (`.pdf`):** Extracted page-by-page using `pypdf`. 1-indexed page numbers are preserved.
- **PowerPoint Presentations (`.pptx`):** Extracted slide-by-slide using `python-pptx`. 1-indexed slide numbers are preserved.
- **Plain Text Files (`.txt`):** Extracted as full text records.
- **Word Documents (`.docx`):** Extracted paragraph by paragraph using `python-docx`.

---

## 3. Metadata Structure & Taxonomy Alignment

Every extracted document page/slide and generated chunk carries structured metadata validated against `data/taxonomy.json` and `data/rag/metadata.csv`:

```json
{
  "chunk_id": "sample_u3_scheduling_p001_chunk01",
  "document_id": "sample_u3_scheduling",
  "filename": "sample_unit3_scheduling.txt",
  "unit": 3,
  "topic": "CPU Scheduling",
  "subtopic": "Round Robin",
  "source_type": "faculty_material",
  "source": "Course Lecture Notes",
  "license_or_access": "Open Educational License",
  "page": 1,
  "slide": null,
  "estimated_tokens": 124,
  "text": "CPU scheduling algorithms decide which process..."
}
```

---

## 4. Text Cleaning Rules (`rag/clean.py`)

1. **Hyphenated Line Splits:** Rejoins words broken across lines (e.g., `schedul-\ning` -> `scheduling`).
2. **Header/Footer Stripping:** Strips repetitive slide/page counts (`Page 12 of 30`), lone line numbers, and confidentiality footers.
3. **Whitespace Normalization:** Collapses tabs and multiple spaces while preserving paragraph breaks (`\n\n`) and markdown headings (`#`).

---

## 5. Chunking Strategy (`rag/chunk.py`)

- **Target Chunk Size:** ~700 tokens (~500 words).
- **Target Overlap Size:** ~100 tokens (~75 words).
- **Chunk ID Generation:** Deterministic ID format `{doc_id}_p{page}_chunk{idx:02d}`.
- **Artifact Persistence:** Processed chunks are saved to `data/rag/processed/chunks.jsonl`.

---

## 6. Embedding Model & Vector Index

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (configurable via `EMBEDDING_MODEL` environment variable).
- **Embedding Dimension:** 384 dimensions.
- **Hardware Acceleration:** Automatic detection of Apple Silicon (`mps`), CUDA, or CPU fallback.
- **Normalization:** Vectors are L2-normalized upon encoding (`||v|| = 1`).
- **FAISS Index Type:** `faiss.IndexFlatIP` (Inner Product over normalized vectors = exact Cosine Similarity).
- **Index Artifacts:** Saved to `data/rag/index/faiss.index` and `data/rag/index/chunks.json`.

---

## 7. Pipeline Execution Commands

### Build RAG Index
Executes discovery, extraction, cleaning, chunking, embedding, and FAISS index construction:
```bash
python3 rag/build_index.py
```

### Test Semantic Retrieval CLI
Query the index interactively:
```bash
python3 rag/test_retrieval.py "What is the difference between preemptive and non-preemptive scheduling?"
```

Query with unit filtering:
```bash
python3 rag/test_retrieval.py --unit 3 --top-k 3 "Round Robin time quantum"
```

### Run Batch Retrieval Evaluation
Evaluates 21 queries across all 7 OS units:
```bash
python3 rag/evaluate_retrieval.py
```

### Run RAG Unit Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 8. Role of RAG vs. Fine-Tuning

- **RAG Role:** Provides dynamic, verifiable external OS knowledge and context at query time to prevent hallucinations and supply exact slide/page source citations.
- **Fine-Tuning Role (Phases 3-5):** Teaches the open-source LLM instructional behavior, step-by-step tutoring tone, and pedagogical problem-solving formats.
