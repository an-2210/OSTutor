# Dataset Strategy & Governance Document

**OSTutorLLM: Operating Systems Data Engineering & Benchmark Specification**

---

## 1. RAG Knowledge Base Strategy

The RAG knowledge base serves as the authoritative, ground-truth domain knowledge repository for OSTutorLLM.

### Approved Material Types:
1. **Institution-Provided Lecture Notes:** Verified course handouts covering Units 1–7.
2. **Faculty Slide Packs:** Course presentations and diagrammatic notes.
3. **Laboratory Manuals:** POSIX C programming guides, IPC exercises, and kernel lab manuals.
4. **Legally Accessible Technical Documentation:** Official Linux kernel docs, POSIX standards, and open educational resources (OER).
5. **Previous Question Papers:** Approved institutional examination questions and sample answer keys (where permitted).

### Data Privacy & Licensing Governance:
> **IMPORTANT SECURITY & LEGAL RULE:**
> Raw document files stored under `data/rag/raw/` must **NEVER** be committed to a public Git repository unless their explicit license permits open redistribution. All proprietary faculty slides or institutional notes remain strictly local. Only `data/rag/metadata.csv` and open-license samples may be committed.

---

## 2. OS Syllabus Taxonomy (`data/taxonomy.json`)

The system categorizes all educational materials, instruction records, and benchmark items into a 7-Unit syllabus taxonomy:

- **Unit 1 — Introduction to Operating Systems** (Focus: Understand)
- **Unit 2 — OS Principles** (Focus: Understand)
- **Unit 3 — Scheduling** (Focus: Apply / Analyze)
- **Unit 4 — Concurrency** (Focus: Analyze / Apply)
- **Unit 5 — Memory Management** (Focus: Analyze / Evaluate)
- **Unit 6 — Virtualization and File System Management** (Focus: Understand / Apply / Analyze)
- **Unit 7 — Storage Management, Protection and Security** (Focus: Analyze / Evaluate)

---

## 3. Bloom's Taxonomy Alignment Strategy

To ensure pedagogical effectiveness, every instruction item and evaluation benchmark question is tagged with its cognitive level based on Bloom's Taxonomy:

| Bloom Level | Cognitive Action | Example OS Task |
| :--- | :--- | :--- |
| **Remember** | Recalling facts and definitions | "What is a Process Control Block (PCB)?" |
| **Understand** | Explaining concepts & trade-offs | "Explain why microkernels use IPC message passing." |
| **Apply** | Solving numericals & calculations | "Calculate average waiting time using Round Robin (q=4ms)." |
| **Analyze** | Comparing algorithms & edge cases | "Compare Indexed vs Linked file allocation for random access." |
| **Evaluate** | Diagnosing race conditions & safety | "Determine if a system state is safe using Banker's Algorithm." |

---

## 4. Instruction-Tuning Dataset Schema (`data/instruction/`)

Instruction dataset records are serialized in JSONL format across `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

### Allowed Task Types:
- `concept_explanation`
- `beginner_explanation`
- `comparison`
- `mcq_generation`
- `viva_generation`
- `numerical_problem`
- `problem_solving`
- `programming`
- `debugging`
- `analysis`
- `evaluation`
- `step_by_step_solution`
- `summarization`

---

## 5. Benchmark Dataset Isolation (`data/benchmark/os_benchmark.json`)

### Isolation Guidelines:
To prevent data contamination and ensure rigorous research evaluation:
- Items in `os_benchmark.json` MUST remain completely isolated from `train.jsonl` and `validation.jsonl`.
- Benchmark items test problem-solving, analytical, and conceptual capabilities across all 7 syllabus units.
- Reference answers in `os_benchmark.json` serve as gold-standard targets for comparative scoring.
