# OS Instruction Dataset Methodology & Governance Document

**OSTutorLLM Phase 3: Instruction-Tuning Data Engineering**

---

## 1. Dataset Purpose & Pedagogical Goal

OSTutorLLM requires instruction tuning to adopt the behavior of an **Operating Systems tutor**. Rather than outputting short generic answers, the model is trained to:
1. Explain OS concepts at specified difficulty levels (beginner vs. advanced).
2. Step-by-step solve numerical CPU scheduling (Gantt charts, WT/TAT), virtual memory paging, and disk seek calculations.
3. Compare OS architectures (Monolithic vs. Microkernel, ULT vs. KLT, Type 1 vs. Type 2 Hypervisors).
4. Identify concurrency race conditions, deadlocks, and debug multi-threaded synchronization pseudocode.
5. Generate structured viva exam questions and MCQs with technical explanations.
6. Align responses with Bloom's Taxonomy cognitive expectations.

---

## 2. Taxonomy & Curriculum Alignment

The dataset maps directly to the 7 OS syllabus units in `data/taxonomy.json`:

| Unit | Name | Target Bloom Emphasis | Key Focus Areas |
| :--- | :--- | :--- | :--- |
| **Unit 1** | Introduction to OS | Understand | OS functionality, Monolithic/Microkernel, Layered/Modular structures, HAL, Security/Networking/Multimedia |
| **Unit 2** | OS Principles | Understand / Apply | System calls, traps, dual-mode, PCB, `fork()`/`exec()`, Thread models (1:1, M:1, M:N) |
| **Unit 3** | Scheduling | Apply / Analyze | FCFS, SJF, SRTF, Priority, Round Robin, MLFQ, Deadlock 4 conditions, RAG, Banker's algorithm |
| **Unit 4** | Concurrency | Analyze / Apply | Critical section requirements, Semaphores, Producer-Consumer, Dining Philosophers, Monitors, IPC, MCS locks |
| **Unit 5** | Memory Management | Analyze / Evaluate | Address binding, First/Best/Worst fit, Virtual memory paging, TLB, Page replacement (FIFO, LRU, OPT), Thrashing |
| **Unit 6** | Virtualization & File Systems | Understand / Apply / Analyze | Type 1/2 Hypervisors, Containers (namespaces, cgroups), Allocation methods, Journaling (WAL), LFS, DFS |
| **Unit 7** | Storage & Security | Analyze / Evaluate | Disk geometry, Disk scheduling (SSTF, SCAN, C-SCAN), Protection Access Matrix, ACLs vs Capabilities, Threats |

---

## 3. Cognitive Levels & Bloom Distribution

- **`Remember`**: Recalling definitions and OS terminology.
- **`Understand`**: Explaining concepts, trade-offs, and architectural principles.
- **`Apply`**: Executing numerical calculations, page translation math, and code tracing.
- **`Analyze`**: Comparing algorithms, evaluating concurrency hazards, and scenario debugging.
- **`Evaluate`**: Assessing deadlock safety matrices, evaluating hypervisor performance, and security risk assessment.

---

## 4. Task Types & Difficulty Levels

The dataset includes 18 distinct task categories:
`concept_explanation`, `beginner_explanation`, `advanced_explanation`, `comparison`, `mcq_generation`, `viva_generation`, `numerical_problem`, `problem_solving`, `algorithm_explanation`, `step_by_step_solution`, `programming`, `debugging`, `analysis`, `evaluation`, `scenario_analysis`, `misconception_correction`, `summarization`, `exam_question_answer`.

Difficulty classification:
- **`easy`**: Single-concept definitions and beginner explanations.
- **`medium`**: Multi-step applications, comparisons, and standard calculations.
- **`hard`**: Complex numerical matrices, multi-threaded code debugging, and evaluation of edge cases.

---

## 5. Verification & Leakage Prevention Methodology

1. **Numerical Verification:** All math outputs (Gantt chart timelines, page translation offsets, Banker's algorithm Need matrices, disk seek totals) are verified step-by-step.
2. **Duplicate & Near-Duplicate Filtering:** Exact duplicates (`instruction` + `input`) and near duplicates (Jaccard n-gram similarity $\ge 0.85$) are audited prior to final splitting.
3. **Train / Test Isolation:** `data/instruction/test.jsonl` and `data/benchmark/os_benchmark.json` are isolated from training splits to prevent data contamination.
