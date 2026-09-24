"""
Batch Retrieval Evaluation Module for OSTutorLLM RAG Pipeline.

Evaluates retrieval quality over a 21-question test set covering all 7 OS syllabus units.
Computes Top-1, Top-3, Top-5 Unit Match Accuracy and Expected Keyword Match Rates.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import INDEX_DIR
from rag.retrieve import load_index, retrieve

logger = logging.getLogger(__name__)

# 21-Question Benchmark Suite across all 7 OS Syllabus Units (3 questions per unit)
EVALUATION_QUESTIONS: List[Dict[str, Any]] = [
    # Unit 1 — Introduction to Operating Systems
    {
        "id": "eval_u1_01",
        "unit": 1,
        "topic": "Functionality of Operating Systems",
        "question": "What are the primary responsibilities and core functions of an Operating System?",
        "expected_keywords": ["resource", "hardware", "abstraction", "interface"],
    },
    {
        "id": "eval_u1_02",
        "unit": 1,
        "topic": "Microkernel Models",
        "question": "How does a microkernel operating system architecture differ from a monolithic kernel?",
        "expected_keywords": ["microkernel", "monolithic", "user space", "ipc"],
    },
    {
        "id": "eval_u1_03",
        "unit": 1,
        "topic": "Layered Systems",
        "question": "Explain the advantages and design principles of a layered operating system structure.",
        "expected_keywords": ["layer", "modular", "interface", "abstraction"],
    },

    # Unit 2 — OS Principles
    {
        "id": "eval_u2_01",
        "unit": 2,
        "topic": "System Calls",
        "question": "Explain how a system call trap mechanism transitions execution from user mode to kernel mode.",
        "expected_keywords": ["system call", "trap", "kernel mode", "user mode"],
    },
    {
        "id": "eval_u2_02",
        "unit": 2,
        "topic": "Process Control Block",
        "question": "What information is stored inside a Process Control Block (PCB)?",
        "expected_keywords": ["pcb", "process state", "register", "program counter"],
    },
    {
        "id": "eval_u2_03",
        "unit": 2,
        "topic": "Process Creation",
        "question": "Describe process creation in Unix using fork() and exec() system calls.",
        "expected_keywords": ["fork", "exec", "child", "process"],
    },

    # Unit 3 — Scheduling
    {
        "id": "eval_u3_01",
        "unit": 3,
        "topic": "Round Robin",
        "question": "How does Round Robin CPU scheduling handle time quantum allocation and preemption?",
        "expected_keywords": ["round robin", "time quantum", "preemptive", "slice"],
    },
    {
        "id": "eval_u3_02",
        "unit": 3,
        "topic": "Preemptive Scheduling",
        "question": "What is the difference between preemptive and non-preemptive CPU scheduling?",
        "expected_keywords": ["preemptive", "non-preemptive", "cpu", "burst"],
    },
    {
        "id": "eval_u3_03",
        "unit": 3,
        "topic": "Deadlocks",
        "question": "What are the four necessary conditions for a deadlock to occur in an operating system?",
        "expected_keywords": ["deadlock", "mutual exclusion", "hold and wait", "circular wait"],
    },

    # Unit 4 — Concurrency
    {
        "id": "eval_u4_01",
        "unit": 4,
        "topic": "Semaphores",
        "question": "Explain counting semaphores and binary semaphores and their wait and signal operations.",
        "expected_keywords": ["semaphore", "wait", "signal", "mutex"],
    },
    {
        "id": "eval_u4_02",
        "unit": 4,
        "topic": "Classical Synchronization Problems",
        "question": "Describe the Bounded Buffer producer consumer synchronization problem.",
        "expected_keywords": ["producer", "consumer", "buffer", "semaphore"],
    },
    {
        "id": "eval_u4_03",
        "unit": 4,
        "topic": "Dining Philosophers",
        "question": "Why does the naive Dining Philosophers implementation deadlock and how can it be solved?",
        "expected_keywords": ["philosopher", "chopstick", "deadlock", "starvation"],
    },

    # Unit 5 — Memory Management
    {
        "id": "eval_u5_01",
        "unit": 5,
        "topic": "Paging",
        "question": "Explain how address translation works in a paged virtual memory architecture.",
        "expected_keywords": ["paging", "page table", "offset", "address translation"],
    },
    {
        "id": "eval_u5_02",
        "unit": 5,
        "topic": "Page Replacement",
        "question": "Compare FIFO and LRU page replacement algorithms in virtual memory.",
        "expected_keywords": ["page replacement", "fifo", "lru", "fault"],
    },
    {
        "id": "eval_u5_03",
        "unit": 5,
        "topic": "Thrashing",
        "question": "What causes thrashing in virtual memory and how does the working set model prevent it?",
        "expected_keywords": ["thrashing", "working set", "page fault", "locality"],
    },

    # Unit 6 — Virtualization and File System Management
    {
        "id": "eval_u6_01",
        "unit": 6,
        "topic": "Hypervisors",
        "question": "What is the difference between Type 1 bare metal and Type 2 hosted hypervisors?",
        "expected_keywords": ["hypervisor", "type 1", "type 2", "virtual machine"],
    },
    {
        "id": "eval_u6_02",
        "unit": 6,
        "topic": "Container Virtualization",
        "question": "How do Linux namespaces and cgroups enable container virtualization?",
        "expected_keywords": ["container", "namespaces", "cgroups", "isolation"],
    },
    {
        "id": "eval_u6_03",
        "unit": 6,
        "topic": "Journaling",
        "question": "How does write ahead logging and journaling protect file systems against crash corruption?",
        "expected_keywords": ["journaling", "write ahead", "file system", "recovery"],
    },

    # Unit 7 — Storage Management, Protection and Security
    {
        "id": "eval_u7_01",
        "unit": 7,
        "topic": "Disk Scheduling Algorithms",
        "question": "Compare SSTF, SCAN, and C-SCAN disk scheduling algorithms.",
        "expected_keywords": ["disk scheduling", "sstf", "scan", "c-scan"],
    },
    {
        "id": "eval_u7_02",
        "unit": 7,
        "topic": "Access Matrix",
        "question": "Explain how an access matrix represents protection domains, objects, and rights.",
        "expected_keywords": ["access matrix", "domain", "protection", "rights"],
    },
    {
        "id": "eval_u7_03",
        "unit": 7,
        "topic": "System Threats and Security",
        "question": "Describe buffer overflow attacks and malware threat vectors in operating systems.",
        "expected_keywords": ["buffer overflow", "malware", "virus", "security"],
    },
]


def run_retrieval_evaluation(index_dir: Path = INDEX_DIR, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate retrieval precision, unit matching rates, and keyword coverage.

    Returns:
        Summary statistics dictionary.
    """
    print("\nOSTutorLLM Batch Retrieval Evaluator")
    print("====================================")
    print(f"Index Directory: {index_dir}")
    print(f"Total Queries  : {len(EVALUATION_QUESTIONS)}\n")

    try:
        vstore = load_index(index_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] Cannot run retrieval evaluation: {e}")
        print("Build index first using `python3 rag/build_index.py`.")
        sys.exit(1)

    top1_unit_hits = 0
    top3_unit_hits = 0
    top5_unit_hits = 0

    total_keyword_matches = 0
    total_keywords_checked = 0

    for idx, qitem in enumerate(EVALUATION_QUESTIONS, start=1):
        q_text = qitem["question"]
        expected_unit = qitem["unit"]
        expected_kws = qitem.get("expected_keywords", [])

        results = retrieve(query=q_text, top_k=top_k, index_dir=index_dir, vector_store=vstore)

        retrieved_units = [r["unit"] for r in results]

        top1_hit = len(retrieved_units) >= 1 and retrieved_units[0] == expected_unit
        top3_hit = expected_unit in retrieved_units[:3]
        top5_hit = expected_unit in retrieved_units[:5]

        if top1_hit:
            top1_unit_hits += 1
        if top3_hit:
            top3_unit_hits += 1
        if top5_hit:
            top5_unit_hits += 1

        # Combine text of retrieved top-K chunks for keyword checking
        combined_text = " ".join([r["text"].lower() for r in results])

        kw_matches = sum(1 for kw in expected_kws if kw.lower() in combined_text)
        total_keyword_matches += kw_matches
        total_keywords_checked += len(expected_kws)

        status_symbol = "✔" if top3_hit else "✘"
        top1_src = results[0]['source'] if results else 'None'
        print(f"[{idx:02d}/{len(EVALUATION_QUESTIONS)}] {status_symbol} Unit {expected_unit} | Top-1 Src: {top1_src} | Keyword Match: {kw_matches}/{len(expected_kws)}")

    total_q = len(EVALUATION_QUESTIONS)
    top1_acc = (top1_unit_hits / total_q) * 100
    top3_acc = (top3_unit_hits / total_q) * 100
    top5_acc = (top5_unit_hits / total_q) * 100
    kw_acc = (total_keyword_matches / total_keywords_checked * 100) if total_keywords_checked > 0 else 0.0

    report = {
        "total_queries": total_q,
        "top1_unit_match_pct": round(top1_acc, 2),
        "top3_unit_match_pct": round(top3_acc, 2),
        "top5_unit_match_pct": round(top5_acc, 2),
        "keyword_coverage_pct": round(kw_acc, 2),
        "disclaimer": "These metrics are baseline retrieval indicators, not final semantic evaluation metrics.",
    }

    print("\nRetrieval Evaluation Report")
    print("===========================")
    print(f"Total Queries Evaluated : {total_q}")
    print(f"Top-1 Unit Match Rate   : {report['top1_unit_match_pct']}%")
    print(f"Top-3 Unit Match Rate   : {report['top3_unit_match_pct']}%")
    print(f"Top-5 Unit Match Rate   : {report['top5_unit_match_pct']}%")
    print(f"Keyword Coverage Rate   : {report['keyword_coverage_pct']}%")
    print(f"Note: {report['disclaimer']}\n")

    return report


if __name__ == "__main__":
    run_retrieval_evaluation()
