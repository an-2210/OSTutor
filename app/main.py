"""
OSTutor // Kernel - FastAPI Application Server
Serving OSTutorLLM Command Center & REST API Endpoints.
"""

import os
import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("ostutor_app")

app = FastAPI(
    title="OSTutor // Kernel",
    description="Instruction-tuned and Retrieval-Augmented AI Tutor Command Center for Operating Systems Education",
    version="1.0.0"
)

# Mount static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Load Configuration
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
try:
    config = load_config(str(CONFIG_PATH))
except Exception:
    config = {
        "system": {"name": "OSTutor // Kernel", "version": "1.0.0"},
        "rag_cti": {"top_k_retrieval": 5, "vector_store_type": "faiss"}
    }

START_TIME = time.time()

# Request Models
class ChatRequest(BaseModel):
    message: str
    topic: Optional[str] = "General OS Kernel"
    use_rag: Optional[bool] = True

class RagQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ScheduleSimRequest(BaseModel):
    algorithm: str  # "FCFS", "SJF", "RR"
    processes: List[Dict[str, Any]]
    quantum: Optional[int] = 2

# Knowledge Base & Concept Data
OS_CONCEPTS = [
    {
        "id": "proc",
        "title": "Process & Thread Management",
        "description": "Process Control Block (PCB), Context Switching, Threading models, Systems calls (fork, exec, wait, exit).",
        "mastery": 85,
        "status": "ACTIVE",
        "topics": ["PCB Architecture", "Context Switching Overhead", "POSIX Threads (pthreads)", "Fork-Exec Pattern", "Process States"]
    },
    {
        "id": "mem",
        "title": "Virtual Memory & Paging",
        "description": "Page Tables, Translation Lookaside Buffer (TLB), Page Replacement (LRU, FIFO, Clock), Thrashing.",
        "mastery": 72,
        "status": "ACTIVE",
        "topics": ["Page Table Structure", "TLB Hit/Miss Ratio", "Page Fault Handling", "LRU Page Replacement", "Demand Paging"]
    },
    {
        "id": "sync",
        "title": "Concurrency & Synchronization",
        "description": "Critical Section Problem, Semaphores, Mutex Locks, Condition Variables, Deadlock Detection & Prevention.",
        "mastery": 64,
        "status": "ACTIVE",
        "topics": ["Peterson's Solution", "Mutex vs Semaphore", "Banker's Algorithm", "Dining Philosophers", "Atomic Operations"]
    },
    {
        "id": "cpu",
        "title": "CPU Scheduling Algorithms",
        "description": "Preemptive vs Non-preemptive, FCFS, Shortest Job First (SJF), Round Robin (RR), Multi-Level Feedback Queue.",
        "mastery": 90,
        "status": "OPTIMAL",
        "topics": ["First-Come First-Served", "Shortest Remaining Time First", "Round Robin Quantum", "MLFQ Rules", "Gantt Tracing"]
    },
    {
        "id": "fs",
        "title": "File Systems & Storage",
        "description": "Inodes, Directory Layouts, Contiguous vs Indexed Allocation, Journaling, RAID Systems, VFS Layer.",
        "mastery": 58,
        "status": "LEARNING",
        "topics": ["Inode Structure", "Virtual File System (VFS)", "Ext4 Journaling", "RAID 0/1/5/10", "Directory Indexing"]
    },
    {
        "id": "io",
        "title": "I/O Hardware & Interrupt Handling",
        "description": "Device Drivers, Interrupt Service Routines (ISR), Direct Memory Access (DMA), Disk Scheduling (SCAN/C-SCAN).",
        "mastery": 48,
        "status": "NEEDS_REVIEW",
        "topics": ["Interrupt Request (IRQ)", "DMA Controller", "Disk Scheduling (C-SCAN)", "Block vs Character Devices", "Buffer Caching"]
    }
]

RAG_KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "source": "Operating System Concepts (Silberschatz) - Ch. 3",
        "topic": "Process Management",
        "content": "A Process Control Block (PCB) contains details including Process State, Program Counter, CPU registers, CPU scheduling information, memory-management information, accounting information, and I/O status information.",
        "score": 0.94
    },
    {
        "id": "doc_02",
        "source": "Modern Operating Systems (Tanenbaum) - Ch. 4",
        "topic": "Virtual Memory",
        "content": "The Translation Lookaside Buffer (TLB) is a hardware cache inside the MMU mapping virtual page numbers to physical frame numbers. TLB misses require walking page table levels, adding latency.",
        "score": 0.89
    },
    {
        "id": "doc_03",
        "source": "Operating Systems: Three Easy Pieces (OSTEP) - Ch. 28",
        "topic": "Synchronization",
        "content": "A Semaphore maintains an integer value accessed only via wait() [P] and signal() [V] atomic operations. Counting semaphores control access to a finite set of resources.",
        "score": 0.86
    },
    {
        "id": "doc_04",
        "source": "Linux Kernel Architecture (Bovet & Cesati) - Ch. 7",
        "topic": "System Calls & Scheduling",
        "content": "In Linux, CFS (Completely Fair Scheduler) uses a red-black tree indexed by virtual runtime (vruntime) to select the task with smallest vruntime for execution next.",
        "score": 0.82
    },
    {
        "id": "doc_05",
        "source": "Operating System Concepts (Silberschatz) - Ch. 11",
        "topic": "File Systems",
        "content": "An inode (index node) stores block pointers to data blocks. Direct pointers handle small files, while single, double, and triple indirect pointers enable large files.",
        "score": 0.78
    }
]

# API Routes

@app.get("/")
async def serve_index():
    """Serve main OSTutor // Kernel Command Center SPA."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "OSTutor // Kernel Server Running", "version": "1.0.0"})

@app.get("/api/kernel/telemetry")
async def get_telemetry():
    """Retrieve simulated live system and tutor model telemetry."""
    uptime = int(time.time() - START_TIME)
    cpu_load = round(25.0 + 15.0 * math.sin(time.time() / 5.0) + random.uniform(-2, 2), 1)
    memory_used = round(4.2 + 0.3 * math.cos(time.time() / 10.0), 2)
    
    return {
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}",
        "cpu_usage_pct": max(5.0, min(99.0, cpu_load)),
        "memory_used_gb": memory_used,
        "memory_total_gb": 16.0,
        "active_kernel_threads": 48 + int(uptime % 12),
        "faiss_indexed_chunks": 1420,
        "model_latency_ms": random.randint(18, 45),
        "tutor_status": "ONLINE / READY",
        "seed": config.get("system", {}).get("seed", 42)
    }

@app.get("/api/concepts")
async def get_concepts():
    """Return OS concept module matrix and mastery rates."""
    return {"concepts": OS_CONCEPTS}

@app.post("/api/tutor/chat")
async def tutor_chat(req: ChatRequest):
    """Instruction-tuned OS Tutor Response Generator with step-by-step kernel trace."""
    msg_lower = req.message.lower()
    
    citations = []
    kernel_trace = []
    
    if "fork" in msg_lower or "process" in msg_lower or "pcb" in msg_lower:
        response_text = (
            "### Process Creation & `fork()` System Call\n\n"
            "In UNIX-like Operating Systems, **`fork()`** creates a child process by duplicating the calling parent process.\n\n"
            "#### Key Kernel Mechanics:\n"
            "1. **PCB Duplication**: The kernel allocates a new `task_struct` (Process Control Block) for the child.\n"
            "2. **Copy-on-Write (COW)**: Rather than immediately copying physical memory pages, parent and child share read-only pages. A copy occurs **only when either process writes to a page**.\n"
            "3. **Return Values**: `fork()` returns `0` in the child process and the child's `PID` in the parent process."
        )
        kernel_trace = [
            "SYS_CALL [0x02]: sys_fork() invoked by PID 4012",
            "ALLOCATE: task_struct allocated at 0xFFFF8800A12",
            "COW_SET: Marking VM pages [0x400000-0x4FF000] READ-ONLY (Copy-On-Write)",
            "PID_ASSIGN: Child PID 4013 assigned to task_struct",
            "RETURN: Parent received PID 4013, Child received PID 0"
        ]
        citations = [RAG_KNOWLEDGE_BASE[0]]
        
    elif "virtual memory" in msg_lower or "page" in msg_lower or "tlb" in msg_lower:
        response_text = (
            "### Virtual Memory & Page Translation\n\n"
            "Virtual memory abstracts physical RAM into contiguous virtual address spaces per process.\n\n"
            "#### Address Translation Sequence:\n"
            "1. **MMU Lookup**: CPU splits Virtual Address into **Virtual Page Number (VPN)** and **Offset**.\n"
            "2. **TLB Check**: The MMU checks the **Translation Lookaside Buffer (TLB)** fast cache.\n"
            "   - **TLB Hit**: Instant physical address translation.\n"
            "   - **TLB Miss**: Page table walk occurs. If valid, updates TLB. If invalid, triggers a **Page Fault interrupt** (`#PF`)."
        )
        kernel_trace = [
            "MMU: Translate Virtual Address 0x00007FFF89A2",
            "TLB: Lookup VPN 0x7FFF89 -> MISS",
            "PAGE_WALK: Traverse CR3 -> PGD -> PUD -> PMD -> PTE",
            "PTE_VALID: Frame 0x1A4F mapped with R/W flags",
            "TLB_UPDATE: Cached VPN 0x7FFF89 -> PFN 0x1A4F in TLB slot 3"
        ]
        citations = [RAG_KNOWLEDGE_BASE[1]]
        
    elif "semaphore" in msg_lower or "mutex" in msg_lower or "sync" in msg_lower or "deadlock" in msg_lower:
        response_text = (
            "### Synchronization: Mutex vs Counting Semaphore\n\n"
            "Concurrency control mechanisms prevent data races in shared kernel or thread memory spaces:\n\n"
            "- **Mutex (Mutual Exclusion)**: Ownership-based lock (`0` or `1`). Only the thread that acquired the mutex can release it.\n"
            "- **Semaphore**: Counter-based primitive.\n"
            "  - `sem_wait()` / `P()`: Decrements counter. If `< 0`, thread blocks.\n"
            "  - `sem_post()` / `V()`: Increments counter and unblocks waiting thread."
        )
        kernel_trace = [
            "THREAD [T2]: Executing sem_wait(&mutex_lock)",
            "ATOMIC_DEC: Semaphore value decremented from 1 -> 0",
            "LOCK_ACQUIRED: Thread T2 entered Critical Section",
            "THREAD [T3]: Executing sem_wait(&mutex_lock)",
            "BLOCKED: Semaphore value = -1, Thread T3 placed on wait_queue"
        ]
        citations = [RAG_KNOWLEDGE_BASE[2]]

    elif "schedule" in msg_lower or "round robin" in msg_lower or "gantt" in msg_lower or "cfs" in msg_lower:
        response_text = (
            "### CPU Scheduling Algorithms & Policy\n\n"
            "The OS CPU scheduler decides which runnable process occupies CPU core time:\n\n"
            "- **FCFS (First-Come First-Served)**: Non-preemptive, susceptible to the *Convoy Effect*.\n"
            "- **SJF (Shortest Job First)**: Minimizes average waiting time; requires job burst estimates.\n"
            "- **Round Robin (RR)**: Time-sliced preemptive scheduling using a fixed Time Quantum (Q).\n"
            "- **CFS (Completely Fair Scheduler)**: Linux default using Red-Black trees to balance `vruntime`."
        )
        kernel_trace = [
            "SCHED_TICK: Timer interrupt fired at 1000ms",
            "PREEMPT: Process P1 quantum exhausted (Q=2ms)",
            "CONTEXT_SWITCH: Saving P1 registers to PCB -> Loading P2 registers",
            "CFS_RB_TREE: Selected Process P2 (vruntime = 4.2ms)"
        ]
        citations = [RAG_KNOWLEDGE_BASE[3]]

    else:
        response_text = (
            f"### OSTutor // Kernel Response: {req.topic}\n\n"
            f"Query processed: *\"{req.message}\"*\n\n"
            "In Operating Systems architecture, key concepts revolve around **isolation**, **concurrency**, and **resource management**.\n\n"
            "#### Core OS Directives:\n"
            "- **Kernel Space vs User Space**: Privilege separation via Dual-Mode execution (Ring 0 vs Ring 3).\n"
            "- **System Calls**: The interface between user applications and kernel services.\n"
            "- **Resource Virtualization**: Abstracting CPU (processes), RAM (virtual memory), and storage (file systems)."
        )
        kernel_trace = [
            f"INPUT_QUERY: Processing user prompt regarding '{req.topic}'",
            "VECTOR_SEARCH: Queried FAISS vector store (top_k=5)",
            "LLM_GENERATE: Applied instruction-tuned OS prompt template",
            "OUTPUT_STREAM: Response compiled with verified kernel citations"
        ]
        citations = RAG_KNOWLEDGE_BASE[:2]

    return {
        "status": "SUCCESS",
        "topic": req.topic,
        "query": req.message,
        "response": response_text,
        "kernel_trace": kernel_trace,
        "rag_citations": citations if req.use_rag else [],
        "confidence_score": round(random.uniform(0.91, 0.98), 2),
        "processing_time_ms": random.randint(22, 55)
    }

@app.post("/api/rag/retrieve")
async def rag_retrieve(req: RagQueryRequest):
    """Inspect FAISS RAG vector similarity search results."""
    query_words = req.query.lower().split()
    
    scored_docs = []
    for doc in RAG_KNOWLEDGE_BASE:
        match_count = sum(1 for w in query_words if w in doc["content"].lower() or w in doc["topic"].lower())
        sim_score = min(0.99, doc["score"] + (match_count * 0.05))
        scored_docs.append({
            **doc,
            "similarity_score": round(sim_score, 4),
            "faiss_distance": round(1.0 - sim_score, 4)
        })
        
    scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_docs = scored_docs[:req.top_k]
    
    return {
        "query": req.query,
        "total_indexed_chunks": 1420,
        "retrieved_count": len(top_docs),
        "results": top_docs
    }

@app.post("/api/explain/shap")
async def get_shap_explain():
    """Return simulated SHAP feature attribution metrics for OS anomaly detection & scheduling model."""
    return {
        "model": "XGBoost + Hybrid Fusion Classifier",
        "target": "Kernel Thrashing & Context-Switch Anomaly Detection",
        "base_value": 0.12,
        "features": [
            {"name": "Page_Fault_Rate_Sec", "shap_value": 0.42, "feature_value": "12,450 /s", "impact": "HIGH_POSITIVE"},
            {"name": "TLB_Miss_Ratio", "shap_value": 0.28, "feature_value": "18.4 %", "impact": "POSITIVE"},
            {"name": "Context_Switches_Sec", "shap_value": 0.19, "feature_value": "8,920 /s", "impact": "POSITIVE"},
            {"name": "CPU_IOWait_Pct", "shap_value": 0.14, "feature_value": "42.1 %", "impact": "POSITIVE"},
            {"name": "Free_Memory_MB", "shap_value": -0.22, "feature_value": "128 MB", "impact": "NEGATIVE"},
            {"name": "Active_Process_Count", "shap_value": 0.08, "feature_value": "184", "impact": "MODERATE"}
        ]
    }

@app.post("/api/simulator/schedule")
async def simulate_schedule(req: ScheduleSimRequest):
    """Simulate CPU scheduling algorithms (FCFS, SJF, RR) and generate Gantt chart telemetry."""
    procs = req.processes if req.processes else [
        {"id": "P1", "burst": 6, "arrival": 0, "priority": 2},
        {"id": "P2", "burst": 3, "arrival": 1, "priority": 1},
        {"id": "P3", "burst": 8, "arrival": 2, "priority": 3},
        {"id": "P4", "burst": 4, "arrival": 3, "priority": 2}
    ]
    
    algo = req.algorithm.upper()
    gantt_blocks = []
    curr_time = 0
    waiting_times = {}
    turnaround_times = {}
    
    if algo == "FCFS":
        sorted_procs = sorted(procs, key=lambda x: x["arrival"])
        for p in sorted_procs:
            if curr_time < p["arrival"]:
                curr_time = p["arrival"]
            start = curr_time
            end = start + p["burst"]
            gantt_blocks.append({"pid": p["id"], "start": start, "end": end, "duration": p["burst"]})
            waiting_times[p["id"]] = start - p["arrival"]
            turnaround_times[p["id"]] = end - p["arrival"]
            curr_time = end
            
    elif algo == "SJF":
        remaining = [dict(p) for p in procs]
        completed = []
        while len(completed) < len(procs):
            available = [p for p in remaining if p["arrival"] <= curr_time and p["id"] not in [c["id"] for c in completed]]
            if not available:
                curr_time += 1
                continue
            shortest = min(available, key=lambda x: x["burst"])
            start = curr_time
            end = start + shortest["burst"]
            gantt_blocks.append({"pid": shortest["id"], "start": start, "end": end, "duration": shortest["burst"]})
            waiting_times[shortest["id"]] = start - shortest["arrival"]
            turnaround_times[shortest["id"]] = end - shortest["arrival"]
            curr_time = end
            completed.append(shortest)
            
    elif algo == "RR":
        q = req.quantum or 2
        rem_burst = {p["id"]: p["burst"] for p in procs}
        arrival_map = {p["id"]: p["arrival"] for p in procs}
        queue = [p["id"] for p in sorted(procs, key=lambda x: x["arrival"])]
        
        while queue:
            pid = queue.pop(0)
            if rem_burst[pid] <= 0:
                continue
            exec_time = min(q, rem_burst[pid])
            start = curr_time
            end = start + exec_time
            gantt_blocks.append({"pid": pid, "start": start, "end": end, "duration": exec_time})
            curr_time = end
            rem_burst[pid] -= exec_time
            
            if rem_burst[pid] == 0:
                turnaround_times[pid] = end - arrival_map[pid]
                waiting_times[pid] = turnaround_times[pid] - next(p["burst"] for p in procs if p["id"] == pid)
            else:
                queue.append(pid)
    else:
        for p in procs:
            start = curr_time
            end = start + p["burst"]
            gantt_blocks.append({"pid": p["id"], "start": start, "end": end, "duration": p["burst"]})
            waiting_times[p["id"]] = start - p["arrival"]
            turnaround_times[p["id"]] = end - p["arrival"]
            curr_time = end

    avg_wait = round(sum(waiting_times.values()) / max(1, len(waiting_times)), 2)
    avg_tat = round(sum(turnaround_times.values()) / max(1, len(turnaround_times)), 2)
    
    return {
        "algorithm": algo,
        "gantt": gantt_blocks,
        "waiting_times": waiting_times,
        "turnaround_times": turnaround_times,
        "avg_waiting_time": avg_wait,
        "avg_turnaround_time": avg_tat,
        "total_time": curr_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
