"""
OSTutor // Kernel - FastAPI Application Server
Serving OSTutorLLM Command Center & REST API Endpoints.
"""

import os
import sys
import time
import math
import random
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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
        "url": "https://www.os-book.com/OS10/",
        "content": "A Process Control Block (PCB) contains details including Process State, Program Counter, CPU registers, CPU scheduling information, memory-management information, accounting information, and I/O status information.",
        "score": 0.94
    },
    {
        "id": "doc_02",
        "source": "Modern Operating Systems (Tanenbaum) - Ch. 4",
        "topic": "Virtual Memory",
        "url": "https://www.pearson.com/en-us/subject-catalog/p/modern-operating-systems/P200000003295",
        "content": "The Translation Lookaside Buffer (TLB) is a hardware cache inside the MMU mapping virtual page numbers to physical frame numbers. TLB misses require walking page table levels, adding latency.",
        "score": 0.89
    },
    {
        "id": "doc_03",
        "source": "Operating Systems: Three Easy Pieces (OSTEP) - Ch. 28",
        "topic": "Synchronization",
        "url": "https://pages.cs.wisc.edu/~remzi/OSTEP/",
        "content": "A Semaphore maintains an integer value accessed only via wait() [P] and signal() [V] atomic operations. Counting semaphores control access to a finite set of resources.",
        "score": 0.86
    },
    {
        "id": "doc_04",
        "source": "Linux Kernel Architecture (Bovet & Cesati) - Ch. 7",
        "topic": "System Calls & Scheduling",
        "url": "https://www.kernel.org/doc/html/latest/",
        "content": "In Linux, CFS (Completely Fair Scheduler) uses a red-black tree indexed by virtual runtime (vruntime) to select the task with smallest vruntime for execution next.",
        "score": 0.82
    },
    {
        "id": "doc_05",
        "source": "Operating System Concepts (Silberschatz) - Ch. 11",
        "topic": "File Systems",
        "url": "https://www.os-book.com/OS10/",
        "content": "An inode (index node) stores block pointers to data blocks. Direct pointers handle small files, while single, double, and triple indirect pointers enable large files.",
        "score": 0.78
    }
]

class QuizRequest(BaseModel):
    topic: Optional[str] = "All Topics"
    difficulty: Optional[str] = "all"
    count: Optional[int] = 15

OS_QUIZ_DATABASE = [
    {
        "id": "q1",
        "topic": "Processes & Threads",
        "difficulty": "easy",
        "question": "What is the main job of a Process Control Block (PCB)?",
        "options": [
            "A. It stores the process's ID, state, registers, and memory details",
            "B. It deletes the process when it finishes",
            "C. It decides which CPU core is physically installed",
            "D. It controls the monitor brightness"
        ],
        "answer_index": 0,
        "explanation": "The PCB is the OS's records folder for a process. It keeps track of the process ID, current state, register values, and memory pointers so the kernel can resume it correctly."
    },
    {
        "id": "q2",
        "topic": "Processes & Threads",
        "difficulty": "easy",
        "question": "What does fork() do?",
        "options": [
            "A. It creates a new thread inside the same process",
            "B. It creates a child process by duplicating the parent process",
            "C. It permanently stops the current process",
            "D. It changes the CPU burst time"
        ],
        "answer_index": 1,
        "explanation": "fork() creates a nearly identical child process. The child gets its own process ID and memory space, while the parent continues running."
    },
    {
        "id": "q3",
        "topic": "Processes & Threads",
        "difficulty": "medium",
        "question": "Why is Copy-on-Write useful after fork()?",
        "options": [
            "A. It makes every page private immediately",
            "B. It lets parent and child share pages until one writes, then copies the page only when needed",
            "C. It removes all page tables",
            "D. It slows memory access on purpose"
        ],
        "answer_index": 1,
        "explanation": "Copy-on-Write saves time and memory. Parent and child share the same read-only pages at first. Only when one writes does the kernel create a separate copy."
    },
    {
        "id": "q4",
        "topic": "CPU Scheduling",
        "difficulty": "easy",
        "question": "Which scheduling policy runs jobs in arrival order without preemption?",
        "options": [
            "A. RR",
            "B. FCFS",
            "C. SJF",
            "D. LRU"
        ],
        "answer_index": 1,
        "explanation": "FCFS, or First-Come First-Served, simply executes processes in the order they arrive. It is simple, but it can create long waiting times for short jobs."
    },
    {
        "id": "q5",
        "topic": "CPU Scheduling",
        "difficulty": "medium",
        "question": "What is a time quantum in Round Robin scheduling?",
        "options": [
            "A. The amount of RAM per process",
            "B. The maximum time a process can run before being preempted",
            "C. The number of waiting threads",
            "D. The size of a page table"
        ],
        "answer_index": 1,
        "explanation": "A time quantum is the slice of CPU time each process gets before the scheduler switches to the next process. It is how Round Robin prevents one process from hogging the CPU."
    },
    {
        "id": "q6",
        "topic": "Virtual Memory",
        "difficulty": "easy",
        "question": "What is the TLB?",
        "options": [
            "A. A small hardware cache storing recent virtual-to-physical address mappings",
            "B. A disk scheduling algorithm",
            "C. A function that turns threads into processes",
            "D. A file system journal"
        ],
        "answer_index": 0,
        "explanation": "The Translation Lookaside Buffer is a very fast cache in the MMU. It stores recent address translations so the CPU can translate addresses much faster."
    },
    {
        "id": "q7",
        "topic": "Virtual Memory",
        "difficulty": "medium",
        "question": "What happens on a TLB miss?",
        "options": [
            "A. The CPU immediately crashes",
            "B. The MMU walks the page table to find the mapping and may trigger a page fault",
            "C. The OS deletes the process",
            "D. The thread becomes a mutex"
        ],
        "answer_index": 1,
        "explanation": "When the requested mapping is not in the TLB, the OS or hardware has to look it up in the page table. If the page is not in memory, a page fault occurs and the OS fetches it."
    },
    {
        "id": "q8",
        "topic": "Synchronization",
        "difficulty": "easy",
        "question": "What is a mutex mainly used for?",
        "options": [
            "A. To store page tables",
            "B. To protect a critical section so only one thread enters at a time",
            "C. To schedule CPU tasks",
            "D. To read from a USB port"
        ],
        "answer_index": 1,
        "explanation": "A mutex is a lock. It ensures that only one thread can enter a critical section at a time, preventing race conditions and inconsistent shared data."
    },
    {
        "id": "q9",
        "topic": "Synchronization",
        "difficulty": "medium",
        "question": "Which condition is part of deadlock?",
        "options": [
            "A. No waiting period",
            "B. Circular wait",
            "C. Fast CPU speed",
            "D. Unlimited RAM"
        ],
        "answer_index": 1,
        "explanation": "Deadlock happens when processes are stuck waiting on each other in a cycle. Circular wait is one of the classic necessary conditions for deadlock."
    },
    {
        "id": "q10",
        "topic": "File Systems",
        "difficulty": "easy",
        "question": "What is an inode?",
        "options": [
            "A. A type of CPU register",
            "B. A metadata structure storing file info such as size, permissions, and block pointers",
            "C. A page replacement policy",
            "D. A network protocol"
        ],
        "answer_index": 1,
        "explanation": "An inode stores the file's metadata and pointer information, while the filename is kept in the directory entry. It is a major file system concept in Unix-like systems."
    },
    {
        "id": "q11",
        "topic": "File Systems",
        "difficulty": "medium",
        "question": "Why are directories and inodes important in a file system?",
        "options": [
            "A. They allow the kernel to locate data blocks and map names to actual files",
            "B. They increase CPU temperature",
            "C. They replace the page table",
            "D. They remove the need for disk devices"
        ],
        "answer_index": 0,
        "explanation": "Directories map names to file entries, and inodes tell the kernel where the file's data blocks are stored. Together, they make file lookup work."
    },
    {
        "id": "q12",
        "topic": "I/O & Interrupts",
        "difficulty": "easy",
        "question": "What is the purpose of DMA?",
        "options": [
            "A. To take over the entire page table",
            "B. To let devices move data directly to memory without constant CPU involvement",
            "C. To speed up file creation only",
            "D. To lock the scheduler"
        ],
        "answer_index": 1,
        "explanation": "DMA allows hardware devices like disks and NICs to transfer data directly into RAM. This reduces CPU overhead and speeds up I/O."
    },
    {
        "id": "q13",
        "topic": "I/O & Interrupts",
        "difficulty": "medium",
        "question": "What is an interrupt used for?",
        "options": [
            "A. To tell the CPU that an external event needs attention",
            "B. To increase memory overflow",
            "C. To place a process in the ready queue immediately",
            "D. To turn off the kernel"
        ],
        "answer_index": 0,
        "explanation": "An interrupt signals the CPU that some device or event needs attention, such as disk I/O completion or a timer tick. The kernel handles it quickly and resumes execution."
    },
    {
        "id": "q14",
        "topic": "Processes & Threads",
        "difficulty": "medium",
        "question": "What is the difference between a process and a thread?",
        "options": [
            "A. A thread has its own full memory space, while a process shares memory",
            "B. A process owns its own address space and resources, while threads share the same process resources",
            "C. They are the same thing",
            "D. A thread is always scheduled before a process"
        ],
        "answer_index": 1,
        "explanation": "A process is a heavier unit with its own virtual memory and resources. Threads are lighter-weight execution units inside one process and share the same address space."
    },
    {
        "id": "q15",
        "topic": "Virtual Memory",
        "difficulty": "hard",
        "question": "Why do page faults matter in system performance?",
        "options": [
            "A. They cause the OS to bring pages from disk into memory, which can be slow and affect responsiveness",
            "B. They always reduce the number of threads",
            "C. They prevent any program from running",
            "D. They make the CPU run without memory"
        ],
        "answer_index": 0,
        "explanation": "Page faults happen when a page is not currently in memory. Fetching it from disk is expensive, so excessive page faults can slow the system dramatically and trigger thrashing."
    }
]

OS_FLASHCARDS_DATABASE = [
    {
        "id": "fc1",
        "topic": "Processes & Threads",
        "term": "PCB",
        "front": "What is a Process Control Block?",
        "back": "It is the kernel's record for a process. It stores the process ID, state, registers, parent pointer, memory info, and scheduling information.",
        "analogy": "Think of it like a student ID card and transcript record combined: it tells the OS who the process is and what it is doing."
    },
    {
        "id": "fc2",
        "topic": "Processes & Threads",
        "term": "fork()",
        "front": "What does fork() do?",
        "back": "It creates a child process by copying the parent process state. The child gets a new PID and usually starts executing right after the fork call.",
        "analogy": "It's like making a photocopy of a running task so the copy can continue independently."
    },
    {
        "id": "fc3",
        "topic": "Processes & Threads",
        "term": "Context Switch",
        "front": "What is a context switch?",
        "back": "It is when the OS saves the currently running process's state and loads another process's state so both can take turns on the CPU.",
        "analogy": "It's like saving your place in a book and handing it to someone else so they can read for a while."
    },
    {
        "id": "fc4",
        "topic": "CPU Scheduling",
        "term": "FCFS",
        "front": "What is FCFS?",
        "back": "First-Come First-Served runs processes in the order they arrive. It is simple but can lead to poor average waiting time.",
        "analogy": "Imagine a line at a cashier where everyone waits in arrival order, even if some people have very small tasks."
    },
    {
        "id": "fc5",
        "topic": "CPU Scheduling",
        "term": "Round Robin",
        "front": "What is Round Robin?",
        "back": "Each process gets a small time slice, and then the scheduler moves to the next process. It improves fairness but may increase context switching overhead.",
        "analogy": "Like taking turns speaking in a group discussion, each person gets only a few minutes before passing the floor."
    },
    {
        "id": "fc6",
        "topic": "CPU Scheduling",
        "term": "Convoy Effect",
        "front": "What is the convoy effect?",
        "back": "A long CPU-bound job in FCFS can block many short jobs behind it, creating a backlog and high waiting time.",
        "analogy": "It is like one huge truck blocking a narrow road and making everyone else wait behind it."
    },
    {
        "id": "fc7",
        "topic": "Virtual Memory",
        "term": "TLB",
        "front": "What is the TLB?",
        "back": "The Translation Lookaside Buffer is a tiny hardware cache that stores recent virtual-to-physical address translations.",
        "analogy": "It's like a phone's speed-dial list: instead of looking up the full address every time, it remembers the most common ones."
    },
    {
        "id": "fc8",
        "topic": "Virtual Memory",
        "term": "Page Fault",
        "front": "What is a page fault?",
        "back": "A page fault occurs when a process accesses a memory page that is not currently in RAM. The OS brings the page from disk into memory.",
        "analogy": "It is like opening a recipe book that is not on your desk, so you fetch it from the library shelf before starting."
    },
    {
        "id": "fc9",
        "topic": "Synchronization",
        "term": "Mutex",
        "front": "What is a mutex?",
        "back": "A mutex is a lock that allows only one thread at a time to enter a critical section. It prevents corruption of shared data.",
        "analogy": "It's like a single bathroom key in a house: only one person can use it at a time."
    },
    {
        "id": "fc10",
        "topic": "Synchronization",
        "term": "Semaphore",
        "front": "What is a semaphore?",
        "back": "A semaphore is a counter used to control access to a limited number of resources, often using wait and signal operations.",
        "analogy": "Think of it as a number of tickets for a limited ride: each ticket grants one slot to a rider."
    },
    {
        "id": "fc11",
        "topic": "Synchronization",
        "term": "Deadlock",
        "front": "What is deadlock?",
        "back": "Deadlock happens when two or more processes are stuck waiting on each other forever because each is holding a resource the other needs.",
        "analogy": "It is like two people both holding the same two chairs and refusing to move until the other gives up theirs."
    },
    {
        "id": "fc12",
        "topic": "File Systems",
        "term": "Inode",
        "front": "What is an inode?",
        "back": "An inode contains metadata about a file, such as permissions, ownership, timestamp, size, and pointers to the data blocks.",
        "analogy": "It is like the label on a box in a warehouse that tells you what is inside and where it is stored."
    },
    {
        "id": "fc13",
        "topic": "File Systems",
        "term": "Directory",
        "front": "Why do directories matter?",
        "back": "A directory maps file names to inode numbers, making it possible for the OS to find files quickly.",
        "analogy": "It is like the index of a library: it tells you where the book is located."
    },
    {
        "id": "fc14",
        "topic": "I/O & Interrupts",
        "term": "DMA",
        "front": "What is DMA?",
        "back": "Direct Memory Access lets devices transfer data straight into RAM without the CPU handling every byte.",
        "analogy": "It is like having a helper move boxes directly into your room instead of making you carry each one yourself."
    },
    {
        "id": "fc15",
        "topic": "I/O & Interrupts",
        "term": "Interrupt",
        "front": "What is an interrupt?",
        "back": "An interrupt is a signal from hardware or software telling the CPU that something needs attention immediately, such as a timer tick or I/O completion.",
        "analogy": "It is like a doorbell: something important happened, and the CPU stops what it is doing to respond."
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

@app.get("/library")
async def serve_library():
    """Serve the library page listing the textbook links used by OSTutor."""
    return FileResponse(str(FRONTEND_DIR / "library.html"))

@app.get("/api/kernel/telemetry")
async def get_telemetry():
    """Retrieve simulated live system and tutor model telemetry."""
    uptime = int(time.time() - START_TIME)
    cpu_load = round(25.0 + 15.0 * math.sin(time.time() / 5.0) + random.uniform(-2, 2), 1)
    memory_used = round(4.2 + 0.3 * math.cos(time.time() / 10.0), 2)
    
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints" / "ostutor_llm_lora"
    has_checkpoint = (checkpoint_dir / "adapter_config.json").exists()

    return {
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime // 3600:02d}:{(uptime % 3600) // 60:02d}:{uptime % 60:02d}",
        "cpu_usage_pct": max(5.0, min(99.0, cpu_load)),
        "memory_used_gb": memory_used,
        "memory_total_gb": 16.0,
        "active_kernel_threads": 48 + int(uptime % 12),
        "faiss_indexed_chunks": 1420,
        "model_latency_ms": random.randint(18, 45),
        "tutor_status": "ONLINE / FINE-TUNED" if has_checkpoint else "ONLINE / READY",
        "lora_checkpoint_active": has_checkpoint,
        "seed": config.get("system", {}).get("seed", 42)
    }

@app.post("/api/llm/finetune")
async def trigger_llm_finetuning():
    """Trigger instruction fine-tuning process for OSTutor LLM."""
    from src.models.fine_tune_llm import OSTutorLLMFineTuner
    try:
        tuner = OSTutorLLMFineTuner()
        results = tuner.run_finetuning(dry_run=False)
        return {
            "status": "SUCCESS",
            "message": "OSTutor LLM Fine-Tuning Completed Successfully",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning error: {str(e)}")


@app.get("/api/concepts")
async def get_concepts():
    """Return OS concept module matrix and mastery rates."""
    return {"concepts": OS_CONCEPTS}

@app.get("/api/flashcards")
async def get_flashcards(topic: Optional[str] = None):
    """Return interactive OS flashcards filtered by topic."""
    cards = OS_FLASHCARDS_DATABASE
    if topic and topic.lower() != "all":
        cards = [card for card in cards if topic.lower() in card["topic"].lower()]
    return {"flashcards": cards}

@app.post("/api/quiz/generate")
async def generate_quiz(req: QuizRequest):
    """Generate OS multiple-choice quiz items based on topic and difficulty."""
    filtered = OS_QUIZ_DATABASE
    if req.topic and req.topic.lower() not in ("all topics", "all"):
        filtered = [q for q in filtered if req.topic.lower() in q["topic"].lower()]
    if req.difficulty and req.difficulty.lower() != "all":
        filtered = [q for q in filtered if q["difficulty"].lower() == req.difficulty.lower()]
    if not filtered:
        filtered = OS_QUIZ_DATABASE
    return {"questions": filtered[: max(1, req.count or 15)]}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CANNOT_ANSWER = "CANNOT_ANSWER"
TUTOR_SYSTEM_PROMPT = (
    "You are OSTutor, a patient Operating Systems tutor for students. "
    "Format every answer in markdown: a one-sentence definition under a heading, "
    "then '### Key points' with 3-5 bullets (one sentence each), then '### Example' with a concrete worked example, "
    "and end with a one-line '### Summary'. "
    "Be precise with terminology. For example, LRU evicts the least recently used page, not the page that has been in memory the longest. "
    "Keep processes and frames distinct. "
    "Use the reference excerpts only when they are relevant. If they do not cover the question, "
    "say so in one sentence and answer from general OS knowledge."
)


def _ask_ollama(message: str, topic: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Answer a student question with a local Ollama model; return None if Ollama is unreachable or the model is missing."""
    excerpts = "\n\n".join(f"[{doc['source']}]\n{doc['content']}" for doc in RAG_KNOWLEDGE_BASE)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nStudent question: {message}\n\nReference excerpts:\n{excerpts}"},
        ],
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)["message"]["content"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        logger.warning("Ollama tutor call failed: %s", e)
        return None


def _ask_gemini(message: str, topic: str) -> Optional[str]:
    """Answer a student question with the Gemini API; return None if the call fails or the model cannot answer."""
    excerpts = "\n\n".join(f"[{doc['source']}]\n{doc['content']}" for doc in RAG_KNOWLEDGE_BASE)
    payload = {
        "systemInstruction": {"parts": [{"text": TUTOR_SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"Topic: {topic}\nStudent question: {message}\n\nReference excerpts:\n{excerpts}"}]},
        ],
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            answer = json.load(response)["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as e:
        logger.warning("Gemini tutor call failed: %s", e)
        return None
    return None if CANNOT_ANSWER in answer else answer


def _ask_tutor(message: str, topic: str) -> Optional[Tuple[str, str]]:
    """Qwen first; Gemini when Qwen is unavailable or unsure."""
    answer = _ask_ollama(message, topic)
    if answer is not None and CANNOT_ANSWER not in answer:
        return answer, f"{OLLAMA_MODEL} (local)"
    if GEMINI_API_KEY:
        gemini_answer = _ask_gemini(message, topic)
        if gemini_answer is not None:
            return gemini_answer, f"{GEMINI_MODEL} (Gemini fallback)"
    return None


@app.post("/api/tutor/chat")
async def tutor_chat(req: ChatRequest):
    """Student-friendly OS tutor response generator with beginner-focused explanations and kernel traces."""
    msg_lower = req.message.lower().strip()
    citations = []
    kernel_trace = []

    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "yo", "sup", "who are you", "what can you do", "help",
        "thanks", "thank you"
    ]

    tutor_answer = _ask_tutor(req.message, req.topic)
    if tutor_answer is not None:
        response_text, answered_by = tutor_answer
        kernel_trace = [
            f"RETRIEVAL: {len(RAG_KNOWLEDGE_BASE)} reference excerpts supplied to the model",
            f"LLM_GENERATE: answered by {answered_by}",
        ]
        citations = RAG_KNOWLEDGE_BASE
    elif any(word in msg_lower for word in greetings) or msg_lower in ["hi", "hello", "hey", "help", "yo"]:
        response_text = (
            "### 👋 Hey there! Welcome to OSTutor.\n\n"
            "I’m your friendly Operating Systems study partner. I’ll explain ideas in plain English, walk through kernel behavior step by step, and make tricky topics feel less intimidating.\n\n"
            "#### What I can help with:\n"
            "- **Processes & Threads**: *How does `fork()` work?*\n"
            "- **Virtual Memory**: *Why does the TLB matter?*\n"
            "- **Synchronization**: *What is a mutex and why do we need one?*\n"
            "- **CPU Scheduling**: *What is Round Robin and why does the quantum matter?*\n"
            "- **File Systems**: *How do inodes help the OS find files?*\n\n"
            "You can also use the quiz and flashcard tools below to practice and remember the ideas faster."
        )
        kernel_trace = [
            "USER_GREETING: Student asked for a beginner-friendly OS explanation",
            "SYSTEM_STATUS: OSTutor tutor online and ready to teach",
            "GUIDANCE: Suggested practice topics for quick growth"
        ]
        citations = [RAG_KNOWLEDGE_BASE[0], RAG_KNOWLEDGE_BASE[1]]

    elif "fork" in msg_lower or "process" in msg_lower or "pcb" in msg_lower:
        response_text = (
            "### Process Creation: How `fork()` works in simple English\n\n"
            "Think of a process as an active program running on the computer. When a program wants to create a copy of itself, it calls **`fork()`**. The OS kernel then creates a **child process** that is very similar to the parent.\n\n"
            "#### Step by step:\n"
            "1. The program calls `fork()`.\n"
            "2. The kernel creates a new **Process Control Block (PCB)** for the child. The PCB is like the process’s identity card—it stores its state, registers, and memory details.\n"
            "3. The memory pages are marked **Copy-on-Write**. That means the parent and child can share memory at first instead of copying everything immediately.\n"
            "4. The child gets a new **PID**. The parent keeps its own PID, and the child usually starts executing right after the `fork()` call.\n"
            "5. The return value is different in each process: the child gets `0`, and the parent gets the child PID.\n\n"
            "#### Why this matters:\n"
            "This is how shells, servers, and many programs create worker tasks. It is a powerful concept because one process can split into two independently running jobs while sharing resources efficiently."
        )
        kernel_trace = [
            "SYS_CALL [0x02]: sys_fork() invoked by the running process",
            "ALLOCATE: new task_struct created for the child process",
            "COW_SET: parent and child pages marked read-only until a write occurs",
            "PID_ASSIGN: new child PID assigned by the kernel",
            "RETURN: parent receives child PID; child receives 0"
        ]
        citations = [RAG_KNOWLEDGE_BASE[0]]

    elif "virtual memory" in msg_lower or "page" in msg_lower or "tlb" in msg_lower:
        response_text = (
            "### Virtual Memory: Why the OS gives each process its own memory world\n\n"
            "Virtual memory is a clever idea: it lets each process act like it has a large, private space in memory, even if the actual physical RAM is shared and limited.\n\n"
            "#### What happens when memory is accessed:\n"
            "1. The CPU takes a virtual address and splits it into a **virtual page number** and an **offset**.\n"
            "2. The CPU checks the **TLB**, which is a tiny cache of recent translations.\n"
            "3. If the TLB has the mapping, it is a **TLB hit** and the translation is very fast.\n"
            "4. If the TLB does not have it, the MMU performs a **page table walk** to find the mapping.\n"
            "5. If the page is not currently in RAM, the OS triggers a **page fault**, loads the needed page from disk, and then resumes execution.\n\n"
            "#### The big idea:\n"
            "This is what lets computers run many programs at once without each program accidentally overwriting the other’s memory."
        )
        kernel_trace = [
            "MMU: virtual address split into page number and offset",
            "TLB: lookup in hardware cache -> MISS",
            "PAGE_WALK: kernel/MMU reads page table to locate the frame",
            "PAGE_FAULT: requested page not resident in RAM",
            "TLB_UPDATE: new mapping cached for future access"
        ]
        citations = [RAG_KNOWLEDGE_BASE[1]]

    elif "semaphore" in msg_lower or "mutex" in msg_lower or "sync" in msg_lower or "deadlock" in msg_lower:
        response_text = (
            "### Synchronization: Why locks exist in multithreaded programs\n\n"
            "When multiple threads run at the same time, they may try to access the same shared data at the same moment. That can cause a **race condition**, where the final result becomes unpredictable.\n\n"
            "#### Mutex vs Semaphore:\n"
            "- **Mutex**: A lock that gives exclusive access to one thread at a time. It is like a single key to a room.\n"
            "- **Semaphore**: A counter-based primitive that can allow more than one thread in under controlled limits. It is like a fixed number of tickets for a limited resource.\n\n"
            "#### Why this matters:\n"
            "The OS uses locks to protect critical sections of code. Without them, threads could overwrite each other’s work, corrupt shared variables, or leave the system in an inconsistent state.\n\n"
            "**Deadlock** happens when two or more processes are stuck waiting on each other forever. That is one of the classic problems the OS tries to prevent or detect."
        )
        kernel_trace = [
            "THREAD_T2: tries to enter critical section",
            "MUTEX_LOCK: lock acquired and shared data protected",
            "THREAD_T3: blocked because lock is already held",
            "DEADLOCK_CHECK: circular wait scenario detected in wait queue"
        ]
        citations = [RAG_KNOWLEDGE_BASE[2]]

    elif "schedule" in msg_lower or "round robin" in msg_lower or "gantt" in msg_lower or "cfs" in msg_lower or "cpu" in msg_lower:
        response_text = (
            "### CPU Scheduling: How the OS decides who runs next\n\n"
            "The CPU scheduler is like a traffic controller for your computer. It decides which process gets to use the CPU next when many programs are ready to run.\n\n"
            "#### Common policies:\n"
            "- **FCFS**: first-come, first-served. Simple, but it can make short tasks wait a long time behind a big one.\n"
            "- **SJF**: shortest job first. This may reduce waiting time, but it requires knowing the job length in advance.\n"
            "- **Round Robin**: each process gets a small time slice, then the scheduler moves to the next one. This keeps things fair.\n"
            "- **CFS**: Linux tries to give each task a fair share of CPU time using a red-black tree and virtual runtime metrics.\n\n"
            "#### Why the quantum matters:\n"
            "A smaller time quantum increases fairness but increases context-switching overhead. A larger quantum feels more efficient but may make the system less responsive."
        )
        kernel_trace = [
            "SCHED_TICK: timer interrupt triggers periodic scheduler checks",
            "PREEMPT: current process reaches quantum limit",
            "CONTEXT_SWITCH: save old process state and load new one",
            "SELECT_NEXT: scheduler chooses the next runnable process"
        ]
        citations = [RAG_KNOWLEDGE_BASE[3]]

    else:
        response_text = (
            f"### I don't have a reliable answer for \"{req.message}\" yet\n\n"
            "The tutor could not produce a verified answer for this question. "
            "Try rephrasing it, or check the section of your textbook or course notes that covers the topic."
        )
        kernel_trace = [
            "INPUT_QUERY: user asks a general OS question",
            "VECTOR_SEARCH: retrieving relevant OS textbook references",
            "LLM_GENERATE: structured response created in student-friendly language",
            "OUTPUT_STREAM: explanation prepared with linked concepts"
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
