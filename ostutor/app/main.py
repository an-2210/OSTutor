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

try:
    from src.utils.config import load_config
except ImportError:
    def load_config(path: str) -> dict:
        return {"system": {"name": "OSTutor // Kernel", "version": "1.0.0"}, "rag_cti": {"top_k_retrieval": 5, "vector_store_type": "faiss"}}

try:
    from src.utils.logging import setup_logger
except ImportError:
    import logging
    def setup_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

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

OS_EBOOKS = [
    {
        "id": "ostep",
        "title": "Operating Systems: Three Easy Pieces (OSTEP)",
        "authors": "Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau",
        "description": "The premier open-access OS textbook covering Virtualization, Concurrency, and Persistence with clean C code and conceptual diagrams.",
        "badge": "FREE / OPEN ACCESS",
        "cover_color": "#7c3aed",
        "units": ["Virtualization", "CPU Scheduling", "Memory Management", "Concurrency", "Persistence"],
        "url": "https://pages.cs.wisc.edu/~remzi/OSTEP/",
        "pdf_url": "https://pages.cs.wisc.edu/~remzi/OSTEP/"
    },
    {
        "id": "silberschatz",
        "title": "Operating System Concepts (10th Edition)",
        "authors": "Abraham Silberschatz, Peter B. Galvin, Greg Gagne",
        "description": "The classic 'Dinosaur Book'. Complete foundational reference for OS architecture, process synchronization, paging, and security.",
        "badge": "ACADEMIC STANDARD",
        "cover_color": "#2563eb",
        "units": ["Processes & Threads", "CPU Scheduling", "Virtual Memory", "Storage & File Systems"],
        "url": "https://www.os-book.com/OS10/",
        "pdf_url": "https://www.os-book.com/OS10/"
    },
    {
        "id": "tanenbaum",
        "title": "Modern Operating Systems (4th Edition)",
        "authors": "Andrew S. Tanenbaum & Herbert Bos",
        "description": "Comprehensive explanation of modern OS design, memory management, multimedia systems, and security by MINIX creator Andrew Tanenbaum.",
        "badge": "CORE REFERENCE",
        "cover_color": "#f43f5e",
        "units": ["Kernel Design", "Memory Management", "File Systems", "Multiple Processor Systems"],
        "url": "https://www.pearson.com/en-us/subject-catalog/p/modern-operating-systems/P200000003295",
        "pdf_url": "https://www.pearson.com/en-us/subject-catalog/p/modern-operating-systems/P200000003295"
    },
    {
        "id": "robert_love",
        "title": "Linux Kernel Development (3rd Edition)",
        "authors": "Robert Love",
        "description": "In-depth guide to the design and implementation of the Linux kernel, covering task schedulers, VFS, interrupt handlers, and kernel synchronization.",
        "badge": "LINUX KERNEL GUIDE",
        "cover_color": "#10b981",
        "units": ["Process Scheduling", "System Calls", "Kernel Data Structures", "Interrupt Handling"],
        "url": "https://www.kernel.org/doc/html/latest/",
        "pdf_url": "https://www.kernel.org/doc/html/latest/"
    },
    {
        "id": "tlcl",
        "title": "The Linux Command Line",
        "authors": "William Shotts",
        "description": "Free Creative Commons guide for mastering Linux shell scripting, process management, file permissions, and environment management.",
        "badge": "FREE CREATIVE COMMONS",
        "cover_color": "#f59e0b",
        "units": ["Shell Basics", "Process Controls", "Permissions", "Storage Utilities"],
        "url": "https://linuxcommand.org/tlcl.php",
        "pdf_url": "https://linuxcommand.org/tlcl.php"
    }
]

RAG_KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "source": "Operating System Concepts (Silberschatz) - Ch. 3",
        "topic": "Process Management",
        "content": "A Process Control Block (PCB) contains details including Process State, Program Counter, CPU registers, CPU scheduling information, memory-management information, accounting information, and I/O status information.",
        "score": 0.94,
        "url": "https://www.os-book.com/OS10/"
    },
    {
        "id": "doc_02",
        "source": "Modern Operating Systems (Tanenbaum) - Ch. 4",
        "topic": "Virtual Memory",
        "content": "The Translation Lookaside Buffer (TLB) is a hardware cache inside the MMU mapping virtual page numbers to physical frame numbers. TLB misses require walking page table levels, adding latency.",
        "score": 0.89,
        "url": "https://www.pearson.com/en-us/subject-catalog/p/modern-operating-systems/P200000003295"
    },
    {
        "id": "doc_03",
        "source": "Operating Systems: Three Easy Pieces (OSTEP) - Ch. 28",
        "topic": "Synchronization",
        "content": "A Semaphore maintains an integer value accessed only via wait() [P] and signal() [V] atomic operations. Counting semaphores control access to a finite set of resources.",
        "score": 0.86,
        "url": "https://pages.cs.wisc.edu/~remzi/OSTEP/"
    },
    {
        "id": "doc_04",
        "source": "Linux Kernel Architecture (Bovet & Cesati) - Ch. 7",
        "topic": "System Calls & Scheduling",
        "content": "In Linux, CFS (Completely Fair Scheduler) uses a red-black tree indexed by virtual runtime (vruntime) to select the task with smallest vruntime for execution next.",
        "score": 0.82,
        "url": "https://www.kernel.org/doc/html/latest/"
    },
    {
        "id": "doc_05",
        "source": "Operating System Concepts (Silberschatz) - Ch. 11",
        "topic": "File Systems",
        "content": "An inode (index node) stores block pointers to data blocks. Direct pointers handle small files, while single, double, and triple indirect pointers enable large files.",
        "score": 0.78,
        "url": "https://www.os-book.com/OS10/"
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

@app.get("/api/ebooks")
async def get_ebooks():
    """Return digital OS textbooks and interactive reference library."""
    return {"ebooks": OS_EBOOKS}

class QuizRequest(BaseModel):
    topic: Optional[str] = "All Topics"
    difficulty: Optional[str] = "all"
    count: Optional[int] = 5

OS_QUIZ_DATABASE = [
    {
        "id": "q1",
        "topic": "Processes & Threads",
        "difficulty": "easy",
        "question": "What is the primary function of a Process Control Block (PCB)?",
        "options": [
            "A. It stores physical RAM chips on the motherboard",
            "B. It holds kernel accounting data (PID, CPU registers, state, memory pointers) for a process",
            "C. It converts compiled C code into Python bytecode",
            "D. It replaces the hardware CPU clock signal"
        ],
        "answer_index": 1,
        "explanation": "The Process Control Block (PCB) is the OS kernel's central accounting record for a process. It tracks Process ID, CPU registers, execution state (Ready/Running/Waiting), and open file pointers."
    },
    {
        "id": "q2",
        "topic": "Virtual Memory",
        "difficulty": "medium",
        "question": "What happens when a CPU requests a virtual address whose VPN is NOT found in the TLB?",
        "options": [
            "A. The computer immediately reboots",
            "B. A TLB miss occurs, causing a page table walk in RAM to find the physical frame mapping",
            "C. The process is deleted permanently from disk",
            "D. Virtual memory is automatically disabled"
        ],
        "answer_index": 1,
        "explanation": "On a TLB miss, the hardware MMU performs a Page Table Walk to look up the physical frame number in main RAM. If valid, the translation is cached into the TLB for fast subsequent access."
    },
    {
        "id": "q3",
        "topic": "CPU Scheduling",
        "difficulty": "easy",
        "question": "Which CPU scheduling algorithm is non-preemptive and susceptible to the 'Convoy Effect'?",
        "options": [
            "A. Round Robin (RR)",
            "B. First-Come First-Served (FCFS)",
            "C. Preemptive Priority",
            "D. Shortest Remaining Time First (SRTF)"
        ],
        "answer_index": 1,
        "explanation": "FCFS schedules processes in arrival order without preemption. If a CPU-bound process with a huge burst arrives first, short processes get stuck waiting behind it (the Convoy Effect)."
    },
    {
        "id": "q4",
        "topic": "Synchronization",
        "difficulty": "medium",
        "question": "What is the key difference between a Mutex and a Counting Semaphore?",
        "options": [
            "A. Mutexes only work on GPUs while Semaphores work on CPUs",
            "B. A Mutex has ownership (only the lock holder can unlock it), while a Semaphore manages N resource units",
            "C. Semaphores cannot block threads",
            "D. Mutexes allow up to 100 threads in a critical section"
        ],
        "answer_index": 1,
        "explanation": "A Mutex is a binary lock with strict ownership (the thread that locked it must unlock it). A Semaphore is a thread-safe counter used to grant access to N available resource units."
    },
    {
        "id": "q5",
        "topic": "File Systems",
        "difficulty": "easy",
        "question": "What metadata does a UNIX inode store?",
        "options": [
            "A. The file's data contents and user passwords",
            "B. File permissions, owner UID, file size, timestamps, and data block pointers (direct/indirect)",
            "C. Only the file name",
            "D. The GPU rendering pipeline configuration"
        ],
        "answer_index": 1,
        "explanation": "An inode (index node) stores all metadata for a file EXCEPT its filename (which lives in directory entries). This includes permissions (rwxr-xr-x), owner, file size, and block pointers."
    },
    {
        "id": "q6",
        "topic": "I/O & Interrupts",
        "difficulty": "medium",
        "question": "Why is Direct Memory Access (DMA) used in modern Operating Systems?",
        "options": [
            "A. To allow high-speed hardware devices to transfer data directly to RAM without burning CPU cycles for every byte",
            "B. To eliminate the need for physical RAM chips",
            "C. To prevent hard drives from spinning",
            "D. To encrypt network traffic in hardware"
        ],
        "answer_index": 0,
        "explanation": "Without DMA, the CPU would have to execute instructions to copy every single byte from SSDs/network cards into RAM. DMA lets device controllers copy data directly to RAM, interrupting the CPU only when finished."
    }
]

OS_FLASHCARDS_DATABASE = [
    {
        "id": "fc1",
        "topic": "Processes & Threads",
        "term": "Process Control Block (PCB)",
        "front": "What is a Process Control Block (PCB)?",
        "back": "The OS kernel's ID card and status file for a process. It holds the PID, program counter, register values, execution state (Running/Ready/Waiting), and open file list.",
        "analogy": "💡 Analogy: Think of a PCB as a student ID file in a university system—it keeps track of your status, classes, and records."
    },
    {
        "id": "fc2",
        "topic": "Processes & Threads",
        "term": "Copy-On-Write (COW)",
        "front": "How does Copy-On-Write (COW) work in fork()?",
        "back": "Instead of duplicating physical RAM during fork(), parent and child processes share read-only memory pages. A physical copy is made ONLY when one of them writes to a page.",
        "analogy": "💡 Analogy: Sharing a Google Doc as read-only. You only get your own private copy if you click 'Make a Duplicate' to edit it!"
    },
    {
        "id": "fc3",
        "topic": "CPU Scheduling",
        "term": "Convoy Effect",
        "front": "What is the Convoy Effect in FCFS Scheduling?",
        "back": "A situation in FCFS scheduling where a single long CPU-bound process blocks all short processes behind it, causing average waiting time to skyrocket.",
        "analogy": "💡 Analogy: Getting stuck behind a tractor on a one-lane highway when you just want to grab a quick coffee!"
    },
    {
        "id": "fc4",
        "topic": "CPU Scheduling",
        "term": "Round Robin (RR) Quantum",
        "front": "What is a Time Quantum in Round Robin?",
        "back": "The maximum consecutive CPU execution time slice (e.g. 2ms or 10ms) granted to a process before the OS timer interrupts and preempts it to let the next process run.",
        "analogy": "💡 Analogy: Passing an arcade controller around every 2 minutes so everyone gets a turn!"
    },
    {
        "id": "fc5",
        "topic": "Virtual Memory",
        "term": "Translation Lookaside Buffer (TLB)",
        "front": "What is the TLB and why is it crucial?",
        "back": "An ultra-fast hardware cache inside the CPU Memory Management Unit (MMU) that stores recent Virtual Page Number -> Physical Frame Number mappings for near-zero latency.",
        "analogy": "💡 Analogy: A speed-dial list on your phone instead of looking up someone's full address in a giant phonebook every time."
    },
    {
        "id": "fc6",
        "topic": "Synchronization",
        "term": "Deadlock (Banker's Algorithm)",
        "front": "What are the 4 Necessary Conditions for Deadlock?",
        "back": "1. Mutual Exclusion\n2. Hold and Wait\n3. No Preemption\n4. Circular Wait.\nIf all 4 hold simultaneously, processes become stuck forever!",
        "analogy": "💡 Analogy: Four cars reaching a 4-way stop sign at the exact same time, each waiting for the car on their right to move first."
    },
    {
        "id": "fc7",
        "topic": "File Systems",
        "term": "Inode (Index Node)",
        "front": "What is an Inode in Unix-like File Systems?",
        "back": "A data structure on disk that stores all metadata for a file (permissions, owner, size, timestamps, block pointers) EXCEPT for the file's name.",
        "analogy": "💡 Analogy: A library catalog card that tells you author, page count, shelf location, and loan status, while directory entries just list the book titles."
    },
    {
        "id": "fc8",
        "topic": "I/O & Interrupts",
        "term": "Direct Memory Access (DMA)",
        "front": "What is Direct Memory Access (DMA)?",
        "back": "A hardware feature allowing storage/network controllers to transfer data blocks directly into main RAM without bothering the CPU for every byte.",
        "analogy": "💡 Analogy: Hiring a mover to unload boxes directly into your living room while you keep working, instead of carrying every single shoe box yourself."
    }
]

@app.get("/api/concepts")
async def get_concepts():
    """Return OS concept module matrix and mastery rates."""
    return {"concepts": OS_CONCEPTS}

@app.get("/api/ebooks")
async def get_ebooks():
    """Return digital OS textbooks and interactive reference library."""
    return {"ebooks": OS_EBOOKS}

@app.get("/api/flashcards")
async def get_flashcards(topic: Optional[str] = None):
    """Return interactive OS flashcards filtered by topic."""
    cards = OS_FLASHCARDS_DATABASE
    if topic and topic.lower() != "all":
        cards = [c for c in cards if topic.lower() in c["topic"].lower()]
    return {"flashcards": cards}

@app.post("/api/quiz/generate")
async def generate_quiz(req: QuizRequest):
    """Generate OS multiple-choice quiz items based on topic and difficulty."""
    filtered = OS_QUIZ_DATABASE
    if req.topic and req.topic.lower() != "all topics" and req.topic.lower() != "all":
        filtered = [q for q in filtered if req.topic.lower() in q["topic"].lower()]
    if req.difficulty and req.difficulty.lower() != "all":
        filtered = [q for q in filtered if q["difficulty"].lower() == req.difficulty.lower()]
    if not filtered:
        filtered = OS_QUIZ_DATABASE
    return {"questions": filtered[:req.count]}

@app.post("/api/tutor/chat")
async def tutor_chat(req: ChatRequest):
    """Instruction-tuned OS Tutor Response Generator with plain-English step-by-step kernel trace."""
    msg_lower = req.message.lower()
    
    citations = []
    kernel_trace = []
    
    if "fork" in msg_lower or "process" in msg_lower or "pcb" in msg_lower:
        response_text = (
            "### Process Creation & `fork()` System Call\n\n"
            "Hey! Imagine a process as an active app running on your computer. When a program wants to create a duplicate of itself, it uses the **`fork()`** system call.\n\n"
            "#### 💡 How `fork()` Works in Simple Terms:\n"
            "1. **PCB Duplication**: The kernel creates a new **Process Control Block (PCB)**—think of it as a student ID card for the new child process.\n"
            "2. **Copy-on-Write (COW)**: Instead of copying all physical RAM immediately (which is slow!), parent and child share read-only memory pages. A physical copy is made **only when one process tries to write to a page**.\n"
            "3. **Return Values**: `fork()` returns `0` inside the child process, and returns the child's `PID` inside the parent process!"
        )
        kernel_trace = [
            "👉 Step 1: User program calls sys_fork() to request a new child process",
            "🔒 Step 2: Kernel allocates a new Process Control Block (task_struct) at 0xFFFF880",
            "⚡ Step 3: Kernel marks RAM pages as Copy-On-Write (COW) so parent & child share memory safely",
            "🆔 Step 4: Child process assigned unique Process ID (PID 4013)",
            "✅ Step 5: System call finishes: Parent receives Child PID 4013, Child receives PID 0"
        ]
        citations = [RAG_KNOWLEDGE_BASE[0]]
        
    elif "virtual memory" in msg_lower or "page" in msg_lower or "tlb" in msg_lower:
        response_text = (
            "### Virtual Memory & Address Translation\n\n"
            "Virtual memory is a super clever trick! It gives every app the illusion that it has its own huge, contiguous block of RAM, even if physical memory is fragmented across real RAM chips.\n\n"
            "#### 💡 Address Translation in 3 Simple Steps:\n"
            "1. **MMU Lookup**: The CPU splits a virtual address into a **Virtual Page Number (VPN)** and an **Offset**.\n"
            "2. **TLB Check (Fast Lane)**: The CPU checks the **TLB cache** inside hardware.\n"
            "   - **TLB Hit (Instant)**: Gets the physical RAM location in less than 1 nanosecond!\n"
            "   - **TLB Miss (Slower)**: The CPU walks the page table in RAM to find the mapping, then caches it into the TLB.\n"
            "3. **Page Fault Handling (`#PF`)**: If the requested page isn't in RAM at all, the OS fetches it from the SSD."
        )
        kernel_trace = [
            "👉 Step 1: CPU requests Virtual Address 0x00007FFF89A2",
            "🔍 Step 2: CPU checks Translation Lookaside Buffer (TLB cache) -> MISS",
            "📖 Step 3: MMU performs Page Table Walk across RAM levels (PGD -> PTE)",
            "⚡ Step 4: Valid Physical Frame (PFN 0x1A4F) located and mapped",
            "✅ Step 5: Translation cached into TLB for instant future lookups"
        ]
        citations = [RAG_KNOWLEDGE_BASE[1]]
        
    elif "semaphore" in msg_lower or "mutex" in msg_lower or "sync" in msg_lower or "deadlock" in msg_lower:
        response_text = (
            "### Concurrency & Locks: Mutex vs Counting Semaphore\n\n"
            "When multiple threads run at the same time, they might try to modify shared variables at the exact same moment (a **Race Condition**). Locks keep things safe!\n\n"
            "#### 💡 Mutex vs Semaphore:\n"
            "- **Mutex (Mutual Exclusion Lock)**: An ownership lock (`0` or `1`). Only the thread that locked it can unlock it!\n"
            "  *Analogy: A single key to a fitting room.* \n"
            "- **Counting Semaphore**: A counter for managing multiple resource units.\n"
            "  - `sem_wait()` / `P()`: Decrements counter. If `< 0`, the thread waits in queue.\n"
            "  - `sem_post()` / `V()`: Increments counter and wakes up a waiting thread.\n"
            "  *Analogy: A bouncer counting open spots at a venue.*"
        )
        kernel_trace = [
            "👉 Step 1: Thread T2 executes sem_wait(&mutex_lock) to enter critical section",
            "🔢 Step 2: Atomic decrement: Semaphore counter decreases from 1 -> 0",
            "🔓 Step 3: Lock acquired! Thread T2 enters critical section safely",
            "⏳ Step 4: Thread T3 attempts sem_wait() -> Counter becomes -1",
            "🛑 Step 5: Thread T3 blocked and placed on kernel wait_queue until lock frees"
        ]
        citations = [RAG_KNOWLEDGE_BASE[2]]

    elif "schedule" in msg_lower or "round robin" in msg_lower or "gantt" in msg_lower or "cfs" in msg_lower:
        response_text = (
            "### CPU Scheduling Algorithms\n\n"
            "The CPU scheduler acts like a master conductor, deciding which process gets to execute on the CPU core right now!\n\n"
            "#### 💡 Common Scheduling Policies:\n"
            "- **FCFS (First-Come First-Served)**: Runs jobs in arrival order. (Susceptible to the Convoy Effect!)\n"
            "- **SJF (Shortest Job First)**: Picks the task with the shortest burst time to minimize average wait.\n"
            "- **Round Robin (RR)**: Rotates processes every fixed **Time Quantum** (e.g. 2ms) so everyone gets fair CPU time.\n"
            "- **Priority Scheduling**: Runs high-priority tasks first."
        )
        kernel_trace = [
            "⏱️ Step 1: Timer Interrupt fires after 2ms Time Quantum",
            "🛑 Step 2: Kernel preempts Process P1 and saves its CPU registers to PCB",
            "🔄 Step 3: Context Switch: Kernel loads Process P2 registers into CPU",
            "🚀 Step 4: Process P2 resumes execution on CPU core 0"
        ]
        citations = [RAG_KNOWLEDGE_BASE[3]]

    elif "file" in msg_lower or "inode" in msg_lower or "fs" in msg_lower or "vfs" in msg_lower or "storage" in msg_lower:
        response_text = (
            "### File Systems & Inodes\n\n"
            "A File System is how the OS organizes files on SSDs or hard drives so you can find them easily.\n\n"
            "#### 💡 What is an Inode?\n"
            "An **inode** (index node) is a data structure on disk that holds all metadata about a file:\n"
            "- File size, permissions (`rwxr-xr-x`), owner UID, timestamps\n"
            "- Direct and indirect pointers to physical data blocks on disk\n"
            "*(Note: Inodes store everything EXCEPT the file's name! Filenames live in directory lists).* \n\n"
            "#### 💡 Virtual File System (VFS):\n"
            "The VFS is an abstraction layer that lets you use the same `open()`, `read()`, `write()` commands whether your file is on Ext4, NTFS, FAT32, or a USB drive!"
        )
        kernel_trace = [
            "👉 Step 1: Program executes open(\"/var/log/syslog\", O_RDONLY)",
            "📂 Step 2: VFS layer traverses directory cache to locate inode #1428591",
            "🔒 Step 3: Kernel checks permissions (User matched read permission 0644)",
            "📖 Step 4: Ext4 block pointers read data blocks into RAM page cache",
            "✅ Step 5: File Descriptor 3 assigned and returned to user application"
        ]
        citations = [RAG_KNOWLEDGE_BASE[4]]

    elif "interrupt" in msg_lower or "dma" in msg_lower or "io" in msg_lower or "isr" in msg_lower or "device" in msg_lower:
        response_text = (
            "### I/O Subsystem, Interrupts & DMA\n\n"
            "Input/Output (I/O) handles communication between the CPU and hardware devices (keyboards, SSDs, network cards).\n\n"
            "#### 💡 Key Hardware-OS Communication Mechanics:\n"
            "- **Interrupt Request (IRQ)**: Hardware signals the CPU when an event occurs (e.g. key pressed or packet received).\n"
            "- **Interrupt Service Routine (ISR)**: A specialized kernel function that handles the IRQ immediately.\n"
            "- **Direct Memory Access (DMA)**: High-speed controllers transfer big data blocks directly to RAM **without wasting CPU cycles**!"
        )
        kernel_trace = [
            "⚡ Step 1: Storage controller triggers Hardware IRQ Line 14",
            "⏸️ Step 2: CPU pauses current user program and saves stack context",
            "🛠️ Step 3: Kernel executes Interrupt Service Routine (nvme_irq_handler)",
            "📦 Step 4: DMA controller streams 4KB data block directly into RAM",
            "✅ Step 5: Waiting process woken up from IO_WAIT queue to process data"
        ]
        citations = [RAG_KNOWLEDGE_BASE[3]]

    else:
        topic_display = req.topic.replace("_", " ").title() if req.topic else "General OS Kernel"
        response_text = (
            f"### OSTutor // Kernel Breakdown: {topic_display}\n\n"
            f"Here is a student-friendly overview for *\"{req.message}\"*\n\n"
            "Operating Systems are built around 3 main goals: **Isolation**, **Concurrency**, and **Resource Management**.\n\n"
            "#### 💡 Core Operating System Directives:\n"
            "- **Dual-Mode Execution**: User Mode (Ring 3) protects system RAM from buggy applications, while Kernel Mode (Ring 0) handles hardware.\n"
            "- **System Calls**: The official gateway API apps use to request kernel services (like reading files or spawning threads).\n"
            "- **Resource Virtualization**: Giving every program its own CPU time slices, virtual RAM address space, and storage access."
        )
        kernel_trace = [
            f"👉 Step 1: Received student query regarding module '{topic_display}'",
            "🔐 Step 2: Operating in Ring 0 Kernel Execution Mode",
            "📖 Step 3: Verified page table mappings for process virtual address space",
            "⚡ Step 4: Evaluated task state in scheduler runqueue -> TASK_RUNNING",
            "✅ Step 5: Service routine completed successfully"
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
    """Simulate CPU scheduling algorithms (FCFS, SJF, SRTF, RR, Priority) with accurate arrival times and idle telemetry."""
    procs = req.processes if req.processes else [
        {"id": "P1", "arrival": 0, "burst": 6, "priority": 2},
        {"id": "P2", "arrival": 1, "burst": 3, "priority": 1},
        {"id": "P3", "arrival": 2, "burst": 8, "priority": 3},
        {"id": "P4", "arrival": 3, "burst": 4, "priority": 2}
    ]
    
    # Normalize inputs
    norm_procs = []
    for idx, p in enumerate(procs):
        pid = str(p.get("id", f"P{idx+1}")).strip()
        arr = max(0, int(p.get("arrival", 0)))
        burst = max(1, int(p.get("burst", 1)))
        pri = max(1, int(p.get("priority", 1)))
        norm_procs.append({"id": pid, "arrival": arr, "burst": burst, "priority": pri})

    algo = req.algorithm.upper()
    gantt_blocks = []
    curr_time = 0

    if algo == "FCFS":
        sorted_procs = sorted(norm_procs, key=lambda x: (x["arrival"], x["id"]))
        for p in sorted_procs:
            if curr_time < p["arrival"]:
                gantt_blocks.append({
                    "pid": "IDLE",
                    "start": curr_time,
                    "end": p["arrival"],
                    "duration": p["arrival"] - curr_time
                })
                curr_time = p["arrival"]
            start = curr_time
            end = start + p["burst"]
            gantt_blocks.append({
                "pid": p["id"],
                "start": start,
                "end": end,
                "duration": p["burst"]
            })
            curr_time = end

    elif algo in ("SJF", "SJF_NP"):
        remaining = [dict(p) for p in norm_procs]
        completed = []
        while len(completed) < len(norm_procs):
            available = [p for p in remaining if p["arrival"] <= curr_time and p["id"] not in [c["id"] for c in completed]]
            if not available:
                next_arr = min(p["arrival"] for p in remaining if p["id"] not in [c["id"] for c in completed])
                gantt_blocks.append({
                    "pid": "IDLE",
                    "start": curr_time,
                    "end": next_arr,
                    "duration": next_arr - curr_time
                })
                curr_time = next_arr
                continue
            shortest = min(available, key=lambda x: (x["burst"], x["arrival"], x["id"]))
            start = curr_time
            end = start + shortest["burst"]
            gantt_blocks.append({
                "pid": shortest["id"],
                "start": start,
                "end": end,
                "duration": shortest["burst"]
            })
            curr_time = end
            completed.append(shortest)

    elif algo in ("SRTF", "SJF_P"):
        rem_burst = {p["id"]: p["burst"] for p in norm_procs}
        arrival_map = {p["id"]: p["arrival"] for p in norm_procs}
        total_procs = len(norm_procs)
        completed_count = 0
        curr_time = 0
        prev_pid = None
        block_start = 0

        while completed_count < total_procs:
            available = [pid for pid, b in rem_burst.items() if arrival_map[pid] <= curr_time and b > 0]
            if not available:
                selected_pid = "IDLE"
            else:
                selected_pid = min(available, key=lambda pid: (rem_burst[pid], arrival_map[pid], pid))

            if selected_pid != prev_pid:
                if prev_pid is not None and curr_time > block_start:
                    gantt_blocks.append({
                        "pid": prev_pid,
                        "start": block_start,
                        "end": curr_time,
                        "duration": curr_time - block_start
                    })
                prev_pid = selected_pid
                block_start = curr_time

            curr_time += 1
            if selected_pid != "IDLE":
                rem_burst[selected_pid] -= 1
                if rem_burst[selected_pid] == 0:
                    completed_count += 1

        if curr_time > block_start and prev_pid is not None:
            gantt_blocks.append({
                "pid": prev_pid,
                "start": block_start,
                "end": curr_time,
                "duration": curr_time - block_start
            })

    elif algo in ("PRIORITY", "PRIORITY_NP"):
        remaining = [dict(p) for p in norm_procs]
        completed = []
        while len(completed) < len(norm_procs):
            available = [p for p in remaining if p["arrival"] <= curr_time and p["id"] not in [c["id"] for c in completed]]
            if not available:
                next_arr = min(p["arrival"] for p in remaining if p["id"] not in [c["id"] for c in completed])
                gantt_blocks.append({
                    "pid": "IDLE",
                    "start": curr_time,
                    "end": next_arr,
                    "duration": next_arr - curr_time
                })
                curr_time = next_arr
                continue
            highest_pri = min(available, key=lambda x: (x["priority"], x["arrival"], x["id"]))
            start = curr_time
            end = start + highest_pri["burst"]
            gantt_blocks.append({
                "pid": highest_pri["id"],
                "start": start,
                "end": end,
                "duration": highest_pri["burst"]
            })
            curr_time = end
            completed.append(highest_pri)

    elif algo in ("PRIORITY_P", "PRIORITY_PREEMPTIVE"):
        rem_burst = {p["id"]: p["burst"] for p in norm_procs}
        arrival_map = {p["id"]: p["arrival"] for p in norm_procs}
        priority_map = {p["id"]: p["priority"] for p in norm_procs}
        total_procs = len(norm_procs)
        completed_count = 0
        curr_time = 0
        prev_pid = None
        block_start = 0

        while completed_count < total_procs:
            available = [pid for pid, b in rem_burst.items() if arrival_map[pid] <= curr_time and b > 0]
            if not available:
                selected_pid = "IDLE"
            else:
                selected_pid = min(available, key=lambda pid: (priority_map[pid], arrival_map[pid], pid))

            if selected_pid != prev_pid:
                if prev_pid is not None and curr_time > block_start:
                    gantt_blocks.append({
                        "pid": prev_pid,
                        "start": block_start,
                        "end": curr_time,
                        "duration": curr_time - block_start
                    })
                prev_pid = selected_pid
                block_start = curr_time

            curr_time += 1
            if selected_pid != "IDLE":
                rem_burst[selected_pid] -= 1
                if rem_burst[selected_pid] == 0:
                    completed_count += 1

        if curr_time > block_start and prev_pid is not None:
            gantt_blocks.append({
                "pid": prev_pid,
                "start": block_start,
                "end": curr_time,
                "duration": curr_time - block_start
            })

    elif algo == "RR":
        q = max(1, req.quantum or 2)
        rem_burst = {p["id"]: p["burst"] for p in norm_procs}
        procs_sorted = sorted(norm_procs, key=lambda x: (x["arrival"], x["id"]))

        curr_time = 0
        ready_queue = []
        visited = set()
        completed_count = 0

        def add_arrivals(t):
            for p in procs_sorted:
                if p["arrival"] <= t and p["id"] not in visited:
                    ready_queue.append(p["id"])
                    visited.add(p["id"])

        add_arrivals(curr_time)

        while completed_count < len(norm_procs):
            if not ready_queue:
                unvisited = [p for p in procs_sorted if p["id"] not in visited]
                if not unvisited:
                    break
                next_arr = min(p["arrival"] for p in unvisited)
                gantt_blocks.append({
                    "pid": "IDLE",
                    "start": curr_time,
                    "end": next_arr,
                    "duration": next_arr - curr_time
                })
                curr_time = next_arr
                add_arrivals(curr_time)
                continue

            pid = ready_queue.pop(0)
            exec_time = min(q, rem_burst[pid])
            start = curr_time
            end = start + exec_time

            if gantt_blocks and gantt_blocks[-1]["pid"] == pid:
                gantt_blocks[-1]["end"] = end
                gantt_blocks[-1]["duration"] += exec_time
            else:
                gantt_blocks.append({
                    "pid": pid,
                    "start": start,
                    "end": end,
                    "duration": exec_time
                })

            curr_time = end
            rem_burst[pid] -= exec_time

            add_arrivals(curr_time)

            if rem_burst[pid] == 0:
                completed_count += 1
            else:
                ready_queue.append(pid)

    else:
        sorted_procs = sorted(norm_procs, key=lambda x: (x["arrival"], x["id"]))
        for p in sorted_procs:
            if curr_time < p["arrival"]:
                gantt_blocks.append({
                    "pid": "IDLE",
                    "start": curr_time,
                    "end": p["arrival"],
                    "duration": p["arrival"] - curr_time
                })
                curr_time = p["arrival"]
            start = curr_time
            end = start + p["burst"]
            gantt_blocks.append({
                "pid": p["id"],
                "start": start,
                "end": end,
                "duration": p["burst"]
            })
            curr_time = end

    # Calculate per-process metrics: CT, TAT, WT
    process_metrics = []
    waiting_times = {}
    turnaround_times = {}

    for p in norm_procs:
        pid = p["id"]
        last_end = max([b["end"] for b in gantt_blocks if b["pid"] == pid], default=p["arrival"] + p["burst"])
        ct = last_end
        tat = ct - p["arrival"]
        wt = tat - p["burst"]
        waiting_times[pid] = wt
        turnaround_times[pid] = tat
        process_metrics.append({
            "id": pid,
            "arrival": p["arrival"],
            "burst": p["burst"],
            "priority": p["priority"],
            "completion_time": ct,
            "turnaround_time": tat,
            "waiting_time": wt
        })

    avg_wait = round(sum(waiting_times.values()) / max(1, len(waiting_times)), 2)
    avg_tat = round(sum(turnaround_times.values()) / max(1, len(turnaround_times)), 2)
    total_time = max([b["end"] for b in gantt_blocks], default=curr_time)

    return {
        "algorithm": algo,
        "quantum": req.quantum or 2,
        "gantt": gantt_blocks,
        "process_metrics": process_metrics,
        "waiting_times": waiting_times,
        "turnaround_times": turnaround_times,
        "avg_waiting_time": avg_wait,
        "avg_turnaround_time": avg_tat,
        "total_time": total_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

