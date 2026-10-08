import json
import random

# Real OS Concepts
CONCEPTS = {
    "Processes & Threads": [
        ("Process Control Block (PCB)", "Kernel data structure tracking PID, state, registers, and memory for a process.", "Like a student ID and transcript."),
        ("Context Switch", "The process of saving the state of the old process and loading the saved state for the new process.", "Like pausing a video game and loading another save file."),
        ("Zombie Process", "A process that has terminated but still has an entry in the process table because its parent hasn't read its exit status.", "Like a worker who quit but is still on the payroll until the boss signs off."),
        ("Orphan Process", "A process whose parent has terminated, usually adopted by the init process (PID 1).", "Like a child whose parents left, so the state takes custody."),
        ("Thread", "The smallest sequence of programmed instructions managed independently by a scheduler.", "Like a single lane in a multi-lane highway."),
        ("User-level vs Kernel-level Threads", "User threads are managed by user libraries; kernel threads are managed directly by the OS.", "Like hiring your own private security vs using the city police."),
        ("fork()", "System call that creates a new process by duplicating the calling process.", "Like cloning a sheep."),
        ("exec()", "System call that replaces the current process image with a new program.", "Like a brain transplant for a process."),
        ("wait()", "System call used by a parent process to wait for a child process to finish.", "Like a parent waiting outside school for their child."),
        ("Multiprogramming", "Keeping multiple programs in main memory at the same time to maximize CPU utilization.", "Like keeping multiple tabs open so you can switch instantly when one loads."),
        ("Time-sharing", "Logical extension of multiprogramming where CPU switches jobs so frequently that users can interact with each job.", "Like a chef cooking 5 meals at once by taking turns."),
        ("Inter-Process Communication (IPC)", "Mechanisms like pipes, sockets, or shared memory allowing processes to communicate.", "Like two different departments passing memos."),
        ("Copy-On-Write (COW)", "Optimization where parent and child share memory pages until one writes to it, at which point a copy is made.", "Like sharing a read-only Google Doc until someone clicks 'Make a copy' to edit."),
        ("Process States", "New, Ready, Running, Waiting/Blocked, Terminated.", "Like the stages of a job application: Applied, Interviewing, Hired."),
        ("Dispatcher", "Module that gives control of the CPU to the process selected by the short-term scheduler.", "Like the usher who physically guides you to your seat.")
    ],
    "Virtual Memory": [
        ("Virtual Address Space", "Logical view of how a process is stored in memory, providing the illusion of a large, contiguous array.", "Like having a massive virtual desk, even if your real desk is tiny."),
        ("Paging", "Memory management scheme that eliminates the need for contiguous allocation of physical memory.", "Like breaking a long book into pages that can be stored anywhere in a library."),
        ("Page Table", "Data structure used to map virtual pages to physical frames.", "Like the library's index catalog mapping book chapters to shelf locations."),
        ("Translation Lookaside Buffer (TLB)", "Hardware cache inside the MMU that stores recent virtual-to-physical address mappings.", "Like a speed-dial list on your phone."),
        ("Page Fault", "An interrupt that occurs when a program attempts to access a page that is mapped in address space, but not loaded in physical memory.", "Like trying to read a book chapter that is in the basement, requiring you to fetch it first."),
        ("Demand Paging", "Bringing a page into memory only when it is needed.", "Like ordering groceries only when you are about to cook the meal."),
        ("Thrashing", "When the system spends more time paging (swapping pages) than executing instructions.", "Like spending more time finding your tools than actually building the house."),
        ("Working Set Model", "Based on the assumption of locality, it defines the set of pages a process is currently using.", "Like the tools you currently have out on your workbench."),
        ("Page Replacement Algorithm", "Algorithm deciding which memory page to page out (e.g., LRU, FIFO, Clock).", "Like deciding which old clothes to donate when your closet is full."),
        ("LRU (Least Recently Used)", "Replaces the page that has not been used for the longest period of time.", "Like throwing away the spice you haven't cooked with in 3 years."),
        ("Belady's Anomaly", "Phenomenon where increasing the number of page frames results in an increase in the number of page faults (happens in FIFO).", "Like buying a bigger closet but somehow losing more clothes."),
        ("Segmentation", "Memory management scheme that supports the user view of memory (logical segments like code, data, stack).", "Like organizing a house by rooms (kitchen, bedroom) rather than equal-sized square blocks."),
        ("Swapping", "Moving entire processes between main memory and a backing store (disk).", "Like packing up your entire desk and putting it in storage to make room for someone else."),
        ("Memory Management Unit (MMU)", "Hardware device that maps virtual to physical address at run time.", "Like an automatic GPS router."),
        ("Dirty Bit", "A bit in the page table indicating if the page has been modified since it was loaded.", "Like a 'needs to be saved' asterisk next to a filename.")
    ],
    "CPU Scheduling": [
        ("CPU Burst vs I/O Burst", "Processes alternate between executing instructions (CPU burst) and waiting for I/O.", "Like alternating between writing code and waiting for the compiler."),
        ("Preemptive Scheduling", "OS can interrupt a currently running process and move it to the Ready queue.", "Like a boss pulling you off a task to handle an urgent fire."),
        ("Non-Preemptive Scheduling", "Once a process gets the CPU, it keeps it until it terminates or switches to the waiting state.", "Like a meeting you can't leave until it's finished."),
        ("First-Come, First-Served (FCFS)", "The process that requests the CPU first is allocated the CPU first.", "Like standing in line at a grocery store checkout."),
        ("Convoy Effect", "Short processes get stuck waiting for one long CPU-bound process to finish in FCFS.", "Like a line of sports cars stuck behind a slow tractor on a one-lane road."),
        ("Shortest Job First (SJF)", "Associates with each process the length of its next CPU burst, picking the shortest.", "Like the express checkout lane for 10 items or fewer."),
        ("Shortest Remaining Time First (SRTF)", "Preemptive version of SJF.", "Like letting someone with 1 item cut in front of you at the register."),
        ("Round Robin (RR)", "Each process gets a small unit of CPU time (time quantum), then is preempted and added to the end of the queue.", "Like passing a controller around so everyone gets a 5-minute turn playing a game."),
        ("Time Quantum", "The fixed time slice allocated to a process in Round Robin.", "Like a 5-minute egg timer."),
        ("Priority Scheduling", "A priority is associated with each process, and the CPU is allocated to the highest priority.", "Like VIP boarding at an airport."),
        ("Starvation", "A low priority process never gets to run because higher priority processes keep arriving.", "Like never getting to ask a question because louder people keep interrupting."),
        ("Aging", "Technique of gradually increasing the priority of processes that wait in the system for a long time (solves starvation).", "Like gaining loyalty points the longer you wait in line."),
        ("Multilevel Queue Scheduling", "Partitions the ready queue into several separate queues (e.g., foreground and background) with their own algorithms.", "Like having separate lines for first-class, economy, and standby."),
        ("Multilevel Feedback Queue", "Allows a process to move between queues; separates processes according to the characteristics of their CPU bursts.", "Like getting demoted to a slower lane if you take too long."),
        ("Turnaround Time", "The interval from the time of submission of a process to the time of completion.", "Like the total time from ordering pizza to eating it.")
    ],
    "Synchronization": [
        ("Race Condition", "When multiple threads access shared data concurrently and the outcome depends on the order of execution.", "Like two people trying to withdraw the last $100 from a joint bank account at the exact same time."),
        ("Critical Section", "A segment of code where shared resources are accessed and must not be concurrently executed by more than one thread.", "Like a single-occupancy public restroom."),
        ("Mutual Exclusion", "If a process is executing in its critical section, no other processes can be executing in theirs.", "Like only one person holding the talking stick."),
        ("Mutex Lock", "A binary lock used to protect critical sections. The thread that acquires it must release it.", "Like a physical key to the bathroom: you take it, use it, return it."),
        ("Semaphore", "An integer variable used to control access to a common resource by multiple processes.", "Like a bouncer holding a clicker, allowing exactly N people into a club."),
        ("Counting Semaphore", "A semaphore that can range over an unrestricted domain (used for managing N identical resources).", "Like a parking garage counter showing available spots."),
        ("Binary Semaphore", "A semaphore whose value can only be 0 or 1, essentially acting like a Mutex.", "Like a simple vacant/occupied sign."),
        ("Deadlock", "Two or more processes are waiting indefinitely for an event that can be caused by only one of the waiting processes.", "Like four cars at a 4-way stop each waiting for the car on their right to go first."),
        ("Conditions for Deadlock", "Mutual exclusion, Hold and wait, No preemption, Circular wait.", "The four ingredients required to bake a deadlock disaster."),
        ("Banker's Algorithm", "A deadlock avoidance algorithm that simulates resource allocation to ensure the system remains in a safe state.", "Like a bank ensuring it has enough cash to satisfy its largest customer's maximum possible withdrawal."),
        ("Livelock", "Processes constantly change state in response to each other without making progress.", "Like two people in a hallway constantly stepping to the same side to let the other pass."),
        ("Condition Variable", "A synchronization primitive that allows threads to wait until a particular condition occurs.", "Like a pager from a restaurant that buzzes when your table is ready."),
        ("Monitor", "A high-level synchronization construct that encapsulates shared variables and the procedures that operate on them.", "Like a class in object-oriented programming where all methods are automatically synchronized."),
        ("Producer-Consumer Problem", "Classic sync problem where producers add data to a buffer and consumers remove it safely.", "Like a chef putting burgers on a finite tray and a waiter taking them away."),
        ("Dining Philosophers Problem", "Classic sync problem illustrating the challenges of allocating multiple resources to multiple processes.", "Like needing two chopsticks to eat, but only having one between each person.")
    ],
    "File Systems": [
        ("File Control Block / Inode", "Storage structure containing information about a file, including ownership, permissions, and location of file contents.", "Like the metadata on a library catalog card."),
        ("Directory Structure", "The organization of files in a file system (flat, two-level, tree, acyclic-graph).", "Like a system of folders and subfolders on your computer."),
        ("Contiguous Allocation", "Each file occupies a set of contiguous blocks on the disk.", "Like reserving an entire block of seats at a movie theater."),
        ("Linked Allocation", "Each file is a linked list of disk blocks, which can be scattered anywhere.", "Like a scavenger hunt where each clue tells you where the next clue is."),
        ("Indexed Allocation", "Brings all the pointers together into one location: the index block.", "Like a table of contents that tells you exactly what page each chapter starts on."),
        ("Virtual File System (VFS)", "An abstraction layer that allows the OS to support multiple file system types transparently.", "Like a universal translator for different hard drive formats."),
        ("Journaling File System", "Keeps a log (journal) of changes not yet committed to the main file system to prevent corruption on crash.", "Like taking notes on what you're going to write in ink before you actually write it."),
        ("Superblock", "A block of metadata that describes the characteristics of a file system (size, status, layout).", "Like the blueprint and status report for the entire city infrastructure."),
        ("Hard Link", "A directory entry that points directly to the inode of a file.", "Like giving a person a second legal name; both names point to the exact same person."),
        ("Symbolic (Soft) Link", "A file whose data contains the path to another file.", "Like a desktop shortcut or a sign pointing to another building."),
        ("Mounting", "The process of making a file system available to the OS at a specific directory (mount point).", "Like plugging a new external hard drive into a specific USB port folder."),
        ("RAID", "Redundant Array of Independent Disks; combines multiple drives into a single logical unit for performance or redundancy.", "Like a team of horses pulling a carriage instead of just one."),
        ("RAID 0 (Striping)", "Splits data evenly across two or more disks with no parity (high performance, no redundancy).", "Like splitting a heavy load between two bags to run faster, but if one bag rips, you lose half your stuff."),
        ("RAID 1 (Mirroring)", "Creates an exact copy of a set of data on two or more disks.", "Like printing two copies of your essay just in case you lose one."),
        ("RAID 5", "Block-level striping with distributed parity, requiring at least 3 disks. Tolerates 1 disk failure.", "Like a magic trick where if one card is lost, you can perfectly guess it using the remaining cards.")
    ],
    "I/O & Interrupts": [
        ("Device Driver", "Software that provides a high-level interface to a specific hardware device.", "Like a translator who converts OS language into specific hardware language."),
        ("Interrupt", "A signal emitted by hardware or software indicating an event that needs immediate attention.", "Like a ringing doorbell interrupting your reading."),
        ("Interrupt Service Routine (ISR)", "A software routine that hardware invokes in response to an interrupt.", "Like the script you follow when you answer the doorbell (e.g., greeting the mailman)."),
        ("Interrupt Vector Table", "An array of pointers to ISRs, indexed by interrupt number.", "Like a phone directory telling you which department handles which type of emergency."),
        ("Direct Memory Access (DMA)", "Allows hardware subsystems to access main system memory independently of the CPU.", "Like hiring a mover to unload boxes directly into the living room without you having to carry each box."),
        ("Memory-Mapped I/O", "Uses the same address space to address both memory and I/O devices.", "Like assigning a specific room in your house to act as a mailbox."),
        ("Polling (Programmed I/O)", "The CPU repeatedly checks the status of a device to see if it is ready.", "Like a kid in the back seat repeatedly asking 'Are we there yet?'"),
        ("Block Device", "A device that moves data in structured blocks (e.g., hard drives, SSDs).", "Like a shipping container that only moves large, fixed-size boxes."),
        ("Character Device", "A device that transmits data character by character (e.g., keyboards, mice).", "Like a garden hose streaming water continuously."),
        ("Spooling", "Putting jobs in a buffer, a special area in memory or on a disk where a device can access them when it is ready.", "Like a printer queue holding documents until the printer is free."),
        ("Buffering", "A memory area that stores data being transferred between two devices or a device and an application.", "Like a waiting room at a doctor's office absorbing the mismatch in arrival times."),
        ("Caching", "Storing a copy of data in a faster memory to speed up subsequent accesses.", "Like keeping your most frequently used tools on your belt instead of in the shed."),
        ("Disk Scheduling", "Algorithms to determine the order in which disk I/O requests will be processed.", "Like planning the most efficient route for a delivery truck."),
        ("SCAN Algorithm (Elevator)", "The disk arm starts at one end, moves toward the other, servicing requests, then reverses direction.", "Like an elevator picking up everyone on the way up, then everyone on the way down."),
        ("C-SCAN Algorithm", "Like SCAN, but when it reaches the end, it returns to the beginning without servicing requests on the return trip.", "Like a typewriter carriage returning quickly to start a new line.")
    ]
}

flashcards = []
fc_id = 1
for topic, items in CONCEPTS.items():
    for term, back, analogy in items:
        flashcards.append({
            "id": f"fc{fc_id}",
            "topic": topic,
            "term": term,
            "front": f"What is {term}?",
            "back": back,
            "analogy": analogy
        })
        fc_id += 1

quizzes = []
q_id = 1
for topic, items in CONCEPTS.items():
    # We have 15 concepts per topic. We will generate 20 questions.
    # To get 20, we loop 20 times and pick concepts.
    for i in range(20):
        # Pick the actual concept for this question
        concept = items[i % 15]
        term = concept[0]
        desc = concept[1]
        
        # Determine difficulty
        diff = "easy" if i < 7 else ("medium" if i < 14 else "hard")
        
        # Pick 3 other random terms to be wrong options
        others = random.sample([c[0] for c in items if c[0] != term], 3)
        
        # Construct the question
        if i % 3 == 0:
            question = f"Which of the following best describes '{term}'?"
            correct_option = desc
            wrong_options = [
                f"It is a {others[0]} optimization technique.",
                f"It refers to the mechanism used by {others[1]}.",
                f"An obsolete feature replaced by {others[2]}."
            ]
        elif i % 3 == 1:
            question = f"What is the primary purpose of {term}?"
            correct_option = desc
            wrong_options = [
                f"To manage {others[0]} automatically.",
                f"To simulate the behavior of {others[1]}.",
                f"To prevent issues related to {others[2]}."
            ]
        else:
            question = f"Which OS concept is defined as: '{desc}'?"
            correct_option = term
            wrong_options = others

        # Shuffle options
        options = wrong_options + [correct_option]
        random.shuffle(options)
        answer_index = options.index(correct_option)
        
        # Format options with A, B, C, D
        letters = ["A. ", "B. ", "C. ", "D. "]
        formatted_options = [letters[idx] + opt for idx, opt in enumerate(options)]
        
        quizzes.append({
            "id": f"q{q_id}",
            "topic": topic,
            "difficulty": diff,
            "question": question,
            "options": formatted_options,
            "answer_index": answer_index,
            "explanation": f"The correct concept is {term}, which is closely associated with this behavior."
        })
        q_id += 1

with open('app/data/flashcards.json', 'w') as f:
    json.dump(flashcards, f, indent=2)

with open('app/data/quizzes.json', 'w') as f:
    json.dump(quizzes, f, indent=2)

print(f"Generated {len(flashcards)} flashcards and {len(quizzes)} quiz questions.")
