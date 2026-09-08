"""Ground-truth evaluation dataset for V2.

Each entry is a question against documents/*.txt plus the source file(s)
that SHOULD be retrieved to answer it. `relevant_sources: []` marks a
question whose answer is deliberately NOT in the corpus -- these test
whether retrieval correctly returns nothing useful, rather than confidently
pointing at the wrong file.

This is hand-written, not generated, on purpose: the entire point of a
ground-truth set is that a human (who knows the material) decided what
"correct" means, independent of whatever the system currently retrieves.

Update this list as documents/ grows. Keep questions single-file where
possible -- that makes it easy to tell WHICH file the retriever confused
whenever a question fails.
"""

QUESTIONS = [
    # processes_and_threads.txt
    {
        "question": "What is a Process Control Block and what information does it store?",
        "relevant_sources": ["processes_and_threads.txt"],
    },
    {
        "question": "What are the five states a process moves through during its lifetime?",
        "relevant_sources": ["processes_and_threads.txt"],
    },
    {
        "question": "What is the difference between the many-to-one and one-to-one threading models?",
        "relevant_sources": ["processes_and_threads.txt"],
    },
    {
        "question": "What is the difference between fork() and exec() when a process is created?",
        "relevant_sources": ["processes_and_threads.txt"],
    },

    # cpu_scheduling.txt
    {
        "question": "Why does Shortest-Job-First scheduling minimize average waiting time?",
        "relevant_sources": ["cpu_scheduling.txt"],
    },
    {
        "question": "What is the convoy effect in FCFS scheduling?",
        "relevant_sources": ["cpu_scheduling.txt"],
    },
    {
        "question": "What is starvation in priority scheduling, and how is it typically solved?",
        "relevant_sources": ["cpu_scheduling.txt"],
    },
    {
        "question": "What happens to Round Robin scheduling if the time quantum is too large or too small?",
        "relevant_sources": ["cpu_scheduling.txt"],
    },
    {
        "question": "What is the difference between multilevel queue and multilevel feedback queue scheduling?",
        "relevant_sources": ["cpu_scheduling.txt"],
    },

    # memory_management.txt
    {
        "question": "What is Belady's Anomaly?",
        "relevant_sources": ["memory_management.txt"],
    },
    {
        "question": "Why is the Optimal page replacement algorithm not used in real operating systems?",
        "relevant_sources": ["memory_management.txt"],
    },
    {
        "question": "What is the purpose of a Translation Lookaside Buffer?",
        "relevant_sources": ["memory_management.txt"],
    },
    {
        "question": "What is thrashing and what causes it?",
        "relevant_sources": ["memory_management.txt"],
    },
    {
        "question": "What is the difference between paging and segmentation?",
        "relevant_sources": ["memory_management.txt"],
    },

    # deadlocks.txt
    {
        "question": "What are the four necessary conditions for a deadlock to occur?",
        "relevant_sources": ["deadlocks.txt"],
    },
    {
        "question": "Explain how the Banker's Algorithm decides whether to grant a resource request.",
        "relevant_sources": ["deadlocks.txt"],
    },
    {
        "question": "What is the most commonly used deadlock prevention technique in practice?",
        "relevant_sources": ["deadlocks.txt"],
    },

    # synchronization_and_disk_scheduling.txt
    {
        "question": "What three requirements must a solution to the critical-section problem satisfy?",
        "relevant_sources": ["synchronization_and_disk_scheduling.txt"],
    },
    {
        "question": "What is the difference between a counting semaphore and a binary semaphore?",
        "relevant_sources": ["synchronization_and_disk_scheduling.txt"],
    },
    {
        "question": "How does the SCAN disk scheduling algorithm work?",
        "relevant_sources": ["synchronization_and_disk_scheduling.txt"],
    },
    {
        "question": "Why can the naive Dining Philosophers solution deadlock?",
        "relevant_sources": ["synchronization_and_disk_scheduling.txt"],
    },

    # Not in corpus -- correct retrieval behavior is to NOT confidently point
    # at an unrelated file just because it shares vocabulary (e.g. "scheduling").
    {
        "question": "What is the Completely Fair Scheduler (CFS) used in the Linux kernel?",
        "relevant_sources": [],
    },
    {
        "question": "Explain how ext4 filesystem journaling works.",
        "relevant_sources": [],
    },
    {
        "question": "What is a hypervisor and how does it enable virtualization?",
        "relevant_sources": [],
    },
]
