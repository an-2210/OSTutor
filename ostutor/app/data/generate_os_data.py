import json
import os

topics = [
    "Processes & Threads",
    "Virtual Memory",
    "CPU Scheduling",
    "Synchronization",
    "File Systems",
    "I/O & Interrupts"
]

flashcards = []
fc_id = 1
for topic in topics:
    for i in range(1, 16):
        flashcards.append({
            "id": f"fc{fc_id}",
            "topic": topic,
            "term": f"{topic} Term {i}",
            "front": f"What is a key concept #{i} in {topic}?",
            "back": f"This is the explanation for concept #{i} in {topic}, covering essential details.",
            "analogy": f"💡 Analogy: Think of it like a real-world scenario #{i} for {topic}."
        })
        fc_id += 1

quizzes = []
q_id = 1
for topic in topics:
    for i in range(1, 21):
        diff = "easy" if i <= 7 else ("medium" if i <= 14 else "hard")
        quizzes.append({
            "id": f"q{q_id}",
            "topic": topic,
            "difficulty": diff,
            "question": f"Question #{i} about {topic}: Which of the following is true?",
            "options": [
                f"A. Incorrect option 1 for {topic}",
                f"B. Correct statement regarding {topic} #{i}",
                f"C. Incorrect option 2 for {topic}",
                f"D. Incorrect option 3 for {topic}"
            ],
            "answer_index": 1,
            "explanation": f"The correct answer is B because it accurately describes the mechanism #{i} in {topic}."
        })
        q_id += 1

with open('app/data/flashcards.json', 'w') as f:
    json.dump(flashcards, f, indent=2)

with open('app/data/quizzes.json', 'w') as f:
    json.dump(quizzes, f, indent=2)

print("Generated flashcards.json and quizzes.json")
