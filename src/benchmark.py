import json
import random
from datasets import load_dataset
import os
from pathlib import Path
from typing import List

def format_question_single(question:dict):
    prompt = "Try to answer it as correct as you can using your knowledge. Put your final answer in the format: Final answer: <answer>."
    return f"The question is {question["question"]}.\n" + prompt


def load_triviaqa_subset(n_questions=100, seed=2026) -> List[dict]:
    filename = "data/triviaqa_100.json"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data =  json.load(f)[:n_questions]
            print(f"number of questions: {len(data)}")
        return data

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        "unfiltered.nocontext",
        split='validation',
        streaming = True
    )

    shuffled = dataset.shuffle(seed=seed, buffer_size=1000).take(n_questions + 20)

    candidates = []

    for idx, item in enumerate(shuffled):
        if len(candidates) >= n_questions:
            break

        question = item['question'].strip()
        answer = item["answer"]
        aliases = answer.get("normalized_aliases", [])

        if question and aliases:
            candidates.append({
                "id": item.get("question_id", str(idx)),
                "question": question,
                "answers": list(set(aliases))
            })

    with open(filename, 'w', encoding='utf-8') as f:
        print(f"number of question: {len(candidates)}")
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    return candidates


def choose_subset_question(questions, n_questions=25, seed=2026):
    return random.Random(seed).sample(questions, n_questions)


def save_questions(questions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)