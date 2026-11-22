import argparse
from pathlib import Path
import json
from src.benchmark import load_triviaqa_subset, format_question_single
from src.llm_client import TAMUChatClient
from src.utils import append_to_json
from src.debate import Debate
from src.agent_config import create_agents
from src.analysis import summarize_results

from src.utils import reformat_string

def run_experiment(n_questions=100, repeats=3, seed=2026,
                   output_dir = "data/task2"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_triviaqa_subset(n_questions, seed=seed)
    all_results_path = output_dir / "results.jsonl"
    completed = set()
    if all_results_path.exists():
        lines = all_results_path.read_text(encoding='utf-8').splitlines()
        for line in lines:
            if line.strip():
                completed.add(json.loads(line)["run_id"])

    client = TAMUChatClient()
    for i, question in enumerate(questions):
        # Each question single
        prompt = format_question_single(question)
        for rep in range(repeats):
            run_id = f"{question['id']}|single|{rep}"
            if run_id in completed:
                continue
            response = client.chat_with_usage(messages=[
                {'role': 'system',
                 'content': "Answer questions carefully and independently."},
                 {'role': 'user', "content": prompt}
            ])
            # print(f"Response: {response}")

            record = {
                "run_id": run_id,
                "question_id": question['id'],
                "condition": "single agent",
                "round": 0,
                "repeat": rep,
                "gold": question["answers"],
                "is_correct": reformat_string(response) in question["answers"],
                "response": response,
            }
            append_to_json(all_results_path, record)

        # Each question run debate
        for n_agents in (2, 3):
            for rep in range(repeats):
                run_id = f"{question['id']}|{n_agents}agents|3rounds|{rep}"

                expected_ids = {
                    f"{run_id}|round{r}" for r in range(1, 4)
                }

                if expected_ids.issubset(completed):
                    continue

                debate = Debate(
                    question=question["question"],
                    n_rounds=3,
                    agent_configs=create_agents(n_agents),
                    q_aliases=question["answers"]
                )

                result = debate.run()
                transcript_dir = output_dir / "transcripts"
                transcript_dir.mkdir(parents=True, exist_ok=True)

                transcript_path = (
                    transcript_dir /
                    f"{question['id']}_{n_agents}agents_rep{rep}.json"
                )

                Debate.save_result(result, transcript_path)
                
                majorities = result.majority_by_round

                # Save only missing rounds
                for round_number in range(1, 4):
                    round_id = f"{run_id}|round{round_number}"

                    if round_id in completed:
                        continue

                    majority = majorities[round_number]

                    debate_result = {
                        "run_id": round_id,
                        "question_id": question["id"],
                        "condition": (
                            f"{n_agents} agents, {round_number} round"
                            + ("" if round_number == 1 else "s")
                        ),
                        "round": round_number,
                        "repeat": rep,
                        "prediction": majority,
                        "gold": question["answers"],
                        "is_correct": (
                            reformat_string(majority) in question["answers"]
                        ),
                        "majority_answer": majority
                    }

                    append_to_json(all_results_path, debate_result)
                    completed.add(round_id)

        print(f"Finished question {i + 1}/{len(questions)}: {question["question"]}")

    summary = summarize_results(str(all_results_path), str(output_dir))
    print(summary.to_string(index=False))
    summary.to_csv(output_dir / "results_summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", '-q', type=int, default=100)
    parser.add_argument("--repeats", '-rep', type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", '-o', default="data/task2")

    args = parser.parse_args()
    run_experiment(args.questions, args.repeats, args.seed, args.output)

    