import argparse
from pathlib import Path

from src.agent_config import create_profile_agents
from src.benchmark import load_triviaqa_subset, choose_subset_question, save_questions
from src.debate import Debate
from src.mechanism import StructuredDebate
from src.run_egta import payoff_from_result
from src.analysis import summarize_by_groups
import src.utils as utils

PROFILES = [
    ("Cooperative", "Cooperative"),
    ("Analytical", "Analytical")
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=25)
    parser.add_argument("--repeats", '-r', type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", '-o', type=str, default="data/task4")

    args = parser.parse_args()

    questions = load_triviaqa_subset(100, args.seed)
    subset = choose_subset_question(questions, args.subset, args.seed)
    output_dir = Path(args.output)
    (output_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    save_questions(subset, output_dir / "task4_questions_25.json")
    results_path = output_dir / "results.jsonl"
    completed = {r["run_id"] for r in utils.read_jsonl(results_path)}

    for s1, s2 in PROFILES:
        for protocol, debate_type in (("baseline", Debate), ("structured", StructuredDebate)):
            for q in subset:
                for rep in range(args.repeats):
                    run_id = f"{protocol}|{s1}|{s2}|{q['id']}|rep{rep}"
                    if run_id in completed:
                        continue
                    debate = debate_type(question=q["question"],
                                         n_rounds=args.n_rounds,
                                         agent_configs=create_profile_agents(s1, s2, args.temperature))
                    result = debate.run()
                    answer = result.final_answer
                    is_correct = int(utils.is_correct(answer, q["answers"]))
                    utility = payoff_from_result(result, q["answers"])
                    saved_data = {
                        "run_id": run_id,
                        "protocol": protocol,
                        "strategy1": s1,
                        "strategy2": s2,
                        "question_id": str(q["id"]),
                        "repeat": rep,
                        "is_correct": is_correct,
                        "payoff": utility,
                        "final_answer": answer,
                        "answers": q["answers"]
                    }
                    utils.append_to_json(results_path, saved_data)
                    completed.add(run_id)
                    debate_result_file = output_dir / "transcripts" / f"{run_id.replace('|', '_')}.json"
                    debate_metadata = {
                        "protocol": protocol,
                        "strategy1": s1,
                        "strategy2": s2,
                        "is_correct": is_correct,
                        "payoff": utility
                    }
                    Debate.save_result(result=result, path=debate_result_file, metadata=debate_metadata)

        print(f"Finished profile {s1} vs {s2}") 

    rows = utils.read_jsonl(results_path)
    summary = summarize_by_groups(records=rows,
                                  group_keys=["strategy1", "strategy2", "protocol"],
                                  metrics=["is_correct", "payoff"],
                                  cluster_key="question_id",
                                  n_bootstrap=5000,
                                  seed=args.seed)
    summary = summary.rename(columns={
        "mean_is_correct": "acc",
        "is_correct_ci95_low": "acc_ci95_low",
        "is_correct_ci95_high": "acc_ci95_high"
    })

    summary["profile"] = summary["strategy1"] + "-" + summary["strategy2"]

    no_consensus_rate = []
    for _, row in summary.iterrows():
        count = 0
        for r in rows:
            if r["strategy1"] == row["strategy1"] and \
                r["strategy2"] == row["strategy2"] and \
                r["protocol"] == row["protocol"]:
                count += r["final_answer"].strip().lower() == "no consensus"

        no_consensus_rate.append(count / row["n_debates"])
    
    summary["no_consensus_rate"] = no_consensus_rate

    summary.to_csv(output_dir / "comparison.csv", index=False)

    summary[summary["protocol"] == "structured"].to_csv(output_dir / "modified_payoff_cells.csv", index=False)

    print("\nTask 4 summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    output_dir = Path("data/task4")
    results_path = output_dir / "results.jsonl"
    rows = utils.read_jsonl(results_path)
    summary = summarize_by_groups(records=rows,
                                  group_keys=["strategy1", "strategy2", "protocol"],
                                  metrics=["is_correct", "payoff"],
                                  cluster_key="question_id",
                                  n_bootstrap=5000,
                                  seed=2026)
    summary = summary.rename(columns={
        "mean_is_correct": "acc",
        "is_correct_ci95_low": "acc_ci95_low",
        "is_correct_ci95_high": "acc_ci95_high"
    })
    
    summary["profile"] = summary["strategy1"] + "-" + summary["strategy2"]
    
    no_consensus_rate = []
    for _, row in summary.iterrows():
        count = 0
        for r in rows:
            if r["strategy1"] == row["strategy1"] and \
                r["strategy2"] == row["strategy2"] and \
                r["protocol"] == row["protocol"]:
                count += r["final_answer"].strip().lower() == "no consensus"
    
        no_consensus_rate.append(count / row["n_debates"])
    
    summary["no_consensus_rate"] = no_consensus_rate
    
    summary.to_csv(output_dir / "comparison.csv", index=False)
    
    summary[summary["protocol"] == "structured"].to_csv(output_dir / "modified_payoff_cells.csv", index=False)
    
    print("\nTask 4 summary:")
    print(summary.to_string(index=False))
    # main()


    
