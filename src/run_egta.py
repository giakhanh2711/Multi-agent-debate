import argparse
import json
from pathlib import Path
from typing import final
import numpy as np

from src.agent_config import create_profile_agents
from src.benchmark import load_triviaqa_subset, choose_subset_question, save_questions
from src.debate import Debate, DebateResult
from src.llm_client import TAMUChatClient
from support_enumeration.support_enumeration import SupportEnumeration
from regret_matching.regret_matching_plus import RegretMatchingPlus
from src.analysis import summarize_payoffs
import src.utils as utils
import src.configs as configs

def payoff_from_result(result: DebateResult, aliases):
    answer = result.final_answer
    if answer == "no consensus":
        return 0.0
    return 1.0 if utils.is_correct(answer, aliases) else -1.0


def unordered_profiles(strategies):
    profiles = []
    for i in range(len(strategies)):
        for j in range(i, len(strategies)):
            profiles.append((strategies[i], strategies[j]))

    return profiles


def estimate_payoffs(args):
    questions = load_triviaqa_subset(args.questions, args.seed)
    subset = choose_subset_question(questions, args.subset, args.seed)
    out = Path(args.output)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    save_questions(subset, out / "25_questions.json")
    results_path = out / "payoffs.jsonl"
    completed = {r["run_id"] for r in utils.read_jsonl(results_path)}

    profiles = unordered_profiles(configs.STRATEGIES)
    print(f"Profiles: {profiles}\n Profiles len: {len(profiles)}")
    for s1, s2 in profiles:
        for q in subset:
            qid = q["id"]
            for rep in range(args.repeats):
                run_id = f"{s1}|{s2}|{qid}|rep{rep}"
                if run_id in completed:
                    continue
                debate = Debate(q["question"], args.n_rounds,
                                create_profile_agents(s1, s2, args.temperature))

                result = debate.run()
                utility = payoff_from_result(result, q["answers"])
                data = {
                    "run_id": run_id,
                    "strategy1": s1,
                    "strategy2": s2,
                    "question_id": qid,
                    "repeat": rep,
                    "payoff": utility,
                    "final_answer": result.final_answer,
                    "consensus": result.final_consensus,
                    "is_correct": utils.is_correct(result.final_answer, q["answers"]),
                    "answers": q["answers"]
                }
                utils.append_to_json(results_path, data)
                completed.add(run_id)
                transcript_db_file = out / "transcripts" / f"{run_id.replace('|', '_')}.json"
                metadata = {
                    "strategy1": s1,
                    "strategy2": s2,
                    "payoff": utility
                }
                Debate.save_result(result, transcript_db_file, metadata=metadata)
        print(f"Finished profile: {s1} vs {s2}")

    matrix, ci_df = summarize_payoffs(input_jsonl=results_path,
                                      strategies=configs.STRATEGIES,
                                      output_dir=out / "payoff_matrix.csv")
    print(matrix, ci_df)
    return matrix, ci_df


def run_rm_plus(payoff_matrix, T=5000, seed=42):
    payoff1 = payoff_matrix[:, :, 0]
    payoff2 = payoff_matrix[:, :, 1].T

    n_actions1 = payoff1.shape[0]
    n_actions2 = payoff2.shape[0]

    p1 = RegretMatchingPlus(n_actions1, payoff1)
    p2 = RegretMatchingPlus(n_actions2, payoff2)

    joint_action_counts = np.zeros(
        (n_actions1, n_actions2), dtype=int
    )

    for t in range(T):
        a1 = p1.make_decision()
        a2 = p2.make_decision()

        joint_action_counts[a1, a2] += 1

        p1.each_round(a2)
        p2.each_round(a1)

    joint_dist = joint_action_counts / joint_action_counts.sum()

    cce_gap1 = p1.cal_cce_violation(joint_dist)
    cce_gap2 = p2.cal_cce_violation(joint_dist.T)

    return {
        "joint_distribution": joint_dist.tolist(),
        "cce_gap": float(max(cce_gap1, cce_gap2)),
        "average_regret_1": float(p1.average_regret()),
        "average_regret_2": float(p2.average_regret()),
        "time_averaged_strategy_1":
            p1.time_averaged_strategy().tolist(),
        "time_averaged_strategy_2":
            p2.time_averaged_strategy().tolist(),
    }

def analyze_matrix(matrix, output):
    payoff = matrix[:, :, 0]
    se = SupportEnumeration(configs.STRATEGIES)
    equilibria = se.support_enumeration(matrix)
    ne_data = []
    for p, q in equilibria:
        expected = float(p @ payoff @ q)
        ne_data.append({
            "player1_strategy": p.tolist(),
            "player2_strategy": q.tolist(),
            "expected_common_payoff": expected,
        })
    dominance = {"strictly_dominated_by_pure_strategy": {}}
    for i, strategy in enumerate(configs.STRATEGIES):
        dominators = [
            configs.STRATEGIES[k] for k in range(len(configs.STRATEGIES))
            if k != i and np.all(payoff[k, :] > payoff[i, :] + 1e-9)
        ]
        dominance["strictly_dominated_by_pure_strategy"][strategy] = dominators
    result = {
        "strategies": configs.STRATEGIES,
        "nash_equilibria": ne_data,
        "rm_plus_cce_approximation": run_rm_plus(matrix, 5000),
        "dominance": dominance,
        "note": "Dominance test checks pure-strategy strict domination only.",
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", '-q', type=int, default=100)
    p.add_argument("--subset", type=int, default=25)
    p.add_argument("--repeats", '-r', type=int, default=5)
    p.add_argument("--n_rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--output", '-o', default="data/task3")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if not args.analyze_only:
        matrix, ci_df = estimate_payoffs(args)
    else:
        matrix = np.load(out / "payoff_matrix.npy")
    analyze_matrix(matrix, out / "equilibrium.json")


if __name__ == "__main__":
    main()
                
