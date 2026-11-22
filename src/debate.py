from src.agent_config import AgentConfig
from typing import List
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from src.llm_client import TAMUChatClient, extract_final_answer
from src.utils import reformat_string

@dataclass
class DebateResult:
    question: str
    n_rounds: int
    agents_answers: list
    majority_by_round: list
    consensus_by_round: list
    final_answer: str
    final_consensus: bool
    transcript: list

def majority_vote(answers: List[str]):
    normalized_answers = [reformat_string(answer) for answer in answers]
    counts = {}
    originals = {}
    for x, y in zip(answers, normalized_answers):
        if not y:
            continue
        counts[y] = counts.get(y, 0) + 1
        originals.setdefault(y, x)

    if not counts:
        return "no consensus"

    win_answer, count = max(counts.items(), key = lambda x: x[1])
    if count > len(answers) / 2: # More than half of the team agree
        return extract_final_answer(originals[win_answer])
    return "no consensus"

def unanimous_correct(
    answers: List[str],
    gold_aliases: List[str]
) -> bool:
    if not answers:
        return False

    normalized_aliases = {
        reformat_string(alias)
        for alias in gold_aliases
        if alias
    }

    normalized_answers = [
        reformat_string(answer)
        for answer in answers
    ]

    # Every agent must provide an answer matching a gold alias.
    return all(
        answer and answer in normalized_aliases
        for answer in normalized_answers
    )

def unanimous_consensus(answers: List[str]) -> bool:
    normalized = [reformat_string(a) for a in answers]
    return bool(normalized) and bool(normalized[0]) and all(
        a == normalized[0] for a in normalized
    )


class Debate:
    def __init__(self, question: str, n_rounds: int, agent_configs: List[AgentConfig], q_aliases:List=None):
        """
        Initialize a debate with len(agent_configs) agents, R rounds, and per-agent configurations.
        """
        self.client = TAMUChatClient()
        self.question = question
        self.n_rounds = n_rounds
        self.agent_configs = agent_configs
        self.q_aliases = q_aliases
        # self.answer_extractor = answer_extractor


    def get_message(self, agent_idx: int, previous_answers: list, round_number: int):
        config = self.agent_configs[agent_idx]
        role = f"Your assigned role is: {config.role}." if config.role else ""
        if round_number == 0:
            user_prompt = (
                f"Question: \n{self.question}\n\n"
                "Answer independently. Try to give a concise answer, and end with exactly one line"
                "in the format: Final answer: <answer>"
            )
        else:
            my_answer = previous_answers[agent_idx]
            others = [
                f"Agent {j + 1} ({self.agent_configs[j].display_name}) answered:\n"
                f"{previous_answers[j]}" for j in range(len(previous_answers)) if j != agent_idx
            ]
            other_str = "\n\n".join(others) if others else "(No other agents.)"
            user_prompt = (
                f"Original question:\n{self.question}\n\n"
                f"Your previous answer:\n{my_answer}\n\n"
                f"Other agents' previous answers:\n"
                f"{other_str}\n\n"
                "Evaluate the other answers critically. Identify useful evidence and possiblel errors. " \
                "You may keep or revise your answer, but do not change it merely to conform to the group. " \
                "End with exactly one line in the format: Final answer: <answer>"
            )

        return [
            {"role": "system", "content": config.system_prompt + role},
            {"role": "user", "content": user_prompt}
        ]

    
    def run(self) -> DebateResult:
        transcript = []
        answers = [None] * len(self.agent_configs)
        all_round_answers = []
        majority_by_round = []
        consensus_by_round = []

        for r in range(self.n_rounds + 1):
            previous_snapshot = list(answers)
            current_answers = []
            for i, config in enumerate(self.agent_configs):
                messages = self.get_message(i, previous_snapshot, r)
                answer = self.client.chat_with_usage(messages, temperature=1.0)
                current_answers.append(answer)
                transcript.append({
                    "round": r,
                    "agent_idx": i,
                    "agent_name": config.display_name,
                    "role": config.role,
                    "system_prompt": config.system_prompt,
                    "temperature": config.temperature,
                    "prompt_messages": messages,
                    "response": answer,
                    "extracted_answer": answer
                })

            answers = current_answers
            all_round_answers.append(list(current_answers))
            majority_by_round.append(majority_vote(answers))
            consensus_by_round.append(unanimous_consensus(answers))

        final_answer = majority_by_round[-1]
        result = DebateResult(
            question=self.question,
            n_rounds=self.n_rounds,
            agents_answers=all_round_answers,
            majority_by_round=majority_by_round,
            final_answer=final_answer,
            final_consensus=consensus_by_round[-1],
            transcript=transcript,
            consensus_by_round=consensus_by_round
        )
        return result


    @staticmethod
    def save_result(result: DebateResult, path:str | Path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(result)
        if metadata:
            data["metadata"] = metadata
            
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)