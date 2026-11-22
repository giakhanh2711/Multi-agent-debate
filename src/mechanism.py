from src.debate import Debate

class StructuredDebate(Debate):

    def get_message(self, agent_idx, previous_answers, round_number):
        config = self.agent_configs[agent_idx]
        system_prompt = (
            config.system_prompt
            + "\nPrioritize factual accuracy over agreement. Distinguish evidence "
            "from speculation and do not invent sources."
        )
        if round_number == 0:
            instruction = (
                "Answer independently. Briefly explain your reasoning and end " \
                "with exactly: Final answer: <short answer>."
            )
            context = ""
        else:
            my_answer = previous_answers[agent_idx]
            others = []
            for j in range(len(previous_answers)):
                if j != agent_idx:
                    others.append(f"{self.agent_configs[j].display_name} said: {previous_answers[j]}")

            context = (
                f"Your previous answer: {my_answer}\n\nOther agents' previous answers:\n"
                + ("\n".join(others) if others else "(No other agents)")
                + "\n\n"
            )

            if round_number == 1:
                instruction = (
                    "ARGUMENT ROUND: Present the strongest evidence and reasoning " \
                    "supporting your answer. Identify what would make it reliable. " \
                    "End with exactly: Final answer: <short answer>."
                )
            elif round_number == 2:
                instruction = (
                    "CRITIQUE ROUND: Critically evaluate the evidence and reasoning " \
                    "presented by the other agents. Identify unsupported claims, " \
                    "possible errors, and alternative interpretations. Do not agree " \
                    "without justification. End with exactly: Final answer: <short answer>."
                )

            else:
                instruction = (
                    "FINAL REVISION ROUND: Consider the arguments and critiques. " \
                    "Choose the answer best supported by the available reasoning. " \
                    "End with exactly: Final answer: <short answer>."
                )

        user_prompt = f"Original question: {self.question}\n\n{context}{instruction}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
            