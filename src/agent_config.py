from dataclasses import dataclass
import src.configs as configs


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str = (
        "You are a careful, independent question-answering agent. " \
        "Prioritize correctness over consensus. Evaluate evidence before revising."
    )
    temperature: float = 1.0
    name: str = None
    role: str = None

    @property
    def display_name(self):
        return self.name



def create_agents(n: int, temperature: float=1.0):
    agents = []
    for i in range(n):
        agent = AgentConfig(system_prompt = configs.COOPERATIVE_PROMPT,
                            temperature=temperature,
                            role="cooperative",
                            name=f"Agent {i + 1}")
        agents.append(agent)
    return agents


def create_profile_agents(strategy1: str, strategy2: str, temperature: 1.0):
    agents = []
    for i, strategy in enumerate((strategy1, strategy2)):
        if strategy not in configs.STRATEGY_PROMPTS:
            raise ValueError(f"Unknown strategy {strategy}")
        agent = AgentConfig(system_prompt="You are a careful factual question-answering agent. " + configs.STRATEGY_PROMPTS[strategy],
                            temperature=temperature,
                            role=strategy,
                            name=f"Agent {i + 1} ({strategy})")
        agents.append(agent)

    return agents