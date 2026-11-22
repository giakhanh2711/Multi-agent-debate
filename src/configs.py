TAMU_BASE_URL = "https://chat.tamu.ai/api"
SONNET_MODEL = "protected.Claude Sonnet 4.5"


COOPERATIVE_PROMPT = (
    "Consider all answers carefully and try to give the correct answer. " \
    "Do no agree just in order to be agreeable."
)


STRATEGIES = [
    "Cooperative",
    "Stubborn",
    "DevilsAdvocate",
    "Analytical",
]


N_ACTIONS = len(STRATEGIES)


STRATEGY_PROMPTS = {
    "Cooperative": COOPERATIVE_PROMPT,
    "Stubborn": "Defend your initial answer unless presented with overwhelming evidence.",
    "DevilsAdvocate": "Argue against the current majority position, even if you agree with it.",
    "Analytical": "Focus on logical reasoning and evidence. Point out flaws in other agents' reasoning."
}