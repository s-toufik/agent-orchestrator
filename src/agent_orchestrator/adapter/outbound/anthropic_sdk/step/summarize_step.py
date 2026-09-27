from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState

MAX_EXCHANGES: int = 10


class SummarizeStep:
    # The model's own context is compacted by the CLI; this bounds the history we keep.
    def __init__(self, max_exchanges: int = MAX_EXCHANGES) -> None:
        self._max_exchanges = max_exchanges

    def run(self, agent_state: AgentState) -> None:
        agent_state.exchanges = agent_state.exchanges[-self._max_exchanges :]
