from agent_orchestrator.adapter.outbound.anthropic_sdk.port.agent_state_store_port import (
    AgentStateStorePort,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState


class IngestStep:
    def __init__(self, states: AgentStateStorePort) -> None:
        self._states = states

    async def run(self, session_id: str) -> AgentState:
        return await self._states.load(session_id) or AgentState(session_id=session_id)
