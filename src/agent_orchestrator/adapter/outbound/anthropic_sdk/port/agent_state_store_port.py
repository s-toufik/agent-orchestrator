from typing import Protocol, runtime_checkable

from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState


@runtime_checkable
class AgentStateStorePort(Protocol):
    async def load(self, session_id: str) -> AgentState | None: ...

    async def save(self, agent_state: AgentState) -> None: ...
