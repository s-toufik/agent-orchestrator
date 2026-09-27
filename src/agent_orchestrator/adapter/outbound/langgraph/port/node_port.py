from typing import Protocol, runtime_checkable

from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


@runtime_checkable
class NodePort(Protocol):
    async def __call__(self, state: GraphState) -> GraphState: ...
