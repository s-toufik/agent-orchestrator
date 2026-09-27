from agent_orchestrator.adapter.outbound.langgraph.store.agent_state import AgentState
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState


def pack_state(state: AgentState) -> GraphState:
    return {"state": state.model_dump(mode="json")}


def unpack_state(graph_state: GraphState) -> AgentState:
    return AgentState.model_validate(graph_state["state"])
