from collections.abc import Sequence
from typing import Any

from langgraph.graph import START, StateGraph

from agent_orchestrator.adapter.outbound.langgraph.graph.policy_router import (
    PolicyRouter,
    node_name,
)
from agent_orchestrator.adapter.outbound.langgraph.node.agent_node import AgentNode
from agent_orchestrator.adapter.outbound.langgraph.state.graph_state import GraphState
from agent_orchestrator.domain.workflow.turn_policy import FIRST_STEP, TurnPolicy


class AgentGraphBuilder:
    def __init__(
        self, nodes: Sequence[AgentNode], policy: TurnPolicy, router: PolicyRouter
    ) -> None:
        self._nodes = nodes
        self._policy = policy
        self._router = router

    def build(self, checkpointer: Any = None) -> Any:
        graph = StateGraph(GraphState)  # ty: ignore[invalid-argument-type]
        for node in self._nodes:
            graph.add_node(node.name, node)
        graph.add_edge(START, node_name(FIRST_STEP))
        for node in self._nodes:
            targets = [node_name(step) for step in self._policy.targets(node.step)]
            if len(targets) == 1:
                graph.add_edge(node.name, targets[0])
            else:
                graph.add_conditional_edges(node.name, self._router.after(node.step), targets)
        return graph.compile(checkpointer=checkpointer)
