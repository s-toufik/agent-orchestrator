from collections.abc import Hashable
from typing import Any

from langgraph.graph import END, StateGraph

from agent_orchestrator.adapter.outbound.langgraph.agent_router import (
    ACT,
    CLARIFY,
    FEEDBACK,
    FINALIZE,
    PLAN,
    REFLECT,
    TOOLS,
    AgentRouter,
)
from agent_orchestrator.adapter.outbound.langgraph.node.act_node import ActNode
from agent_orchestrator.adapter.outbound.langgraph.node.clarify_node import ClarifyNode
from agent_orchestrator.adapter.outbound.langgraph.node.context_node import ContextNode
from agent_orchestrator.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent_orchestrator.adapter.outbound.langgraph.node.finalize_node import FinalizeNode
from agent_orchestrator.adapter.outbound.langgraph.node.ingest_node import IngestNode
from agent_orchestrator.adapter.outbound.langgraph.node.plan_node import PlanNode
from agent_orchestrator.adapter.outbound.langgraph.node.reflect_node import ReflectNode
from agent_orchestrator.adapter.outbound.langgraph.node.summarize_node import SummarizeNode
from agent_orchestrator.adapter.outbound.langgraph.node.tools_node import ToolsNode
from agent_orchestrator.adapter.outbound.langgraph.port.node_port import NodePort
from agent_orchestrator.adapter.outbound.langgraph.store.graph_state import GraphState

INGEST: str = "ingest"
CONTEXT: str = "context"
SUMMARIZE: str = "summarize"


class AgentGraph:
    def __init__(
        self,
        ingest: IngestNode,
        context: ContextNode,
        clarify: ClarifyNode,
        plan: PlanNode,
        act: ActNode,
        tools: ToolsNode,
        reflect: ReflectNode,
        feedback: FeedbackNode,
        finalize: FinalizeNode,
        summarize: SummarizeNode,
    ) -> None:
        self._nodes: dict[str, NodePort] = {
            INGEST: ingest,
            CONTEXT: context,
            CLARIFY: clarify,
            PLAN: plan,
            ACT: act,
            TOOLS: tools,
            REFLECT: reflect,
            FEEDBACK: feedback,
            FINALIZE: finalize,
            SUMMARIZE: summarize,
        }

    # noinspection PyTypeChecker
    def build(self, checkpointer: Any = None) -> Any:
        graph = StateGraph(GraphState)  # ty: ignore[invalid-argument-type]
        for name, node in self._nodes.items():
            graph.add_node(name, node)

        graph.set_entry_point(INGEST)
        graph.add_edge(INGEST, CONTEXT)

        graph.add_conditional_edges(
            CONTEXT, AgentRouter.after_context, _targets(CLARIFY, PLAN, ACT)
        )
        graph.add_conditional_edges(ACT, AgentRouter.after_act, _targets(TOOLS, REFLECT, FINALIZE))
        graph.add_conditional_edges(
            REFLECT, AgentRouter.after_reflect, _targets(FEEDBACK, FINALIZE)
        )

        graph.add_edge(CLARIFY, FINALIZE)
        graph.add_edge(PLAN, FINALIZE)
        graph.add_edge(TOOLS, ACT)
        graph.add_edge(FEEDBACK, ACT)
        graph.add_edge(FINALIZE, SUMMARIZE)
        graph.add_edge(SUMMARIZE, END)

        return graph.compile(checkpointer=checkpointer)


def _targets(*names: str) -> dict[Hashable, str]:
    return {name: name for name in names}
