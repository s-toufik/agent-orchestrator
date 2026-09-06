from collections.abc import Hashable
from typing import Any

from langgraph.graph import END, StateGraph

from agent.adapter.outbound.langgraph.node.execution_node import ExecutorNode
from agent.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent.adapter.outbound.langgraph.node.final_node import FinalNode
from agent.adapter.outbound.langgraph.node.memory_node import MemoryNode
from agent.adapter.outbound.langgraph.node.planner_node import PlannerNode
from agent.adapter.outbound.langgraph.node.reflection_node import ReflectionNode
from agent.adapter.outbound.langgraph.node.router_node import RouterNode
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState


class AgentGraph:
    def __init__(
        self,
        planner: PlannerNode,
        router: RouterNode,
        executor: ExecutorNode,
        memory: MemoryNode,
        reflection: ReflectionNode,
        feedback: FeedbackNode,
        final: FinalNode,
    ) -> None:
        self._planner = planner
        self._router = router
        self._executor = executor
        self._memory = memory
        self._reflection = reflection
        self._feedback = feedback
        self._final = final

    # noinspection PyTypeChecker
    def build(self, checkpointer: Any = None) -> Any:
        graph = StateGraph(GraphState)  # ty: ignore[invalid-argument-type]
        graph.add_node("planner", self._planner)
        graph.add_node("executor", self._executor)
        graph.add_node("memory", self._memory)
        graph.add_node("memory_pre_reflection", self._memory)
        graph.add_node("reflection", self._reflection)
        graph.add_node("feedback", self._feedback)
        graph.add_node("final", self._final)

        graph.set_entry_point("memory")

        # Only planner and reflection still need runtime branching.
        planner_targets: dict[Hashable, str] = {
            "executor": "executor",
            "reflection": "memory_pre_reflection",  # planner's "no tool" branch
            "final": "final",
        }
        reflection_targets: dict[Hashable, str] = {
            "feedback": "feedback",
            "final": "final",
        }

        graph.add_conditional_edges("planner", self._router, planner_targets)
        graph.add_conditional_edges("reflection", self._router, reflection_targets)

        # Deterministic: every path into memory has a fixed destination.
        graph.add_edge("memory", "planner")
        graph.add_edge("memory_pre_reflection", "reflection")
        graph.add_edge("executor", "memory")
        graph.add_edge("feedback", "memory")
        graph.add_edge("final", END)

        return graph.compile(checkpointer=checkpointer)
