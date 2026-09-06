from collections.abc import Awaitable, Callable
from typing import Any

from pycraftcore.logger.port import Logger

from agent.adapter.outbound.langgraph.agent_graph import AgentGraph
from agent.adapter.outbound.langgraph.node.execution_node import ExecutorNode
from agent.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent.adapter.outbound.langgraph.node.final_node import FinalNode
from agent.adapter.outbound.langgraph.node.memory_node import MemoryNode
from agent.adapter.outbound.langgraph.node.planner_node import PlannerNode
from agent.adapter.outbound.langgraph.node.reflection_node import ReflectionNode
from agent.adapter.outbound.langgraph.node.router_node import RouterNode
from agent.adapter.outbound.langgraph.schema.agent_state import AgentState
from agent.adapter.outbound.langgraph.schema.graph_state import GraphState
from agent.adapter.outbound.langgraph.service.prompt_service import PromptService
from agent.adapter.outbound.langgraph.service.state_serialization import pack_state
from agent.adapter.outbound.llm.schema import ModelParameters
from agent.application.port.outbound.tool_port import ToolRegistryPort


def build_agent(
    planner_llm: Any,
    reflection_llm: Any,
    tool_registry: ToolRegistryPort,
    model_parameters: ModelParameters,
    logger: Logger,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    checkpointer: Any = None,
) -> tuple[Any, GraphState]:
    use_streaming: bool = model_parameters.use_streaming

    graph = AgentGraph(
        planner=PlannerNode(
            llm=planner_llm,
            tool_registry=tool_registry,
            prompt_service=PromptService(),
            on_token=on_token if use_streaming else None,
        ),
        router=RouterNode(),
        executor=ExecutorNode(tool_registry, logger),
        memory=MemoryNode(max_context_tokens=model_parameters.max_context_tokens),
        reflection=ReflectionNode(reflection_llm, PromptService()),
        feedback=FeedbackNode(PromptService()),
        final=FinalNode(),
    ).build(checkpointer=checkpointer)

    initial_state = AgentState(max_iterations=model_parameters.max_iterations)
    return graph, pack_state(initial_state)
