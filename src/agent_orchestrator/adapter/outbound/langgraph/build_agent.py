from typing import Any

from langchain_core.language_models import BaseChatModel
from pycraftcore.logger.port import Logger

from agent_orchestrator.adapter.outbound.langgraph.agent_graph import AgentGraph
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
from agent_orchestrator.adapter.outbound.langgraph.schema.agent_limits import AgentLimits
from agent_orchestrator.adapter.outbound.langgraph.service.context_window import ContextWindow
from agent_orchestrator.adapter.outbound.llm.schema import ModelParameters
from agent_orchestrator.adapter.outbound.tool.tool_port import ToolRegistryPort


def build_agent(
    act_llm: BaseChatModel,
    context_llm: BaseChatModel,
    plan_llm: BaseChatModel,
    reflection_llm: BaseChatModel,
    summary_llm: BaseChatModel,
    tool_registry: ToolRegistryPort,
    model_parameters: ModelParameters,
    logger: Logger,
    checkpointer: Any = None,
) -> Any:
    window = ContextWindow(max_tokens=model_parameters.max_context_tokens)
    limits = AgentLimits(
        max_iterations=model_parameters.max_iterations,
        max_retries=model_parameters.max_reflection_retries,
    )

    return AgentGraph(
        ingest=IngestNode(limits),
        context=ContextNode(context_llm, window, logger),
        clarify=ClarifyNode(),
        plan=PlanNode(plan_llm, tool_registry, window, logger),
        act=ActNode(act_llm, tool_registry, window, logger),
        tools=ToolsNode(tool_registry, logger),
        reflect=ReflectNode(reflection_llm, logger),
        feedback=FeedbackNode(logger),
        finalize=FinalizeNode(stream_answer=model_parameters.use_streaming),
        summarize=SummarizeNode(
            summary_llm, logger, trigger_tokens=model_parameters.max_context_tokens // 2
        ),
    ).build(checkpointer=checkpointer)
