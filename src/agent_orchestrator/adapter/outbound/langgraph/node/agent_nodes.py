from collections.abc import Sequence

from agent_orchestrator.adapter.outbound.langgraph.node.act_node import ActNode
from agent_orchestrator.adapter.outbound.langgraph.node.agent_node import AgentNode
from agent_orchestrator.adapter.outbound.langgraph.node.clarify_node import ClarifyNode
from agent_orchestrator.adapter.outbound.langgraph.node.feedback_node import FeedbackNode
from agent_orchestrator.adapter.outbound.langgraph.node.finish_node import FinishNode
from agent_orchestrator.adapter.outbound.langgraph.node.plan_node import PlanNode
from agent_orchestrator.adapter.outbound.langgraph.node.review_node import ReviewNode
from agent_orchestrator.adapter.outbound.langgraph.node.run_tools_node import RunToolsNode
from agent_orchestrator.adapter.outbound.langgraph.node.summarize_node import SummarizeNode
from agent_orchestrator.adapter.outbound.langgraph.node.understand_node import UnderstandNode
from agent_orchestrator.adapter.outbound.langgraph.state.turn_state_codec import TurnStateCodec
from agent_orchestrator.application.step.step_handler import StepHandler
from agent_orchestrator.domain.workflow.step import Step

# The node class that runs each step.
NODE_TYPES: tuple[type[AgentNode], ...] = (
    UnderstandNode,
    PlanNode,
    ClarifyNode,
    ActNode,
    RunToolsNode,
    ReviewNode,
    FeedbackNode,
    FinishNode,
    SummarizeNode,
)


def agent_nodes(handlers: Sequence[StepHandler], codec: TurnStateCodec) -> list[AgentNode]:
    by_step: dict[Step, StepHandler] = {handler.step: handler for handler in handlers}
    return [node_type(by_step[node_type.step], codec) for node_type in NODE_TYPES]
