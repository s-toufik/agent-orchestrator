from agent_orchestrator.adapter.outbound.langgraph.node.agent_node import AgentNode
from agent_orchestrator.domain.workflow.step import Step


class SummarizeNode(AgentNode):
    step = Step.SUMMARIZE
