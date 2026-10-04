from agent_orchestrator.adapter.outbound.langgraph.node.agent_node import AgentNode
from agent_orchestrator.domain.workflow.step import Step


class RunToolsNode(AgentNode):
    step = Step.RUN_TOOLS
