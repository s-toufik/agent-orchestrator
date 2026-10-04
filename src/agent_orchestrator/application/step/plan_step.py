from agent_orchestrator.application.port.outbound.planner import Planner
from agent_orchestrator.application.port.outbound.tool_catalog import ToolCatalog
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class PlanStep:
    step = Step.PLAN

    def __init__(self, planner: Planner, catalog: ToolCatalog) -> None:
        self._planner = planner
        self._catalog = catalog

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        plan = await self._planner.plan(conversation, turn, self._catalog.tools())
        conversation.submit(turn, plan)
