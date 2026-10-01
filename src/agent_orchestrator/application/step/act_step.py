from agent_orchestrator.application.port.outbound.actor import Actor
from agent_orchestrator.application.port.outbound.tool_catalog import ToolCatalog
from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class ActStep:
    step = Step.ACT

    def __init__(self, actor: Actor, catalog: ToolCatalog) -> None:
        self._actor = actor
        self._catalog = catalog

    async def run(self, conversation: Conversation, turn: Turn) -> None:
        turn.drafted(await self._actor.act(conversation, turn, self._catalog.tools()))
