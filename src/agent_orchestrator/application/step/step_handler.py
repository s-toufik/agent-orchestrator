from typing import Protocol

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class StepHandler(Protocol):
    @property
    def step(self) -> Step: ...

    async def run(self, conversation: Conversation, turn: Turn) -> None: ...
