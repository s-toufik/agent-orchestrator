from collections.abc import AsyncIterator
from typing import Protocol

from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.turn.turn import Turn


class WorkflowRunner(Protocol):
    def run(self, conversation_id: str, turn: Turn) -> AsyncIterator[TurnEvent]: ...
