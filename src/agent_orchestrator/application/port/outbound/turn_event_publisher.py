from typing import Protocol

from agent_orchestrator.domain.event.turn_event import TurnEvent


class TurnEventPublisher(Protocol):
    async def publish(self, event: TurnEvent) -> None: ...
