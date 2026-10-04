from collections.abc import AsyncIterator
from typing import Protocol

from agent_orchestrator.domain.event.turn_event import TurnEvent


class TurnEventStream(Protocol):
    async def publish(self, event: TurnEvent) -> None: ...

    async def complete(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[TurnEvent]: ...
