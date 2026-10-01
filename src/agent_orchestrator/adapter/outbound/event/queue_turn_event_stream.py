import asyncio
from collections.abc import AsyncIterator

from agent_orchestrator.domain.event.turn_event import TurnEvent


class QueueTurnEventStream:
    """One request's events, handed from the running turn to the HTTP response."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()

    async def publish(self, event: TurnEvent) -> None:
        await self._queue.put(event)

    async def complete(self) -> None:
        await self._queue.put(None)

    async def __aiter__(self) -> AsyncIterator[TurnEvent]:
        while (event := await self._queue.get()) is not None:
            yield event
