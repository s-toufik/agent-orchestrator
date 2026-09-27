import asyncio
from collections.abc import AsyncIterator

from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class EventChannel:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentMessageStream | None] = asyncio.Queue()

    async def emit(self, event: AgentMessageStream) -> None:
        await self._queue.put(event)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def drain(self) -> AsyncIterator[AgentMessageStream]:
        while (event := await self._queue.get()) is not None:
            yield event
