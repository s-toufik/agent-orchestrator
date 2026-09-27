import asyncio
from collections.abc import AsyncIterator

from agent_orchestrator.domain.enum.agent_message_status import MessageStreamType
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


class SSEQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentMessageStream] = asyncio.Queue()

    async def publish(self, event: AgentMessageStream) -> None:
        await self._queue.put(event)

    async def complete(self) -> None:
        await self._queue.put(AgentMessageStream.complete())

    async def __aiter__(self) -> AsyncIterator[AgentMessageStream]:
        while True:
            event: AgentMessageStream = await self._queue.get()
            yield event
            if event.type is MessageStreamType.COMPLETE:
                return
