import asyncio
from typing import Any

from agent.domain.enum.agent_message_status import MessageStreamType
from agent.domain.model.agent_message_stream import AgentMessageStream


class SSEQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentMessageStream] = asyncio.Queue()

    @property
    def queue(self) -> asyncio.Queue[AgentMessageStream]:
        return self._queue

    async def token(self, value: str) -> None:
        await self._queue.put(AgentMessageStream(type=MessageStreamType.TOKEN, content=value))

    async def final(self, value: str, metadata: dict[str, Any] | None = None) -> None:
        await self._queue.put(
            AgentMessageStream(type=MessageStreamType.FINAL, content=value, metadata=metadata or {})
        )

    async def error(self, value: str) -> None:
        await self._queue.put(AgentMessageStream(type=MessageStreamType.ERROR, content=value))

    async def complete(self) -> None:
        await self._queue.put(AgentMessageStream(type=MessageStreamType.COMPLETE, content=""))
