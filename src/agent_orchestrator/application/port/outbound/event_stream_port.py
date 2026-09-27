from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream


@runtime_checkable
class EventStreamPort(Protocol):
    async def publish(self, event: AgentMessageStream) -> None: ...

    async def complete(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[AgentMessageStream]: ...
