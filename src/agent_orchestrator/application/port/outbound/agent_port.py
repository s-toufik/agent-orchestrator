from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest


@runtime_checkable
class AgentPort(Protocol):
    def stream(self, request: AgentRequest) -> AsyncIterator[AgentMessageStream]: ...
