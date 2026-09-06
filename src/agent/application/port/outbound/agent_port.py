from typing import Protocol, runtime_checkable

from agent.domain.model.agent_message import AgentMessage
from agent.domain.model.agent_request import AgentRequest


@runtime_checkable
class AgentPort(Protocol):
    async def run(self, request: AgentRequest) -> AgentMessage: ...
