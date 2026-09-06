from typing import Protocol, runtime_checkable

from agent_orchestrator.application.port.outbound.sse_queue_port import SSEQueuePort
from agent_orchestrator.domain.model.agent_request import AgentRequest


@runtime_checkable
class StreamAgentPort(Protocol):
    async def execute(self, request: AgentRequest, events: SSEQueuePort) -> None: ...
