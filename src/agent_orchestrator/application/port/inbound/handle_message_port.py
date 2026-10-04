from typing import Protocol

from agent_orchestrator.application.port.inbound.agent_request import AgentRequest
from agent_orchestrator.application.port.outbound.turn_event_stream import TurnEventStream


class HandleMessagePort(Protocol):
    async def handle(self, request: AgentRequest, events: TurnEventStream) -> None: ...
