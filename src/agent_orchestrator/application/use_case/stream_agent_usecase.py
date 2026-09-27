import traceback

from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.outbound.agent_port import AgentPort
from agent_orchestrator.application.port.outbound.event_stream_port import EventStreamPort
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.model.agent_message_stream import AgentMessageStream
from agent_orchestrator.domain.model.agent_request import AgentRequest

AGENT_UNAVAILABLE_MESSAGE = (
    "The assistant is temporarily unavailable due to repeated upstream failures/maintenance/unavailable resources. "
    "Please try again in a moment."
)


class StreamAgentUseCase:
    def __init__(self, agent: AgentPort, logger: Logger) -> None:
        self._agent = agent
        self._logger = logger

    async def execute(self, request: AgentRequest, events: EventStreamPort) -> None:
        try:
            async for event in self._agent.stream(request):
                await events.publish(event)
        except AgentUnavailableException as exception:
            self._logger.warning(f"[{request.request_id}] agent unavailable: {exception}")
            await events.publish(AgentMessageStream.error(AGENT_UNAVAILABLE_MESSAGE))
        except Exception as exception:
            traceback_str: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"[{request.request_id}] unhandled agent error:\n{traceback_str}")
            await events.publish(AgentMessageStream.error(traceback_str))
        finally:
            await events.complete()
