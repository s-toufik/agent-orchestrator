import contextvars
import traceback

from pycraftcore.logger.port import Logger

from agent.application.port.outbound.agent_port import AgentPort
from agent.application.port.outbound.sse_queue_port import SSEQueuePort
from agent.domain.exception.agent_unavailable_exception import AgentUnavailableException
from agent.domain.model.agent_message import AgentMessage
from agent.domain.model.agent_request import AgentRequest

AGENT_UNAVAILABLE_MESSAGE = (
    "The assistant is temporarily unavailable due to repeated upstream failures/maintenance/unavailable resources. "
    "Please try again in a moment."
)

_current_events: contextvars.ContextVar[SSEQueuePort | None] = contextvars.ContextVar(
    "current_events", default=None
)


async def on_token(value: str) -> None:

    events = _current_events.get()
    if events is not None:
        await events.token(value)


class StreamAgentUseCase:
    def __init__(self, agent: AgentPort, logger: Logger) -> None:
        self._agent = agent
        self._logger = logger

    async def execute(self, request: AgentRequest, events: SSEQueuePort) -> None:
        context_token: contextvars.Token = _current_events.set(events)
        try:
            message: AgentMessage = await self._agent.run(request)
            await events.final(message.content, metadata=message.metadata)
        except AgentUnavailableException as exception:
            self._logger.warning(f"[{request.request_id}] agent unavailable: {exception}")
            await events.error(AGENT_UNAVAILABLE_MESSAGE)
        except Exception as exception:
            traceback_str: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"[{request.request_id}] unhandled agent error:\n{traceback_str}")
            await events.error(traceback_str)
        finally:
            _current_events.reset(context_token)
            await events.complete()
