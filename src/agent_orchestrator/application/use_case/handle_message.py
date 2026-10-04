import traceback

from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.inbound.agent_request import AgentRequest
from agent_orchestrator.application.port.outbound.model_registry import ModelRegistry
from agent_orchestrator.application.port.outbound.turn_event_stream import TurnEventStream
from agent_orchestrator.application.port.outbound.workflow_runner import WorkflowRunner
from agent_orchestrator.domain.event.turn_event import TurnEvent
from agent_orchestrator.domain.exception.agent_unavailable_exception import (
    AgentUnavailableException,
)
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.turn_options import TurnOptions

AGENT_UNAVAILABLE_MESSAGE: str = (
    "The assistant is temporarily unavailable due to repeated upstream "
    "failures/maintenance/unavailable resources. Please try again in a moment."
)


class HandleMessage:
    def __init__(self, workflow: WorkflowRunner, models: ModelRegistry, logger: Logger) -> None:
        self._workflow = workflow
        self._models = models
        self._logger = logger

    async def handle(self, request: AgentRequest, events: TurnEventStream) -> None:
        try:
            turn = Turn(
                request=request.message,
                model=request.model_name,
                settings=self._models.turn_settings(request.model_name),
                options=TurnOptions(auto_approve=request.auto_approve),
            )
            async for event in self._workflow.run(request.request_id, turn):
                await events.publish(event)
        except AgentUnavailableException as exception:
            self._logger.warning(f"agent unavailable: {exception}")
            await events.publish(TurnEvent.failed(AGENT_UNAVAILABLE_MESSAGE))
        except Exception as exception:
            trace: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"unhandled agent error:\n{trace}")
            await events.publish(TurnEvent.failed(trace))
        finally:
            await events.complete()
