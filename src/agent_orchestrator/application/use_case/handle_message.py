import asyncio
import time
import traceback

from pycraftcore.logger.port import Logger

from agent_orchestrator.application.port.inbound.agent_request import AgentRequest
from agent_orchestrator.application.port.outbound.model_registry import ModelRegistry
from agent_orchestrator.application.port.outbound.turn_event_stream import TurnEventStream
from agent_orchestrator.application.port.outbound.workflow_runner import WorkflowRunner
from agent_orchestrator.domain.event.turn_event import TurnEvent, TurnEventKind
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
        started = time.monotonic()
        try:
            turn = Turn(
                request=request.message,
                model=request.model_name,
                settings=self._models.turn_settings(request.model_name),
                options=TurnOptions(auto_approve=request.auto_approve),
            )
            async for event in self._workflow.run(request.request_id, turn):
                await events.publish(event)
                if event.kind is TurnEventKind.FINISHED:
                    self._logger.info(
                        f"turn finished: {_outcome(event)}, {_summary(event, started)}"
                    )
        except asyncio.CancelledError:
            self._logger.warning(f"turn cancelled after {_elapsed(started)}")
            raise
        except AgentUnavailableException as exception:
            self._logger.warning(f"agent unavailable: {exception}")
            await events.publish(TurnEvent.failed(AGENT_UNAVAILABLE_MESSAGE))
        except Exception as exception:
            trace: str = "".join(traceback.format_exception(exception))
            self._logger.error(f"unhandled agent error:\n{trace}")
            await events.publish(TurnEvent.failed(trace))
        finally:
            await events.complete()


def _outcome(event: TurnEvent) -> str:
    return str(event.answer.outcome) if event.answer else "no answer"


def _summary(event: TurnEvent, started: float) -> str:
    return f"{event.steps_taken}/{event.max_steps} steps in {_elapsed(started)}"


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"
