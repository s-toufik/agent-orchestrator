from agent_orchestrator.adapter.inbound.web.schema.agent_message_schema import AgentMessageSchema
from agent_orchestrator.adapter.inbound.web.schema.agent_message_stream_schema import (
    AgentMessageStreamSchema,
)
from agent_orchestrator.adapter.inbound.web.schema.message_stream_type import MessageStreamType
from agent_orchestrator.domain.event.turn_event import TurnEvent, TurnEventKind
from agent_orchestrator.domain.workflow.step import Step

STEP_STATUS: dict[Step, str] = {
    Step.UNDERSTAND: "Understanding your request",
    Step.PLAN: "Preparing a plan",
    Step.REVIEW: "Checking the answer",
}


class SsePresenter:
    """Turns what happens in a turn into the SSE events homelab-ui reads."""

    def present(self, event: TurnEvent, session_id: str) -> list[bytes]:
        match event.kind:
            case TurnEventKind.STEP_STARTED:
                status = self._status(event)
                return [_stream(MessageStreamType.STATUS, status)] if status else []
            case TurnEventKind.FINISHED:
                return self._finished(event, session_id)
            case TurnEventKind.FAILED:
                return [_stream(MessageStreamType.ERROR, event.error)]

    @staticmethod
    def complete() -> bytes:
        return _stream(MessageStreamType.COMPLETE, "")

    @staticmethod
    def error(session_id: str, trace: str) -> bytes:
        return AgentMessageSchema(session_id=session_id, content="", error=trace).serialize()

    @staticmethod
    def _status(event: TurnEvent) -> str | None:
        if event.step is Step.RUN_TOOLS:
            return f"Running {', '.join(event.tools)}"
        return STEP_STATUS.get(event.step) if event.step else None

    @staticmethod
    def _finished(event: TurnEvent, session_id: str) -> list[bytes]:
        answer = event.answer
        text = answer.text if answer else ""
        final = AgentMessageSchema(
            session_id=session_id,
            content=text,
            metadata={
                "iteration": str(event.steps_taken),
                "max_iteration": str(event.max_steps),
                "outcome": str(answer.outcome) if answer else "",
            },
        ).serialize()
        return [_stream(MessageStreamType.TOKEN, text), final] if event.stream_answer else [final]


def _stream(event_type: MessageStreamType, content: str) -> bytes:
    return AgentMessageStreamSchema(type=event_type, content=content).serialize()
