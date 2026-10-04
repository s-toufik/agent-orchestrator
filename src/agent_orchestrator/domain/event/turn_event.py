from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


class TurnEventKind(StrEnum):
    STEP_STARTED = "step_started"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TurnEvent:
    kind: TurnEventKind
    step: Step | None = None
    tools: tuple[str, ...] = field(default_factory=tuple)
    answer: Answer | None = None
    steps_taken: int = 0
    max_steps: int = 0
    stream_answer: bool = False
    error: str = ""

    @classmethod
    def entering(cls, step: Step, turn: Turn) -> TurnEvent:
        draft = turn.last_draft
        tools = tuple(call.name for call in draft.tool_calls) if draft else ()
        return cls(
            TurnEventKind.STEP_STARTED, step=step, tools=tools if step is Step.RUN_TOOLS else ()
        )

    @classmethod
    def finished(cls, turn: Turn) -> TurnEvent:
        return cls(
            TurnEventKind.FINISHED,
            answer=turn.answer,
            steps_taken=turn.steps_taken,
            max_steps=turn.settings.max_steps,
            stream_answer=turn.settings.stream_answer,
        )

    @classmethod
    def failed(cls, error: str) -> TurnEvent:
        return cls(TurnEventKind.FAILED, error=error)
