from dataclasses import dataclass

from agent_orchestrator.domain.conversation.conversation import Conversation
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.outcome import Outcome
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.workflow.step import Step


def names(route: tuple[Step, ...]) -> list[str]:
    return [str(step) for step in route]


@dataclass(frozen=True, slots=True)
class Exchange:
    message: str
    route: tuple[Step, ...]
    turn: Turn
    conversation: Conversation

    @property
    def answer(self) -> Answer | None:
        return self.turn.answer

    @property
    def outcome(self) -> Outcome | None:
        return self.answer.outcome if self.answer else None

    @property
    def intent(self) -> Intent | None:
        return self.turn.intent

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(action.tool for action in self.turn.actions)

    @property
    def plan(self) -> Plan | None:
        return self.turn.plan or self.conversation.pending_plan
