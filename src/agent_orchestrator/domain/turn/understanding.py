from __future__ import annotations

from dataclasses import dataclass, field, replace

from agent_orchestrator.domain.turn.intent import Intent


@dataclass(frozen=True, slots=True)
class Understanding:
    intent: Intent
    query: str
    success_criteria: tuple[str, ...] = field(default_factory=tuple)
    clarification_question: str | None = None

    @classmethod
    def fallback(cls, message: str) -> Understanding:
        # Treating an unreadable message as a task is safe: it leads to a plan to approve.
        return cls(intent=Intent.TASK, query=message)

    def normalized(self, has_pending_plan: bool) -> Understanding:
        if self.intent is Intent.PLAN_APPROVAL and not has_pending_plan:
            return replace(self, intent=Intent.DIRECT)
        if self.intent is Intent.AMBIGUOUS and not self.clarification_question:
            return replace(self, intent=Intent.TASK)
        return self

    def about(self, query: str) -> Understanding:
        return replace(self, query=query)
