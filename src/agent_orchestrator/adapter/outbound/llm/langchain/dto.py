"""What the structured-output models return, mapped onto the domain."""

from pydantic import BaseModel, Field

from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict, VerdictAction


class UnderstandingDto(BaseModel):
    intent: Intent
    standalone_query: str = Field(
        description="The latest user message rewritten so it makes sense on its own."
    )
    success_criteria: list[str] = Field(default_factory=list)
    clarification_question: str | None = None

    def to_domain(self) -> Understanding:
        return Understanding(
            intent=self.intent,
            query=self.standalone_query,
            success_criteria=tuple(self.success_criteria),
            clarification_question=self.clarification_question,
        )


class VerdictDto(BaseModel):
    action: VerdictAction
    critique: str = ""

    def to_domain(self) -> Verdict:
        return Verdict(self.action, self.critique)
