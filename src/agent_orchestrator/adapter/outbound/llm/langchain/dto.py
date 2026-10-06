from collections.abc import Collection

from pydantic import BaseModel, Field

from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.plan import Plan, PlannedStep
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


class PlanStepDto(BaseModel):
    action: str = Field(description="What this step does, in one sentence.")
    tool: str | None = Field(
        default=None, description="The one tool this step calls, or null if it calls none."
    )


class PlanDto(BaseModel):
    steps: list[PlanStepDto]
    expected_result: str = Field(description="One sentence: what the user gets.")

    def to_domain(self, task: str, tools: Collection[str]) -> Plan:
        return Plan(
            task=task,
            steps=tuple(
                PlannedStep(step.action, step.tool if step.tool in tools else None)
                for step in self.steps
            ),
            expected_result=self.expected_result,
        )
