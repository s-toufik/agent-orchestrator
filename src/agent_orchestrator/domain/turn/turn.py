from __future__ import annotations

from dataclasses import dataclass, field

from agent_orchestrator.domain.tool.tool_result import ToolResult
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.draft import Draft
from agent_orchestrator.domain.turn.feedback import Feedback
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn_settings import TurnSettings
from agent_orchestrator.domain.turn.understanding import Understanding
from agent_orchestrator.domain.turn.verdict import Verdict

WorkItem = Draft | ToolResult | Feedback


@dataclass
class Turn:
    request: str
    model: str
    settings: TurnSettings = field(default_factory=TurnSettings)
    understanding: Understanding | None = None
    plan: Plan | None = None
    work: list[WorkItem] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    answer: Answer | None = None

    # ------------------------------------------------------------------ changes
    def understood(self, understanding: Understanding) -> None:
        self.understanding = understanding

    def execute(self, plan: Plan) -> None:
        self.plan = plan

    def drafted(self, draft: Draft) -> None:
        self.work.append(draft)

    def observed(self, results: list[ToolResult]) -> None:
        self.work.extend(results)

    def reviewed(self, verdict: Verdict) -> None:
        self.verdicts.append(verdict)

    def give_feedback(self, critique: str) -> None:
        self.work.append(Feedback(critique))

    def finish(self, answer: Answer) -> None:
        self.answer = answer

    def conclude(self) -> Answer:
        draft = self.last_draft
        if draft is None:
            return Answer.nothing()
        if draft.asks_for_tools:
            return Answer.budget_exhausted()
        if self.last_verdict is not None and self.last_verdict.rejects:
            return Answer.best_effort(draft.text)
        return Answer.answered(draft.text)

    # ---------------------------------------------------------------- questions
    @property
    def query(self) -> str:
        return self.understanding.query if self.understanding else self.request

    @property
    def intent(self) -> Intent | None:
        return self.understanding.intent if self.understanding else None

    @property
    def drafts(self) -> list[Draft]:
        return [item for item in self.work if isinstance(item, Draft)]

    @property
    def evidence(self) -> list[ToolResult]:
        return [item for item in self.work if isinstance(item, ToolResult)]

    @property
    def last_draft(self) -> Draft | None:
        drafts = self.drafts
        return drafts[-1] if drafts else None

    @property
    def last_verdict(self) -> Verdict | None:
        return self.verdicts[-1] if self.verdicts else None

    @property
    def steps_taken(self) -> int:
        return len(self.drafts)

    @property
    def retries(self) -> int:
        return sum(1 for item in self.work if isinstance(item, Feedback))

    @property
    def used_tools(self) -> bool:
        return bool(self.evidence)

    @property
    def is_plain_direct_answer(self) -> bool:
        # A direct answer made without tools has no evidence to be checked against.
        return self.intent is Intent.DIRECT and not self.used_tools
