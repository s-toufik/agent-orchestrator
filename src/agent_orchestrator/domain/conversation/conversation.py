from __future__ import annotations

from dataclasses import dataclass, field

from agent_orchestrator.domain.conversation.message import Message, Speaker
from agent_orchestrator.domain.turn.answer import Answer
from agent_orchestrator.domain.turn.intent import Intent
from agent_orchestrator.domain.turn.plan import Plan
from agent_orchestrator.domain.turn.turn import Turn
from agent_orchestrator.domain.turn.understanding import Understanding


@dataclass
class Conversation:

    id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    pending_plan: Plan | None = None

    def interpret(self, turn: Turn, understanding: Understanding) -> None:
        understanding = understanding.normalized(has_pending_plan=self.pending_plan is not None)
        pending, self.pending_plan = self.pending_plan, None
        if understanding.intent is Intent.PLAN_APPROVAL and pending is not None:
            turn.understood(understanding.about(pending.task))
            turn.execute(pending)
            return
        if understanding.intent.keeps_pending_plan:
            self.pending_plan = pending
        turn.understood(understanding)

    def submit(self, turn: Turn, plan: Plan) -> None:
        if turn.options.auto_approve:
            turn.execute(plan)
            return
        self.propose(plan)
        turn.finish(Answer.approval_request(plan))

    def propose(self, plan: Plan) -> None:
        self.pending_plan = plan

    def close(self, turn: Turn) -> None:
        answer = turn.answer.text if turn.answer else ""
        self.messages.extend(
            [Message(Speaker.USER, turn.request), Message(Speaker.ASSISTANT, answer)]
        )

    def older_than(self, keep_last: int) -> list[Message]:
        return self.messages[:-keep_last] if keep_last else list(self.messages)

    def compact(self, summary: str, keep_last: int) -> None:
        self.summary = summary
        self.messages = self.messages[-keep_last:]
