from dataclasses import dataclass, field

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_mode import TurnMode
from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.turn_context import TurnContext
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.agent_state import AgentState


@dataclass
class TurnState:
    user_message: str
    agent_state: AgentState
    context: TurnContext
    mode: TurnMode
    plan: Plan | None = None
    evidence: list[str] = field(default_factory=list)
    tools_used: bool = False
    retries: int = 0
    reflection: ReflectionDecision | None = None
    answer: str | None = None
    outcome: TurnOutcome | None = None
    _draft: list[str] = field(default_factory=list, repr=False)

    @property
    def draft(self) -> str:
        return "".join(self._draft).strip()

    def write(self, text: str) -> None:
        self._draft.append(text)

    def restart_draft(self) -> None:
        self._draft = []
