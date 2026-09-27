from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.enum.turn_outcome import TurnOutcome
from agent_orchestrator.adapter.outbound.langgraph.schema.conversation import Conversation
from agent_orchestrator.adapter.outbound.langgraph.schema.plan import Plan
from agent_orchestrator.adapter.outbound.langgraph.schema.reflection_decision import (
    ReflectionDecision,
)
from agent_orchestrator.adapter.outbound.langgraph.schema.turn_context import TurnContext


class TurnState(BaseModel):
    user_message: str = ""
    context: TurnContext | None = None
    plan: Plan | None = None
    scratch: Conversation = Field(default_factory=Conversation)
    reflection: ReflectionDecision | None = None
    iteration: int = 0
    retries: int = 0
    answer: str | None = None
    outcome: TurnOutcome | None = None
