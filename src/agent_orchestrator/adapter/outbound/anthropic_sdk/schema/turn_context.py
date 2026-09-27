from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.anthropic_sdk.enum.intent import Intent


class TurnContext(BaseModel):
    intent: Intent
    standalone_query: str
    success_criteria: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
