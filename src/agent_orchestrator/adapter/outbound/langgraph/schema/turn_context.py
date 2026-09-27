from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.enum.intent import Intent


class TurnContext(BaseModel):
    intent: Intent
    standalone_query: str = Field(
        description="The latest user message rewritten so it makes sense on its own."
    )
    success_criteria: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
