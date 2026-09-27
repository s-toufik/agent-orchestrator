from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.anthropic_sdk.schema.plan import Plan
from agent_orchestrator.adapter.outbound.anthropic_sdk.store.exchange import Exchange


class AgentState(BaseModel):
    session_id: str
    sdk_session_id: str | None = None
    pending_plan: Plan | None = None
    exchanges: list[Exchange] = Field(default_factory=list)

    def record(self, user: str, assistant: str) -> None:
        self.exchanges = [*self.exchanges, Exchange(user=user, assistant=assistant)]

    def last_exchange(self) -> Exchange | None:
        return self.exchanges[-1] if self.exchanges else None
