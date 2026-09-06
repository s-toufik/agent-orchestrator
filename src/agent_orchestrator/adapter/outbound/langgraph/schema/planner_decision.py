from pydantic import BaseModel, Field

from agent_orchestrator.adapter.outbound.langgraph.schema.tool_call import ToolCall


class PlannerDecision(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    answer: str | None = None

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0
